"""
zone_extraction.py
==================
Connected-component zone extraction: turns a labelled voxel grid into a graph
with the same schema as the generator (graph_builder.py).

  1. Detect the hull / superstructure boundary (footprint heuristic)
  2. Connected-component zone extraction per Comp class; zone types
     (AFT_PEAK, ENGINE_REGION, etc.) inferred from spatial position
  3. Node features x (8 + N_ZONE_TYPES dims) — geometry incl. cy_norm
  4. Node labels y
  5. Edges from voxel-level adjacency, typed by boundary faces (7 dims)
  6. Conditioning vector cond (18 dims)
  7. Package as PyG Data with matching metadata fields

The real general arrangements in ``data/real_ga`` were converted to graphs
with these steps; ``cc_reextract`` applies them to generated arrangements.

Coordinate convention (matches the generator):
    x: 0 = aft, nx-1 = bow
    y: 0 = port, ny-1 = starboard
    z: 0 = keel, increasing upward
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict

import scipy.ndimage as ndi

try:
    import torch
    from torch_geometric.data import Data
    HAS_PYG = True
except ImportError:
    HAS_PYG = False
    torch = None
    Data = None

from validation_constants import (
    Comp, ShipType, ZoneType, N_ZONE_TYPES, N_NODE_FEATURES, NORM_L, NORM_B,
    NORM_D, BUDGET_KEYS, EDGE_LONGITUDINAL, EDGE_VERTICAL, EDGE_SS_HULL,
    EDGE_TRANSVERSE, N_EDGE_FEATURES, MIN_ZONE_VOXELS, SPATIAL_RULES,
    SS_FOOTPRINT_THRESHOLD, SS_ACCOM_THRESHOLD, SS_ACCOM_FOOTPRINT_THRESHOLD,
    N_SHIP_TYPES,
)



def _norm(val: float, lo: float, hi: float) -> float:
    return float(np.clip((val - lo) / max(hi - lo, 1e-6), 0.0, 1.0))


# ─────────────────────────────────────────────────────────────────
# Zone dataclass
# ─────────────────────────────────────────────────────────────────

@dataclass
class ConvertedZone:
    zone_id:          int
    zone_type:        ZoneType
    comp:             Comp        # assigned compartment label
    x0: int;  x1: int
    y0: int;  y1: int
    z0: int;  z1: int
    voxel_indices:    np.ndarray  # (N, 3) array of (ix, iy, iz)
    available_cells:  int
    cx_norm:          float
    cy_norm:          float
    cz_norm:          float
    hull_avail:       float
    zone_length_norm: float
    zone_width_norm:  float
    zone_height_norm: float
    deck_idx:         int


# ─────────────────────────────────────────────────────────────────
# Step 1: Detect hull / superstructure boundary
# ─────────────────────────────────────────────────────────────────

def detect_ss_boundary(
    label_grid: np.ndarray,
    hull_info: dict,
    footprint_threshold: float = SS_FOOTPRINT_THRESHOLD,
    accom_threshold: float = SS_ACCOM_THRESHOLD,
    accom_footprint_threshold: float = SS_ACCOM_FOOTPRINT_THRESHOLD,
) -> Tuple[int, float]:
    """
    Detect the hull-superstructure boundary using dual criteria.

    Two independent triggers (earliest one wins):

    Criterion A — Footprint drop:
        The first level where occupied xy-footprint drops below
        `footprint_threshold` (40%) of the maximum footprint seen
        in the hull levels below. Works well for transports/OSVs
        with sharp structural step-changes.

    Criterion B — Accommodation dominance:
        The first level where accommodation fraction exceeds
        `accom_threshold` (60%) AND footprint is below
        `accom_footprint_threshold` (85%) of maximum. Catches
        yacht-type ships where the hull tapers gradually but
        the functional transition to superstructure (accommodation-
        dominant decks) is clear.

    The hull/SS boundary is set to the MINIMUM of the two,
    ensuring we catch gradual tapers that pure footprint misses.

    Returns
    -------
    nz_hull : int   — number of hull deck levels (iz = 0..nz_hull-1)
    D_hull  : float — hull depth in metres (nz_hull * gs)
    """
    nx, ny, nz = label_grid.shape
    gs = hull_info["gs"]

    # Per-level statistics
    occupied_per_level = np.zeros(nz, dtype=int)
    accom_frac_per_level = np.zeros(nz, dtype=float)

    for iz in range(nz):
        level_slice = label_grid[:, :, iz]
        occupied = np.sum(level_slice != Comp.EMPTY)
        occupied_per_level[iz] = occupied
        if occupied > 0:
            accom_count = np.sum(level_slice == Comp.ACCOMMODATION)
            accom_frac_per_level[iz] = accom_count / occupied

    if occupied_per_level.max() == 0:
        return nz, nz * gs

    max_footprint = occupied_per_level.max()

    # Criterion A: footprint drop
    nz_hull_footprint = nz
    running_max = 0
    for iz in range(nz):
        occ = occupied_per_level[iz]
        if occ == 0:
            nz_hull_footprint = iz
            break
        if iz == 0:
            running_max = occ
            continue
        running_max = max(running_max, occupied_per_level[iz - 1])
        if occ < footprint_threshold * running_max and iz >= 2:
            nz_hull_footprint = iz
            break

    # Criterion B: accommodation dominance
    nz_hull_accom = nz
    for iz in range(2, nz):  # skip bottom levels (DB, lower hold)
        fp_ratio = occupied_per_level[iz] / max(max_footprint, 1)
        if (accom_frac_per_level[iz] > accom_threshold and
                fp_ratio < accom_footprint_threshold):
            nz_hull_accom = iz
            break

    # Take earliest trigger
    nz_hull = min(nz_hull_footprint, nz_hull_accom)

    # Safety: at least 2 hull levels
    nz_hull = max(nz_hull, 2)

    D_hull = nz_hull * gs
    return nz_hull, D_hull


# ─────────────────────────────────────────────────────────────────
# Step 2: Connected-component zone extraction
# ─────────────────────────────────────────────────────────────────

def extract_zones(
    label_grid: np.ndarray,
    hull_info: dict,
    nz_hull: int,
) -> List[ConvertedZone]:
    """
    Extract zones via connected-component labelling per Comp class.

    Each connected component of same-label voxels becomes one zone.
    Zone types are inferred from spatial position.
    Small zones (< MIN_ZONE_VOXELS) are merged into their nearest
    same-label neighbor.
    """
    nx, ny, nz = label_grid.shape
    gs = hull_info["gs"]
    L = hull_info["L"]
    nz_total = nz   # total including SS

    # 6-connected structure for 3D
    structure = ndi.generate_binary_structure(3, 1)

    zones: List[ConvertedZone] = []
    zone_id = 0

    # Process each Comp class except EMPTY
    active_comps = set(np.unique(label_grid)) - {Comp.EMPTY}

    for comp_val in sorted(active_comps):
        comp = Comp(comp_val)
        binary_mask = (label_grid == comp_val)
        labelled, n_components = ndi.label(binary_mask, structure=structure)

        for cc in range(1, n_components + 1):
            voxel_mask = (labelled == cc)
            voxel_indices = np.argwhere(voxel_mask)  # (N, 3)

            if len(voxel_indices) < MIN_ZONE_VOXELS:
                continue  # skip tiny fragments

            # Bounding box
            x0, y0, z0 = voxel_indices.min(axis=0)
            x1, y1, z1 = voxel_indices.max(axis=0) + 1  # exclusive

            # Centroid (normalised)
            cx_norm = float(voxel_indices[:, 0].mean() + 0.5) / nx
            cy_norm = float(voxel_indices[:, 1].mean() + 0.5) / ny
            cz_norm = float(voxel_indices[:, 2].mean() + 0.5) / nz_total

            # Available cells and hull availability
            available = len(voxel_indices)
            total_bb = (x1 - x0) * (y1 - y0) * (z1 - z0)
            hull_avail = available / max(1, total_bb)

            # Normalised extents
            zone_length_norm = (x1 - x0) / nx
            zone_width_norm  = (y1 - y0) / ny
            zone_height_norm = (z1 - z0) / nz_total

            # Deck index — based on the lowest z level of this zone
            # relative to the hull/SS boundary
            deck_idx = _infer_deck_idx(z0, z1, nz_hull)

            # Zone type — inferred from spatial position and comp
            zone_type = _infer_zone_type(
                cx_norm, cz_norm, z0, z1,
                nz_hull, deck_idx, comp
            )

            zones.append(ConvertedZone(
                zone_id=zone_id,
                zone_type=zone_type,
                comp=comp,
                x0=int(x0), x1=int(x1),
                y0=int(y0), y1=int(y1),
                z0=int(z0), z1=int(z1),
                voxel_indices=voxel_indices,
                available_cells=available,
                cx_norm=round(cx_norm, 5),
                cy_norm=round(cy_norm, 5),
                cz_norm=round(cz_norm, 5),
                hull_avail=round(hull_avail, 5),
                zone_length_norm=round(zone_length_norm, 5),
                zone_width_norm=round(zone_width_norm, 5),
                zone_height_norm=round(zone_height_norm, 5),
                deck_idx=deck_idx,
            ))
            zone_id += 1

    return zones


def _infer_deck_idx(z0: int, z1: int, nz_hull: int) -> int:
    """
    Infer a deck index for the zone.

    Matches the synthetic convention:
      0 = double bottom (z0 == 0)
      1 = lower hold
      2 = upper hold
      3+ = superstructure levels
    """
    if z0 >= nz_hull:
        # Superstructure: 3 + level above hull
        return 3 + (z0 - nz_hull)

    # Hull zones — DB is always iz=0
    if z0 == 0:
        return 0   # double bottom

    # Approximate lower/upper split: lower if below 40% of hull
    mid_z = (z0 + z1) / 2
    frac = mid_z / max(nz_hull, 1)
    if frac < SPATIAL_RULES["acc_min_deck_frac"]:
        return 1   # lower hold
    return 2       # upper hold


def _infer_zone_type(
    cx_norm: float,
    cz_norm: float,
    z0: int, z1: int,
    nz_hull: int,
    deck_idx: int,
    comp: Comp,
) -> ZoneType:
    """
    Infer the ZoneType from spatial position and compartment label.

    Priority order (matches synthetic logic):
      1. Superstructure (above hull)
      2. Aft peak (extreme aft)
      3. Forward peak (extreme forward)
      4. Engine upper (top hull tier over ER x-band)
      5. Engine region (aft zone with ER/machinery/fuel)
      6. Main DB / lower / upper by deck index
    """
    # 1. Superstructure
    if z0 >= nz_hull:
        return ZoneType.SUPERSTRUCTURE

    # 2. Aft peak — extreme aft position + steering gear or ballast
    if cx_norm < SPATIAL_RULES["aft_peak_max_cx"] + 0.02:
        if comp in (Comp.STEERING_GEAR, Comp.BALLAST_TANKS, Comp.VOID):
            return ZoneType.AFT_PEAK

    # 3. Forward peak — extreme forward position
    if cx_norm > SPATIAL_RULES["fwd_peak_min_cx"] - 0.02:
        if comp in (Comp.BALLAST_TANKS, Comp.VOID):
            return ZoneType.FWD_PEAK

    # 4. Engine upper — top hull hold over ER longitudinal band
    if z0 < nz_hull and cx_norm < SPATIAL_RULES["er_max_cx"] + 0.05:
        if deck_idx >= 2 and comp in (
            Comp.CARGO, Comp.ACCOMMODATION, Comp.MACHINERY, Comp.VOID,
        ):
            return ZoneType.ENGINE_UPPER

    # 5. Engine region — aft hull with ER / machinery / fuel
    if cx_norm < SPATIAL_RULES["er_max_cx"] + 0.05:
        if comp in (Comp.ENGINE_ROOM, Comp.MACHINERY, Comp.FUEL_TANKS):
            return ZoneType.ENGINE_REGION

    # 6. Main hull zones by deck index
    if deck_idx == 0:
        return ZoneType.MAIN_DB
    if deck_idx == 1:
        return ZoneType.MAIN_LOWER
    return ZoneType.MAIN_UPPER


# ─────────────────────────────────────────────────────────────────
# Step 3: Build node features (matches graph_builder.N_NODE_FEATURES)
# ─────────────────────────────────────────────────────────────────

def build_node_features(zones: List[ConvertedZone], nz_total: int) -> np.ndarray:
    """
    Build (n_zones, N_NODE_FEATURES) node input features — geometry only, no labels.

    Matches graph_builder._build_node_input_features:
        [0]   cx_norm
        [1]   cy_norm
        [2]   cz_norm
        [3]   hull_avail
        [4]   zone_length_norm
        [5]   zone_width_norm
        [6]   zone_height_norm
        [7]   deck_idx_norm
        [8:8+N_ZONE_TYPES] zone_type_onehot
    """
    n = len(zones)
    max_deck = max((z.deck_idx for z in zones), default=1)
    max_deck = max(max_deck, 1)

    features = np.zeros((n, N_NODE_FEATURES), dtype=np.float32)
    for i, z in enumerate(zones):
        features[i, 0] = z.cx_norm
        features[i, 1] = z.cy_norm
        features[i, 2] = z.cz_norm
        features[i, 3] = z.hull_avail
        features[i, 4] = z.zone_length_norm
        features[i, 5] = z.zone_width_norm
        features[i, 6] = z.zone_height_norm
        features[i, 7] = z.deck_idx / max_deck

        zt_idx = int(z.zone_type)
        if 0 <= zt_idx < N_ZONE_TYPES:
            features[i, 8 + zt_idx] = 1.0

    return features


# ─────────────────────────────────────────────────────────────────
# Step 4: Build node labels
# ─────────────────────────────────────────────────────────────────

def build_node_labels(zones: List[ConvertedZone]) -> np.ndarray:
    return np.array([z.comp.value for z in zones], dtype=np.int64)


# ─────────────────────────────────────────────────────────────────
# Step 5: Build edges from voxel-level adjacency
# ─────────────────────────────────────────────────────────────────

def build_edges(
    zones: List[ConvertedZone],
    label_grid: np.ndarray,
    nz_hull: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build edge_index (2, n_edges) and edge_attr (n_edges, N_EDGE_FEATURES).

    Two zones are adjacent if any of their voxels are 6-connected.
    Edge schema matches graph_builder:
        [0] dx_norm         signed x-centroid separation
        [1] dy_norm         signed y-centroid separation
        [2] dz_norm         signed z-centroid separation
        [3] is_longitudinal (0/1)
        [4] is_vertical     (0/1)
        [5] is_ss_hull      (0/1)
        [6] is_transverse   (0/1)
    """
    nx, ny, nz = label_grid.shape
    n = len(zones)

    # Build a zone_id grid for efficient adjacency detection
    zone_grid = np.full((nx, ny, nz), -1, dtype=np.int32)
    for z in zones:
        for vi in range(len(z.voxel_indices)):
            ix, iy, iz = z.voxel_indices[vi]
            zone_grid[ix, iy, iz] = z.zone_id

    # Find adjacent zone pairs by scanning all voxels
    # For each voxel, check its 6 neighbors; if neighbor belongs to
    # a different zone, record the adjacency and boundary direction
    adj_info: Dict[Tuple[int, int], Dict[str, int]] = defaultdict(
        lambda: {"x_faces": 0, "y_faces": 0, "z_faces": 0}
    )

    # 6-connectivity offsets
    offsets = [(1,0,0), (-1,0,0), (0,1,0), (0,-1,0), (0,0,1), (0,0,-1)]

    for z in zones:
        zid = z.zone_id
        for vi in range(len(z.voxel_indices)):
            ix, iy, iz = z.voxel_indices[vi]
            for dx, dy, dz in offsets:
                nx_, ny_, nz_ = ix+dx, iy+dy, iz+dz
                if 0 <= nx_ < nx and 0 <= ny_ < ny and 0 <= nz_ < nz:
                    neighbor_zid = zone_grid[nx_, ny_, nz_]
                    if neighbor_zid >= 0 and neighbor_zid != zid:
                        pair = (min(zid, neighbor_zid), max(zid, neighbor_zid))
                        if dx != 0:
                            adj_info[pair]["x_faces"] += 1
                        elif dy != 0:
                            adj_info[pair]["y_faces"] += 1
                        else:
                            adj_info[pair]["z_faces"] += 1

    # Build zone lookup
    zone_by_id = {z.zone_id: z for z in zones}

    src: List[int] = []
    dst: List[int] = []
    attrs: List[List[float]] = []

    # Map zone_id to index in zones list
    id_to_idx = {z.zone_id: i for i, z in enumerate(zones)}

    for (za_id, zb_id), faces in adj_info.items():
        za = zone_by_id[za_id]
        zb = zone_by_id[zb_id]

        i = id_to_idx[za_id]
        j = id_to_idx[zb_id]

        dx = zb.cx_norm - za.cx_norm
        dy = zb.cy_norm - za.cy_norm
        dz = zb.cz_norm - za.cz_norm

        # Classify edge type
        edge_type = _classify_edge_from_faces(za, zb, faces, nz_hull)
        type_vec = [0.0, 0.0, 0.0, 0.0]
        type_vec[edge_type] = 1.0

        # Add both directions (undirected)
        src.extend([i, j])
        dst.extend([j, i])
        attrs.append([dx, dy, dz] + type_vec)
        attrs.append([-dx, -dy, -dz] + type_vec)

    # Isolated-node fallback (match graph_builder)
    connected = set(src)
    for i, za in enumerate(zones):
        if i in connected:
            continue
        best_j, best_d = -1, 1e18
        for j, zb in enumerate(zones):
            if j == i:
                continue
            d = ((za.cx_norm - zb.cx_norm) ** 2
                 + (za.cy_norm - zb.cy_norm) ** 2
                 + (za.cz_norm - zb.cz_norm) ** 2)
            if d < best_d:
                best_d, best_j = d, j
        if best_j < 0:
            continue
        zb = zones[best_j]
        dx = zb.cx_norm - za.cx_norm
        dy = zb.cy_norm - za.cy_norm
        dz = zb.cz_norm - za.cz_norm
        adx, ady, adz = abs(dx), abs(dy), abs(dz)
        if adx >= ady and adx >= adz:
            et = EDGE_LONGITUDINAL
        elif adz >= ady:
            et = EDGE_VERTICAL
        else:
            et = EDGE_TRANSVERSE
        tv = [0.0, 0.0, 0.0, 0.0]
        tv[et] = 1.0
        src.extend([i, best_j])
        dst.extend([best_j, i])
        attrs.append([dx, dy, dz] + tv)
        attrs.append([-dx, -dy, -dz] + tv)
        connected.add(i)
        connected.add(best_j)

    if not src:
        return (np.zeros((2, 0), dtype=np.int64),
                np.zeros((0, N_EDGE_FEATURES), dtype=np.float32))

    edge_index = np.array([src, dst], dtype=np.int64)
    edge_attr = np.array(attrs, dtype=np.float32)
    return edge_index, edge_attr


def _classify_edge_from_faces(
    za: ConvertedZone,
    zb: ConvertedZone,
    faces: Dict[str, int],
    nz_hull: int,
) -> int:
    """
    Classify edge type from boundary face counts (matches graph_builder).
    """
    a_ss = za.zone_type == ZoneType.SUPERSTRUCTURE
    b_ss = zb.zone_type == ZoneType.SUPERSTRUCTURE

    if a_ss != b_ss:
        return EDGE_SS_HULL

    y_only = (
        faces["y_faces"] > 0
        and faces["x_faces"] == 0
        and faces["z_faces"] == 0
    )
    if y_only or (
        faces["y_faces"] > faces["x_faces"]
        and faces["y_faces"] >= faces["z_faces"]
    ):
        return EDGE_TRANSVERSE

    if faces["z_faces"] > faces["x_faces"] + faces["y_faces"]:
        return EDGE_VERTICAL
    return EDGE_LONGITUDINAL


# ─────────────────────────────────────────────────────────────────
# Step 6: Build conditioning vector (18 dims)
# ─────────────────────────────────────────────────────────────────

def build_conditioning(
    ship_type: ShipType,
    L: float, B: float, D_hull: float,
    lcg_frac: float, kg_frac: float,
    budget_fracs: Dict[str, float],
) -> np.ndarray:
    """
    Build global conditioning vector (18 dims).
    Matches _build_conditioning in graph_builder.py.

        [0-5]   ship_type one-hot (6 dims)
        [6]     L_norm
        [7]     B_norm
        [8]     D_norm
        [9]     target_lcg_frac  (= actual for real GAs)
        [10]    target_kg_frac   (= actual for real GAs)
        [11-17] budget fractions (7 dims)
    """
    cond = np.zeros(18, dtype=np.float32)

    # Ship type one-hot
    st_idx = int(ship_type)
    if 0 <= st_idx < N_SHIP_TYPES:
        cond[st_idx] = 1.0

    # Normalised dimensions
    cond[6] = _norm(L, *NORM_L)
    cond[7] = _norm(B, *NORM_B)
    cond[8] = _norm(D_hull, *NORM_D)

    # Physics targets (for real GAs, target = actual)
    cond[9]  = lcg_frac
    cond[10] = kg_frac

    # Budget fractions
    for k, key in enumerate(BUDGET_KEYS):
        cond[11 + k] = budget_fracs.get(key, 0.0)

    return cond


# ─────────────────────────────────────────────────────────────────
# Step 7: Package as PyG Data
# ─────────────────────────────────────────────────────────────────
def package_graph(
    x: np.ndarray,
    y: np.ndarray,
    edge_index: np.ndarray,
    edge_attr: np.ndarray,
    cond: np.ndarray,
    ship_type: int,
    n_zones: int,
    lcg_actual: float,
    kg_actual: float,
    gm_t: float,
    ship_name: str,
    voxel_labels: Optional[np.ndarray] = None,
    voxel_hull_mask: Optional[np.ndarray] = None,
    voxel_volume_m3: Optional[np.ndarray] = None,
    native_grid_shape: Optional[Tuple[int, int, int]] = None,
    measurement_basis: str = "native_volume",
) -> Any:
    """Package arrays into a PyG Data object or plain dict."""
    meta = {
        "ship_type":  ship_type,
        "n_zones":    n_zones,
        "lcg_actual": lcg_actual,
        "kg_actual":  kg_actual,
        "gm_t":       gm_t,
        "hull_source": "real_ga",
        "ship_name":  ship_name,
        "measurement_basis": measurement_basis,
    }

    if native_grid_shape is not None:
        meta["native_grid_shape"] = tuple(int(v) for v in native_grid_shape)

    if HAS_PYG:
        data = Data(
            x=torch.tensor(x, dtype=torch.float32),
            y=torch.tensor(y, dtype=torch.long),
            edge_index=torch.tensor(edge_index, dtype=torch.long),
            edge_attr=torch.tensor(edge_attr, dtype=torch.float32),
            cond=torch.tensor(cond, dtype=torch.float32),
        )
        for k, v in meta.items():
            setattr(data, k, v)

        if voxel_labels is not None:
            data.voxel_labels = torch.tensor(voxel_labels, dtype=torch.int8)
        if voxel_hull_mask is not None:
            data.voxel_hull_mask = torch.tensor(voxel_hull_mask, dtype=torch.bool)
        if voxel_volume_m3 is not None:
            data.voxel_volume_m3 = torch.tensor(voxel_volume_m3, dtype=torch.float32)

        return data

    else:
        out = {
            "x": x,
            "y": y,
            "edge_index": edge_index,
            "edge_attr": edge_attr,
            "cond": cond,
            **meta,
        }
        if voxel_labels is not None:
            out["voxel_labels"] = voxel_labels
        if voxel_hull_mask is not None:
            out["voxel_hull_mask"] = voxel_hull_mask
        if voxel_volume_m3 is not None:
            out["voxel_volume_m3"] = voxel_volume_m3
        return out

# ─────────────────────────────────────────────────────────────────
# Validation
# ─────────────────────────────────────────────────────────────────

def validate_converted_graph(graph: Any) -> Tuple[bool, List[str]]:
    """Run the same sanity checks as graph_builder.validate_graph."""
    warns: List[str] = []

    if HAS_PYG:
        x = graph.x.numpy()
        ei = graph.edge_index.numpy()
        ea = graph.edge_attr.numpy()
        y_lbl = graph.y.numpy()
        cond = graph.cond.numpy()
    else:
        x = graph["x"]
        ei = graph["edge_index"]
        ea = graph["edge_attr"]
        y_lbl = graph["y"]
        cond = graph["cond"]

    n = x.shape[0]

    if x.shape[1] != N_NODE_FEATURES:
        warns.append(
            f"Node features: {x.shape[1]} dims, expected {N_NODE_FEATURES}"
        )
    if np.isnan(x).any() or np.isinf(x).any():
        warns.append("NaN/Inf in node features")
    if ea.size > 0 and (np.isnan(ea).any() or np.isinf(ea).any()):
        warns.append("NaN/Inf in edge attributes")
    if ei.size > 0:
        if ei.max() >= n:
            warns.append(f"Edge index max {ei.max()} >= n_nodes {n}")
        if ei.min() < 0:
            warns.append(f"Edge index min {ei.min()} < 0")
        connected = set(ei[0].tolist()) | set(ei[1].tolist())
        isolated = set(range(n)) - connected
        if isolated:
            warns.append(f"{len(isolated)} isolated nodes: {sorted(isolated)[:5]}")
    n_edges = ei.shape[1] if ei.ndim == 2 else 0
    if n_edges > 0 and ea.shape != (n_edges, N_EDGE_FEATURES):
        warns.append(f"Edge attr shape {ea.shape}, expected ({n_edges}, {N_EDGE_FEATURES})")
    if y_lbl.min() < 0 or y_lbl.max() >= int(Comp.EMPTY):
        warns.append(f"Labels out of range: [{y_lbl.min()}, {y_lbl.max()}]")
    if cond.shape != (18,):
        warns.append(f"Cond shape {cond.shape}, expected (18,)")
    if n < 3:
        warns.append(f"Only {n} zones (min expected ~8)")

    return len(warns) == 0, warns
