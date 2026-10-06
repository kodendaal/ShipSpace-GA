"""
graph_builder.py
================
Stage 5: Graph Construction for Ship Layout Generation
-------------------------------------------------------
Converts a BulkheadLayout + CompartmentAssignment into a graph suitable
for training a VAE/diffusion model on the compartment label space.

Architecture alignment
----------------------
The target generative model is a **graph-conditional VAE or diffusion model**:

    Encoder:   (graph_structure, node_labels)  -->  latent z
    Decoder:   (graph_structure, latent z)      -->  node_labels
    Condition: graph-level requirements (ship type, dimensions, targets)

This means the graph has a clean separation between:

    x_input  — FIXED geometric features per node (what the model conditions on).
               These describe WHERE each zone is, HOW BIG it is, and WHAT TYPE
               of zone it is.  They do NOT contain the compartment label.

    y        — TARGET labels per node (what the model generates).
               One Comp class index per zone.  During training the encoder
               sees these; during inference the decoder produces them.

    cond     — GLOBAL conditioning vector (ship-level requirements).
               Ship type (one-hot), normalised dimensions, physics targets,
               volume budget fractions.  Fed into both encoder and decoder.

    edge_index, edge_attr — graph topology and edge features (fixed).

Each ship instance is one torch_geometric.data.Data object (or plain dict
if PyG is not installed).

Coordinate convention (inherited from Stages 1-3):
    x : 0 = aft,   nx-1 = bow
    y : 0 = port,  ny-1 = starboard
    z : 0 = keel,  increasing upward

Dependencies: numpy, (optional: torch, torch_geometric)
"""

from __future__ import annotations
import numpy as np
from typing import Any, List, Tuple

from ship_params import (
    ShipParameterization, ShipType, Comp,
)
from bulkhead_placement import (
    BulkheadLayout, ZoneType,
)
from compartment_assignment import (
    CompartmentAssignment,
    QC_KG_TOL,
    QC_LCG_TOL,
    lcg_kg_tracking_metrics,
)

# Try importing PyG; fall back to plain dict if unavailable.
try:
    import torch
    from torch_geometric.data import Data
    HAS_PYG = True
except ImportError:
    HAS_PYG = False
    torch = None
    Data = None


# ─────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────

# Number of ZoneType classes (for one-hot encoding)
N_ZONE_TYPES = len(ZoneType)

# Node input: scalar geometry (8) + zone-type one-hot (N_ZONE_TYPES)
N_NODE_FEATURE_SCALAR = 8
N_NODE_FEATURES = N_NODE_FEATURE_SCALAR + N_ZONE_TYPES

# Number of Comp classes (for label encoding)
N_COMP_CLASSES = len(Comp)     # 11: VOID .. EMPTY (NAVIGATION=9, EMPTY=10)

# Number of ShipType classes (for conditioning one-hot)
N_SHIP_TYPES = len(ShipType)   # 6: BULKER .. YACHT

# Number of budget keys (for conditioning vector)
BUDGET_KEYS = [
    "engine", "machinery", "cargo", "stores",
    "accommodation", "fuel_tanks", "ballast_tanks",
]
N_BUDGET_KEYS = len(BUDGET_KEYS)

# Conditioning contract: 6 ship types + 3 dimensions + 7 budgets.
N_COND_DIMS = N_SHIP_TYPES + 3 + N_BUDGET_KEYS
GENERATOR_VERSION = "v5.4-voidfix-hardened"

# Normalisation ranges for L, B, D (union of all ship type ranges)
NORM_L = (51.0, 280.0)
NORM_B = (9.0, 54.0)
NORM_D = (6.0, 36.0)

# Edge type indices (one-hot in edge_attr cols 3–6)
EDGE_LONGITUDINAL = 0
EDGE_VERTICAL = 1
EDGE_SS_HULL = 2
EDGE_TRANSVERSE = 3
N_EDGE_FEATURES = 7


def _norm(val: float, lo: float, hi: float) -> float:
    """Normalise val to [0, 1] given range [lo, hi]."""
    return float(np.clip((val - lo) / max(hi - lo, 1e-6), 0.0, 1.0))


# ─────────────────────────────────────────────────────────────────
# Node input features (geometric — NO labels)
# ─────────────────────────────────────────────────────────────────

def _build_node_input_features(layout: BulkheadLayout) -> np.ndarray:
    """
    Build (n_zones, N_NODE_FEATURES) input feature matrix.

    These features describe the graph STRUCTURE — zone geometry and type.
    They are the conditioning signal for each node.  No compartment
    assignment information is included here.

    Feature vector per zone (8 + N_ZONE_TYPES dims):
        [0]   cx_norm           — longitudinal centroid (0=aft, 1=bow)
        [1]   cy_norm           — transverse centroid (0=port, 1=stbd)
        [2]   cz_norm           — vertical centroid (0=keel, 1=top)
        [3]   hull_avail        — volume fraction (available/total cells)
        [4]   zone_length_norm  — x-extent normalised by nx
        [5]   zone_width_norm   — y-extent normalised by ny
        [6]   zone_height_norm  — z-extent normalised by nz_total
        [7]   deck_idx_norm     — vertical tier normalised by max deck index
        [8:8+N_ZONE_TYPES] zone_type_onehot — one-hot encoding of ZoneType
    """
    zones = layout.zones
    p = layout.params
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
        features[i, 6] = (z.z1 - z.z0) / max(p.nz_total, 1)
        features[i, 7] = z.deck_idx / max_deck

        # One-hot zone type
        zt_idx = int(z.zone_type)
        if 0 <= zt_idx < N_ZONE_TYPES:
            features[i, 8 + zt_idx] = 1.0

    return features


# ─────────────────────────────────────────────────────────────────
# Node labels (compartment assignments — the generation target)
# ─────────────────────────────────────────────────────────────────

def _build_node_labels(
    layout: BulkheadLayout,
    assignment: CompartmentAssignment,
) -> np.ndarray:
    """
    Build (n_zones,) label vector — one Comp class index per node.

    This is what the VAE encoder reads and the decoder produces.
    """
    return np.array([
        assignment.assignments.get(z.zone_id, Comp.VOID).value
        for z in layout.zones
    ], dtype=np.int64)


# ─────────────────────────────────────────────────────────────────
# Edge construction
# ─────────────────────────────────────────────────────────────────

def _build_edges(layout: BulkheadLayout) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build edge_index (2, n_edges) and edge_attr (n_edges, 7).

    Adjacency is detected directly from the **zone_mask voxel grid**:
    two zones are adjacent if any voxel in zone A is face-adjacent
    (6-connected) to a voxel in zone B.  This is robust to fractional
    hull boundaries and variable deck heights — no bounding-box geometry.

    Four edge types:
      1. LONGITUDINAL — zones adjacent in x
      2. VERTICAL     — zones adjacent in z (both hull, or both SS)
      3. SS_TO_HULL   — zones adjacent in z where one is SS, other is hull
      4. TRANSVERSE   — zones adjacent in y (centre↔port/stbd)

    Edge feature vector (7 dims):
        [0]  dx_norm          signed x-centroid separation (b - a)
        [1]  dy_norm          signed y-centroid separation (b - a)
        [2]  dz_norm          signed z-centroid separation (b - a)
        [3]  is_longitudinal  (0/1)
        [4]  is_vertical      (0/1)
        [5]  is_ss_hull       (0/1)
        [6]  is_transverse    (0/1)
    """
    zones = layout.zones
    zm = layout.zone_mask  # (nx, ny, nz_total) int32, -1 = empty

    # ── Step 1: find all face-adjacent zone pairs from the voxel grid ──
    # For each of 3 axes, shift the grid by 1 and find where
    # adjacent voxels belong to different (valid) zones.
    from collections import defaultdict
    adj_dirs: dict = defaultdict(set)  # (min_id, max_id) -> {'x','y','z'}

    for axis, label in [(0, 'x'), (1, 'y'), (2, 'z')]:
        lo = np.take(zm, range(zm.shape[axis] - 1), axis=axis)
        hi = np.take(zm, range(1, zm.shape[axis]),  axis=axis)
        mask = (lo >= 0) & (hi >= 0) & (lo != hi)
        if not mask.any():
            continue
        pairs = np.column_stack([lo[mask].ravel(), hi[mask].ravel()])
        pairs = np.sort(pairs, axis=1)
        unique_pairs = np.unique(pairs, axis=0)
        for a, b in unique_pairs:
            adj_dirs[(int(a), int(b))].add(label)

    # ── Step 2: build zone_id → zone index mapping ────────────────
    zid_to_idx = {z.zone_id: i for i, z in enumerate(zones)}
    zone_map = layout.zone_by_id

    # ── Step 3: classify each adjacent pair and build edge lists ──
    src: List[int] = []
    dst: List[int] = []
    attrs: List[List[float]] = []

    for (zid_a, zid_b), dirs in adj_dirs.items():
        if zid_a not in zid_to_idx or zid_b not in zid_to_idx:
            continue

        za = zone_map[zid_a]
        zb = zone_map[zid_b]
        idx_a = zid_to_idx[zid_a]
        idx_b = zid_to_idx[zid_b]

        # Classify edge type
        a_ss = za.zone_type == ZoneType.SUPERSTRUCTURE
        b_ss = zb.zone_type == ZoneType.SUPERSTRUCTURE

        # SS hull boundary takes priority over transverse side edges
        if 'z' in dirs and a_ss != b_ss:
            edge_type = EDGE_SS_HULL
        else:
            y_only = 'y' in dirs and 'x' not in dirs and 'z' not in dirs
            if y_only or ('y' in dirs and za.side != zb.side):
                edge_type = EDGE_TRANSVERSE
            elif 'z' in dirs:
                edge_type = EDGE_VERTICAL
            else:
                edge_type = EDGE_LONGITUDINAL

        dx = zb.cx_norm - za.cx_norm
        dy = zb.cy_norm - za.cy_norm
        dz = zb.cz_norm - za.cz_norm

        type_vec = [0.0, 0.0, 0.0, 0.0]
        type_vec[edge_type] = 1.0

        src.extend([idx_a, idx_b])
        dst.extend([idx_b, idx_a])
        attrs.append([dx, dy, dz] + type_vec)
        attrs.append([-dx, -dy, -dz] + type_vec)

    if not src:
        return (np.zeros((2, 0), dtype=np.int64),
                np.zeros((0, N_EDGE_FEATURES), dtype=np.float32))

    # ── Isolated-node fallback ────────────────────────────────────────
    # Thin decks on small hulls can leave a zone with no face-adjacent
    # neighbour. Connect any such orphan to its nearest zone (by centroid)
    # so the graph stays connected. Edge type by dominant separation axis.
    connected = set(src)
    n_zones = len(zones)
    for i in range(n_zones):
        if i in connected:
            continue
        za = zones[i]
        best_j, best_d = -1, 1e18
        for j in range(n_zones):
            if j == i:
                continue
            zb = zones[j]
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
        tv = [0.0, 0.0, 0.0, 0.0]; tv[et] = 1.0
        src.extend([i, best_j]); dst.extend([best_j, i])
        attrs.append([dx, dy, dz] + tv)
        attrs.append([-dx, -dy, -dz] + tv)
        connected.add(i); connected.add(best_j)

    edge_index = np.array([src, dst], dtype=np.int64)
    edge_attr = np.array(attrs, dtype=np.float32)

    return edge_index, edge_attr


# ─────────────────────────────────────────────────────────────────
# Graph-level conditioning vector
# ─────────────────────────────────────────────────────────────────

def _build_conditioning(
    p: ShipParameterization,
    assignment: CompartmentAssignment,
    d_occupied_m: float,
    *,
    cond_source: str = "achieved",
) -> np.ndarray:
    """
    Global conditioning vector (16 dims).

    LCG/KG are not part of the conditioning:
    CG is an OUTPUT of the layout, not a steering input. Physics control is
    deferred to differential post-optimisation. Targets and achieved values
    are still STORED on the graph as `aux_physics` (see build_graph) so they
    can be re-introduced as conditioning later without regenerating data.

    Layout (cond_version = 2):
        [0-5]   ship_type one-hot                       (6)
        [6]     L_norm                                  (1)
        [7]     B_norm                                  (1)
        [8]     D_norm                                  (1)
        [9-15]  budget signal selected by ``cond_source``   (7)
                (achieved by production default; effective post-S3.5
                request when ``cond_source=target``), in the order:
                engine, machinery, cargo, stores, accommodation,
                fuel_tanks, ballast_tanks.
    Total: 16 dims. Sampled, effective, and achieved programmes are all
    stored separately on every graph so downstream studies can rebuild
    alternative conditioning without regenerating the corpus.
    """
    cond = np.zeros(N_COND_DIMS, dtype=np.float32)

    st_idx = int(p.ship_type)
    if 0 <= st_idx < N_SHIP_TYPES:
        cond[st_idx] = 1.0

    cond[6] = _norm(p.L, *NORM_L)
    cond[7] = _norm(p.B, *NORM_B)
    cond[8] = _norm(d_occupied_m, *NORM_D)

    for j, bkey in enumerate(BUDGET_KEYS):
        if cond_source == "achieved":
            cond[9 + j] = float(assignment.budget_fracs.get(bkey, 0.0))
        else:
            cond[9 + j] = float(p.budget.get(bkey, 0.0))
    return cond


def _deck_z_m_from_layout(layout: BulkheadLayout) -> List[float]:
    """Physical deck/tier heights (m) from keel, from zone z-bounds."""
    p = layout.params
    dz = p.dz_m
    z_breaks = {0, int(p.nz_total)}
    for z in layout.zones:
        z_breaks.add(int(z.z0))
        z_breaks.add(int(z.z1))
    return [round(k * dz, 4) for k in sorted(z_breaks)]


def _zone_bboxes_from_layout(layout: BulkheadLayout) -> np.ndarray:
    """Per-zone integer bounding boxes aligned with graph node order."""
    return np.array(
        [[z.x0, z.x1, z.y0, z.y1, z.z0, z.z1] for z in layout.zones],
        dtype=np.int16,
    )


def _deck_tiers_from_layout(layout: BulkheadLayout) -> np.ndarray:
    """
    Hull vertical tier bands as placed in Stage 3.

    Returns (n_tiers, 4) int16: [z0, z1, deck_idx, tier_role] per row,
    upper z bound exclusive. Reconstructed from zone geometry (no SS tiers).
    """
    nz_hull = layout.params.nz_hull
    seen: set = set()
    rows: List[List[int]] = []
    for z in layout.zones:
        if z.zone_type == ZoneType.SUPERSTRUCTURE:
            continue
        if z.z0 >= nz_hull:
            continue
        key = (int(z.z0), int(z.z1), int(z.deck_idx), int(z.tier_role))
        if key in seen:
            continue
        seen.add(key)
        rows.append(list(key))
    rows.sort(key=lambda r: (r[0], r[2]))
    if not rows:
        return np.zeros((0, 4), dtype=np.int16)
    return np.array(rows, dtype=np.int16)


# ─────────────────────────────────────────────────────────────────
# Canonical physics (shared with CC + real validation)
# ─────────────────────────────────────────────────────────────────

_physics_from_assignment_fn = None


def _canonical_physics_from_assignment(assignment: CompartmentAssignment):
    """Lazy import from validation.volume_metrics (same path as cc_reextract)."""
    global _physics_from_assignment_fn
    if _physics_from_assignment_fn is None:
        import sys
        from pathlib import Path

        val_root = Path(__file__).resolve().parent.parent / "validation"
        val_s = str(val_root)
        if val_s not in sys.path:
            sys.path.insert(0, val_s)
        from volume_metrics import compute_physics_from_assignment as _fn

        _physics_from_assignment_fn = _fn
    return _physics_from_assignment_fn(assignment)


def build_graph(
    assignment: CompartmentAssignment,
    *,
    cond_source: str = "achieved",
    lcg_kg_qc_gate: bool = False,
    lcg_tol: float = QC_LCG_TOL,
    kg_tol: float = QC_KG_TOL,
) -> Any:
    """
    Build a graph from a completed Stage 4 CompartmentAssignment.

    Returns a torch_geometric.data.Data object (or plain dict if PyG
    is not installed) with clean separation between input features,
    labels, conditioning, and topology.

    Data fields
    -----------
    x           : (n_zones, N_NODE_FEATURES) float  — node input features (geometry only)
    y           : (n_zones,)    long   — compartment labels (Comp class index)
    edge_index  : (2, n_edges)  long   — undirected adjacency
    edge_attr   : (n_edges, 7)  float  — edge features [dx, dy, dz, type_onehot]
    cond        : (16,)         float  — global conditioning vector (budgets, no CG)

    Metadata (graph-level scalars, for QC and analysis):
    ship_type   : int
    n_zones     : int
    lcg_actual  : float
    kg_actual   : float
    gm_t        : float
    hull_source : str

    Additional metadata:
    program              : dict   — alias for effective budget fractions
    budget_sampled       : dict   — original Stage-1 programme draw
    budget_effective     : dict   — post-S3.5 feasible programme request
    budget_achieved      : dict   — realised functional fractions
    bulkhead_x_positions : list   — transverse bulkhead x-indices (voxel grid coords)
    n_holds              : int    — number of longitudinal holds
    db_layers            : int    — double-bottom z-layers
    acc_min_z            : int    — lowest z for accommodation
    nz_hull              : int    — hull z-layers (excl. superstructure)
    deck_z_m             : list   — physical tier heights from keel (m); n_decks = len-1
    zone_bboxes          : (n_zones, 6) int16 — [x0,x1,y0,y1,z0,z1] per node
    deck_tiers           : (n_tiers, 4) int16 — [z0,z1,deck_idx,tier_role] hull bands
    """
    layout = assignment.layout
    p = layout.params

    zone_bboxes = _zone_bboxes_from_layout(layout)
    deck_tiers = _deck_tiers_from_layout(layout)

    # Node features (geometry only — no labels)
    x = _build_node_input_features(layout)

    # Labels (what the generative model produces)
    y = _build_node_labels(layout, assignment)

    # Edges
    edge_index, edge_attr = _build_edges(layout)

    # Volumetric physics + occupied-hull-band denominator (KG_SPEC)
    lcg_frac, kg_frac, gm_t, d_occupied_m = _canonical_physics_from_assignment(assignment)

    kb_m = float(getattr(assignment, "kb_m", 0.0) or 0.0)
    bm_m = float(getattr(assignment, "bm_m", 0.0) or 0.0)
    if kb_m == 0.0 and bm_m == 0.0:
        from compartment_assignment import compute_hydrostatic_primitives

        kb_m, bm_m, _ = compute_hydrostatic_primitives(p)

    from compartment_assignment import (
        QC_GM_MAX,
        gm_feasible_at_convention,
        gm_min_for_type,
    )
    gm_lo = gm_min_for_type(p.ship_type)
    gm_feasible = gm_feasible_at_convention(gm_t, p.ship_type, gm_min=gm_lo)

    cg_track = lcg_kg_tracking_metrics(
        float(lcg_frac), float(p.target_lcg_frac),
        float(kg_frac), float(p.target_kg_frac),
        lcg_tol=lcg_tol, kg_tol=kg_tol,
    )

    # Conditioning vector (D_norm uses occupied span, not moulded p.D)
    cond = _build_conditioning(p, assignment, d_occupied_m, cond_source=cond_source)

    # ── 3D connectivity metrics ──────────
    # Per-axis edge counts from the 4-way type one-hot in edge_attr[:, 3:7]
    # (0=longitudinal/x, 1=vertical/z, 2=ss-hull, 3=transverse/y), plus a
    # structural coverage check: every mirrored side node must reach the
    # centreline via at least one transverse edge.
    if edge_attr.shape[0] > 0:
        etype = edge_attr[:, 3:7].argmax(axis=1)
        n_e = edge_attr.shape[0] // 2          # undirected pairs stored twice
        n_long = int((etype == EDGE_LONGITUDINAL).sum()) // 2
        n_vert = int((etype == EDGE_VERTICAL).sum()) // 2
        n_ss = int((etype == EDGE_SS_HULL).sum()) // 2
        n_trans = int((etype == EDGE_TRANSVERSE).sum()) // 2
    else:
        n_e = n_long = n_vert = n_ss = n_trans = 0
    side_idx = [i for i, z in enumerate(layout.zones)
                if getattr(z, "side", None) in ("port", "stbd")
                and z.mirror_id >= 0]
    side_with_trans = set()
    if edge_attr.shape[0] > 0 and side_idx:
        trans_mask = etype == EDGE_TRANSVERSE
        for col in range(edge_index.shape[1]):
            if trans_mask[col]:
                side_with_trans.add(int(edge_index[0, col]))
                side_with_trans.add(int(edge_index[1, col]))
    side_missing_trans = [i for i in side_idx if i not in side_with_trans]

    # Metadata
    meta = {
        "ship_type":     int(p.ship_type),
        "n_zones":       len(layout.zones),
        "lcg_actual":    float(lcg_frac),
        "kg_actual":     float(kg_frac),
        "gm_t":          float(gm_t),
        "D_occupied_m":  float(d_occupied_m),
        "hull_source":   getattr(layout.hull_result, "hull_source", "parametric"),
        "cond_version":  2,
        "cond_source":   cond_source,
        "lcg_kg_qc_gate": bool(lcg_kg_qc_gate),
        # ── CG is not in cond but is STORED for future
        #    differential physics-control work. Targets are the S1 draws the
        #    optimiser pursued; actuals are the realised layout physics.
        "aux_physics": {
            "target_lcg": float(p.target_lcg_frac),
            "target_kg":  float(p.target_kg_frac),
            "actual_lcg": float(lcg_frac),
            "actual_kg":  float(kg_frac),
            "gm_t":       float(gm_t),
            "kb_m":       kb_m,
            "bm_m":       bm_m,
            "gm_feasible": bool(gm_feasible),
            "gm_convention": {
                "gm_min": float(gm_lo),
                "gm_max_reference": float(QC_GM_MAX),
                "feasible_rule": "gm_t >= gm_min (lower bound only)",
            },
            "lcg_tracking_error": cg_track["lcg_tracking_error"],
            "kg_tracking_error": cg_track["kg_tracking_error"],
            "lcg_tracking_within_tol": cg_track["lcg_tracking_within_tol"],
            "kg_tracking_within_tol": cg_track["kg_tracking_within_tol"],
            "lcg_kg_tracking_convention": {
                "lcg_tol": float(lcg_tol),
                "kg_tol": float(kg_tol),
                "feasible_rule": "|actual - sampled_target| <= tol",
                "qc_gate_active_at_build": bool(lcg_kg_qc_gate),
            },
        },
        "gen_version": GENERATOR_VERSION,
        # Raw S1 budget draw (pre-S3.5); cond carries achieved or effective values
        # according to cond_source. The effective programme is stored explicitly.
        "budget_sampled": {k: float(v) for k, v in
                           (p.budget_sampled or p.budget).items()},
        "budget_effective": {k: float(p.budget.get(k, 0.0))
                             for k in BUDGET_KEYS},
        "budget_achieved": {k: float(assignment.budget_fracs.get(k, 0.0))
                            for k in BUDGET_KEYS},
        # ── 3D connectivity (x=longitudinal, z=vertical, y=transverse) ──
        "n_edges":               n_e,
        "n_edges_longitudinal":  n_long,
        "n_edges_vertical":      n_vert,
        "n_edges_transverse":    n_trans,
        "n_edges_ss_hull":       n_ss,
        "frac_edges_longitudinal": (n_long / n_e) if n_e else 0.0,
        "frac_edges_vertical":     (n_vert / n_e) if n_e else 0.0,
        "frac_edges_transverse":   (n_trans / n_e) if n_e else 0.0,
        "n_side_nodes":            len(side_idx),
        "n_side_missing_transverse": len(side_missing_trans),
        # Geometry & hydrostatics
        "L":             float(p.L),
        "B":             float(p.B),
        "D":             float(p.D),
        "T":             float(p.T),
        "Cb":            float(p.Cb),
        "displacement":  float(p.displacement),
        "voxel_size":    float(p.voxel_size),
        "dx":            float(p.dx_m),
        "dy":            float(p.dy_m),
        "dz":            float(p.dz_m),
        "cell_volume":   float(p.cell_volume),
        # ── Structured program specification ────────────
        # Alias for the effective post-S3.5 programme. New code should
        # prefer the explicit ``budget_effective`` field above.
        "program":       {k: float(p.budget.get(k, 0.0)) for k in BUDGET_KEYS},
        # ── Partition structure ─────────────────────────
        # Transverse bulkhead x-indices (voxel grid coordinates).
        # Stored so future models can treat partition as a target
        # without re-deriving from zone geometry.
        "bulkhead_x_positions": [int(bx) for bx in layout.transverse_bulkheads_x],
        # Vertical tier parameters — fully determine the tier
        # boundaries via _compute_vertical_tiers(p).
        "n_holds":       int(layout.n_holds),
        "db_layers":     int(p.db_layers),
        "acc_min_z":     int(p.acc_min_z),
        "nz_hull":       int(p.nz_hull),
        "deck_z_m":      _deck_z_m_from_layout(layout),
        "mirror_ids":    [int(z.mirror_id) for z in layout.zones],
        "zone_sides":    [z.side for z in layout.zones],
    }

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
        data.zone_bboxes = torch.tensor(zone_bboxes, dtype=torch.int16)
        data.deck_tiers = torch.tensor(deck_tiers, dtype=torch.int16)
        return data
    else:
        return {"x": x, "y": y, "edge_index": edge_index,
                "edge_attr": edge_attr, "cond": cond,
                "zone_bboxes": zone_bboxes, "deck_tiers": deck_tiers, **meta}


# ─────────────────────────────────────────────────────────────────
# Validation
# ─────────────────────────────────────────────────────────────────

def validate_graph(graph: Any) -> Tuple[bool, List[str]]:
    """
    Sanity checks on a constructed graph.

    Checks:
      1. Node feature shape is (n, N_NODE_FEATURES)
      2. No NaN/Inf in node features or edge attributes
      3. Edge indices within bounds [0, n_zones)
      4. No isolated nodes
      5. Edge attribute shape is (n_edges, N_EDGE_FEATURES)
      6. Label values in valid Comp range [0, N_COMP_CLASSES)
      7. Conditioning vector shape is (16,)
      8. At least 2 edges exist
      9. Every mirrored side node has >=1 transverse edge (3D guarantee)
    """
    warns: List[str] = []

    if HAS_PYG:
        x = graph.x.numpy()
        ei = graph.edge_index.numpy()
        ea = graph.edge_attr.numpy()
        y = graph.y.numpy()
        cond = graph.cond.numpy()
    else:
        x = graph["x"]
        ei = graph["edge_index"]
        ea = graph["edge_attr"]
        y = graph["y"]
        cond = graph["cond"]

    n = x.shape[0]

    # 1. Node feature shape
    if x.shape[1] != N_NODE_FEATURES:
        warns.append(
            f"Node features: {x.shape[1]} dims, expected {N_NODE_FEATURES}"
        )

    # 2. NaN/Inf
    if np.isnan(x).any() or np.isinf(x).any():
        warns.append("NaN/Inf in node features")
    if ea.size > 0 and (np.isnan(ea).any() or np.isinf(ea).any()):
        warns.append("NaN/Inf in edge attributes")

    # 3. Edge bounds
    if ei.size > 0:
        if ei.max() >= n:
            warns.append(f"Edge index max {ei.max()} >= n_nodes {n}")
        if ei.min() < 0:
            warns.append(f"Edge index min {ei.min()} < 0")

    # 4. Isolated nodes
    if ei.size > 0:
        connected = set(ei[0].tolist()) | set(ei[1].tolist())
        isolated = set(range(n)) - connected
        if isolated:
            warns.append(f"{len(isolated)} isolated nodes: {sorted(isolated)[:5]}")

    # 5. Edge attr shape
    n_edges = ei.shape[1] if ei.ndim == 2 else 0
    if n_edges > 0 and ea.shape != (n_edges, N_EDGE_FEATURES):
        warns.append(
            f"Edge attr shape {ea.shape}, expected ({n_edges}, {N_EDGE_FEATURES})"
        )

    # 6. Label range
    if y.min() < 0 or y.max() >= N_COMP_CLASSES:
        warns.append(f"Labels out of range [{y.min()}, {y.max()}], "
                     f"expected [0, {N_COMP_CLASSES - 1}]")

    # 7. Conditioning shape
    if cond.shape != (N_COND_DIMS,):
        warns.append(f"Conditioning shape {cond.shape}, expected (16,)")

    # 9. Every mirrored side node must have >=1 transverse (port/stbd) edge —
    #    the structural guarantee that the graph is genuinely 3D, not an
    #    extruded 2D section.
    if HAS_PYG:
        n_miss = int(getattr(graph, "n_side_missing_transverse", 0) or 0)
    else:
        n_miss = int(graph.get("n_side_missing_transverse", 0) or 0)
    if n_miss > 0:
        warns.append(f"{n_miss} mirrored side node(s) lack a transverse edge")

    # 8. Minimum edges
    if n_edges < 2:
        warns.append(f"Only {n_edges} edges, expected >= 2")

    return len(warns) == 0, warns


# ─────────────────────────────────────────────────────────────────
# Dataset format names
# ─────────────────────────────────────────────────────────────────

NODE_FEATURE_NAMES = [
    "cx_norm", "cy_norm", "cz_norm", "hull_avail",
    "zone_len", "zone_wid", "zone_hgt", "deck_idx",
    "zt_AFT_PK", "zt_ENGINE", "zt_MAIN_DB",
    "zt_MAIN_LO", "zt_MAIN_UP", "zt_FWD_PK", "zt_SS",
    "zt_ENG_UP", "zt_SIDE_DB", "zt_SIDE_LO", "zt_SIDE_UP",
]

EDGE_LABELS = {
    EDGE_LONGITUDINAL: 'Longitudinal',
    EDGE_VERTICAL:     'Vertical',
    EDGE_SS_HULL:      'SS-Hull',
    EDGE_TRANSVERSE:   'Transverse',
}
