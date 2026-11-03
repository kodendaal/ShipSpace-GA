from __future__ import annotations
from dataclasses import dataclass
from typing import Any
from collections import defaultdict
import math
import numpy as np
import networkx as nx
from scipy import ndimage

# Frozen v1.0 semantic labels
VOID = 0
ENGINE_ROOM = 1
MACHINERY = 2
CARGO = 3
ACCOMMODATION = 4
FUEL = 5
BALLAST = 6
STEERING = 7
STORES = 8
NAVIGATION = 9
EMPTY = 10

LABEL_NAMES = {
    0: "VOID", 1: "ENGINE_ROOM", 2: "MACHINERY", 3: "CARGO",
    4: "ACCOMMODATION", 5: "FUEL_TANKS", 6: "BALLAST_TANKS",
    7: "STEERING_GEAR", 8: "STORES", 9: "NAVIGATION", 10: "EMPTY",
}

@dataclass
class VolumeGeometry:
    labels: np.ndarray
    volume_m3: np.ndarray
    x0: np.ndarray
    x1: np.ndarray
    y0: np.ndarray
    y1: np.ndarray
    z0: np.ndarray
    z1: np.ndarray
    L: float
    B: float
    D: float

    def validate(self) -> None:
        shape = self.labels.shape
        for name in ("volume_m3", "x0", "x1", "y0", "y1", "z0", "z1"):
            arr = getattr(self, name)
            if arr.shape != shape:
                raise ValueError(f"{name} shape {arr.shape} != labels shape {shape}")
        if self.labels.ndim != 3:
            raise ValueError("labels must be a 3-D array")
        if not (self.L > 0 and self.B > 0 and self.D > 0):
            raise ValueError("L, B and D must be positive")


def _component_nodes(vg: VolumeGeometry) -> tuple[np.ndarray, list[dict[str, Any]]]:
    structure = ndimage.generate_binary_structure(3, 1)  # 6-connectivity
    node_grid = np.full(vg.labels.shape, -1, dtype=np.int32)
    nodes: list[dict[str, Any]] = []
    next_id = 0
    retained = sorted(int(v) for v in np.unique(vg.labels) if int(v) != EMPTY)
    xc = 0.5 * (vg.x0 + vg.x1)
    yc = 0.5 * (vg.y0 + vg.y1)
    zc = 0.5 * (vg.z0 + vg.z1)
    for label_value in retained:
        cc, ncc = ndimage.label(vg.labels == label_value, structure=structure)
        for cc_id in range(1, ncc + 1):
            mask = cc == cc_id
            vol = np.where(mask, vg.volume_m3, 0.0)
            w = float(vol.sum())
            if w <= 0:
                w = float(mask.sum())
                cx = float(xc[mask].mean())
                cy = float(yc[mask].mean())
                cz = float(zc[mask].mean())
            else:
                cx = float((xc * vol).sum() / w)
                cy = float((yc * vol).sum() / w)
                cz = float((zc * vol).sum() / w)
            node_grid[mask] = next_id
            nodes.append({
                "node_id": next_id,
                "label": label_value,
                "label_name": LABEL_NAMES.get(label_value, str(label_value)),
                "voxel_count": int(mask.sum()),
                "volume_m3": float(vg.volume_m3[mask].sum()),
                "centroid": np.array([cx, cy, cz], dtype=float),
            })
            next_id += 1
    return node_grid, nodes


def _face_geometry(vg: VolumeGeometry, axis: int, lo_idx: tuple[np.ndarray, ...], hi_idx: tuple[np.ndarray, ...]) -> tuple[np.ndarray, np.ndarray]:
    if axis == 0:
        plane = 0.5 * (vg.x1[lo_idx] + vg.x0[hi_idx])
        a0, a1 = np.maximum(vg.y0[lo_idx], vg.y0[hi_idx]), np.minimum(vg.y1[lo_idx], vg.y1[hi_idx])
        b0, b1 = np.maximum(vg.z0[lo_idx], vg.z0[hi_idx]), np.minimum(vg.z1[lo_idx], vg.z1[hi_idx])
        centers = np.column_stack([plane, 0.5*(a0+a1), 0.5*(b0+b1)])
    elif axis == 1:
        plane = 0.5 * (vg.y1[lo_idx] + vg.y0[hi_idx])
        a0, a1 = np.maximum(vg.x0[lo_idx], vg.x0[hi_idx]), np.minimum(vg.x1[lo_idx], vg.x1[hi_idx])
        b0, b1 = np.maximum(vg.z0[lo_idx], vg.z0[hi_idx]), np.minimum(vg.z1[lo_idx], vg.z1[hi_idx])
        centers = np.column_stack([0.5*(a0+a1), plane, 0.5*(b0+b1)])
    else:
        plane = 0.5 * (vg.z1[lo_idx] + vg.z0[hi_idx])
        a0, a1 = np.maximum(vg.x0[lo_idx], vg.x0[hi_idx]), np.minimum(vg.x1[lo_idx], vg.x1[hi_idx])
        b0, b1 = np.maximum(vg.y0[lo_idx], vg.y0[hi_idx]), np.minimum(vg.y1[lo_idx], vg.y1[hi_idx])
        centers = np.column_stack([0.5*(a0+a1), 0.5*(b0+b1), plane])
    area = np.maximum(a1-a0, 0.0) * np.maximum(b1-b0, 0.0)
    return centers, area


def _component_edges(vg: VolumeGeometry, node_grid: np.ndarray, nodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    # Accumulate area-weighted shared-interface centroids for each node pair.
    acc: dict[tuple[int, int], dict[str, Any]] = defaultdict(lambda: {"area": 0.0, "sum": np.zeros(3), "faces": 0})
    shape = node_grid.shape
    for axis in range(3):
        lo_sl = [slice(None)] * 3
        hi_sl = [slice(None)] * 3
        lo_sl[axis] = slice(0, shape[axis]-1)
        hi_sl[axis] = slice(1, shape[axis])
        lo = node_grid[tuple(lo_sl)]
        hi = node_grid[tuple(hi_sl)]
        mask = (lo >= 0) & (hi >= 0) & (lo != hi)
        if not np.any(mask):
            continue
        local_idx = np.where(mask)
        lo_idx_list = list(local_idx)
        hi_idx_list = list(local_idx)
        hi_idx_list[axis] = hi_idx_list[axis] + 1
        lo_idx = tuple(lo_idx_list)
        hi_idx = tuple(hi_idx_list)
        centers, areas = _face_geometry(vg, axis, lo_idx, hi_idx)
        uvals = lo[mask].astype(int)
        vvals = hi[mask].astype(int)
        for u, v, c, a in zip(uvals, vvals, centers, areas):
            if a <= 0:
                continue
            pair = (u, v) if u < v else (v, u)
            rec = acc[pair]
            rec["area"] += float(a)
            rec["sum"] += c * float(a)
            rec["faces"] += 1
    edges: list[dict[str, Any]] = []
    for (u, v), rec in acc.items():
        interface = rec["sum"] / max(rec["area"], 1e-12)
        cu, cv = nodes[u]["centroid"], nodes[v]["centroid"]
        length = float(np.linalg.norm(cu-interface) + np.linalg.norm(cv-interface))
        edges.append({
            "u": u, "v": v, "length_m": length,
            "interface_area_m2": float(rec["area"]),
            "interface_centroid": interface,
            "n_faces": int(rec["faces"]),
        })
    return edges


def _mean_or_nan(values: list[float]) -> float:
    return float(np.mean(values)) if values else float("nan")


def compute_functional_adjacency(
    vg: VolumeGeometry,
    *,
    void_traversable: bool = True,
    ship_id: str = "unknown",
    source: str = "unknown",
    family: str = "unknown",
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Calculate the simple Functional Adjacency Cost on a 3-D semantic volume.

    Required relationships (each terminal contributes one equal-weight pair):
      fuel -> nearest engine-room island
      machinery -> nearest engine-room island
      accommodation -> nearest engine-room island
      steering -> nearest engine-room island
      accommodation -> nearest machinery island

    Each path distance is divided by ship length L, then all are averaged once.
    """
    vg.validate()
    node_grid, nodes = _component_nodes(vg)
    edges = _component_edges(vg, node_grid, nodes)

    G = nx.Graph()
    for n in nodes:
        if (not void_traversable) and n["label"] == VOID:
            continue
        G.add_node(n["node_id"], **{k:v for k,v in n.items() if k != "centroid"})
    for e in edges:
        if e["u"] in G and e["v"] in G:
            G.add_edge(e["u"], e["v"], weight=e["length_m"], length_m=e["length_m"])

    by_label: dict[int, list[int]] = defaultdict(list)
    for n in nodes:
        if n["node_id"] in G:
            by_label[n["label"]].append(n["node_id"])

    required = {
        "engine_room": ENGINE_ROOM,
        "machinery": MACHINERY,
        "accommodation": ACCOMMODATION,
        "fuel": FUEL,
        "steering": STEERING,
    }
    missing = [name for name, lbl in required.items() if not by_label.get(lbl)]
    invalid_reasons: list[str] = []
    if missing:
        invalid_reasons.append("missing_labels:" + ",".join(missing))

    distances_by_group: dict[str, list[float]] = {
        "fuel_er": [], "machinery_er": [], "accommodation_er": [],
        "steering_er": [], "accommodation_machinery": [],
    }
    disconnected: list[dict[str, Any]] = []

    def multi_source_dist(sources: list[int]) -> dict[int, float]:
        if not sources:
            return {}
        return nx.multi_source_dijkstra_path_length(G, sources, weight="weight")

    dist_er = multi_source_dist(by_label.get(ENGINE_ROOM, []))
    dist_m = multi_source_dist(by_label.get(MACHINERY, []))

    terminal_specs = [
        (FUEL, "fuel_er", dist_er),
        (MACHINERY, "machinery_er", dist_er),
        (ACCOMMODATION, "accommodation_er", dist_er),
        (STEERING, "steering_er", dist_er),
        (ACCOMMODATION, "accommodation_machinery", dist_m),
    ]
    for lbl, group, dist_map in terminal_specs:
        for node_id in by_label.get(lbl, []):
            d = dist_map.get(node_id, math.inf)
            if not math.isfinite(d):
                disconnected.append({"group": group, "node_id": int(node_id), "label": LABEL_NAMES[lbl]})
            else:
                distances_by_group[group].append(float(d) / vg.L)

    if disconnected:
        invalid_reasons.append(f"disconnected_required_paths:{len(disconnected)}")

    all_values = [d for vals in distances_by_group.values() for d in vals]
    valid = (not missing) and (not disconnected) and bool(all_values)
    j = _mean_or_nan(all_values) if valid else float("nan")
    graph_connected = nx.is_connected(G) if G.number_of_nodes() > 0 else False

    row = {
        "ship_id": ship_id, "source": source, "family": family,
        "L": float(vg.L), "B": float(vg.B), "D": float(vg.D),
        "n_nodes": int(G.number_of_nodes()), "n_edges": int(G.number_of_edges()),
        "n_fuel_islands": len(by_label.get(FUEL, [])),
        "n_machinery_islands": len(by_label.get(MACHINERY, [])),
        "n_accommodation_islands": len(by_label.get(ACCOMMODATION, [])),
        "n_steering_islands": len(by_label.get(STEERING, [])),
        "n_engine_room_islands": len(by_label.get(ENGINE_ROOM, [])),
        "D_fuel_ER": _mean_or_nan(distances_by_group["fuel_er"]),
        "D_machinery_ER": _mean_or_nan(distances_by_group["machinery_er"]),
        "D_accommodation_ER": _mean_or_nan(distances_by_group["accommodation_er"]),
        "D_steering_ER": _mean_or_nan(distances_by_group["steering_er"]),
        "D_accommodation_machinery": _mean_or_nan(distances_by_group["accommodation_machinery"]),
        "n_required_pairs": len(all_values) + len(disconnected),
        "J_adj": j,
        "graph_connected": bool(graph_connected),
        "all_services_valid": bool(valid),
        "invalid_reason": ";".join(invalid_reasons),
        "void_traversable": bool(void_traversable),
    }
    details = {"nodes": nodes, "edges": edges, "distances_by_group": distances_by_group, "disconnected": disconnected}
    return row, details
