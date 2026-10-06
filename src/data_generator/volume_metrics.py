"""
volume_metrics.py
=================
Single source of truth for volumetric budget fractions, physics (LCG/KG/GM),
and volume-weighted label mix — the same rules for the real GA graphs and the generated graphs.

Denominator rule (identical both sides)
---------------------------------------
* Include every in-hull voxel (label != EMPTY), **including VOID and SS**.
* Exclude EMPTY (outside represented ship).
* Budget numerators: the seven ``BUDGET_KEYS`` classes only.

Positions
---------
Mass-weighted LCG/KG and zone centroids use **effective cell centres in metres**
(``(i+0.5)*d`` on the native grid).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from validation_constants import (
    BUDGET_KEYS, COMP_DENSITY, COMP_TO_BUDGET_KEY, Comp,
)


@dataclass
class VoxelFields:
    """Per-cell physical geometry aligned with a label grid."""

    labels: np.ndarray       # (nx, ny, nz) int — Comp enum
    volume_m3: np.ndarray    # (nx, ny, nz) float — 0 outside hull
    x_m: np.ndarray          # effective centre x (m), aft → bow
    y_m: np.ndarray
    z_m: np.ndarray          # effective centre z (m), keel ↑
    L: float
    B: float
    D_total_m: float
    z0_m: Optional[np.ndarray] = None   # optional per-cell bottom z (m)
    z1_m: Optional[np.ndarray] = None   # cell top z (m)

    @property
    def shape(self) -> Tuple[int, int, int]:
        return self.labels.shape

    def in_hull_mask(self) -> np.ndarray:
        return self.labels != int(Comp.EMPTY)

    def has_metric_z_bounds(self) -> bool:
        """True when per-cell z0/z1 faces are set."""
        if self.z0_m is None or self.z1_m is None:
            return False
        return bool(np.isfinite(self.z0_m[self.in_hull_mask()]).any())


def build_voxel_fields_from_assignment(assignment) -> VoxelFields:
    """Native synth grid: uniform dx/dy/dz per ship, fractional hull occupancy."""
    layout = assignment.layout
    p = layout.params
    zone_mask = layout.zone_mask
    full_mask = layout.hull_result.full_mask
    nx, ny, nz_total = zone_mask.shape
    dx, dy, dz = p.dx_m, p.dy_m, p.dz_m
    cell_vol = dx * dy * dz

    labels = np.full((nx, ny, nz_total), int(Comp.EMPTY), dtype=np.int32)
    volume_m3 = np.zeros((nx, ny, nz_total), dtype=np.float64)
    x_m = np.zeros((nx, ny, nz_total), dtype=np.float64)
    y_m = np.zeros((nx, ny, nz_total), dtype=np.float64)
    z_m = np.zeros((nx, ny, nz_total), dtype=np.float64)

    zone_to_class = {}
    for zone_id, comp in assignment.assignments.items():
        zone_to_class[int(zone_id)] = int(comp)

    for ix in range(nx):
        for iy in range(ny):
            for iz in range(nz_total):
                occ = float(full_mask[ix, iy, iz])
                if occ <= 0.0:
                    continue
                zid = int(zone_mask[ix, iy, iz])
                if zid < 0:
                    continue
                labels[ix, iy, iz] = zone_to_class.get(zid, int(Comp.VOID))
                volume_m3[ix, iy, iz] = cell_vol * occ
                x_m[ix, iy, iz] = (ix + 0.5) * dx
                y_m[ix, iy, iz] = (iy + 0.5) * dy
                z_m[ix, iy, iz] = (iz + 0.5) * dz

    D_total = p.D + p.nz_ss * dz
    return VoxelFields(
        labels=labels,
        volume_m3=volume_m3,
        x_m=x_m,
        y_m=y_m,
        z_m=z_m,
        L=float(p.L),
        B=float(p.B),
        D_total_m=float(D_total),
    )


def build_voxel_fields_from_saved_graph(graph: Any) -> VoxelFields:
    """
    Rebuild volumetric fields from a saved Stage-5 PyG graph (normalised or native grid).
    """
    if hasattr(graph, "voxel_labels"):
        labels_t = graph.voxel_labels
        labels = (
            labels_t.cpu().numpy() if hasattr(labels_t, "cpu") else np.asarray(labels_t)
        ).astype(np.int32)
    else:
        raise ValueError("graph has no voxel_labels")

    nx, ny, nz = labels.shape
    dx = float(getattr(graph, "dx", 1.0))
    dy = float(getattr(graph, "dy", 1.0))
    dz = float(getattr(graph, "dz", 1.0))
    cell_vol = dx * dy * dz

    if getattr(graph, "voxel_volume_m3", None) is not None:
        vol_t = graph.voxel_volume_m3
        volume_m3 = (
            vol_t.cpu().numpy() if hasattr(vol_t, "cpu") else np.asarray(vol_t)
        ).astype(np.float64)
    elif getattr(graph, "voxel_hull_mask", None) is not None:
        mask_t = graph.voxel_hull_mask
        mask = mask_t.cpu().numpy() if hasattr(mask_t, "cpu") else np.asarray(mask_t)
        volume_m3 = np.where(mask, cell_vol, 0.0)
    else:
        volume_m3 = np.where(labels != int(Comp.EMPTY), cell_vol, 0.0)

    x_m = (np.arange(nx, dtype=np.float64) + 0.5)[:, None, None] * dx
    y_m = (np.arange(ny, dtype=np.float64) + 0.5)[None, :, None] * dy
    z_m = (np.arange(nz, dtype=np.float64) + 0.5)[None, None, :] * dz
    x_m = np.broadcast_to(x_m, (nx, ny, nz)).copy()
    y_m = np.broadcast_to(y_m, (nx, ny, nz)).copy()
    z_m = np.broadcast_to(z_m, (nx, ny, nz)).copy()

    L = float(getattr(graph, "L", nx * dx))
    B = float(getattr(graph, "B", ny * dy))
    nz_hull = int(getattr(graph, "nz_hull", nz))
    D_hull = float(getattr(graph, "D", nz_hull * dz))
    nz_ss = max(nz - nz_hull, 0)
    D_total_m = D_hull + nz_ss * dz

    return VoxelFields(
        labels=labels,
        volume_m3=volume_m3,
        x_m=x_m,
        y_m=y_m,
        z_m=z_m,
        L=L,
        B=B,
        D_total_m=D_total_m,
    )


def hull_band_depth_m(
    nz_hull: int,
    dz_m: float,
    vf: Optional[VoxelFields] = None,
) -> float:
    """
    Keel → main-deck depth (m). Same rule both sides:

    * **Per-cell faces** (``z0``/``z1`` set): ``max(z1) − min(z0)`` over
      in-hull voxels with ``iz < nz_hull`` — respects anisotropic layer heights.
    * **Uniform synth grid**: ``nz_hull × dz_m`` — cell faces ``0 … nz_hull·dz``
      (= moulded ``p.D`` when ``dz = D / nz_hull``).

    ``nz_hull``: detected main deck on real GAs; on synth use parametric
    ``p.nz_hull``.
    """
    nz_hull = max(int(nz_hull), 1)
    dz_m = float(dz_m)
    if vf is not None and vf.has_metric_z_bounds():
        nx, ny, nz = vf.shape
        nz_cap = int(np.clip(nz_hull, 1, nz))
        iz_grid = np.arange(nz)[None, None, :]
        mask = vf.in_hull_mask() & (iz_grid < nz_cap)
        if mask.any():
            z0 = vf.z0_m[mask]
            z1 = vf.z1_m[mask]
            ok = np.isfinite(z0) & np.isfinite(z1)
            if ok.any():
                return max(float(z1[ok].max() - z0[ok].min()), 1e-6)
    return nz_hull * dz_m


def compute_budget_fracs_volumetric(
    vf: VoxelFields,
    nz_hull: int = 0,  # noqa: ARG001 — API symmetry; denominator is full ship
) -> Dict[str, float]:
    """Volume-weighted budget fractions (VOID in denominator, SS included)."""
    mask = vf.in_hull_mask()
    total_vol = float(vf.volume_m3[mask].sum())
    if total_vol <= 0.0:
        return {k: 0.0 for k in BUDGET_KEYS}

    comp_vol: Dict[str, float] = {k: 0.0 for k in BUDGET_KEYS}
    for comp_val, bkey in COMP_TO_BUDGET_KEY.items():
        cmask = mask & (vf.labels == int(comp_val))
        comp_vol[bkey] += float(vf.volume_m3[cmask].sum())

    return {k: comp_vol[k] / total_vol for k in BUDGET_KEYS}


def compute_physics_volumetric(
    vf: VoxelFields,
    nz_hull: int,
    dz_m: float,
) -> Tuple[float, float, float]:
    """
    Mass-weighted LCG/KG and crude GM from volumetric fields.

    KG denominator: ``hull_band_depth_m(nz_hull, dz_m)`` (deck-template depth).

    Returns (lcg_frac, kg_frac, gm_t).
    """
    mask = vf.in_hull_mask()
    if not mask.any():
        return 0.5, 0.5, 0.0

    L = vf.L
    D_hull = hull_band_depth_m(nz_hull, dz_m, vf)

    sum_mass = 0.0
    sum_mx = 0.0
    sum_mz = 0.0
    nx, ny, nz = vf.shape
    nz_hull_i = int(np.clip(nz_hull, 1, nz))
    iz_grid = np.arange(nz)[None, None, :]
    hull_only = iz_grid < nz_hull_i
    hull_mask = mask & hull_only
    hull_vol = float(vf.volume_m3[hull_mask].sum()) if hull_mask.any() else 0.0

    for comp_val, density in COMP_DENSITY.items():
        cmask = mask & (vf.labels == int(comp_val))
        if not cmask.any():
            continue
        vols = vf.volume_m3[cmask]
        mass = density * vols
        xs = vf.x_m[cmask]
        zs = vf.z_m[cmask]
        sum_mass += float(mass.sum())
        sum_mx += float((mass * xs).sum())
        sum_mz += float((mass * zs).sum())

    if sum_mass <= 0.0:
        return 0.5, 0.5, 0.0

    lcg_m = sum_mx / sum_mass
    kg_m = sum_mz / sum_mass
    lcg_frac = float(np.clip(lcg_m / L if L > 0 else 0.5, 0.0, 1.0))
    kg_frac = float(np.clip(kg_m / D_hull if D_hull > 0 else 0.5, 0.0, 1.0))

    # Crude GM (volumetric Cb)
    total_box = L * vf.B * D_hull
    Cb = float(np.clip(hull_vol / max(total_box, 1e-6), 0.4, 0.9))
    T = D_hull * 0.7
    if T > 0 and Cb > 0:
        Cw = 0.70 + 0.30 * Cb
        KB = T * (5.0 / 6.0 - Cb / (3.0 * Cw))
        BM = (Cw / Cb) * vf.B ** 2 / (12.0 * T)
        gm_t = float(KB + BM - kg_m)
    else:
        gm_t = 0.0

    return lcg_frac, kg_frac, gm_t


def compute_physics_from_assignment(
    assignment,
) -> Tuple[float, float, float, float]:
    """
    Canonical synth physics — identical to ``cc_reextract`` native_volume path.

    Returns (lcg_frac, kg_frac, gm_t, d_hull_m).
    """
    p = assignment.layout.params
    vf = build_voxel_fields_from_assignment(assignment)
    nz_hull_depth = int(p.nz_hull)
    d_hull_m = hull_band_depth_m(nz_hull_depth, p.dz_m, vf)
    lcg_frac, kg_frac, gm_t = compute_physics_volumetric(vf, nz_hull_depth, p.dz_m)
    return lcg_frac, kg_frac, gm_t, d_hull_m


def refine_zone_centroids_volumetric(
    zones: List,
    vf: VoxelFields,
) -> None:
    """Overwrite zone cx_norm/cz_norm/cy_norm with volume-weighted effective centres."""
    L, D = vf.L, vf.D_total_m
    for zone in zones:
        idx = zone.voxel_indices
        if len(idx) == 0:
            continue
        vols = vf.volume_m3[idx[:, 0], idx[:, 1], idx[:, 2]]
        w = vols.sum()
        if w <= 0.0:
            continue
        cx = float((vf.x_m[idx[:, 0], idx[:, 1], idx[:, 2]] * vols).sum() / w)
        cy = float((vf.y_m[idx[:, 0], idx[:, 1], idx[:, 2]] * vols).sum() / w)
        cz = float((vf.z_m[idx[:, 0], idx[:, 1], idx[:, 2]] * vols).sum() / w)
        zone.cx_norm = round(cx / L if L > 0 else 0.5, 5)
        zone.cy_norm = round(cy / vf.B if vf.B > 0 else 0.5, 5)
        zone.cz_norm = round(cz / D if D > 0 else 0.5, 5)


def label_freq_from_graph_voxels(
    g,
    active_comp_indices: Tuple[int, ...],
    _get=None,
) -> Optional[np.ndarray]:
    """Volume-weighted label mix from graph voxel tensors, if attached."""
    if _get is None:
        def _get(obj, attr):
            v = getattr(obj, attr) if hasattr(obj, attr) else obj[attr]
            try:
                import torch
                if isinstance(v, torch.Tensor):
                    return v.numpy()
            except ImportError:
                pass
            return v

    has_vol = hasattr(g, "voxel_volume_m3") or (
        isinstance(g, dict) and "voxel_volume_m3" in g
    )
    has_lab = hasattr(g, "voxel_labels") or (
        isinstance(g, dict) and "voxel_labels" in g
    )
    if not (has_vol and has_lab):
        return None

    vl = _get(g, "voxel_labels").astype(np.int64)
    vol = _get(g, "voxel_volume_m3").astype(np.float64)
    mask = vl != int(Comp.EMPTY)
    total = float(vol[mask].sum())
    if total <= 0.0:
        return None
    return np.array([
        float(vol[mask & (vl == int(c))].sum()) / total
        for c in active_comp_indices
    ], dtype=np.float64)
