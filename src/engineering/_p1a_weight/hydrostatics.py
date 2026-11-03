from __future__ import annotations

import math
from typing import Any
import numpy as np

from .models import Hydrostatics

RHO_SW_DEFAULT = 1.025


def waterline_profile(p: Any, mean_draft_m: float, trim_m: float) -> tuple[np.ndarray, np.ndarray]:
    """Return x-centres and local waterline z(x).

    Positive ``trim_m`` means the forward draft is greater than the aft draft.
    """
    nx = int(p.nx)
    x = (np.arange(nx, dtype=float) + 0.5) * float(p.dx_m)
    slope = float(trim_m) / max(float(p.L), 1e-12)
    zw = float(mean_draft_m) + slope * (x - 0.5 * float(p.L))
    return x, zw


def _waterplane_occupancy(mask: np.ndarray, dz: float, zw: np.ndarray) -> np.ndarray:
    """Interpolate occupancy at a longitudinally varying waterline."""
    nx, ny, nz = mask.shape
    zc = (np.arange(nz, dtype=float) + 0.5) * dz
    out = np.zeros((nx, ny), dtype=float)
    for i in range(nx):
        z = float(zw[i])
        if z <= zc[0]:
            out[i] = mask[i, :, 0]
            continue
        if z >= zc[-1]:
            out[i] = mask[i, :, -1]
            continue
        hi = int(np.searchsorted(zc, z, side="right"))
        lo = hi - 1
        alpha = (z - zc[lo]) / max(zc[hi] - zc[lo], 1e-12)
        out[i] = (1.0 - alpha) * mask[i, :, lo] + alpha * mask[i, :, hi]
    return np.clip(out, 0.0, 1.0)


def integrate_hydrostatics(
    mask: np.ndarray,
    p: Any,
    mean_draft_m: float,
    trim_m: float = 0.0,
    rho_sw_t_m3: float = RHO_SW_DEFAULT,
) -> Hydrostatics:
    """Integrate hydrostatics from the main-hull voxel mask.

    The superstructure must not be included in ``mask``. A longitudinally tilted
    waterplane is supported, allowing simultaneous heave and trim closure.
    """
    mask = np.asarray(mask, dtype=float)
    if mask.ndim != 3:
        raise ValueError("mask must have shape (nx, ny, nz)")
    if mask.shape != (int(p.nx), int(p.ny), int(p.nz_hull)):
        raise ValueError(
            f"mask shape {mask.shape} does not match params "
            f"{(int(p.nx), int(p.ny), int(p.nz_hull))}"
        )

    dx, dy, dz = float(p.dx_m), float(p.dy_m), float(p.dz_m)
    L, B = float(p.L), float(p.B)
    x, zw = waterline_profile(p, mean_draft_m, trim_m)

    cell_plan_area = dx * dy
    z0 = np.arange(mask.shape[2], dtype=float) * dz
    submerged_height = np.clip(zw[:, None] - z0[None, :], 0.0, dz)
    occupancy_xz = mask.sum(axis=1)
    dv_xz = occupancy_xz * cell_plan_area * submerged_height
    volume = float(dv_xz.sum())
    mx = float(np.sum(dv_xz * x[:, None]))
    mz = float(np.sum(dv_xz * (z0[None, :] + 0.5 * submerged_height)))

    if volume <= 1e-12:
        return Hydrostatics(
            mean_draft_m=float(mean_draft_m),
            aft_draft_m=float(zw[0]),
            forward_draft_m=float(zw[-1]),
            trim_m=float(trim_m),
            trim_angle_deg=math.degrees(math.atan2(float(trim_m), max(L, 1e-12))),
            volume_m3=0.0,
            displacement_t=0.0,
            kb_m=math.nan,
            lcb_m=math.nan,
            awp_m2=0.0,
            iwp_transverse_m4=0.0,
            bm_transverse_m=math.nan,
            km_transverse_m=math.nan,
            lcf_m=math.nan,
            cb=math.nan,
        )

    occ_wp = _waterplane_occupancy(mask, dz, zw)
    yc = (np.arange(mask.shape[1], dtype=float) + 0.5) * dy
    yrel2 = (yc - 0.5 * B) ** 2
    area_cells = occ_wp * dx * dy
    awp = float(area_cells.sum())
    # Cell-centre term plus each rectangular cell's own second moment.
    iwp = float(np.sum(area_cells * (yrel2[None, :] + dy**2 / 12.0)))
    xwp = np.broadcast_to(x[:, None], area_cells.shape)
    lcf = float(np.sum(area_cells * xwp) / awp) if awp > 1e-12 else math.nan

    kb = mz / volume
    lcb = mx / volume
    bm = iwp / volume
    km = kb + bm
    mean_positive_draft = max(float(mean_draft_m), 1e-12)
    cb = volume / max(L * B * mean_positive_draft, 1e-12)

    return Hydrostatics(
        mean_draft_m=float(mean_draft_m),
        aft_draft_m=float(zw[0]),
        forward_draft_m=float(zw[-1]),
        trim_m=float(trim_m),
        trim_angle_deg=math.degrees(math.atan2(float(trim_m), max(L, 1e-12))),
        volume_m3=float(volume),
        displacement_t=float(rho_sw_t_m3 * volume),
        kb_m=float(kb),
        lcb_m=float(lcb),
        awp_m2=float(awp),
        iwp_transverse_m4=float(iwp),
        bm_transverse_m=float(bm),
        km_transverse_m=float(km),
        lcf_m=float(lcf),
        cb=float(cb),
    )
