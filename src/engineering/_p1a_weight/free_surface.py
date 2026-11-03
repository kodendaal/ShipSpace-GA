from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any
import numpy as np
from scipy.ndimage import label as cc_label


@dataclass(frozen=True)
class FreeSurfaceResult:
    evaluated: bool
    correction_m: float
    total_fsm_t_m: float
    n_slack_tanks: int
    fuel_fsm_t_m: float
    ballast_fsm_t_m: float
    message: str

    def to_dict(self) -> dict:
        return asdict(self)


def _component_i_free_surface_m4(
    component: np.ndarray,
    p: Any,
    fill_fraction: float,
) -> float:
    """Approximate transverse free-surface inertia at the fill plane.

    The semantic tank island is treated as the tank envelope. The fill plane is
    selected by the cumulative occupied volume in z. The returned inertia is
    about the free surface's own longitudinal centroidal axis, which is the
    appropriate tank free-surface inertia for the virtual-rise correction.
    """
    idx = np.argwhere(component)
    if idx.size == 0:
        return 0.0
    z_levels = np.unique(idx[:, 2])
    counts = np.array([(idx[:, 2] == z).sum() for z in z_levels], dtype=float)
    cum = np.cumsum(counts)
    target = float(np.clip(fill_fraction, 0.0, 1.0)) * cum[-1]
    z_sel = int(z_levels[int(np.searchsorted(cum, target, side="left"))])
    plan = component[:, :, z_sel]
    xy = np.argwhere(plan)
    if xy.shape[0] < 2:
        return 0.0
    dy = float(p.dy_m)
    dx = float(p.dx_m)
    y = (xy[:, 1].astype(float) + 0.5) * dy
    ybar = float(np.mean(y))
    dA = dx * dy
    return float(np.sum(dA * ((y - ybar) ** 2 + dy**2 / 12.0)))


def estimate_free_surface(
    assignment: Any,
    generator_api: Any,
    family_cfg: dict,
    displacement_t: float,
) -> FreeSurfaceResult:
    if not bool(family_cfg.get("free_surface_enabled", False)):
        return FreeSurfaceResult(
            evaluated=False,
            correction_m=0.0,
            total_fsm_t_m=0.0,
            n_slack_tanks=0,
            fuel_fsm_t_m=0.0,
            ballast_fsm_t_m=0.0,
            message="not evaluated: loading-state tank fill fractions not enabled",
        )
    if displacement_t <= 0.0:
        return FreeSurfaceResult(False, 0.0, 0.0, 0, 0.0, 0.0, "invalid displacement")

    vf = generator_api.build_voxel_fields_from_assignment(assignment)
    labels = np.asarray(vf.labels)
    p = assignment.layout.params
    structure = np.zeros((3, 3, 3), dtype=np.uint8)
    structure[1, 1, 1] = 1
    structure[0, 1, 1] = structure[2, 1, 1] = 1
    structure[1, 0, 1] = structure[1, 2, 1] = 1
    structure[1, 1, 0] = structure[1, 1, 2] = 1

    total_fsm = 0.0
    fuel_fsm = 0.0
    ballast_fsm = 0.0
    n_slack = 0
    for comp, cfg_key, rho in (
        (generator_api.Comp.FUEL_TANKS, "FUEL_TANKS", float(generator_api.density_for(generator_api.Comp.FUEL_TANKS, p.ship_type))),
        (generator_api.Comp.BALLAST_TANKS, "BALLAST_TANKS", 1.025),
    ):
        fill = float((family_cfg.get("compartment_fill_factors", {}) or {}).get(cfg_key, 1.0))
        if fill <= 1e-6 or fill >= 1.0 - 1e-6:
            continue
        cc, n = cc_label(labels == int(comp), structure=structure)
        for k in range(1, n + 1):
            i_m4 = _component_i_free_surface_m4(cc == k, p, fill)
            if i_m4 <= 0.0:
                continue
            fsm = rho * i_m4
            total_fsm += fsm
            n_slack += 1
            if comp == generator_api.Comp.FUEL_TANKS:
                fuel_fsm += fsm
            else:
                ballast_fsm += fsm

    correction = total_fsm / float(displacement_t)
    return FreeSurfaceResult(
        evaluated=True,
        correction_m=float(correction),
        total_fsm_t_m=float(total_fsm),
        n_slack_tanks=int(n_slack),
        fuel_fsm_t_m=float(fuel_fsm),
        ballast_fsm_t_m=float(ballast_fsm),
        message="approximate semantic-tank free-surface correction",
    )
