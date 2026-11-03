from __future__ import annotations

import math
from typing import Any, Callable
import numpy as np
from scipy.optimize import brentq, minimize_scalar

from .deck_guard import check_deck_clearance
from .hydrostatics import integrate_hydrostatics, waterline_profile
from .models import DeckEnvelope, EquilibriumResult, Hydrostatics, WeightBreakdown


def _mean_draft_bounds(
    p: Any,
    envelope: DeckEnvelope,
    trim_m: float,
    required_freeboard_m: float,
) -> tuple[float, float] | None:
    _, offset = waterline_profile(p, 0.0, trim_m)
    valid = envelope.valid & np.isfinite(envelope.deck_z_m)
    if not np.any(valid):
        return None

    # All local waterlines must remain above the keel.
    lower = float(np.max(-offset)) + 1e-5
    # Every retained local deck segment, including lower split-deck plateaus,
    # must retain the configured freeboard margin.
    upper = float(
        np.min(envelope.deck_z_m[valid] - required_freeboard_m - offset[valid])
    )
    # Never exceed the top of the main-hull voxel band.
    upper = min(upper, float(p.nz_hull * p.dz_m) - 1e-5)
    if upper <= lower:
        return None
    return lower, upper


def _solve_heave_for_trim(
    mask: np.ndarray,
    p: Any,
    weight_t: float,
    trim_m: float,
    envelope: DeckEnvelope,
    required_freeboard_m: float,
    rho_sw_t_m3: float,
    max_iter: int,
) -> tuple[Hydrostatics | None, str, int]:
    bounds = _mean_draft_bounds(p, envelope, trim_m, required_freeboard_m)
    if bounds is None:
        return None, "no_safe_draft_interval", 0
    lo, hi = bounds
    hlo = integrate_hydrostatics(mask, p, lo, trim_m, rho_sw_t_m3)
    hhi = integrate_hydrostatics(mask, p, hi, trim_m, rho_sw_t_m3)
    if weight_t < hlo.displacement_t - 1e-7:
        return None, "weight_below_keel_solution", 0
    if weight_t > hhi.displacement_t + 1e-7:
        return None, "overloaded_before_deck_limit", 0

    iterations = 0
    for iterations in range(1, max_iter + 1):
        mid = 0.5 * (lo + hi)
        hm = integrate_hydrostatics(mask, p, mid, trim_m, rho_sw_t_m3)
        if hm.displacement_t < weight_t:
            lo = mid
        else:
            hi = mid
        if abs(hm.displacement_t - weight_t) <= max(1e-6, 1e-8 * weight_t):
            return hm, "ok", iterations
    h = integrate_hydrostatics(mask, p, 0.5 * (lo + hi), trim_m, rho_sw_t_m3)
    return h, "ok", iterations


def solve_heave_and_trim(
    mask: np.ndarray,
    p: Any,
    weights: WeightBreakdown,
    envelope: DeckEnvelope,
    *,
    required_freeboard_m: float,
    rho_sw_t_m3: float = 1.025,
    trim_limit_L: float = 0.04,
    trim_grid_points: int = 17,
    weight_tolerance_fraction: float = 1e-6,
    lcb_tolerance_L: float = 0.002,
    max_heave_iterations: int = 45,
) -> EquilibriumResult:
    """Solve vertical-force and longitudinal-moment equilibrium.

    The solver enforces the local split-deck envelope during every heave solve.
    It never uses superstructure as buoyant volume or as the deck reference.
    """
    if weights.total_t <= 0.0:
        return EquilibriumResult(
            status="invalid_weight",
            hydro=None,
            clearance=None,
            weight_residual_t=math.nan,
            lcb_lcg_residual_m=math.nan,
            iterations=0,
            message="Total vessel weight must be positive.",
        )

    trim_limit_m = abs(float(trim_limit_L) * float(p.L))
    trim_values = np.linspace(-trim_limit_m, trim_limit_m, max(5, int(trim_grid_points)))
    cache: dict[float, tuple[Hydrostatics | None, str, int]] = {}

    def evaluate(trim_m: float) -> tuple[Hydrostatics | None, str, int]:
        key = float(trim_m)
        if key not in cache:
            cache[key] = _solve_heave_for_trim(
                mask,
                p,
                weights.total_t,
                key,
                envelope,
                required_freeboard_m,
                rho_sw_t_m3,
                max_heave_iterations,
            )
        return cache[key]

    valid: list[tuple[float, Hydrostatics, float, int]] = []
    for tr in trim_values:
        hydro, status, iterations = evaluate(float(tr))
        if hydro is None or status != "ok":
            continue
        residual = float(hydro.lcb_m - weights.lcg_m)
        valid.append((float(tr), hydro, residual, iterations))

    if not valid:
        # Distinguish a deck-limited overload from other failures.
        statuses = [v[1] for v in cache.values()]
        status = (
            "overloaded_before_deck_limit"
            if "overloaded_before_deck_limit" in statuses
            else "no_equilibrium_solution"
        )
        return EquilibriumResult(
            status=status,
            hydro=None,
            clearance=None,
            weight_residual_t=math.nan,
            lcb_lcg_residual_m=math.nan,
            iterations=sum(v[2] for v in cache.values()),
            message="No safe heave/trim solution exists inside the local deck envelope.",
        )

    # Find a sign change in LCB-LCG and solve trim exactly where possible.
    valid.sort(key=lambda x: x[0])
    final_hydro: Hydrostatics | None = None
    total_iterations = sum(v[3] for v in valid)
    for left, right in zip(valid[:-1], valid[1:]):
        if left[2] == 0.0:
            final_hydro = left[1]
            break
        if left[2] * right[2] <= 0.0:
            def root_fn(trim: float) -> float:
                h, status, _ = evaluate(float(trim))
                if h is None or status != "ok":
                    raise ValueError("invalid trim point inside root bracket")
                return h.lcb_m - weights.lcg_m

            try:
                root_trim = brentq(root_fn, left[0], right[0], xtol=1e-8, rtol=1e-10, maxiter=80)
                final_hydro, _, it = evaluate(float(root_trim))
                total_iterations += it
                break
            except (ValueError, RuntimeError):
                pass

    if final_hydro is None:
        # No sign change: return the best bounded solution, but flag moment closure.
        lo, hi = valid[0][0], valid[-1][0]

        def objective(trim: float) -> float:
            h, status, _ = evaluate(float(trim))
            if h is None or status != "ok":
                return 1e6
            return abs(h.lcb_m - weights.lcg_m)

        opt = minimize_scalar(objective, bounds=(lo, hi), method="bounded", options={"xatol": 1e-7})
        final_hydro, _, it = evaluate(float(opt.x))
        total_iterations += it

    assert final_hydro is not None
    clearance = check_deck_clearance(
        envelope,
        p,
        final_hydro.mean_draft_m,
        final_hydro.trim_m,
        required_freeboard_m,
    )
    wres = float(final_hydro.displacement_t - weights.total_t)
    lres = float(final_hydro.lcb_m - weights.lcg_m)
    wt_tol = max(0.01, abs(weights.total_t) * float(weight_tolerance_fraction))
    l_tol = max(0.02, float(p.L) * float(lcb_tolerance_L))

    # Repaired contract: both heave and longitudinal moment equilibrium are
    # required for status=ok. The best bounded trim remains available as a
    # diagnostic but is explicitly labelled trim_not_closed.
    if not clearance.passes:
        status = "deck_breach"
        message = "Solved waterline violates the local split-deck freeboard envelope."
    elif abs(wres) > wt_tol:
        status = "heave_not_closed"
        message = "Displacement and total weight did not close within tolerance."
    elif abs(lres) > l_tol:
        status = "trim_not_closed"
        message = (
            "Heave closed, but LCB and LCG did not close within tolerance: "
            f"residual {lres:+.3f} m, tolerance {l_tol:.3f} m."
        )
    else:
        status = "ok"
        message = "Heave and trim equilibrium closed with local deck clearance."

    return EquilibriumResult(
        status=status,
        hydro=final_hydro,
        clearance=clearance,
        weight_residual_t=wres,
        lcb_lcg_residual_m=lres,
        iterations=int(total_iterations),
        message=message,
    )
