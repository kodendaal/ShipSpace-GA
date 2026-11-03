from __future__ import annotations

from typing import Any
import math
import numpy as np

from .models import Hydrostatics, WeightBreakdown


def _fill_factor_for(comp: Any, family_cfg: dict) -> float:
    """Return loading/fill multiplier for a semantic class.

    The generator densities are effective class intensities. Fill factors make
    operating-state assumptions explicit instead of silently treating every tank
    or cargo space as full.
    """
    factors = family_cfg.get("compartment_fill_factors", {}) or {}
    name = getattr(comp, "name", str(comp))
    value = factors.get(name, factors.get(str(int(comp)), 1.0))
    return float(np.clip(float(value), 0.0, 1.0))


def raw_compartment_moments(
    assignment: Any,
    generator_api: Any,
    family_cfg: dict | None = None,
) -> tuple[float, float, float]:
    """Return density-proxy mass, LCG and KG from the assignment.

    The density-proxy mass is retained for transparency. It is not automatically
    interpreted as complete vessel mass because the class densities were created
    primarily as relative mass-intensity proxies.
    """
    family_cfg = family_cfg or {}
    vf = generator_api.build_voxel_fields_from_assignment(assignment)
    labels = np.asarray(vf.labels)
    volume = np.asarray(vf.volume_m3, dtype=float)
    valid = (labels != int(generator_api.Comp.EMPTY)) & (volume > 0.0)

    max_label = max(int(c) for c in generator_api.Comp)
    density_lut = np.zeros(max_label + 1, dtype=float)
    ship_type = assignment.layout.params.ship_type
    for c in generator_api.Comp:
        density_lut[int(c)] = (
            float(generator_api.density_for(c, ship_type))
            * _fill_factor_for(c, family_cfg)
        )

    masses = density_lut[labels[valid]] * volume[valid]
    total = float(masses.sum())
    if total <= 0.0:
        p = assignment.layout.params
        return (
            0.0,
            float(assignment.actual_lcg_frac * p.L),
            float(assignment.actual_kg_frac * p.D),
        )
    lcg = float(np.dot(masses, np.asarray(vf.x_m)[valid]) / total)
    kg = float(np.dot(masses, np.asarray(vf.z_m)[valid]) / total)
    return total, lcg, kg


def total_mass_from_basis(
    basis: str,
    p: Any,
    *,
    declared_mass_t: float,
    reference_displacement_t: float | None = None,
    fixed_mass_t: float | None = None,
) -> float:
    """Legacy mass-basis helper.

    `declared_capped_to_freeboard` is deliberately rejected in repaired runs:
    freeboard is a constraint on a fixed weight, not a device for reducing mass.
    """
    if basis == "declared_displacement":
        return float(declared_mass_t)
    if basis in {"reference_loading_displacement", "reference_freeboard_displacement"}:
        if reference_displacement_t is None:
            raise ValueError("reference_displacement_t is required for this mass basis")
        return float(reference_displacement_t)
    if basis == "declared_capped_to_freeboard":
        raise ValueError(
            "declared_capped_to_freeboard is disabled in the repaired model: "
            "freeboard must not redefine vessel mass"
        )
    if basis == "fixed_mass_t":
        if fixed_mass_t is None:
            raise ValueError("fixed_mass_t is required for this mass basis")
        return float(fixed_mass_t)
    raise ValueError(f"unsupported total mass basis: {basis}")


def _mass_from_cfg(
    prefix: str,
    family_cfg: dict,
    reference_displacement_t: float,
) -> float | None:
    fixed = family_cfg.get(f"{prefix}_mass_t")
    if fixed is not None:
        return float(fixed)
    frac = family_cfg.get(f"{prefix}_mass_fraction")
    if frac is not None:
        return float(frac) * float(reference_displacement_t)
    return None


def _invalid_breakdown(
    *,
    profile: str,
    basis: str,
    raw_t: float,
    raw_lcg: float,
    raw_kg: float,
    reference_hydro: Hydrostatics | None,
    reason: str,
) -> WeightBreakdown:
    return WeightBreakdown(
        total_t=0.0,
        kg_m=float(raw_kg),
        lcg_m=float(raw_lcg),
        compartment_t=0.0,
        compartment_kg_m=float(raw_kg),
        compartment_lcg_m=float(raw_lcg),
        lightship_t=0.0,
        lightship_kg_m=0.0,
        lightship_lcg_m=0.0,
        residual_t=0.0,
        residual_kg_m=0.0,
        residual_lcg_m=0.0,
        profile_name=profile,
        total_mass_basis=basis,
        raw_compartment_proxy_t=float(raw_t),
        compartment_scale=0.0,
        reference_displacement_t=(
            float(reference_hydro.displacement_t) if reference_hydro is not None else math.nan
        ),
        reference_lcb_m=(float(reference_hydro.lcb_m) if reference_hydro is not None else math.nan),
        compartment_fraction=math.nan,
        lightship_fraction=math.nan,
        residual_fraction=math.nan,
        mass_model_valid=False,
        invalid_reason=reason,
    )


def build_weight_model(
    assignment: Any,
    p: Any,
    generator_api: Any,
    family_cfg: dict,
    *,
    reference_hydro: Hydrostatics | None = None,
) -> WeightBreakdown:
    """Build the configured fixed-weight model.

    Repaired (`absolute_closure`) sequence:
      1. establish one fixed reference loading displacement from the actual hull;
      2. assign compartment and lightship masses explicitly;
      3. derive or specify residual mass;
      4. optionally derive residual LCG to balance the reference loading state;
      5. hold all masses/moments fixed during the equilibrium solve.

    Family residual closure (`family_residual_closure`) keeps the generator's
    density-derived compartment mass in absolute form using one family-level
    scale. Lightship is fixed by a family mass relation and the remaining
    displacement becomes residual mass. All LCG/VCG assumptions are fixed at
    family level; no per-arrangement residual centre is derived.
    """
    raw_t, raw_lcg, raw_kg = raw_compartment_moments(
        assignment, generator_api, family_cfg
    )
    profile = str(family_cfg.get("profile_name", "unvalidated"))
    mode = str(family_cfg.get("weight_model_mode", "legacy_fractional"))
    basis = str(family_cfg.get("total_mass_basis", "declared_displacement"))

    if mode == "legacy_fractional":
        reference_disp = reference_hydro.displacement_t if reference_hydro else None
        total_t = total_mass_from_basis(
            basis,
            p,
            declared_mass_t=float(p.displacement),
            reference_displacement_t=reference_disp,
            fixed_mass_t=family_cfg.get("fixed_mass_t"),
        )
        comp_frac = float(family_cfg.get("compartment_mass_fraction", 0.50))
        light_frac = float(family_cfg.get("lightship_mass_fraction", 0.35))
        residual_frac_cfg = family_cfg.get("residual_mass_fraction")
        residual_frac = (
            float(residual_frac_cfg)
            if residual_frac_cfg is not None
            else 1.0 - comp_frac - light_frac
        )
        fractions = np.array([comp_frac, light_frac, residual_frac], dtype=float)
        if np.any(fractions < -1e-12) or not np.isclose(fractions.sum(), 1.0, atol=1e-8):
            raise ValueError("legacy weight fractions must be non-negative and sum to 1")
        comp_t, light_t, residual_t = total_t * fractions
        scale = comp_t / max(raw_t, 1e-12)
        residual_lcg_derived = False
    elif mode == "absolute_closure":
        if reference_hydro is None or not np.isfinite(reference_hydro.displacement_t):
            return _invalid_breakdown(
                profile=profile, basis=basis, raw_t=raw_t, raw_lcg=raw_lcg,
                raw_kg=raw_kg, reference_hydro=reference_hydro,
                reason="absolute_closure requires a valid actual-hull reference loading state",
            )
        reference_disp = float(reference_hydro.displacement_t)

        comp_mode = str(family_cfg.get("compartment_mass_mode", "raw_proxy_scaled"))
        if comp_mode == "raw_proxy_absolute":
            scale = float(family_cfg.get("compartment_mass_scale", 1.0))
            comp_t = raw_t * scale
        elif comp_mode == "raw_proxy_scaled":
            target_fraction = family_cfg.get("compartment_mass_fraction")
            if target_fraction is None:
                scale = float(family_cfg.get("compartment_mass_scale", 1.0))
                comp_t = raw_t * scale
            else:
                comp_t = float(target_fraction) * reference_disp
                scale = comp_t / max(raw_t, 1e-12)
        else:
            raise ValueError(f"unknown compartment_mass_mode: {comp_mode}")

        light_t = _mass_from_cfg("lightship", family_cfg, reference_disp)
        if light_t is None:
            return _invalid_breakdown(
                profile=profile, basis=basis, raw_t=raw_t, raw_lcg=raw_lcg,
                raw_kg=raw_kg, reference_hydro=reference_hydro,
                reason="absolute_closure requires lightship_mass_t or lightship_mass_fraction",
            )

        residual_mode = str(family_cfg.get("residual_mass_mode", "closure_to_reference"))
        if residual_mode == "closure_to_reference":
            residual_t = reference_disp - comp_t - light_t
        elif residual_mode == "explicit":
            residual_t = _mass_from_cfg("residual", family_cfg, reference_disp)
            if residual_t is None:
                return _invalid_breakdown(
                    profile=profile, basis=basis, raw_t=raw_t, raw_lcg=raw_lcg,
                    raw_kg=raw_kg, reference_hydro=reference_hydro,
                    reason="explicit residual requires residual_mass_t or residual_mass_fraction",
                )
        else:
            raise ValueError(f"unknown residual_mass_mode: {residual_mode}")

        if min(comp_t, light_t, residual_t) < -1e-8:
            return _invalid_breakdown(
                profile=profile, basis=basis, raw_t=raw_t, raw_lcg=raw_lcg,
                raw_kg=raw_kg, reference_hydro=reference_hydro,
                reason=(
                    "negative component mass: compartment/lightship exceed the "
                    "reference loading displacement"
                ),
            )
        comp_t, light_t, residual_t = map(lambda x: max(0.0, float(x)), (comp_t, light_t, residual_t))
        total_t = comp_t + light_t + residual_t
        residual_lcg_derived = str(family_cfg.get("residual_lcg_mode", "fixed")) == "close_reference_lcb"
    elif mode == "family_residual_closure":
        if reference_hydro is None or not np.isfinite(reference_hydro.displacement_t):
            return _invalid_breakdown(
                profile=profile, basis=basis, raw_t=raw_t, raw_lcg=raw_lcg,
                raw_kg=raw_kg, reference_hydro=reference_hydro,
                reason="family_residual_closure requires a valid actual-hull reference loading state",
            )
        reference_disp = float(reference_hydro.displacement_t)
        scale_cfg = family_cfg.get("compartment_mass_scale")
        if scale_cfg is None or not np.isfinite(float(scale_cfg)) or float(scale_cfg) <= 0.0:
            return _invalid_breakdown(
                profile=profile, basis=basis, raw_t=raw_t, raw_lcg=raw_lcg,
                raw_kg=raw_kg, reference_hydro=reference_hydro,
                reason="family_residual_closure requires a positive family-level compartment_mass_scale",
            )
        scale = float(scale_cfg)
        comp_t = float(raw_t) * scale
        light_t = _mass_from_cfg("lightship", family_cfg, reference_disp)
        if light_t is None:
            return _invalid_breakdown(
                profile=profile, basis=basis, raw_t=raw_t, raw_lcg=raw_lcg,
                raw_kg=raw_kg, reference_hydro=reference_hydro,
                reason="family_residual_closure requires lightship_mass_t or lightship_mass_fraction",
            )
        residual_t = reference_disp - comp_t - light_t
        if residual_t < -1e-8:
            return _invalid_breakdown(
                profile=profile, basis=basis, raw_t=raw_t, raw_lcg=raw_lcg,
                raw_kg=raw_kg, reference_hydro=reference_hydro,
                reason=(
                    "negative residual mass: family compartment scale plus lightship "
                    "exceed the reference loading displacement"
                ),
            )
        residual_t = max(0.0, float(residual_t))
        total_t = float(comp_t + light_t + residual_t)
        residual_lcg_derived = False
        if str(family_cfg.get("residual_lcg_mode", "fixed")) != "fixed":
            return _invalid_breakdown(
                profile=profile, basis=basis, raw_t=raw_t, raw_lcg=raw_lcg,
                raw_kg=raw_kg, reference_hydro=reference_hydro,
                reason=(
                    "family_residual_closure requires residual_lcg_mode=fixed; "
                    "per-arrangement LCG closure is not allowed"
                ),
            )
    else:
        raise ValueError(f"unknown weight_model_mode: {mode}")

    light_kg = float(family_cfg.get("lightship_vcg_D", 0.55)) * float(p.D)
    residual_kg = float(family_cfg.get("residual_vcg_D", 0.62)) * float(p.D)
    light_lcg_mode = str(family_cfg.get("lightship_lcg_mode", "fixed"))
    if light_lcg_mode == "reference_lcb":
        if reference_hydro is None:
            return _invalid_breakdown(
                profile=profile, basis=basis, raw_t=raw_t, raw_lcg=raw_lcg,
                raw_kg=raw_kg, reference_hydro=reference_hydro,
                reason="lightship_lcg_mode=reference_lcb requires reference hydrostatics",
            )
        light_lcg = float(reference_hydro.lcb_m)
    elif light_lcg_mode == "fixed":
        light_lcg = float(family_cfg.get("lightship_lcg_L", 0.50)) * float(p.L)
    else:
        raise ValueError(f"unknown lightship_lcg_mode: {light_lcg_mode}")

    if residual_lcg_derived:
        if residual_t <= 1e-9 or reference_hydro is None:
            residual_lcg = float(family_cfg.get("residual_lcg_L", 0.50)) * float(p.L)
            residual_lcg_derived = False
        else:
            residual_lcg = (
                total_t * float(reference_hydro.lcb_m)
                - comp_t * raw_lcg
                - light_t * light_lcg
            ) / residual_t
    else:
        residual_lcg = float(family_cfg.get("residual_lcg_L", 0.50)) * float(p.L)

    lcg_min = float(family_cfg.get("residual_lcg_L_min", 0.20)) * float(p.L)
    lcg_max = float(family_cfg.get("residual_lcg_L_max", 0.80)) * float(p.L)
    valid = True
    reason = ""
    if not (lcg_min <= residual_lcg <= lcg_max):
        valid = False
        reason = (
            f"residual LCG {residual_lcg / max(float(p.L),1e-12):.3f}L "
            f"outside [{lcg_min/float(p.L):.3f}, {lcg_max/float(p.L):.3f}]L"
        )

    compartment_fraction = float(comp_t / max(total_t, 1e-12))
    lightship_fraction = float(light_t / max(total_t, 1e-12))
    residual_fraction = float(residual_t / max(total_t, 1e-12))
    min_residual_fraction = family_cfg.get("minimum_residual_mass_fraction")
    max_residual_fraction = family_cfg.get("maximum_residual_mass_fraction")
    if min_residual_fraction is not None and residual_fraction < float(min_residual_fraction) - 1e-10:
        valid = False
        reason = (
            f"residual fraction {residual_fraction:.4f} below configured minimum "
            f"{float(min_residual_fraction):.4f}"
        )
    if max_residual_fraction is not None and residual_fraction > float(max_residual_fraction) + 1e-10:
        valid = False
        reason = (
            f"residual fraction {residual_fraction:.4f} above configured maximum "
            f"{float(max_residual_fraction):.4f}"
        )

    total_mz = comp_t * raw_kg + light_t * light_kg + residual_t * residual_kg
    total_mx = comp_t * raw_lcg + light_t * light_lcg + residual_t * residual_lcg
    kg = total_mz / max(total_t, 1e-12)
    lcg = total_mx / max(total_t, 1e-12)

    return WeightBreakdown(
        total_t=float(total_t),
        kg_m=float(kg),
        lcg_m=float(lcg),
        compartment_t=float(comp_t),
        compartment_kg_m=float(raw_kg),
        compartment_lcg_m=float(raw_lcg),
        lightship_t=float(light_t),
        lightship_kg_m=float(light_kg),
        lightship_lcg_m=float(light_lcg),
        residual_t=float(residual_t),
        residual_kg_m=float(residual_kg),
        residual_lcg_m=float(residual_lcg),
        profile_name=profile,
        total_mass_basis=basis,
        raw_compartment_proxy_t=float(raw_t),
        compartment_scale=float(scale),
        reference_displacement_t=(
            float(reference_hydro.displacement_t) if reference_hydro is not None else math.nan
        ),
        reference_lcb_m=(float(reference_hydro.lcb_m) if reference_hydro is not None else math.nan),
        residual_lcg_derived=bool(residual_lcg_derived),
        compartment_fraction=compartment_fraction,
        lightship_fraction=lightship_fraction,
        residual_fraction=residual_fraction,
        mass_model_valid=bool(valid),
        invalid_reason=reason,
    )
