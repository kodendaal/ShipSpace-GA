from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any
import numpy as np


@dataclass(frozen=True)
class Hydrostatics:
    mean_draft_m: float
    aft_draft_m: float
    forward_draft_m: float
    trim_m: float
    trim_angle_deg: float
    volume_m3: float
    displacement_t: float
    kb_m: float
    lcb_m: float
    awp_m2: float
    iwp_transverse_m4: float
    bm_transverse_m: float
    km_transverse_m: float
    lcf_m: float
    cb: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True)
class DeckEnvelope:
    x_m: np.ndarray
    deck_z_m: np.ndarray
    valid: np.ndarray
    raw_deck_z_m: np.ndarray
    step_indices: tuple[int, ...]
    source: str

    @property
    def step_count(self) -> int:
        return len(self.step_indices)


@dataclass(frozen=True)
class ClearanceResult:
    min_freeboard_m: float
    p05_freeboard_m: float
    median_freeboard_m: float
    violation_count: int
    violation_fraction: float
    worst_x_m: float
    required_freeboard_m: float

    @property
    def passes(self) -> bool:
        return self.violation_count == 0


@dataclass(frozen=True)
class WeightBreakdown:
    total_t: float
    kg_m: float
    lcg_m: float
    compartment_t: float
    compartment_kg_m: float
    compartment_lcg_m: float
    lightship_t: float
    lightship_kg_m: float
    lightship_lcg_m: float
    residual_t: float
    residual_kg_m: float
    residual_lcg_m: float
    profile_name: str
    total_mass_basis: str
    raw_compartment_proxy_t: float = 0.0
    compartment_scale: float = 1.0
    reference_displacement_t: float = float("nan")
    reference_lcb_m: float = float("nan")
    residual_lcg_derived: bool = False
    compartment_fraction: float = float("nan")
    lightship_fraction: float = float("nan")
    residual_fraction: float = float("nan")
    mass_model_valid: bool = True
    invalid_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EquilibriumResult:
    status: str
    hydro: Hydrostatics | None
    clearance: ClearanceResult | None
    weight_residual_t: float
    lcb_lcg_residual_m: float
    iterations: int
    message: str

    @property
    def converged(self) -> bool:
        return self.status == "ok"

    @property
    def trim_closed(self) -> bool:
        return self.status == "ok"
