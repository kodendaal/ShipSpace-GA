"""
ship_params.py
==============
Stage 1: Ship Parameterisation Library
---------------------------------------
Defines ship-type configurations and samples physically consistent
parameterisations for synthetic dataset generation.

Design decisions:
- Fixed cell counts per hull (64 x 32 x 24 for the hull library), so the
  cell sizes dx, dy, dz scale with L, B, D
- Superstructure added programmatically above STL hull bounding box
- Deck layers emerge naturally from voxel z-index, not separately defined
- 3D full-grid representation; 2D mid-plane projection derived as needed
- Six ship types across three structural families
- All dimensional ranges validated against published naval architecture references

Compartment taxonomy (Comp enum, indices 0–10):
    VOID, ENGINE_ROOM, MACHINERY, CARGO, ACCOMMODATION, FUEL_TANKS, BALLAST_TANKS,
    STEERING_GEAR, STORES, NAVIGATION (not assigned; the bridge is part of
    ACCOMMODATION), EMPTY (outside the ship)
    Freshwater is folded into machinery/engine at zone resolution (no separate label).
"""

from __future__ import annotations
import os
import numpy as np
from dataclasses import dataclass, field
from typing import Tuple, Dict, Optional, List
from enum import IntEnum


import math


def _env_bool(key: str, default: bool = True) -> bool:
    """Parse a boolean feature flag from the environment."""
    val = os.environ.get(key, "1" if default else "0").strip().lower()
    return val in ("1", "true", "yes", "on")


ENABLE_YACHT_DENSITY = _env_bool("ENABLE_YACHT_DENSITY", True)


# ─────────────────────────────────────────────
# Compartment taxonomy
# ─────────────────────────────────────────────

class Comp(IntEnum):
    VOID             = 0   # structural / empty space
    ENGINE_ROOM      = 1   # main propulsion machinery
    MACHINERY        = 2   # auxiliary machinery, pump rooms, HVAC
    CARGO            = 3   # cargo holds, weapons stores (patrol), tender garage (yacht)
    ACCOMMODATION    = 4   # crew / passenger living spaces
    FUEL_TANKS       = 5   # fuel oil (HFO / MDO) — near ER, low placement
    BALLAST_TANKS    = 6   # segregated ballast water — wing tanks, DB, peaks
    STEERING_GEAR    = 7   # steering gear room — always aft peak, heavy
    STORES           = 8   # provisions, spare parts, bosun stores — near midship
    NAVIGATION       = 9   # not assigned; bridge = top-forward ACCOMMODATION
    EMPTY            = 10  # hull-forced outside (outside hull envelope)


COMP_DENSITY: Dict[Comp, float] = {
    Comp.VOID:             0.05,
    Comp.ENGINE_ROOM:      5.00,
    Comp.MACHINERY:        2.50,
    Comp.CARGO:            2.00,
    Comp.ACCOMMODATION:    0.50,
    Comp.FUEL_TANKS:       0.85,   # HFO/MDO density
    Comp.BALLAST_TANKS:    1.025,  # seawater
    Comp.STEERING_GEAR:    3.50,   # heavy hydraulic / electric gear
    Comp.STORES:           1.50,   # provisions + spares, moderate density
    Comp.NAVIGATION:       0.50,   # bridge / chart room (light)
    Comp.EMPTY:            0.00,
}


def density_for(comp: Comp, ship_type: "ShipType") -> float:
    """
    Effective design density (t/m³). Yacht CARGO denotes crew quarters at 0.5
    when ENABLE_YACHT_DENSITY is set.
    """
    if (
        ENABLE_YACHT_DENSITY
        and comp == Comp.CARGO
        and ship_type == ShipType.YACHT
    ):
        return 0.5
    return COMP_DENSITY.get(comp, 0.0)


# ─────────────────────────────────────────────
# Ship family definitions
# ─────────────────────────────────────────────

class ShipFamily(IntEnum):
    """
    Three structural families that determine deck architecture
    and superstructure placement logic.
    """
    DEEP_CARGO    = 0   # Bulker, Tanker, General Cargo
    WORKING       = 1   # OSV, Patrol
    SLENDER_DISP  = 2   # Motor Yacht


class ShipType(IntEnum):
    BULKER    = 0
    TANKER    = 1
    CARGO     = 2   # General cargo / multipurpose
    OSV       = 3   # Offshore support vessel
    PATROL    = 4
    YACHT     = 5   # Displacement motor yacht


SHIP_FAMILY: Dict[ShipType, ShipFamily] = {
    ShipType.BULKER:   ShipFamily.DEEP_CARGO,
    ShipType.TANKER:   ShipFamily.DEEP_CARGO,
    ShipType.CARGO:    ShipFamily.DEEP_CARGO,
    ShipType.OSV:      ShipFamily.WORKING,
    ShipType.PATROL:  ShipFamily.WORKING,
    ShipType.YACHT:    ShipFamily.SLENDER_DISP,
}


# Per-type transverse side-zone configuration.
# side_beam_frac: target share of full beam for port+stbd bands combined (≈30–45%).
# min_ny_for_scaled_width: below this ny, keep w=1 cell/side (narrow yachts/OSVs).
# n_cargo_blocks: longitudinal cargo side-zone blocks (2–3), snapped to holds.
# Metric wing default (ShipLayout): w_side_m = clip(coeff*B, min, max) per side.
_METRIC_SIDE_WING: Dict[str, object] = {
    "metric_side_width": True,
    "side_width_B_coeff": 0.06,
    "side_width_m_min": 0.76,
    "side_width_m_max": 2.0,
}

SIDE_ZONE_CONFIG: Dict[ShipType, Dict] = {
    ShipType.BULKER:  {
        "enabled": True,
        **_METRIC_SIDE_WING,
        "side_beam_frac": 0.28,  # fallback when B/dy not passed
        "min_ny_for_scaled_width": 8,
        "n_cargo_blocks": 3,
    },
    ShipType.TANKER:  {
        "enabled": True,
        **_METRIC_SIDE_WING,
        "side_beam_frac": 0.20,
        "min_ny_for_scaled_width": 8,
        "n_cargo_blocks": 3,
    },
    ShipType.CARGO:   {
        "enabled": True,
        **_METRIC_SIDE_WING,
        "side_beam_frac": 0.30,
        "min_ny_for_scaled_width": 8,
        "n_cargo_blocks": 2,
    },
    ShipType.OSV:     {
        "enabled": True,
        **_METRIC_SIDE_WING,
        "side_beam_frac": 0.34,
        "min_ny_for_scaled_width": 8,
        "n_cargo_blocks": 2,
    },
    ShipType.PATROL: {
        "enabled": True,
        **_METRIC_SIDE_WING,
        "side_beam_frac": 0.34,
        "min_ny_for_scaled_width": 8,
        "n_cargo_blocks": 2,
    },
    ShipType.YACHT:   {
        "enabled": True,
        **_METRIC_SIDE_WING,
        "side_beam_frac": 0.35,
        "min_ny_for_scaled_width": 8,
        "n_cargo_blocks": 2,
    },
}


def side_zone_width_cells(
    ny: int,
    ship_type: ShipType,
    *,
    B: Optional[float] = None,
    dy_m: Optional[float] = None,
) -> int:
    """
    Transverse side-band width in voxels (per side).

    Narrow hulls (ny < min_ny_for_scaled_width): fixed 1 cell/side.
    Types with ``metric_side_width``: w_m = clip(coeff*B, min, max) per side.
    Otherwise: w ≈ round(side_beam_frac * ny / 2), centre ≥ 1 cell.
    """
    cfg = SIDE_ZONE_CONFIG[ship_type]
    w_max = max(1, (ny - 1) // 2)
    if ny < int(cfg.get("min_ny_for_scaled_width", 8)):
        return min(1, w_max)

    if cfg.get("metric_side_width") and B is not None and dy_m is not None and dy_m > 0:
        coeff = float(cfg.get("side_width_B_coeff", 0.05))
        w_min = float(cfg.get("side_width_m_min", 0.76))
        w_max_m = float(cfg.get("side_width_m_max", 2.0))
        w_m = float(np.clip(coeff * float(B), w_min, w_max_m))
        w = int(round(w_m / float(dy_m)))
        return max(1, min(w_max, w))

    side_frac = float(cfg.get("side_beam_frac", 0.35))
    w = int(round(side_frac * ny / 2.0))
    return max(1, min(w_max, w))


def side_zones_enabled(ship_type: ShipType) -> bool:
    return bool(SIDE_ZONE_CONFIG.get(ship_type, {}).get("enabled", True))


# ─────────────────────────────────────────────
# Ship type dimensional ranges
# ─────────────────────────────────────────────
# All ranges validated against Lloyd's Register rules and
# published reference ships.
# Ratios: B/L and D/L expressed as (min, max) tuples.
# L in metres.

SHIP_DIM_RANGES: Dict[ShipType, Dict] = {
    ShipType.BULKER: {
        "L":   (150, 280),
        "B_L": (0.14, 0.19),  # widened from 0.17: real bulker B/L up to 0.183
        "D_L": (0.058, 0.081),  # real hull-band D/L 0.060–0.079 (n=3 GAs); +margin on max
        "T_D": (0.70, 0.82),     # design draft / D ratio
        "Cb":  (0.80, 0.87),     # block coefficient
    },
    ShipType.TANKER: {
        "L":   (75, 150),   # validation tankers 87–135 m; caps long crude tankers
        "B_L": (0.14, 0.18),
        "D_L": (0.080, 0.095),  # real hull-band D/L 0.083–0.092 (n=3 GAs)
        "T_D": (0.72, 0.84),
        "Cb":  (0.78, 0.86),
    },
    ShipType.CARGO: {
        "L":   (230, 300),  # validation containers 242–286 m; min raised from 80
        "B_L": (0.12, 0.18),  # widened from (0.14,0.17): container_1 B/L=0.125
        "D_L": (0.068, 0.082),  # real hull-band D/L 0.071–0.077 (n=3 GAs); was 0.065–0.120 → +28% deep bias
        "T_D": (0.68, 0.82),
        "Cb":  (0.68, 0.82),  # raised: container ships have high Cb
    },
    ShipType.OSV: {
        "L":   (50, 95),   # extended from 90: real OSV_1 at L=91
        "B_L": (0.18, 0.27),  # widened from 0.24: real OSVs up to 0.266
        "D_L": (0.084, 0.113),  # real hull-band D/L 0.088–0.109 (n=5 GAs); was 0.09–0.17 → +36% deep bias
        "T_D": (0.60, 0.75),
        "Cb":  (0.60, 0.72),
    },
    ShipType.PATROL: {
        "L":   (55, 85),   # validation patrol 60–70 m; upper capped from 165→100→85
        "B_L": (0.12, 0.15),
        "D_L": (0.115, 0.145),  # raised: real edge D/L ~0.12–0.14; synth L-match was ~0.64× shallow
        "T_D": (0.58, 0.72),
        "Cb":  (0.48, 0.60),
    },
    ShipType.YACHT: {
        "L":   (50, 100),  # extended from 60: real yachts 56-99m (validation)
        "B_L": (0.13, 0.22),  # widened from 0.18: real yachts up to 0.214
        "D_L": (0.095, 0.155),  # real hull-band D/L 0.099–0.147 (n=6 GAs); floor raised from 0.085
        "T_D": (0.55, 0.70),
        "Cb":  (0.45, 0.58),
    },
}


# ─────────────────────────────────────────────
# Double-bottom height: beam-scaled metric, not D-fraction
# h_db = clip(coeff * B, min, max) — parallel to transverse side-wing width.
# ─────────────────────────────────────────────

_DB_HEIGHT_B_COEFF = 0.06
_DB_HEIGHT_M_MIN = 1.0
_DB_HEIGHT_M_MAX = 2.5


def sample_db_height_m(B: float) -> float:
    """Physical double-bottom height (metres) from beam."""
    return float(np.clip(_DB_HEIGHT_B_COEFF * float(B), _DB_HEIGHT_M_MIN, _DB_HEIGHT_M_MAX))


def sample_db_height_voxels(
    ship_type: ShipType,
    B: float,
    voxel_size: float = 3.0,
    dz: float = 0.0,
    anisotropic: bool = False,
) -> int:
    """
    Return double-bottom height in number of voxel layers.

    Metric target: h_db = clip(0.06 * B, 1.0, 2.5) m, then ceil(h_db / dz).
    Anisotropic mode enforces >=2 layers for graph visibility.
    DB height does not scale with moulded depth D.
    """
    h_db = sample_db_height_m(B)
    cell_h = dz if (anisotropic and dz > 0) else max(float(voxel_size), 1e-6)
    n = int(math.ceil(h_db / cell_h))
    if anisotropic and dz > 0:
        return max(2, n)
    return max(1, n)


# ─────────────────────────────────────────────
# Superstructure parameterisation
# ─────────────────────────────────────────────
# STLs cover hull up to main deck only.
# Superstructure is added programmatically.
SUPERSTRUCTURE_CONFIG: Dict[ShipType, Dict] = {
    # ── Coordinate convention: lcg_frac = 0 is AFT, 1 is BOW ──────────────
    # Cargo/tanker/bulker: bridge far aft -> lcg_frac near 0
    # OSV: SS forward -> lcg_frac 0.55–0.80 (see ShipType.OSV entry)
    # Patrol / yacht: island or large SS -> lcg_frac ranges on those entries
    #
    # Taper params:
    #   y_taper_per_level : fractional y-width reduction per SS level going up
    #                       0.0 = blunt rectangle, 0.15 = strongly tapered
    #   x_taper_per_level : fractional x-length reduction per SS level going up
    #                       produces raked/swept forward face

    ShipType.BULKER: {
        "lcg_frac":          (0.05, 0.15),  # bridge far aft
        "length_frac":       (0.08, 0.15),
        "width_frac":        (0.70, 0.90),
        "n_decks":           (4, 6),
        "comp_stack":        [Comp.ACCOMMODATION, Comp.ACCOMMODATION, Comp.ACCOMMODATION],
        "y_taper_per_level": 0.05,   # blunt rectangular box — no taper
        "x_taper_per_level": 0.10,
    },
    ShipType.TANKER: {
        "lcg_frac":          (0.05, 0.15),
        "length_frac":       (0.08, 0.15),
        "width_frac":        (0.70, 0.90),
        "n_decks":           (3, 4),
        "comp_stack":        [Comp.ACCOMMODATION, Comp.ACCOMMODATION, Comp.ACCOMMODATION],
        "y_taper_per_level": 0.05,
        "x_taper_per_level": 0.10,
    },
    ShipType.CARGO: {
        "lcg_frac":          (0.15, 0.30),
        "length_frac":       (0.05, 0.08),
        "width_frac":        (0.50, 0.70),
        "n_decks":           (7, 10),
        "comp_stack":        [Comp.ACCOMMODATION, Comp.ACCOMMODATION,
                              Comp.ACCOMMODATION, Comp.ACCOMMODATION],
        "y_taper_per_level": 0.05,   # very slight chamfer only
        "x_taper_per_level": 0.05,
    },
    ShipType.OSV: {
        "lcg_frac":          (0.70, 0.85),  # FIXED: real OSVs have SS forward (was 0.20-0.45)
        "length_frac":       (0.15, 0.28),
        "width_frac":        (0.75, 0.95),
        "n_decks":           (2, 3),
        "comp_stack":        [Comp.ACCOMMODATION, Comp.ACCOMMODATION,
                              Comp.ACCOMMODATION, Comp.ACCOMMODATION],
        "y_taper_per_level": 0.20,   # noticeably streamlined cross-section
        "x_taper_per_level": 0.20,   # raked forward face
    },
    ShipType.PATROL: {
        "lcg_frac":          (0.15, 0.40),  # island aft; synth LCG +0.07 vs real
        "length_frac":       (0.35, 0.60),
        "width_frac":        (0.65, 0.85),
        "n_decks":           (1, 2),   # 1-deck-only regressed yield; real nz_ss=1 is encoding artefact
        "comp_stack":        [Comp.ACCOMMODATION, Comp.ACCOMMODATION,
                              Comp.ACCOMMODATION, Comp.ACCOMMODATION],
        "y_taper_per_level": 0.30,   # was 0.20 — stronger beam taper per deck
        "x_taper_per_level": 0.30,   # was 0.20 — more rake on upper decks
    },
    ShipType.YACHT: {
        "lcg_frac":          (0.40, 0.60),  # forward SS — helps yacht LCG vs (0.30, 0.60)
        "length_frac":       (0.35, 0.60),
        "width_frac":        (0.75, 0.85),
        # Keep 2–3 deck variability. KG control is taper + Stage-4
        # main-deck-first preference (decay on higher SS levels), not a
        # fixed n_decks=2 (that killed programme diversity).
        "n_decks":           (2, 3),
        "comp_stack":        [Comp.ACCOMMODATION, Comp.ACCOMMODATION,
                              Comp.ACCOMMODATION, Comp.ACCOMMODATION],
        "y_taper_per_level": 0.28,   # was 0.20 — stronger upper-deck taper (KG)
        "x_taper_per_level": 0.28,   # was 0.20
    },
}


# Superstructure deck clear heights (metres) — aligned with hull _sample_deck_plan.
SS_DECK_CLEAR_H_M: Tuple[float, float] = (2.4, 2.7)


def sample_ss_layer_stack(
    n_logical_decks: int,
    dz: float,
    rng: np.random.Generator,
    h_bounds: Tuple[float, float] = SS_DECK_CLEAR_H_M,
) -> Tuple[int, List[int]]:
    """
    Convert logical SS deck count to z-layer stack height.

    Isotropic (dz ≈ 3 m): one clear deck ≈ one cell.
    Anisotropic (fine dz): each logical deck spans multiple z-layers so
    physical accommodation height stays ~2.4–2.7 m per deck.
    """
    import math
    if n_logical_decks <= 0:
        return 0, []
    dz = max(float(dz), 1e-6)
    heights_m = rng.uniform(h_bounds[0], h_bounds[1], size=n_logical_decks)
    layers_per = [max(1, int(math.ceil(float(h) / dz))) for h in heights_m]
    return int(sum(layers_per)), layers_per


def ss_logical_deck_index(level: int, layers_per_deck: List[int]) -> int:
    """Map SS z-layer offset (0..nz_ss-1) to logical deck index for taper/comp."""
    if not layers_per_deck:
        return max(0, level)
    acc = 0
    for i, n in enumerate(layers_per_deck):
        if level < acc + n:
            return i
        acc += n
    return len(layers_per_deck) - 1


# ─────────────────────────────────────────────
# Volume budget ranges by ship type
# ─────────────────────────────────────────────
# Sampling targets for each budget key. Compared to realised volume in Stage 4
# via volumetric fractions (native grid; validation uses volume_metrics).

# Targets = volumetric real-GA means.
# Water-fill does not top up ballast; slack goes to cargo (deep-cargo) or accommodation (working types).

VOLUME_BUDGETS: Dict[ShipType, Dict] = {
    ShipType.BULKER: {
        # Real bulker machinery ~1.9%, stores ~1.8%,
        # cargo ~57%. Tighten aux brackets; cargo band shifted to real mean.
        "v_engine":        (0.05, 0.08),
        "v_machinery":     (0.008, 0.020),
        "v_cargo":         (0.53, 0.63),
        "v_stores":        (0.008, 0.018),
        "v_fuel_tanks":    (0.04, 0.08),
        "v_ballast_tanks": (0.13, 0.20),
        "v_accommodation": (0.04, 0.07),
    },
    ShipType.TANKER: {
        "v_engine":        (0.05, 0.08),
        "v_machinery":     (0.02, 0.06),
        "v_cargo":         (0.45, 0.58),
        "v_stores":        (0.02, 0.04),
        "v_fuel_tanks":    (0.035, 0.07),
        "v_ballast_tanks": (0.11, 0.20),
        # Containment: real accom range is wide (0.043–0.137);
        # raise ceiling so the synth envelope reaches the upper real ships.
        "v_accommodation": (0.03, 0.10),
    },
    ShipType.CARGO: {
        "v_engine":        (0.03, 0.07),
        "v_machinery":     (0.015, 0.035),
        "v_cargo":         (0.55, 0.70),
        "v_stores":        (0.005, 0.025),
        "v_fuel_tanks":    (0.04, 0.08),
        "v_ballast_tanks": (0.08, 0.14),
        "v_accommodation": (0.03, 0.06),
    },
    ShipType.OSV: {
        # Containment: lower engine floor to reach small real OSV
        # engines (real min 0.041); raise cargo ceiling to real max (~0.395)
        # and ballast ceiling toward real max (0.168).
        "v_engine":        (0.04, 0.08),
        "v_machinery":     (0.06, 0.10),
        "v_cargo":         (0.16, 0.42),
        "v_stores":        (0.02, 0.07),
        "v_fuel_tanks":    (0.08, 0.16),
        "v_ballast_tanks": (0.10, 0.19),
        "v_accommodation": (0.18, 0.28),
    },
    ShipType.PATROL: {
        # Default patrol cargo ~0.04 (CC real); narrow cargo, raise stores.
        "v_engine":        (0.08, 0.12),
        "v_machinery":     (0.14, 0.24),
        "v_cargo":         (0.03, 0.08),
        "v_stores":        (0.10, 0.15),
        "v_fuel_tanks":    (0.03, 0.07),
        "v_ballast_tanks": (0.04, 0.08),
        # Real GAs (n=3): accom 0.313–0.328 with nz_ss=1; budget must fit
        # MAIN_UPPER + 1–2 SS decks, not SS-island-only deep-cargo grammar.
        "v_accommodation": (0.28, 0.34),
    },
    ShipType.YACHT: {
        # Floor kept at synth p5≈0.070 (real min≈0.062).
        # Trim ceiling only — upper tail was above real max; do not lower floor (GAP risk).
        "v_engine":        (0.045, 0.085),
        "v_machinery":     (0.08, 0.14),
        "v_cargo":         (0.12, 0.22),
        "v_stores":        (0.04, 0.10),
        "v_fuel_tanks":    (0.02, 0.06),
        "v_ballast_tanks": (0.04, 0.09),
        "v_accommodation": (0.28, 0.42),
    },
}

# Water-fill: keys receive slack below budget_cap (ballast excluded).
WATERFILL_PRIORITY: Dict[ShipType, Tuple[str, ...]] = {
    ShipType.BULKER:  ("v_cargo", "v_fuel_tanks", "v_engine"),
    ShipType.TANKER:  ("v_cargo", "v_fuel_tanks"),
    ShipType.CARGO:   ("v_cargo", "v_fuel_tanks"),
    # Slack below cap → productive volume (cargo then accom), not VOID.
    ShipType.OSV:     ("v_cargo", "v_accommodation"),
    ShipType.PATROL:  ("v_cargo", "v_accommodation"),
    ShipType.YACHT:   ("v_cargo", "v_accommodation"),
}

# Type-dependent assignable-volume cap (remainder → VOID).
# Deep-cargo tends high; working types allow more structural slack.
BUDGET_CAP_RANGE: Dict[ShipType, Tuple[float, float]] = {
    ShipType.BULKER:  (0.92, 0.98),
    ShipType.TANKER:  (0.92, 0.98),
    ShipType.CARGO:   (0.90, 0.98),
    ShipType.OSV:     (0.88, 0.96),
    ShipType.PATROL:  (0.90, 0.98),
    ShipType.YACHT:   (0.88, 0.95),
}


# ─────────────────────────────────────────────
# Hard spatial rules
# ─────────────────────────────────────────────
# Expressed as fractions of ship length from aft (cx: 0=aft, 1=bow)
# These are enforced during compartment assignment, not sampling.

SPATIAL_RULES = {
    # ENGINE_ROOM must occupy only the aft region (default for deep-cargo)
    "er_max_cx":       0.30,   # ENGINE_ROOM forward limit: aft 30% of ship

    # Collision bulkhead (IMO SOLAS II-1): forward of this = fore peak tank only
    "fwd_peak_min_cx": 0.90,   # forward 7% reserved for fore peak
    "fwd_peak_max_cx": 1.00,

    # Aft peak (steering gear compartment)
    "aft_peak_max_cx": 0.04,   # aft 4% reserved

    # ACCOMMODATION deck restrictions
    "acc_min_deck_frac": 0.60,  # accommodation / MAIN_UPPER / ENGINE_UPPER
                                 # only at or above 60% of hull depth

    # ENGINE_ROOM contiguity: enforced in generator (no splitting)

    # FUEL_TANKS proximity to ENGINE_ROOM (supply piping)
    # Soft preference: fuel tanks should be in the aft half of the ship
    "fuel_tanks_max_cx": 0.55,  # fuel tanks strongly preferred aft of midship

    # BALLAST_TANKS placement preferences
    # Ballast tanks are preferred in the double-bottom tier and wing positions
    # This is a soft preference applied via scoring, not a hard constraint
    "ballast_prefer_db": True,

    # STORES placement — prefer midship area for ease of access
    "stores_prefer_cx_range": (0.30, 0.70),

    # FUEL_TANKS must not sit directly above ENGINE_ROOM (fire safety).
    # Greedy scoring excludes those zones; validate_assignment fails if any remain.
    "tanks_above_er_penalty": True,

    # STEERING_GEAR: always in AFT_PEAK (enforced via zone eligibility)
    "steering_gear_aft_only": True,
}

# Per-ship sampled ranges for structural peak boundaries (fraction of L, aft->bow).
# These are sampled in Stage 1 and carried in ShipParameterization.
PEAK_BOUNDARY_RANGES = {
    "aft_peak_max_cx": (0.04, 0.08),  # aft peak spans 4–8% of L
    "fwd_peak_min_cx": (0.90, 0.95),  # fore peak starts at 90–95% of L
}

# Type-dependent overrides for spatial rules
# ER position varies by ship type: deep-cargo ships have ER far aft,
# working vessels and yachts can have ER at midship
ER_MAX_CX_BY_TYPE: Dict[ShipType, float] = {
    ShipType.BULKER:  0.30,  # ER always aft
    ShipType.TANKER:  0.30,
    ShipType.CARGO:   0.30,
    ShipType.OSV:     0.65,  # ER can be midship, cargo deck aft
    ShipType.YACHT:   0.50,  # ER amidships is common
    ShipType.PATROL: 0.55,  # distributed propulsion
}

# Per-ship ER forward-boundary jitter (deep-cargo + OSV). Default types use
# the fixed ER_MAX_CX_BY_TYPE value.
ER_MAX_CX_RANGE: Dict[ShipType, Tuple[float, float]] = {
    ShipType.BULKER: (0.22, 0.34),
    ShipType.TANKER: (0.22, 0.34),
    ShipType.CARGO:  (0.22, 0.36),
    ShipType.OSV:    (0.58, 0.72),  # forward machinery block; real ER cx ~0.55–0.65
}


# ─────────────────────────────────────────────
# LCG and KG target ranges by ship type
# ─────────────────────────────────────────────
# LCG expressed as fraction of L from aft (0=aft, 1=bow)
# KG expressed as fraction of D
# These are sampling targets — the generator tries to hit them.

PHYSICS_TARGETS: Dict[ShipType, Dict] = {
    # Explicit wing/double-bottom ballast volumes lower the achievable KG.
    # Ranges below straddle the measured achieved p10-p90 (parametric,
    # n=50/type, post budget-rescale) with headroom so steering is still
    # exercised, while staying within kg_tol of the reachable centre. These
    # ranges govern generation yield only; they are not a claim about the
    # reachable CG envelope.
    ShipType.BULKER: {
        "lcg_frac": (0.42, 0.52),     # achieved p10-p90: 0.42-0.44
        "kg_D_frac": (0.45, 0.62),    # achieved p10-p90: 0.53-0.56
    },
    ShipType.TANKER: {
        "lcg_frac": (0.42, 0.52),     # achieved: 0.44-0.46
        "kg_D_frac": (0.45, 0.62),    # achieved: 0.53-0.55
    },
    ShipType.CARGO: {
        "lcg_frac": (0.42, 0.53),     # achieved: 0.42-0.45
        "kg_D_frac": (0.46, 0.64),    # achieved: 0.54-0.58
    },
    ShipType.OSV: {
        "lcg_frac": (0.40, 0.55),     # achieved: 0.40-0.49
        "kg_D_frac": (0.44, 0.62),    # achieved: 0.50-0.55
    },
    ShipType.PATROL: {
        "lcg_frac": (0.40, 0.52),     # achieved p10-p90 0.43-0.46 (aft machinery/stores clustering)
        "kg_D_frac": (0.46, 0.64),    # achieved: 0.52-0.58
    },
    ShipType.YACHT: {
        # Containment: real yacht LCG is 0.437–0.528 (more
        # forward); shift+widen target so achieved (lags aft) can reach it.
        "lcg_frac": (0.42, 0.56),     # real 0.437-0.528
        "kg_D_frac": (0.46, 0.63),    # achieved p10-p90 0.51-0.57
    },
}


# ─────────────────────────────────────────────
# ShipParameterization dataclass
# ─────────────────────────────────────────────

@dataclass
class ShipParameterization:
    """
    Complete parameterisation for one ship instance.

    All geometric values are in metres.
    Voxel grid dimensions: nx (longitudinal), ny (transverse), nz (vertical).
    Coordinate convention: 
        x: 0=aft, nx-1=bow  (matches cx_norm = x/nx)
        y: 0=port, ny-1=starboard (symmetric about y=ny/2)
        z: 0=keel, nz_hull-1=main deck, nz_hull+ = superstructure
    """
    # Identity
    ship_type:      ShipType
    ship_family:    ShipFamily
    stl_id:         Optional[str]    = None   # which STL was used (None = parametric)

    # Principal dimensions
    L:  float = 0.0   # length between perpendiculars (m)
    B:  float = 0.0   # moulded breadth (m)
    D:  float = 0.0   # moulded depth to main deck (m)
    T:  float = 0.0   # design draft (m)
    Cb: float = 0.0   # block coefficient

    # Bow and stern taper parameters (used for parametric hull generation)
    bow_taper_frac:   float = 0.15   # fraction of L where bow taper starts (from fwd)
    stern_taper_frac: float = 0.08   # fraction of L where stern taper starts (from aft)
    bow_taper_angle:  float = 25.0   # half-angle of bow taper (degrees)
    stern_taper_angle: float = 20.0  # half-angle of stern taper

    # Voxel grid
    voxel_size:  float = 3.0   # uniform voxel edge length (m), isotropic grids
    # Anisotropic per-axis spacing (m). In isotropic mode all three equal
    # voxel_size. In anisotropic mode they are derived per-ship from fixed grid
    # counts: dx=L/nx, dy=B/ny, dz=D/nz_hull. Physics and volume MUST use these,
    # not voxel_size**3, once anisotropy is active.
    dx: float = 0.0            # longitudinal cell size (0 → falls back to voxel_size)
    dy: float = 0.0            # transverse cell size
    dz: float = 0.0            # vertical cell size
    anisotropic: bool = False  # True when grid built from fixed counts (deck plan active)
    nx: int = 0                # longitudinal voxels (aft to bow)
    ny: int = 0                # transverse voxels (port to starboard)
    nz_hull: int = 0           # vertical voxels within main hull (keel to main deck)
    nz_ss:   int = 0           # z-layers stacked above hull (metric-derived)
    nz_total: int = 0          # nz_hull + nz_ss
    ss_n_decks: int = 0        # logical accommodation decks (SUPERSTRUCTURE_CONFIG n_decks)
    ss_layers_per_deck: list = field(default_factory=list)  # z cells per logical SS deck

    # Double bottom layer index (z index of inner bottom deck)
    db_layers: int = 1         # number of voxel layers in double bottom

    # Sampled peak boundaries (fractions of ship length from aft).
    # If left as None, fallback defaults from SPATIAL_RULES are used.
    aft_peak_max_cx_sampled: Optional[float] = None
    fwd_peak_min_cx_sampled: Optional[float] = None
    er_max_cx_sampled: Optional[float] = None  # sampled per ship; else ER_MAX_CX_BY_TYPE

    # Superstructure parameters
    ss_lcg_frac:    float = 0.12   # SS centroid from aft / L  (0=aft, 1=bow)
    ss_length_frac: float = 0.10   # SS length / L
    ss_width_frac:  float = 0.80   # SS width / B (at base level)
    ss_comp_stack:  list  = field(default_factory=list)

    # Superstructure taper (how much each successive level shrinks)
    # 0.0 = pure rectangle; 0.15 = strongly streamlined upper decks
    ss_y_taper_per_level: float = 0.00  # fractional width reduction per level
    ss_x_taper_per_level: float = 0.00  # fractional length reduction per level
    # Note: forecastle is captured directly from STL hull geometry via
    # top_deck_z in HullMaskResult. No forecastle parameters are stored here.

    # Physics targets
    target_lcg_frac: float = 0.48  # target LCG / L from aft
    target_kg_frac:  float = 0.62  # target KG / D

    # Volume budget targets (fractions of hull volume)
    budget: Dict[str, float] = field(default_factory=dict)
    # Stage-1 budget as originally sampled (provenance). After the Stage-3.5
    # capacity rescaler runs, `budget` holds the EFFECTIVE (feasible) targets
    # used for assignment, QC, and the `cond` vector; `budget_sampled` keeps
    # the raw draw for analysis. None until the rescaler runs.
    budget_sampled: Optional[Dict[str, float]] = None

    # Displacement (tonnes, approximate)
    displacement: float = 0.0

    def __post_init__(self):
        if self.ss_n_decks <= 0 and self.nz_ss > 0:
            self.ss_n_decks = self.nz_ss
        if not self.ss_layers_per_deck and self.nz_ss > 0 and self.ss_n_decks > 0:
            base, rem = divmod(self.nz_ss, self.ss_n_decks)
            self.ss_layers_per_deck = [
                base + (1 if i < rem else 0) for i in range(self.ss_n_decks)
            ]
        self.nz_total = self.nz_hull + self.nz_ss

    @property
    def ss_height_m(self) -> float:
        """Physical height of the superstructure stack above main deck (m)."""
        return self.nz_ss * self.dz_m

    # ── resolved physical cell spacing (single source of truth) ──────────
    # In isotropic mode dx/dy/dz are 0 and these resolve to voxel_size.
    # In anisotropic mode they return the per-axis
    # spacing. ALL physics/volume code should read these, never voxel_size**3.

    @property
    def dx_m(self) -> float:
        return self.dx if self.dx > 0 else self.voxel_size

    @property
    def dy_m(self) -> float:
        return self.dy if self.dy > 0 else self.voxel_size

    @property
    def dz_m(self) -> float:
        return self.dz if self.dz > 0 else self.voxel_size

    @property
    def cell_volume(self) -> float:
        """Physical volume of one grid cell (m³)."""
        return self.dx_m * self.dy_m * self.dz_m

    # ── derived grid properties ──────────────────────────────────────────

    @property
    def grid_shape(self) -> Tuple[int, int, int]:
        return (self.nx, self.ny, self.nz_total)

    @property
    def hull_shape(self) -> Tuple[int, int, int]:
        """Grid shape for main hull only (excluding superstructure)."""
        return (self.nx, self.ny, self.nz_hull)

    @property
    def ss_x_start(self) -> int:
        """Voxel x-index where superstructure starts (aft end)."""
        cx_aft = self.ss_lcg_frac - self.ss_length_frac / 2
        return max(0, int(cx_aft * self.nx))

    @property
    def ss_x_end(self) -> int:
        """Voxel x-index where superstructure ends (fwd end)."""
        cx_fwd = self.ss_lcg_frac + self.ss_length_frac / 2
        return min(self.nx, int(cx_fwd * self.nx))

    @property
    def ss_y_start(self) -> int:
        """Voxel y-index where superstructure starts (port side)."""
        margin = int(self.ny * (1 - self.ss_width_frac) / 2)
        return max(0, margin)

    @property
    def ss_y_end(self) -> int:
        """Voxel y-index where superstructure ends (starboard)."""
        return min(self.ny, self.ny - self.ss_y_start)

    @property
    def er_max_x(self) -> int:
        """Maximum x-index ENGINE_ROOM is allowed to occupy (type-dependent)."""
        if self.er_max_cx_sampled is not None:
            er_cx = self.er_max_cx_sampled
        else:
            er_cx = ER_MAX_CX_BY_TYPE.get(self.ship_type, SPATIAL_RULES["er_max_cx"])
        return int(er_cx * self.nx)

    @property
    def fwd_peak_x_start(self) -> int:
        """x-index where fore peak begins (reserved for fore peak tank)."""
        fwd_pk = self.fwd_peak_min_cx_sampled
        if fwd_pk is None:
            fwd_pk = SPATIAL_RULES["fwd_peak_min_cx"]
        return int(fwd_pk * self.nx)

    @property
    def aft_peak_x_end(self) -> int:
        """x-index where aft peak ends (reserved for aft peak)."""
        aft_pk = self.aft_peak_max_cx_sampled
        if aft_pk is None:
            aft_pk = SPATIAL_RULES["aft_peak_max_cx"]
        return int(aft_pk * self.nx)

    @property
    def acc_min_z(self) -> int:
        """
        Minimum z-index ACCOMMODATION is allowed to occupy.

        Also defines the MAIN_LOWER / MAIN_UPPER split in Stage 3. For nz_hull >= 3,
        ensures acc_min_z > db_layers so a lower-hold tier exists (avoids int(frac*nz)
        colliding with db). Capped at nz_hull - 1 so one z-layer remains for MAIN_UPPER.
        For nz_hull < 3, returns db_layers (cannot form three hull tiers).
        """
        nz = self.nz_hull
        db = self.db_layers
        frac = SPATIAL_RULES["acc_min_deck_frac"]
        if nz < 3:
            return db
        acc = max(db + 1, int(frac * nz))
        return min(acc, nz - 1)

    def summary(self) -> str:
        lines = [
            f"Ship type:    {self.ship_type.name}",
            f"Dimensions:   L={self.L:.1f}m  B={self.B:.1f}m  D={self.D:.1f}m  T={self.T:.1f}m",
            f"Block coeff:  Cb={self.Cb:.3f}",
            f"Displacement: {self.displacement:.0f} t",
            f"Voxel grid:   {self.nx} x {self.ny} x {self.nz_total}  "
            f"(hull: {self.nz_hull} layers, SS: {self.ss_n_decks} decks / {self.nz_ss} layers)",
            f"Voxel size:   {self.voxel_size:.1f}m",
            f"DB layers:    {self.db_layers}",
            f"Targets:      LCG={self.target_lcg_frac:.3f}L  KG={self.target_kg_frac:.3f}D",
            f"SS position:  {self.ss_lcg_frac:.2f}L  len={self.ss_length_frac:.2f}L",
        ]
        return "\n".join(lines)


# ─────────────────────────────────────────────
# Sampler
# ─────────────────────────────────────────────

class ShipParameterizationSampler:
    """
    Samples a complete ShipParameterization for a given ship type.

    Usage:
        sampler = ShipParameterizationSampler(voxel_size=3.0, rng_seed=42)
        params = sampler.sample(ShipType.BULKER)
    """

    def __init__(self, voxel_size: float = 3.0, rng_seed: Optional[int] = None,
                 grid_counts: Optional[Tuple[int, int, int]] = None):
        """
        voxel_size  : isotropic cell edge (m). Used when grid_counts is None.
        grid_counts : (Nx, Ny, Nz_hull) for ANISOTROPIC mode. When given, the grid
                      has fixed per-axis counts and per-ship spacing dx=L/Nx,
                      dy=B/Ny, dz=D/Nz_hull — so decks/DB/wings get a consistent
                      number of cells regardless of vessel size. voxel_size is then
                      only a nominal fallback.
        """
        self.voxel_size = voxel_size
        self.grid_counts = grid_counts
        self.rng = np.random.default_rng(rng_seed)

    def _u(self, lo: float, hi: float) -> float:
        """Uniform sample in [lo, hi]."""
        return float(self.rng.uniform(lo, hi))

    def _ui(self, lo: int, hi: int) -> int:
        """Uniform integer sample in [lo, hi] inclusive."""
        return int(self.rng.integers(lo, hi + 1))

    def _round_to_voxel(self, val: float) -> float:
        """Round a dimension to the nearest voxel boundary."""
        return round(val / self.voxel_size) * self.voxel_size

    def sample(self, ship_type: ShipType,
               stl_id: Optional[str] = None) -> ShipParameterization:
        """
        Sample a physically consistent parameterisation for the given ship type.
        Returns a fully populated ShipParameterization.
        """
        dim = SHIP_DIM_RANGES[ship_type]
        ss_cfg = SUPERSTRUCTURE_CONFIG[ship_type]
        phys = PHYSICS_TARGETS[ship_type]
        budget_cfg = VOLUME_BUDGETS[ship_type]

        # ── Principal dimensions ──────────────────────────────────────
        L_raw = self._u(*dim["L"])
        B_raw = L_raw * self._u(*dim["B_L"])
        D_raw = L_raw * self._u(*dim["D_L"])
        # Grid dimensions computed below; L, B, D will be set from grid.
        T_frac = self._u(*dim["T_D"])
        Cb = self._u(*dim["Cb"])

        # ── Voxel grid dimensions ─────────────────────────────────────
        import math
        if self.grid_counts is not None:
            # ANISOTROPIC mode: fixed per-axis counts, per-ship spacing.
            nx, ny, nz_mid = self.grid_counts
            nx = max(8, int(nx)); ny = max(2, int(ny)); nz_mid = max(3, int(nz_mid))
            dx = L_raw / nx
            dy = B_raw / ny
            dz = D_raw / nz_mid
            L = L_raw; B = B_raw; D = D_raw   # dimensions preserved exactly
            nz_hull = nz_mid
        else:
            # Isotropic mode: derive counts from a single cell size.
            nx = max(8,  math.ceil(L_raw / self.voxel_size))
            ny = max(2,  math.ceil(B_raw / self.voxel_size))
            nz_mid = max(3, math.ceil(D_raw / self.voxel_size))  # midship deck layers
            dx = dy = dz = self.voxel_size
            L = nx * self.voxel_size
            B = ny * self.voxel_size
            D = nz_mid * self.voxel_size  # D = midship moulded depth (flat deck level)
            nz_hull = nz_mid  # will be updated after forecastle is sampled

        # ── Double bottom layers ──────────────────────────────────────
        db_layers = sample_db_height_voxels(
            ship_type, B, self.voxel_size,
            dz=dz, anisotropic=self.grid_counts is not None,
        )

        # ── Superstructure ────────────────────────────────────────────
        ss_lcg  = self._u(*ss_cfg["lcg_frac"])
        ss_len  = self._u(*ss_cfg["length_frac"])
        ss_wid  = self._u(*ss_cfg["width_frac"])
        ss_n_decks = self._ui(*ss_cfg["n_decks"])
        nz_ss, ss_layers_per_deck = sample_ss_layer_stack(ss_n_decks, dz, self.rng)
        ss_comp = list(ss_cfg["comp_stack"][:ss_n_decks])
        ss_y_taper = float(ss_cfg["y_taper_per_level"])
        ss_x_taper = float(ss_cfg["x_taper_per_level"])
        # Forecastle is captured from STL geometry — no sampled parameters needed.

        nz_hull = nz_mid

        # ── Physics targets ───────────────────────────────────────────
        target_lcg = self._u(*phys["lcg_frac"])
        target_kg  = self._u(*phys["kg_D_frac"])

        # ── Volume budgets ────────────────────────────────────────────
        budget = self._sample_budget(budget_cfg, target_kg, ship_type=ship_type)

        T = D * T_frac

        # ── Displacement (approximate) ────────────────────────────────
        # Δ = Cb x L x B x T x ρ_sw
        rho_sw = 1.025   # t/m³
        displacement = Cb * L * B * T * rho_sw

        # ── Bow / stern taper (for parametric hulls) ──────────────────
        bow_taper_frac   = self._u(0.10, 0.20)
        stern_taper_frac = self._u(0.05, 0.12)
        bow_taper_angle  = self._u(20.0, 35.0)
        stern_taper_angle = self._u(15.0, 28.0)

        # ── Structural peak boundaries (sampled per ship) ─────────────
        aft_peak_max_cx = self._u(*PEAK_BOUNDARY_RANGES["aft_peak_max_cx"])
        fwd_peak_min_cx = self._u(*PEAK_BOUNDARY_RANGES["fwd_peak_min_cx"])
        er_max_cx = None
        if ship_type in ER_MAX_CX_RANGE:
            er_max_cx = self._u(*ER_MAX_CX_RANGE[ship_type])

        return ShipParameterization(
            ship_type=ship_type,
            ship_family=SHIP_FAMILY[ship_type],
            stl_id=stl_id,
            L=L, B=B, D=D, T=T, Cb=Cb,
            bow_taper_frac=bow_taper_frac,
            stern_taper_frac=stern_taper_frac,
            bow_taper_angle=bow_taper_angle,
            stern_taper_angle=stern_taper_angle,
            voxel_size=self.voxel_size,
            dx=dx, dy=dy, dz=dz,
            anisotropic=self.grid_counts is not None,
            nx=nx, ny=ny,
            nz_hull=nz_hull,
            nz_ss=nz_ss,
            nz_total=nz_hull + nz_ss,
            ss_n_decks=ss_n_decks,
            ss_layers_per_deck=ss_layers_per_deck,
            db_layers=db_layers,
            aft_peak_max_cx_sampled=aft_peak_max_cx,
            fwd_peak_min_cx_sampled=fwd_peak_min_cx,
            er_max_cx_sampled=er_max_cx,
            ss_lcg_frac=ss_lcg,
            ss_length_frac=ss_len,
            ss_width_frac=ss_wid,
            ss_comp_stack=ss_comp,
            ss_y_taper_per_level=ss_y_taper,
            ss_x_taper_per_level=ss_x_taper,
            target_lcg_frac=target_lcg,
            target_kg_frac=target_kg,
            budget=budget,
            displacement=displacement,
        )

    def _sample_budget(self, cfg: Dict, target_kg: float,
                       ship_type: "ShipType" = None) -> Dict[str, float]:
        """
        Sample volume budgets with KG-responsive scaling.

        High target_kg -> more ACCOMMODATION (light, high up)
        Low  target_kg -> more heavy tanks and cargo (low down)

        Returns dict of {comp_name: volume_fraction}.
        Steering gear is not budgeted — mandatory aft-peak assignment in Stage 4.

        Target fractions are compared to realised volume in Stage 4 using
        sum(available_cells) over all zones (hull + superstructure).

        Budget cap is type-dependent: sampled from BUDGET_CAP_RANGE.
        """
        deep_cargo_types = {ShipType.BULKER, ShipType.TANKER, ShipType.CARGO}
        cap_lo, cap_hi = BUDGET_CAP_RANGE.get(ship_type, (0.88, 0.98))
        budget_cap = self._u(cap_lo, cap_hi)

        # KG response factor: 0 = low KG target, 1 = high KG target
        kg_resp = np.clip((target_kg - 0.50) / (0.75 - 0.50), 0.0, 1.0)

        # Base samples — engine and machinery are KG-neutral
        v_eng  = self._u(*cfg["v_engine"])
        v_mach = self._u(*cfg["v_machinery"])

        # Cargo: for deep-cargo types, sample from upper portion of range
        # first (cargo-dominant allocation). For others, standard sampling.
        cargo_lo, cargo_hi = cfg["v_cargo"]
        if ship_type in deep_cargo_types:
            # Cargo-first: bias toward upper 60% of range
            v_cargo = self._u(cargo_lo * 0.4 + cargo_hi * 0.6, cargo_hi)
        else:
            v_cargo = self._u(cargo_lo, cargo_hi)
        v_cargo = v_cargo - kg_resp * (cargo_hi - cargo_lo) * 0.15
        v_cargo = np.clip(v_cargo, cargo_lo * 0.85, cargo_hi)

        # Stores: KG-neutral (moderate density, mid placement)
        v_stores = self._u(*cfg["v_stores"])

        # KG-responsive: accommodation scales up with kg_resp
        acc_lo, acc_hi = cfg["v_accommodation"]
        acc_range = acc_hi - acc_lo
        v_acc = np.clip(self._u(acc_lo, acc_hi) + kg_resp * acc_range * 0.5,
                        acc_lo, acc_hi * 1.4)

        # KG-responsive: fuel tanks scale down with kg_resp (heavy, low placement)
        fuel_lo, fuel_hi = cfg["v_fuel_tanks"]
        fuel_range = fuel_hi - fuel_lo
        v_fuel = np.clip(self._u(fuel_lo, fuel_hi) - kg_resp * fuel_range * 0.3,
                         fuel_lo * 0.7, fuel_hi)

        # KG-responsive: ballast tanks scale down with kg_resp (heavy, low placement)
        # Soften KG-down so high target-KG ships still reach ballast lo.
        ballast_lo, ballast_hi = cfg["v_ballast_tanks"]
        ballast_range = ballast_hi - ballast_lo
        v_ballast = np.clip(self._u(ballast_lo, ballast_hi) - kg_resp * ballast_range * 0.18,
                            ballast_lo * 0.85, ballast_hi)

        # Combined cap (type-dependent — see docstring)
        keys = ["v_engine", "v_machinery", "v_cargo", "v_stores",
                "v_accommodation", "v_fuel_tanks", "v_ballast_tanks"]
        vals = {"v_engine": v_eng, "v_machinery": v_mach, "v_cargo": v_cargo,
                "v_stores": v_stores, "v_accommodation": v_acc,
                "v_fuel_tanks": v_fuel, "v_ballast_tanks": v_ballast}
        # Per-key upper limits for the fill-up pass: never push a key beyond
        # its documented real-ship range max (a uniform scale-up would
        # produce e.g. patrol machinery 0.24 > range hi 0.22).
        his = {k: float(cfg[k][1]) for k in keys}
        # accommodation already allows a KG-responsive stretch to hi*1.4
        his["v_accommodation"] = float(cfg["v_accommodation"][1]) * 1.4

        total = sum(vals.values())
        if total > budget_cap:
            scale = budget_cap / total
            for k in keys:
                vals[k] *= scale
            total = budget_cap
        elif total < budget_cap - 0.05:
            # Fill productive comps (cargo / accom / fuel), not ballast.
            if ship_type in deep_cargo_types:
                void_frac = self._u(0.04, 0.12)
            else:
                void_frac = self._u(0.03, 0.08)
            target_assigned = budget_cap - void_frac

            priority = WATERFILL_PRIORITY.get(ship_type, ("v_cargo",))
            fill_keys = list(priority) + [
                k for k in keys
                if k not in priority and k != "v_ballast_tanks"
            ]
            for _ in range(6):
                deficit = target_assigned - sum(vals.values())
                if deficit <= 1e-6:
                    break
                headroom = {k: max(0.0, his[k] - vals[k]) for k in fill_keys}
                room_total = sum(headroom.values())
                if room_total <= 1e-9:
                    break
                take = min(deficit, room_total)
                for k in fill_keys:
                    vals[k] += take * headroom[k] / room_total
            total = sum(vals.values())

        v_eng, v_mach, v_cargo = vals["v_engine"], vals["v_machinery"], vals["v_cargo"]
        v_stores, v_acc = vals["v_stores"], vals["v_accommodation"]
        v_fuel, v_ballast = vals["v_fuel_tanks"], vals["v_ballast_tanks"]

        return {
            "engine":           float(v_eng),
            "machinery":        float(v_mach),
            "cargo":            float(v_cargo),
            "stores":           float(v_stores),
            "accommodation":    float(v_acc),
            "fuel_tanks":       float(v_fuel),
            "ballast_tanks":    float(v_ballast),
            "void":             float(1.0 - total),
        }

    def sample_batch(self, n: int,
                     type_weights: Optional[Dict[ShipType, float]] = None
                     ) -> list[ShipParameterization]:
        """
        Sample n parameterisations with optional ship type weighting.

        Default weights produce roughly equal representation across types.
        Adjust to oversample rare types (e.g., patrol).
        """
        types = list(ShipType)

        if type_weights is None:
            # Equal weighting — 15 STLs: 6 cargo, 6 work, 2 yacht, 1 patrol
            # Boost patrol and yacht slightly to compensate for fewer STLs
            type_weights = {
                ShipType.BULKER:  2.0,
                ShipType.TANKER:  2.0,
                ShipType.CARGO:   2.0,
                ShipType.OSV:     2.0,
                ShipType.PATROL: 2.5,  # boosted
                ShipType.YACHT:   2.5,  # boosted
            }

        weights = np.array([type_weights.get(t, 1.0) for t in types])
        weights /= weights.sum()

        chosen_types = self.rng.choice(len(types), size=n, p=weights)
        return [self.sample(ShipType(int(idx))) for idx in chosen_types]


# ─────────────────────────────────────────────
# Quick validation and diagnostics
# ─────────────────────────────────────────────

def validate_parameterization(p: ShipParameterization) -> Tuple[bool, list[str]]:
    """
    Run sanity checks on a sampled parameterisation.
    Returns (is_valid, list_of_warnings).
    """
    warnings = []

    # Dimension ratios — use SHIP_DIM_RANGES (same source as sampler)
    dim = SHIP_DIM_RANGES[p.ship_type]
    bl_lo, bl_hi = dim["B_L"]
    dl_lo, dl_hi = dim["D_L"]
    td_lo, td_hi = dim["T_D"]
    _eps = 1e-3
    if not (bl_lo - _eps <= p.B / p.L <= bl_hi + _eps):
        warnings.append(f"B/L={p.B/p.L:.3f} outside expected range [{bl_lo}, {bl_hi}]")
    if not (dl_lo - _eps <= p.D / p.L <= dl_hi + _eps):
        warnings.append(f"D/L={p.D/p.L:.3f} outside expected range [{dl_lo}, {dl_hi}]")
    if not (td_lo - _eps <= p.T / p.D <= td_hi + _eps):
        warnings.append(f"T/D={p.T/p.D:.3f} outside expected range [{td_lo}, {td_hi}]")

    # Grid size sensibility
    if p.nx < 8:
        warnings.append(f"nx={p.nx} is very coarse (< 8 longitudinal zones)")
    if p.ny < 2:
        warnings.append(f"ny={p.ny} is too narrow (< 2 transverse voxels)")
    if p.nz_hull < 3:
        warnings.append(f"nz_hull={p.nz_hull} is very shallow (< 3 deck layers)")

    # Budget sum — only check that it doesn't exceed 1.0 (physically impossible)
    budget_sum = sum(v for k, v in p.budget.items() if k != "void")
    if budget_sum > 1.0:
        warnings.append(f"Budget sum (excl. void) = {budget_sum:.3f} > 1.0")


    # Superstructure position within ship
    # SS can be far aft on cargo ships (lcg_frac ~0.05) or centred on yachts (~0.60)
    if not (0.03 <= p.ss_lcg_frac <= 0.98):
        warnings.append(f"ss_lcg_frac={p.ss_lcg_frac:.3f} is outside hull bounds")

    is_valid = len(warnings) == 0
    return is_valid, warnings
