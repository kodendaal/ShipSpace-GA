"""
validation_constants.py
=======================
Single source of truth for the real-vs-synthetic validation pipeline.

Re-exports taxonomy, graph schema, and normalisation from the generator
modules so ``zone_extraction``, ``cc_reextract``, ``similarity_metrics``,
and ``validation_report`` measure the same quantities on both sides.
"""

from __future__ import annotations

from typing import Dict, Union

import numpy as np

from ship_params import Comp, COMP_DENSITY, ShipType
from bulkhead_placement import ZoneType
from graph_builder import (
    N_ZONE_TYPES, N_NODE_FEATURES, N_EDGE_FEATURES, N_SHIP_TYPES, NORM_L,
    NORM_B, NORM_D, BUDGET_KEYS, EDGE_LONGITUDINAL, EDGE_VERTICAL,
    EDGE_SS_HULL, EDGE_TRANSVERSE,
)

# ── Parameter conditioning block (coverage / conditioning-space metrics) ──
N_COND_PARAM = 16  # type(6) + L/B/D(3) + budgets(7); LCG/KG excluded (outputs)


def parameter_cond_vector(cond: Union[np.ndarray, list]) -> np.ndarray:
    """
    Extract the 16-d *parameter* conditioning block.

    Layout: ship_type one-hot (6), L/B/D norm (3), effective budget fracs (7).
    LCG/KG are layout *outputs* and are excluded from coverage distance.

    Accepts 16-d vectors or 18-d vectors (LCG/KG at idx 9–10).
    """
    c = np.asarray(cond, dtype=np.float64).ravel()
    if c.shape[0] == N_COND_PARAM:
        return c
    if c.shape[0] == 18:
        return np.concatenate([c[:9], c[11:]])
    raise ValueError(
        f"Unexpected cond length {c.shape[0]}; expected {N_COND_PARAM} or 18"
    )


# ── CC extraction (shared real + synthetic) ───────────────────────────
MIN_ZONE_VOXELS = 2
SS_FOOTPRINT_THRESHOLD = 0.40
SS_ACCOM_THRESHOLD = 0.60
SS_ACCOM_FOOTPRINT_THRESHOLD = 0.85

SPATIAL_RULES = {
    "aft_peak_max_cx": 0.04,
    "er_max_cx": 0.30,
    "fwd_peak_min_cx": 0.93,
    "acc_min_deck_frac": 0.40,
}

# Nine budgeted / reported compartment classes (excl. NAVIGATION, EMPTY)
ACTIVE_LABEL_COMPS = [
    Comp.VOID,
    Comp.ENGINE_ROOM,
    Comp.MACHINERY,
    Comp.CARGO,
    Comp.ACCOMMODATION,
    Comp.FUEL_TANKS,
    Comp.BALLAST_TANKS,
    Comp.STEERING_GEAR,
    Comp.STORES,
]
ACTIVE_LABEL_INDICES = [int(c) for c in ACTIVE_LABEL_COMPS]

COMP_SHORT: Dict[int, str] = {
    int(Comp.VOID): "VOID",
    int(Comp.ENGINE_ROOM): "ER",
    int(Comp.MACHINERY): "MACH",
    int(Comp.CARGO): "CARGO",
    int(Comp.ACCOMMODATION): "ACCOM",
    int(Comp.FUEL_TANKS): "FUEL",
    int(Comp.BALLAST_TANKS): "BALL",
    int(Comp.STEERING_GEAR): "STEER",
    int(Comp.STORES): "STORES",
    int(Comp.NAVIGATION): "NAV",
    int(Comp.EMPTY): "EMPTY",
}

TYPE_NAMES: Dict[int, str] = {
    int(ShipType.BULKER): "Bulker",
    int(ShipType.TANKER): "Tanker",
    int(ShipType.CARGO): "Cargo",
    int(ShipType.OSV): "OSV",
    int(ShipType.PATROL): "Patrol",
    int(ShipType.YACHT): "Yacht",
}
FAMILY_NAME_TO_TYPE: Dict[str, ShipType] = {
    name: ShipType(st) for st, name in TYPE_NAMES.items()
}
COMPARABLE_TYPES = list(TYPE_NAMES.keys())


# Pre-registered thresholds
METRIC_THRESHOLDS = {
    "JSD_label": {
        "excellent": 0.05,
        "good": 0.10,
        "acceptable": 0.20,
    },
    "KS_pvalue_pass": 0.05,
    "coverage_target": 0.80,
    # The generator goal is a broad, plausible
    # synthetic envelope that *contains* real operating points, not
    # distribution-shape matching. KS is demoted to a one-sided bias diagnostic.
    "containment": {
        # Robust synthetic band used to judge containment (percentiles).
        "robust_band": [5.0, 95.0],
        # Synth spread must be at least this fraction of real spread (anti-collapse).
        "diversity_min_ratio": 0.5,
        # Flag an implausible high tail when synth max exceeds this × real max.
        "implausible_tail_ratio": 2.0,
    },
}

KS_FEATURE_KEYS = [
    "zones", "edges", "LCG", "KG", "GM",
    *BUDGET_KEYS,
]

# Features judged for containment (broad envelope that brackets real values).
# Budgets are the primary target; LCG/KG/GM are physics plausibility.
# Zones/edges are granularity diagnostics, excluded from containment.
CONTAINMENT_FEATURE_KEYS = [
    *BUDGET_KEYS,
    "LCG", "KG", "GM",
]

COMP_TO_BUDGET_KEY: Dict[int, str] = {
    int(Comp.ENGINE_ROOM): "engine",
    int(Comp.MACHINERY): "machinery",
    int(Comp.CARGO): "cargo",
    int(Comp.STORES): "stores",
    int(Comp.ACCOMMODATION): "accommodation",
    int(Comp.FUEL_TANKS): "fuel_tanks",
    int(Comp.BALLAST_TANKS): "ballast_tanks",
}

# Plot colours (NAVIGATION ≠ EMPTY)
COMP_PALETTE: Dict[int, str] = {
    int(Comp.VOID): "#607d8b",
    int(Comp.ENGINE_ROOM): "#e74c3c",
    int(Comp.MACHINERY): "#e67e22",
    int(Comp.CARGO): "#f1c40f",
    int(Comp.ACCOMMODATION): "#2ecc71",
    int(Comp.FUEL_TANKS): "#9b59b6",
    int(Comp.BALLAST_TANKS): "#3498db",
    int(Comp.STEERING_GEAR): "#e91e63",
    int(Comp.STORES): "#8d6e63",
    int(Comp.NAVIGATION): "#00bcd4",
    int(Comp.EMPTY): "#F0F0F0",
}

# All compartment classes that can appear on graph nodes (incl. NAVIGATION)
PLOT_ASSIGNED_COMPS = list(range(int(Comp.EMPTY)))  # 0..9
# Graph legends: show assignable comps; never EMPTY (hull exterior)
GRAPH_LEGEND_EXCLUDE = {int(Comp.EMPTY)}
