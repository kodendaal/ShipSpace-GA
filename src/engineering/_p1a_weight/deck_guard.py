from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import numpy as np
from scipy.ndimage import median_filter

from .hydrostatics import waterline_profile
from .models import ClearanceResult, DeckEnvelope


def _top_cell_indices(mask: np.ndarray, occupancy_threshold: float) -> np.ndarray:
    occupied = np.asarray(mask, dtype=float) > float(occupancy_threshold)
    nz = occupied.shape[2]
    reverse_first = np.argmax(occupied[:, :, ::-1], axis=2)
    has = occupied.any(axis=2)
    top = (nz - 1 - reverse_first).astype(np.int32)
    top[~has] = -1
    return top


def extract_local_deck_envelope(
    hull_mask: np.ndarray,
    p: Any,
    *,
    occupancy_threshold: float = 0.10,
    central_beam_fraction: float = 0.60,
    station_quantile: float = 0.25,
    end_exclusion_fraction: float = 0.03,
    min_station_breadth_fraction: float = 0.18,
    smoothing_stations: int = 3,
    step_threshold_m: float | None = None,
    source: str = "hull_mask",
) -> DeckEnvelope:
    """Extract a *local* deck-height profile, preserving split/stepped decks.

    A single global top-deck height is deliberately not used. For each
    longitudinal station, the profile is taken from the central part of the
    main-hull top surface. This avoids superstructure, bulwarks and very narrow
    shell-edge columns while retaining lower aft working decks and other deck
    steps.
    """
    mask = np.asarray(hull_mask, dtype=float)
    nx, ny, _ = mask.shape
    top = _top_cell_indices(mask, occupancy_threshold)

    half = 0.5 * float(central_beam_fraction)
    y0 = max(0, int(np.floor((0.5 - half) * ny)))
    y1 = min(ny, int(np.ceil((0.5 + half) * ny)))
    if y1 <= y0:
        y0, y1 = 0, ny

    raw = np.full(nx, np.nan, dtype=float)
    valid = np.zeros(nx, dtype=bool)
    min_cols = max(2, int(np.ceil(min_station_breadth_fraction * ny)))
    i0 = int(np.floor(end_exclusion_fraction * nx))
    i1 = int(np.ceil((1.0 - end_exclusion_fraction) * nx))

    for i in range(nx):
        all_vals = top[i, top[i] >= 0]
        central_vals = top[i, y0:y1]
        central_vals = central_vals[central_vals >= 0]
        if central_vals.size >= min_cols:
            vals = central_vals
        elif all_vals.size >= min_cols:
            vals = all_vals
        else:
            continue
        raw[i] = (float(np.quantile(vals, station_quantile)) + 1.0) * float(p.dz_m)
        valid[i] = i0 <= i < i1

    # Fill short gaps for plotting/solver continuity, but keep validity separate.
    filled = raw.copy()
    finite = np.flatnonzero(np.isfinite(filled))
    if finite.size >= 2:
        missing = np.flatnonzero(~np.isfinite(filled))
        filled[missing] = np.interp(missing, finite, filled[finite])
    elif finite.size == 1:
        filled[:] = filled[finite[0]]
    else:
        filled[:] = float(p.D)
        valid[:] = False

    size = int(max(1, smoothing_stations))
    if size % 2 == 0:
        size += 1
    smoothed = median_filter(filled, size=size, mode="nearest") if size > 1 else filled

    threshold = (
        float(step_threshold_m)
        if step_threshold_m is not None
        else max(0.50, 1.25 * float(p.dz_m))
    )
    diffs = np.abs(np.diff(smoothed))
    steps = tuple(int(i + 1) for i in np.flatnonzero(diffs >= threshold))
    x = (np.arange(nx, dtype=float) + 0.5) * float(p.dx_m)

    return DeckEnvelope(
        x_m=x,
        deck_z_m=smoothed.astype(float),
        valid=valid & np.isfinite(smoothed),
        raw_deck_z_m=raw,
        step_indices=steps,
        source=source,
    )


def required_freeboard_m(p: Any, family_cfg: dict) -> float:
    return max(
        float(family_cfg.get("minimum_freeboard_m", 0.0)),
        float(family_cfg.get("minimum_freeboard_D", 0.0)) * float(p.D),
    )


def check_deck_clearance(
    envelope: DeckEnvelope,
    p: Any,
    mean_draft_m: float,
    trim_m: float,
    required_m: float,
    tolerance_m: float = 1e-6,
) -> ClearanceResult:
    _, zw = waterline_profile(p, mean_draft_m, trim_m)
    valid = envelope.valid & np.isfinite(envelope.deck_z_m)
    if not np.any(valid):
        return ClearanceResult(
            min_freeboard_m=float("nan"),
            p05_freeboard_m=float("nan"),
            median_freeboard_m=float("nan"),
            violation_count=0,
            violation_fraction=float("nan"),
            worst_x_m=float("nan"),
            required_freeboard_m=float(required_m),
        )
    fb = envelope.deck_z_m[valid] - zw[valid]
    violations = fb < (float(required_m) - tolerance_m)
    idx_valid = np.flatnonzero(valid)
    worst_local = int(np.argmin(fb))
    return ClearanceResult(
        min_freeboard_m=float(np.min(fb)),
        p05_freeboard_m=float(np.quantile(fb, 0.05)),
        median_freeboard_m=float(np.median(fb)),
        violation_count=int(np.count_nonzero(violations)),
        violation_fraction=float(np.mean(violations)),
        worst_x_m=float(envelope.x_m[idx_valid[worst_local]]),
        required_freeboard_m=float(required_m),
    )
