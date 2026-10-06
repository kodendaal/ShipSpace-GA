"""
companion.py — compact companion written next to every accepted arrangement.

The companion keeps the native zone map of the arrangement and adds a fixed
64 x 32 x 32 deck-stacked view of its functional labels. The compact view is
built from geometry only (hull occupancy, the pre-assignment hull bands and the
superstructure decks; see ``deck_codec``). It is accepted only when it is
exact: decoding it restores every native label, every compact cell keeps the
native cell counts, no side zone shares a compact cell with another zone, and
the stored hull availability and LCG/KG/GM of the graph are reproduced from
the companion. An arrangement that fails any check is rejected.

Arrays (``np.savez_compressed``, loadable with ``allow_pickle=False``):

    compact_labels            uint8   (64, 32, 32)  function per compact cell (10 = empty)
    compact_mask              bool    (64, 32, 32)  compact cell holds native cells
    compact_native_cell_count int32   (64, 32, 32)  native cells per compact cell
    native_zone_id            int16   (nx, ny, nz)  graph node per native cell (-1 outside)
    native_zone_labels        int8    (n_nodes,)    function per graph node (= graph.y)
    source_zone_ids           int32   (n_nodes,)    layout zone id per graph node
    native_occupied_fraction  float32 (nx, ny, nz)  occupied volume fraction per native cell
    native_spacing_m          float64 (3,)          native cell size dx, dy, dz
    native_to_compact         int16   (nx, ny, nz)  compact deck slot per native cell (-1 outside)
    hull_plan_native_bounds   int16   (n_bands, 3)  hull bands [z0, z1, role]
    ss_cells_per_deck         int16   (n_ss_decks,) native z-cells per superstructure deck
    hull_groups               int16   (n_groups, 3) native hull z-ranges per compact slot
    used_slots                int16   ()            compact slots in use
    compact_volume_m3         float64 (64, 32, 32)  occupied volume per compact cell
    compact_x_first_moment_m4 float64 (64, 32, 32)  volume * x per compact cell
    compact_z_first_moment_m4 float64 (64, 32, 32)  volume * z per compact cell
"""
from __future__ import annotations

import copy
import hashlib
import io
from typing import Any, Dict, Tuple

import numpy as np

from bulkhead_placement import (
    ZoneType,
    _compute_longitudinal_boundaries,
    _compute_vertical_tiers,
    _sample_deck_plan,
)
from compartment_assignment import Comp
from deck_codec import build_geometry_map, decode_decks, encode_decks, reduce_geometry
from hull_mask import EPS_EMPTY
from ship_params import ShipType
from volume_metrics import VoxelFields, compute_physics_volumetric

COMPACT_SHAPE = (64, 32, 32)
HULL_GRID = (64, 32, 24)
EMPTY = 10
SIDE_ZONE_TYPES = (int(ZoneType.SIDE_DB), int(ZoneType.SIDE_LOWER), int(ZoneType.SIDE_UPPER))
PHYSICS_TOLERANCE = 1e-12


class CompanionError(RuntimeError):
    """The arrangement cannot be stored as an exact compact companion."""


def hull_bands(params, seed: int) -> np.ndarray:
    """Hull bands [z0, z1, role] drawn by ``place_bulkheads`` for this hull seed."""
    p = copy.deepcopy(params)
    rng = np.random.default_rng(int(seed))
    _compute_longitudinal_boundaries(p, rng)
    if p.anisotropic and p.ship_type != ShipType.YACHT:
        raw = _sample_deck_plan(p, rng)
    else:
        raw = _compute_vertical_tiers(p)
    return np.asarray([(int(a), int(b), int(c)) for a, b, c in raw], dtype=np.int16)


def _numpy(value: Any) -> np.ndarray:
    return value.numpy() if hasattr(value, "numpy") else np.asarray(value)


def _attribute(graph: Any, key: str) -> Any:
    return graph[key] if isinstance(graph, dict) else getattr(graph, key)


def build_companion(graph: Any, assignment, params, hull_seed: int) -> Dict[str, Any]:
    """Companion arrays for one accepted graph, in storage order.

    ``params`` is the parameterisation as sampled (before the pipeline adjusts
    it); the hull bands and superstructure decks are replayed from it.
    """
    layout = assignment.layout
    hull = layout.hull_result.hull_fraction
    if hull is None:
        raise CompanionError("hull builder did not record the fractional hull")
    hull = np.asarray(hull, dtype=np.float32)
    if hull.shape != HULL_GRID:
        raise CompanionError(f"companions need a {HULL_GRID} hull grid, got {hull.shape}")

    zones = layout.zones
    node = np.full(layout.zone_mask.shape, -1, dtype=np.int16)
    for index, zone in enumerate(zones):
        node[layout.zone_mask == zone.zone_id] = index
    functions = np.asarray(
        [assignment.assignments.get(zone.zone_id, Comp.VOID).value for zone in zones],
        dtype=np.int8,
    )
    source_ids = np.asarray([zone.zone_id for zone in zones], dtype=np.int32)
    zone_types = np.asarray([int(zone.zone_type) for zone in zones], dtype=np.int64)

    inside = node >= 0
    labels = np.full(node.shape, EMPTY, dtype=np.uint8)
    labels[inside] = functions[node[inside]]

    bands = hull_bands(params, hull_seed)
    try:
        mapping = build_geometry_map(
            hull, inside, bands, params.ss_layers_per_deck, mode="compact32_protected",
        )
        deck = mapping["deck_index"]
        slots = int(mapping["slot_count"])
        encoded = encode_decks(labels, deck, slots)
        compact = np.full(COMPACT_SHAPE, EMPTY, dtype=np.uint8)
        compact[:, :, :slots] = encoded["labels"]
        decoded = decode_decks(compact, deck)
    except ValueError as exc:
        raise CompanionError(str(exc)) from exc
    changed = int(np.count_nonzero(decoded != labels))
    if changed:
        raise CompanionError(f"{changed} functional cells change in the compact view")

    alpha = np.zeros(node.shape, dtype=np.float32)
    alpha[:, :, :hull.shape[2]] = hull
    alpha[inside & (zone_types[node.clip(min=0)] == int(ZoneType.SUPERSTRUCTURE))] = 1
    alpha[~inside] = 0
    if not np.array_equal(alpha > EPS_EMPTY, inside):
        raise CompanionError("occupied cells do not match the native zone map")

    availability = []
    for index, box in enumerate(_numpy(_attribute(graph, "zone_bboxes"))):
        total = int(box[1] - box[0]) * int(box[3] - box[2]) * int(box[5] - box[4])
        availability.append(round(float(alpha[node == index].sum()) / max(total, 1), 5))
    stored = _numpy(_attribute(graph, "x"))[:, 3]
    if not np.array_equal(np.asarray(availability, dtype=np.float32), stored):
        raise CompanionError("hull availability of the graph nodes is not reproduced")

    spacing = np.asarray(
        [float(_attribute(graph, key)) for key in ("dx", "dy", "dz")], dtype=np.float64,
    )
    shape = node.shape
    dz = float(spacing[2])
    centres = [
        np.broadcast_to(
            (np.arange(shape[axis]) + 0.5).reshape(
                tuple(shape[axis] if other == axis else 1 for other in range(3))
            ) * spacing[axis],
            shape,
        )
        for axis in range(3)
    ]
    fields = VoxelFields(
        labels=labels.astype(np.int32),
        volume_m3=alpha.astype(np.float64) * float(_attribute(graph, "cell_volume")),
        x_m=centres[0], y_m=centres[1], z_m=centres[2],
        L=float(_attribute(graph, "L")), B=float(_attribute(graph, "B")),
        D_total_m=shape[2] * dz,
    )
    lcg, kg, gm = compute_physics_volumetric(fields, HULL_GRID[2], dz)
    physics = {
        key: abs(float(value) - float(_attribute(graph, key)))
        for key, value in (("lcg_actual", lcg), ("kg_actual", kg), ("gm_t", gm))
    }
    if max(physics.values()) > PHYSICS_TOLERANCE:
        raise CompanionError("stored LCG/KG/GM are not reproduced from the companion")

    weights = np.zeros(COMPACT_SHAPE, dtype=np.int32)
    weights[:, :, :slots] = encoded["occupied_cell_count"]
    native_counts = np.bincount(labels[inside], minlength=11)
    compact_counts = np.array([weights[compact == c].sum() for c in range(11)])
    if not np.array_equal(native_counts, compact_counts):
        raise CompanionError("compact cell counts differ from the native labels")

    x, y, _z = np.nonzero(inside)
    keys = (x * shape[1] + y) * slots + deck[inside]
    size = shape[0] * shape[1] * slots
    lo = np.full(size, len(functions), dtype=np.int32)
    hi = np.full(size, -1, dtype=np.int32)
    np.minimum.at(lo, keys, node[inside])
    np.maximum.at(hi, keys, node[inside])
    mixed = (hi >= 0) & (lo != hi)
    side = np.isin(node[inside], np.flatnonzero(np.isin(zone_types, SIDE_ZONE_TYPES)))
    if np.any(mixed[keys][side]):
        raise CompanionError("a side zone shares a compact cell with another zone")

    geometry = {}
    for key, value in reduce_geometry(deck, alpha, spacing, slots).items():
        canvas = np.zeros(COMPACT_SHAPE, dtype=np.float64)
        canvas[:, :, :slots] = value
        geometry["compact_" + key] = canvas

    arrays = {
        "compact_labels": compact,
        "compact_mask": weights > 0,
        "compact_native_cell_count": weights,
        "native_zone_id": node,
        "native_zone_labels": functions,
        "source_zone_ids": source_ids,
        "native_occupied_fraction": alpha,
        "native_spacing_m": spacing,
        "native_to_compact": deck,
        "hull_plan_native_bounds": bands,
        "ss_cells_per_deck": np.asarray(params.ss_layers_per_deck, dtype=np.int16),
        "hull_groups": np.asarray(mapping["hull_groups"], dtype=np.int16),
        "used_slots": np.asarray(slots, dtype=np.int16),
        **geometry,
    }
    record = {
        "native_z": int(shape[2]),
        "slot_count": slots,
        "native_zones": int(len(functions)),
        "side_zones": int(np.isin(zone_types, SIDE_ZONE_TYPES).sum()),
        "mixed_instance_columns": int(mixed.sum()),
        "functional_changed_cells": changed,
        "physics_replay_absolute_difference": physics,
        "native_zone_sha256": hashlib.sha256(node.tobytes()).hexdigest(),
    }
    return {"arrays": arrays, "record": record}


def companion_bytes(graph: Any, assignment, params, hull_seed: int) -> Tuple[bytes, Dict[str, Any]]:
    """Serialised companion plus the per-record facts written to records.json."""
    built = build_companion(graph, assignment, params, hull_seed)
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **built["arrays"])
    data = buffer.getvalue()
    record = dict(built["record"], companion_sha256=hashlib.sha256(data).hexdigest())
    return data, record


def attach_companion(graph: Any, assignment, params, hull_seed: int) -> None:
    """Keep the serialised companion on the graph until its shard is written."""
    payload = companion_bytes(graph, assignment, params, hull_seed)
    if isinstance(graph, dict):
        graph["companion"] = payload
    else:
        graph.companion = payload
