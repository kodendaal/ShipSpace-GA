"""
representation_converters.py
============================
Voxel tensors stored on every generated graph.

``attach_voxel_and_deck`` resamples the native label grid of a Stage 4
assignment (nearest neighbour) to a fixed 64 x 32 x 32 grid and stores it
on the graph:

    voxel_labels      (64, 32, 32) int8   compartment class per cell (10 = empty)
    voxel_hull_mask   (64, 32, 32) bool   cell inside the hull envelope
    zone_node_mask    (64, 32, 32) int16  graph node per cell (-1 outside)
    deck_labels, deck_hull_masks          the same grids ordered (z, x, y)
    native_grid_shape, target_grid_shape

Coordinate convention (inherited from Stages 1-4):
    x : 0 = aft,    nx-1 = bow         → normalised dim 0
    y : 0 = port,   ny-1 = starboard   → normalised dim 1
    z : 0 = keel,   increasing upward  → normalised dim 2

Dependencies: numpy, torch.
"""

from __future__ import annotations

import numpy as np
from typing import Any, Dict, Tuple

# ── Optional imports ──────────────────────────────────────────────
try:
    import torch
    HAS_TORCH = True
except ImportError:
    torch = None
    HAS_TORCH = False

try:
    from torch_geometric.data import Data
    HAS_PYG = True
except ImportError:
    Data = None
    HAS_PYG = False

# ── Pipeline imports ─
from compartment_assignment import CompartmentAssignment
from hull_mask import EPS_EMPTY
from ship_params import Comp

# ─────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────

# Default normalised grid shape (length × beam × height).
# Matches production native nx/ny (64×32); z=32 covers typical nz_total (hull 24
# + SS) with ~97% round-trip label accuracy vs ~87% at (64, 24, 12).
# Ships with nz_total > 32 still incur mild height downsampling only.
DEFAULT_TARGET_SHAPE = (64, 32, 32)


# Class index used for voxels outside the hull
EMPTY_CLASS = Comp.EMPTY.value  # 10 (NAVIGATION=9, EMPTY=10)


# ─────────────────────────────────────────────────────────────────
# Core: Native-resolution label grid
# ─────────────────────────────────────────────────────────────────

def build_label_grid(
    assignment: CompartmentAssignment,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Construct the full 3D label grid and binary hull mask from Stage 4 output.

    For each voxel, look up its zone_id from the zone_mask, then map
    to the compartment label via assignment.assignments.  Voxels outside
    the hull (zone_mask == -1) receive EMPTY_CLASS (10).

    Parameters
    ----------
    assignment : CompartmentAssignment
        Completed Stage 4 output with .layout.zone_mask and .assignments.

    Returns
    -------
    label_grid : np.ndarray, shape (nx, ny, nz_total), dtype int8
        Compartment class index per voxel (0-10).
    hull_mask : np.ndarray, shape (nx, ny, nz_total), dtype bool
        True for voxels inside the hull envelope.
    """
    layout = assignment.layout
    zone_mask = layout.zone_mask              # (nx, ny, nz_total) int32, -1 outside
    full_mask = layout.hull_result.full_mask   # (nx, ny, nz_total) float [0,1]

    # Build zone_id → class index lookup (vectorised via array indexing)
    max_zone_id = int(zone_mask.max())
    zone_to_class = np.full(max_zone_id + 1, EMPTY_CLASS, dtype=np.int8)
    for zone_id, comp in assignment.assignments.items():
        if 0 <= zone_id <= max_zone_id:
            zone_to_class[zone_id] = comp.value

    # Map zone_mask → label_grid
    label_grid = np.where(
        zone_mask >= 0,
        zone_to_class[zone_mask.clip(min=0)],
        EMPTY_CLASS,
    ).astype(np.int8)

    # Binary hull envelope (same EPS_EMPTY occupancy rule as Stages 2–3)
    hull_mask = full_mask > EPS_EMPTY

    return label_grid, hull_mask


# ─────────────────────────────────────────────────────────────────
# Core: Nearest-neighbour resampling
# ─────────────────────────────────────────────────────────────────

def _nearest_resample_3d(
    grid: np.ndarray,
    target_shape: Tuple[int, int, int],
) -> np.ndarray:
    """
    Resample a 3D array to target_shape using nearest-neighbour interpolation.

    For each cell in the target grid, compute the corresponding source cell
    via linear coordinate mapping and take its value.  This preserves
    discrete class labels exactly (no interpolation artefacts).

    Parameters
    ----------
    grid : np.ndarray, shape (sx, sy, sz)
        Source grid (any dtype).
    target_shape : tuple of 3 ints
        Target dimensions (tx, ty, tz).

    Returns
    -------
    np.ndarray, shape target_shape, same dtype as input.
    """
    src = np.array(grid.shape, dtype=np.float64)
    tgt = np.array(target_shape, dtype=np.int64)

    # For each target index, compute the nearest source index
    # linspace from 0 to src_dim-1, sampled at tgt_dim points
    coords = [
        np.round(np.linspace(0, s - 1, int(t))).astype(np.intp)
        for s, t in zip(src, tgt)
    ]

    # Build index arrays via meshgrid (ij indexing = no transposition)
    ix, iy, iz = np.meshgrid(coords[0], coords[1], coords[2], indexing="ij")

    return grid[ix, iy, iz]


# ─────────────────────────────────────────────────────────────────
# Normalised voxel grid
# ─────────────────────────────────────────────────────────────────

def build_voxel_representation(
    assignment: CompartmentAssignment,
    target_shape: Tuple[int, int, int] = DEFAULT_TARGET_SHAPE,
) -> Dict[str, np.ndarray]:
    """
    Build the normalised voxel grid.

    Pipeline: Stage 4 → native label grid → nearest-neighbour resample
    to fixed target_shape.

    Parameters
    ----------
    assignment : CompartmentAssignment
        Completed Stage 4 output.
    target_shape : tuple of 3 ints
        Normalised grid dimensions (default 64 x 32 x 32).

    Returns
    -------
    dict with:
        'voxel_labels'    : np.ndarray (tx, ty, tz) int8    — class indices 0-10
        'voxel_hull_mask' : np.ndarray (tx, ty, tz) bool    — True inside hull
        'native_shape'    : tuple (nx, ny, nz_total)         — original grid dims
    """
    label_grid, hull_mask = build_label_grid(assignment)
    native_shape = label_grid.shape

    norm_labels = _nearest_resample_3d(label_grid, target_shape)
    norm_hull   = _nearest_resample_3d(hull_mask.astype(np.uint8), target_shape).astype(bool)

    # Enforce consistency: voxels outside hull must be EMPTY
    norm_labels[~norm_hull] = EMPTY_CLASS

    return {
        "voxel_labels":    norm_labels,
        "voxel_hull_mask": norm_hull,
        "native_shape":    native_shape,
    }


# ─────────────────────────────────────────────────────────────────
# Deck-stacked view
# ─────────────────────────────────────────────────────────────────

def build_deck_representation(
    assignment: CompartmentAssignment,
    target_shape: Tuple[int, int, int] = DEFAULT_TARGET_SHAPE,
) -> Dict[str, Any]:
    """
    Build the deck-stacked view: the same normalised voxel grid, reorganised
    as a sequence of 2D deck slices (bottom to top).

    Parameters
    ----------
    assignment : CompartmentAssignment
        Completed Stage 4 output.
    target_shape : tuple of 3 ints
        Normalised grid dimensions (default 64 x 32 x 32).

    Returns
    -------
    dict with:
        'deck_labels'     : np.ndarray (tz, tx, ty) int8    — class per deck
        'deck_hull_masks' : np.ndarray (tz, tx, ty) bool    — hull mask per deck
        'n_decks'         : int                              — number of z-levels
        'native_shape'    : tuple (nx, ny, nz_total)
    """
    voxel = build_voxel_representation(assignment, target_shape)

    tx, ty, tz = target_shape

    # Transpose to (z, x, y) — deck-first ordering, bottom to top
    deck_labels    = np.ascontiguousarray(voxel["voxel_labels"].transpose(2, 0, 1))
    deck_hull_masks = np.ascontiguousarray(voxel["voxel_hull_mask"].transpose(2, 0, 1))

    return {
        "deck_labels":     deck_labels,       # (tz, tx, ty)
        "deck_hull_masks": deck_hull_masks,   # (tz, tx, ty)
        "n_decks":         tz,
        "native_shape":    voxel["native_shape"],
    }


# ─────────────────────────────────────────────────────────────────
# Attach to the PyG Data object
# ─────────────────────────────────────────────────────────────────

def attach_voxel_and_deck(
    pyg_data: Any,
    assignment: CompartmentAssignment,
    target_shape: Tuple[int, int, int] = DEFAULT_TARGET_SHAPE,
    include_deck: bool = True,
) -> None:
    """
    Attach the voxel and deck tensors to a graph built by build_graph().

    Adds the following fields to the Data object:
        .voxel_labels      : tensor (tx, ty, tz) int8
        .voxel_hull_mask   : tensor (tx, ty, tz) bool
        .zone_node_mask    : tensor (tx, ty, tz) int16 — graph node index per
                             voxel (-1 outside hull envelope); aligns with
                             ``voxel_labels`` after the same nearest-neighbour
                             resample (graph→voxel decode bridge).
        .native_grid_shape : tuple (nx, ny, nz_total)
        .target_grid_shape : tuple (tx, ty, tz)

    If include_deck=True (default), also adds:
        .deck_labels       : tensor (tz, tx, ty) int8
        .deck_hull_masks   : tensor (tz, tx, ty) bool

    Note: deck tensors are a transposed view of the voxel tensors.  To
    save ~36 KB/sample, set include_deck=False and compute the deck view
    on-the-fly in your dataloader: deck = voxel_labels.permute(2, 0, 1).

    Parameters
    ----------
    pyg_data : torch_geometric.data.Data
        Existing graph data (modified in-place).
    assignment : CompartmentAssignment
        Same assignment used to build the graph.
    target_shape : tuple of 3 ints
        Normalised grid dimensions.
    include_deck : bool
        If True, store pre-transposed deck tensors (convenient but redundant).
    """
    if not HAS_TORCH:
        raise RuntimeError("PyTorch required for attach_voxel_and_deck()")

    voxel = build_voxel_representation(assignment, target_shape)

    # Voxel grid
    pyg_data.voxel_labels    = torch.tensor(
        voxel["voxel_labels"], dtype=torch.int8,
    )
    pyg_data.voxel_hull_mask = torch.tensor(
        voxel["voxel_hull_mask"], dtype=torch.bool,
    )

    # Deck-stacked view (optional, redundant with voxel)
    if include_deck:
        deck = build_deck_representation(assignment, target_shape)
        pyg_data.deck_labels     = torch.tensor(
            deck["deck_labels"], dtype=torch.int8,
        )
        pyg_data.deck_hull_masks = torch.tensor(
            deck["deck_hull_masks"], dtype=torch.bool,
        )

    # Metadata
    pyg_data.native_grid_shape = voxel["native_shape"]
    pyg_data.target_grid_shape = target_shape

    # Graph→voxel decode bridge: same node order as graph_builder (enumerate zones)
    zone_mask_native = assignment.layout.zone_mask  # (nx, ny, nz) int32, -1 outside
    zones = assignment.layout.zones
    zid_to_node = {z.zone_id: i for i, z in enumerate(zones)}
    node_mask = np.full(zone_mask_native.shape, -1, dtype=np.int16)
    for zid, node_idx in zid_to_node.items():
        node_mask[zone_mask_native == zid] = node_idx
    norm_node_mask = _nearest_resample_3d(node_mask, target_shape)
    pyg_data.zone_node_mask = torch.tensor(norm_node_mask, dtype=torch.int16)
