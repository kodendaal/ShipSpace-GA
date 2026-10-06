"""
bulkhead_placement.py
=====================
Stage 3: Bulkhead and Zone Placement
--------------------------------------
Divides the available hull cells (from Stage 2) into named zones by placing
transverse bulkheads. Each zone becomes one node in the GNN graph.

Design mirrors real naval architecture:
  - Mandatory bulkheads at aft peak, engine room forward limit, collision bulkhead
  - Variable transverse bulkheads divide the main cargo/machinery body
  - Vertical subdivision into DB / lower / upper tiers where depth permits
  - Superstructure: one zone per 6-connected SS component per relative level

Inputs:   HullMaskResult (from Stage 2)
Outputs:  BulkheadLayout
            .zones          — List[Zone], one per region → GNN nodes
            .zone_mask      — (nx, ny, nz_total) int32, -1 outside hull
            .transverse_bulkheads_x — x-indices of placed bulkheads

Coordinate convention (inherited from Stages 1-2):
    x : 0 = aft,   nx-1 = bow
    y : 0 = port,  ny-1 = starboard
    z : 0 = keel,  increasing upward;  nz_hull = first SS layer
"""

from __future__ import annotations
import os
import numpy as np
from dataclasses import dataclass
from enum import IntEnum
from typing import Dict, List, FrozenSet, Optional, Set, Tuple

from ship_params import (
    ShipParameterization, ShipType, Comp,
    SPATIAL_RULES, ER_MAX_CX_BY_TYPE,
    SIDE_ZONE_CONFIG, side_zone_width_cells, side_zones_enabled,
    ss_logical_deck_index,
)
from scipy.ndimage import label, generate_binary_structure

from hull_mask import (
    HullMaskResult, EPS_EMPTY,
)

# Side-zone transverse carving mode (override via SIDE_CARVE_MODE env var).
#   fixed_band     — grid-edge strips y∈[0,w) and y∈[ny-w,ny)
#   skin_relative  — w-cell band inward from occupied hull skin per (x,z) column
SIDE_CARVE_MODES = ("fixed_band", "skin_relative")
DEFAULT_SIDE_CARVE_MODE = os.environ.get("SIDE_CARVE_MODE", "skin_relative")

# ─────────────────────────────────────────────────────────────────
# Zone taxonomy
# ─────────────────────────────────────────────────────────────────

class ZoneType(IntEnum):
    AFT_PEAK       = 0   # extreme aft — steering gear / aft peak tank
    ENGINE_REGION  = 1   # engine room and auxiliary machinery aft region
    MAIN_DB        = 2   # main body, double-bottom tier (z < db_layers)
    MAIN_LOWER     = 3   # main body, lower hold (db_layers ≤ z < acc_min_z)
    MAIN_UPPER     = 4   # main body, upper hold / tweendeck (z ≥ acc_min_z)
    FWD_PEAK       = 5   # extreme fwd — fore peak tank (collision bulkhead fwd)
    SUPERSTRUCTURE = 6   # above main deck — accommodation decks
    ENGINE_UPPER   = 7   # engine region, upper hold (z >= acc_min_z)
    SIDE_DB        = 8   # port/stbd double-bottom wing
    SIDE_LOWER     = 9   # port/stbd lower hold wing
    SIDE_UPPER     = 10  # port/stbd upper hold wing

# Eligible compartment types per zone type.
# These constrain Stage 4 assignment — a zone may only be assigned
# a compartment type from its eligibility set.
ZONE_ELIGIBILITY: dict[ZoneType, FrozenSet[Comp]] = {
    ZoneType.AFT_PEAK:       frozenset({Comp.STEERING_GEAR, Comp.BALLAST_TANKS, Comp.VOID}),
    ZoneType.ENGINE_REGION:  frozenset({Comp.ENGINE_ROOM, Comp.MACHINERY, Comp.FUEL_TANKS, Comp.VOID}),
    ZoneType.MAIN_DB:        frozenset({Comp.BALLAST_TANKS, Comp.FUEL_TANKS, Comp.VOID}),
    ZoneType.MAIN_LOWER:     frozenset({Comp.CARGO, Comp.FUEL_TANKS, Comp.BALLAST_TANKS, Comp.MACHINERY,
                                         Comp.STORES, Comp.VOID}),
    # Superseded at zone build time by :func:`_main_upper_eligible_comps`
    # (deep-cargo drops ACCOMMODATION).
    ZoneType.MAIN_UPPER:     frozenset({Comp.CARGO, Comp.STORES, Comp.MACHINERY,
                                         Comp.ACCOMMODATION, Comp.VOID}),
    ZoneType.FWD_PEAK:       frozenset({Comp.BALLAST_TANKS, Comp.VOID, Comp.MACHINERY}),
    ZoneType.SUPERSTRUCTURE: frozenset({Comp.ACCOMMODATION, Comp.VOID}),
    # Superseded at zone build time by :func:`_engine_upper_eligible_comps`.
    ZoneType.ENGINE_UPPER:   frozenset({Comp.MACHINERY, Comp.CARGO, Comp.VOID}),
    ZoneType.SIDE_DB:        frozenset({Comp.BALLAST_TANKS, Comp.FUEL_TANKS, Comp.VOID}),
    ZoneType.SIDE_LOWER:     frozenset({Comp.BALLAST_TANKS, Comp.FUEL_TANKS, Comp.VOID}),
    ZoneType.SIDE_UPPER:     frozenset({Comp.BALLAST_TANKS, Comp.FUEL_TANKS, Comp.VOID}),
}


class TierRole(IntEnum):
    """Structural role of a vertical band, decoupled from its physical index.

    `deck_idx` is the physical band index; with a deck plan, hull bands can
    range past 2. `tier_role` carries the structural meaning (DB, LOWER,
    UPPER, SS) so role logic is independent of how many physical decks exist.
    """
    DB = 0       # double bottom
    LOWER = 1    # lower hold (below accommodation floor)
    UPPER = 2    # upper hold / tweendeck
    SS = 3       # superstructure


def _tier_role_from_deck_idx(deck_idx: int) -> "TierRole":
    """Map a deck_idx to its structural role (0=DB, 1=LOWER, 2=UPPER, 3+=SS)."""
    if deck_idx == 0:
        return TierRole.DB
    if deck_idx == 1:
        return TierRole.LOWER
    if deck_idx == 2:
        return TierRole.UPPER
    return TierRole.SS


def _side_tier_carving_enabled(ship_type: ShipType, role: "TierRole") -> bool:
    return not (role == TierRole.UPPER and ship_type in (
        ShipType.CARGO, ShipType.OSV, ShipType.PATROL, ShipType.YACHT,
    ))


def _side_zone_type_for_tier(deck_idx: int,
                             tier_role: Optional["TierRole"] = None) -> ZoneType:
    """Map a vertical tier to its side-zone type (engine + main body).

    Keyed off ``tier_role`` so upper hull decks (deck_idx 3,4,5 under the deck
    plan) type as SIDE_UPPER. SS never reaches this path (only main/engine are
    side-carved)."""
    role = tier_role if tier_role is not None else _tier_role_from_deck_idx(deck_idx)
    return {
        TierRole.DB:    ZoneType.SIDE_DB,
        TierRole.LOWER: ZoneType.SIDE_LOWER,
        TierRole.UPPER: ZoneType.SIDE_UPPER,
    }.get(role, ZoneType.SIDE_UPPER)


def _engine_upper_eligible_comps(st: ShipType) -> FrozenSet[Comp]:
    """
    ENGINE_UPPER (deckhouse over ER): working ships carry cargo aft; transport
    and yachts use the space for accommodation / hotel services, not cargo.
    """
    if st in (ShipType.OSV, ShipType.PATROL):
        return frozenset({Comp.MACHINERY, Comp.CARGO, Comp.VOID})
    return frozenset({Comp.MACHINERY, Comp.ACCOMMODATION, Comp.STORES, Comp.VOID})


def _main_upper_eligible_comps(st: ShipType) -> FrozenSet[Comp]:
    """
    MAIN_UPPER eligibility — vessel-specific.

    Deep-cargo (bulker/tanker/cargo): accommodation is SS + ENGINE_UPPER only.
    Working / slender types keep
    MAIN_UPPER accommodation for main-deck living spaces.
    """
    base = frozenset({Comp.CARGO, Comp.STORES, Comp.MACHINERY, Comp.VOID})
    if st in (ShipType.BULKER, ShipType.TANKER, ShipType.CARGO):
        return base
    return base | {Comp.ACCOMMODATION}


def _side_eligible_for_centre_zone(
    ship_type: ShipType,
    side_type: ZoneType,
    centre_zone_type: ZoneType,
    block_cx: float = 0.5,
) -> FrozenSet[Comp]:
    """
    Side-zone eligibility — liquids-only wings.

    Wing tanks hold ballast, fuel, and void only.  Solids (cargo, stores,
    accommodation, machinery) stay in centre zones / peaks / SS.

    block_cx = carve-block centre x / nx (0 = aft, 1 = fwd).

    Fuel policy by tier:
      SIDE_DB:     non-tanker — fuel eligible along full length (DB bunkers);
                   tanker — fuel only abreast ENGINE_REGION (MARPOL).
      SIDE_LOWER / SIDE_UPPER: fuel only abreast ER or in aft blocks
                   (block_cx < 0.45) — not along midship cargo holds.
    """
    abreast_er = centre_zone_type == ZoneType.ENGINE_REGION
    aft_block = block_cx < 0.45

    base = {Comp.BALLAST_TANKS, Comp.VOID}

    if side_type == ZoneType.SIDE_DB:
        if ship_type == ShipType.TANKER:
            if abreast_er:
                base.add(Comp.FUEL_TANKS)
        else:
            base.add(Comp.FUEL_TANKS)
        return frozenset(base)

    if side_type == ZoneType.SIDE_LOWER:
        if abreast_er or aft_block:
            base.add(Comp.FUEL_TANKS)
        return frozenset(base)
    if side_type == ZoneType.SIDE_UPPER:
        return frozenset(base)

    return ZONE_ELIGIBILITY.get(side_type, frozenset({Comp.VOID}))


# Number-of-holds sampling range per ship type for the main cargo region.
# Holds are the longitudinal subdivisions between the engine room forward
# bulkhead and the collision bulkhead.
N_HOLDS_CONFIG: dict[ShipType, Tuple[int, int]] = {
    ShipType.BULKER:  (5, 9),    # bulk carriers have many hold sub-divisions
    ShipType.TANKER:  (6, 12),   # tankers have many narrow cargo tanks
    ShipType.CARGO:   (3, 5),    # general cargo has 3-5 holds
    ShipType.OSV:     (2, 4),    # OSVs have simpler hold arrangement
    ShipType.PATROL: (3, 6),    # patrols have more zones due to systems
    ShipType.YACHT:   (2, 4),    # yachts have fewer, larger zones
}


# Fraction of the total engine region that is ER proper (aft portion).
# The remainder is auxiliary machinery / switchboard room (forward portion).
# Grounded in real proportions:
#   Transport ships have large ER / small aux space.
#   OSVs and patrols have proportionally more auxiliary and combat systems.
ER_SPLIT_CONFIG: dict[ShipType, Tuple[float, float]] = {
    ShipType.BULKER:  (0.60, 0.75),
    ShipType.TANKER:  (0.60, 0.75),
    ShipType.CARGO:   (0.55, 0.70),
    ShipType.OSV:     (0.40, 0.55),  # large aux machinery proportion
    ShipType.PATROL: (0.35, 0.50),  # extensive combat / power systems
    ShipType.YACHT:   (0.50, 0.65),
}


# Maximum jitter applied to each interior hold boundary, as a fraction
# of the base (equal) hold width.  Controls how regular the hold spacing is.
#   Tankers: very regular — cargo tanks designed for grade separation.
#   Bulkers: fairly regular — hatch spacing near-standardised per class.
#   Cargo / patrol: moderate variation.
#   OSV / yacht: more flexible arrangement.
HOLD_JITTER_CONFIG: dict[ShipType, float] = {
    ShipType.BULKER:  0.12,
    ShipType.TANKER:  0.08,
    ShipType.CARGO:   0.18,
    ShipType.OSV:     0.22,
    ShipType.PATROL: 0.15,
    ShipType.YACHT:   0.20,
}


# Uniformity bias: fraction by which the raw jitter is blended back toward
# the ideal (equal) boundary position.  0.0 = pure jitter, 1.0 = no jitter.
# Tankers and bulkers have a strong pull toward equal spacing because
# hold/tank widths are design-constrained by cargo management.
HOLD_UNIFORMITY_CONFIG: dict[ShipType, float] = {
    ShipType.BULKER:  0.40,   # moderate pull toward equal hatch spacing
    ShipType.TANKER:  0.60,   # strong pull — grade separation tanks near-equal
    ShipType.CARGO:   0.20,
    ShipType.OSV:     0.10,
    ShipType.PATROL: 0.20,
    ShipType.YACHT:   0.10,
}


# ─────────────────────────────────────────────────────────────────
# Zone dataclass
# ─────────────────────────────────────────────────────────────────

@dataclass
class Zone:
    """
    One zone = one GNN node.

    Bounding box [x0:x1, y0:y1, z0:z1] — upper bound exclusive.
    available_cells: number of cells with occupancy > EPS_EMPTY within the box
        (Stage 2 constant ``EPS_EMPTY`` in ``hull_mask``).
    total_cells:     (x1-x0)*(y1-y0)*(z1-z0) — the full bounding volume.

    hull_avail = available_cells / total_cells  (≤ 1.0).
    At tapered bow/stern, hull_avail < 1.0 because the hull doesn't fill
    the full bounding box. This is a core node feature for the GNN.

    Node features (partial — physics targets added in Stage 5):
        cx_norm         zone centroid x / nx
        cy_norm         zone centroid y / ny
        cz_norm         zone centroid z / nz_total
        hull_avail      available_cells / total_cells
        zone_width_norm  (y1-y0) / ny
        zone_length_norm (x1-x0) / nx
        deck_idx        integer deck level (0=keel, hull-nz_hull=first SS deck)
    """
    zone_id:          int
    zone_type:        ZoneType
    hold_idx:         int          # which hold band this belongs to (-1 = peak/ER)
    x0: int;  x1: int              # x bounds [x0, x1)
    y0: int;  y1: int              # y bounds [y0, y1)
    z0: int;  z1: int              # z bounds [z0, z1)

    available_cells:  int
    total_cells:      int
    eligible_comps:   FrozenSet[Comp]

    # Pre-computed node features
    cx_norm:          float
    cy_norm:          float
    cz_norm:          float
    hull_avail:       float
    zone_width_norm:  float
    zone_length_norm: float
    deck_idx:         int          # vertical tier index (0=DB, 1=lower, 2=upper, 3+=SS);
                                   # SS deck_idx = 3 + logical accommodation deck
    tier_role:        TierRole = TierRole.UPPER  # structural role (decoupled from deck_idx)
    side:             str = "centre"   # centre | port | stbd
    mirror_id:        int = -1         # pairing tag for port/stbd twins (-1 = none)

    def volume_m3(self, cell_volume: float) -> float:
        """Physical volume (m³); pass ``ShipParameterization.cell_volume``."""
        return self.available_cells * cell_volume

    def __repr__(self) -> str:
        return (
            f"Zone(id={self.zone_id} {self.zone_type.name} hold={self.hold_idx} "
            f"x=[{self.x0},{self.x1}) z=[{self.z0},{self.z1}) "
            f"avail={self.available_cells}/{self.total_cells} "
            f"cx={self.cx_norm:.2f} cz={self.cz_norm:.2f} "
            f"hull_avail={self.hull_avail:.2f})"
        )


# ─────────────────────────────────────────────────────────────────
# BulkheadLayout dataclass
# ─────────────────────────────────────────────────────────────────

@dataclass
class BulkheadLayout:
    """
    Complete zone layout for one ship instance.

    zones         : list of Zone objects — GNN nodes.
    zone_mask     : (nx, ny, nz_total) int32.
                    -1  = cell outside hull envelope (full_mask <= EPS_EMPTY)
                     k  = zone_id of the zone this cell belongs to
    transverse_bulkheads_x : x-indices where transverse bulkheads are placed.
    n_holds       : number of main-body longitudinal holds.
    params        : ShipParameterization
    hull_result   : HullMaskResult from Stage 2
    """
    zones:                    List[Zone]
    zone_mask:                np.ndarray    # (nx, ny, nz_total) int32
    transverse_bulkheads_x:   List[int]
    n_holds:                  int
    params:                   ShipParameterization
    hull_result:              HullMaskResult
    side_carve_mode:          str = DEFAULT_SIDE_CARVE_MODE

    @property
    def n_zones(self) -> int:
        return len(self.zones)

    @property
    def zone_by_id(self) -> dict[int, Zone]:
        return {z.zone_id: z for z in self.zones}

    def zones_of_type(self, zt: ZoneType) -> List[Zone]:
        return [z for z in self.zones if z.zone_type == zt]

    def summary(self) -> str:
        p = self.params
        lines = [
            f"BulkheadLayout  {p.ship_type.name}  "
            f"L={p.L:.0f}m  grid={p.nx}x{p.ny}x{p.nz_total}",
            f"  n_zones  : {self.n_zones}",
            f"  n_holds  : {self.n_holds}",
            f"  bulkheads: {self.transverse_bulkheads_x}",
            f"",
            f"  {'ID':>3}  {'Type':15}  {'Hold':>4}  "
            f"{'x':>8}  {'z':>6}  {'Avail':>6}  {'HullAvail':>9}",
            f"  {'─'*62}",
        ]
        for z in self.zones:
            lines.append(
                f"  {z.zone_id:>3}  {z.zone_type.name:15}  {z.hold_idx:>4}  "
                f"[{z.x0:2d},{z.x1:2d})  "
                f"[{z.z0},{z.z1})  "
                f"{z.available_cells:>6}  "
                f"{z.hull_avail:>9.3f}"
            )
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────

def place_bulkheads(
    hull_result: HullMaskResult,
    rng: Optional[np.random.Generator] = None,
    seed: Optional[int] = None,
    side_carve_mode: Optional[str] = None,
) -> BulkheadLayout:
    """
    Divide hull cells into zones by placing transverse and vertical bulkheads.

    Parameters
    ----------
    hull_result : HullMaskResult
        Output of Stage 2. Must have full_mask and params populated.
    rng : numpy Generator, optional
        For reproducible hold-width sampling. If None, created from seed.
    seed : int, optional
        Seed for the RNG if rng is not provided.
    side_carve_mode : str, optional
        ``skin_relative`` (default) — w-cell band inward from occupied hull
        skin per (x,z) column; ``fixed_band`` — grid-edge strips.
        Override default via ``SIDE_CARVE_MODE`` env var.

    Returns
    -------
    BulkheadLayout
    """
    if rng is None:
        rng = np.random.default_rng(seed)

    if side_carve_mode is None:
        side_carve_mode = DEFAULT_SIDE_CARVE_MODE
    if side_carve_mode not in SIDE_CARVE_MODES:
        raise ValueError(
            f"side_carve_mode must be one of {SIDE_CARVE_MODES}, got {side_carve_mode!r}"
        )
    skin_relative = side_carve_mode == "skin_relative"

    p   = hull_result.params
    fm  = hull_result.full_mask    # (nx, ny, nz_total) ∈ [0,1]
    hm  = hull_result.hull_mask    # hull-only (excludes SS voxels)
    sm  = hull_result.ss_mask      # SS-only   (1 where SS, 0 else)
    nx, ny, nz_total = fm.shape
    nz_hull = p.nz_hull

    # ── 1. Longitudinal boundary x-indices ───────────────────────────
    bh_x = _compute_longitudinal_boundaries(p, rng)
    # bh_x is a list of (x_start, x_end, region_label, hold_idx) tuples

    # ── 2. Vertical tier boundaries ──────────────────────────────────
    # Yachts and other sloped-deck STLs: keep classic 3-tier split even in
    # anisotropic mode — multi-band deck plan fragments the side profile.
    if p.anisotropic and p.ship_type != ShipType.YACHT:
        _deck_plan = _sample_deck_plan(p, rng)
        tiers = [(z0, z1, i) for i, (z0, z1, _r) in enumerate(_deck_plan)]
        tier_roles = {i: r for i, (_z0, _z1, r) in enumerate(_deck_plan)}
    else:
        tiers = _compute_vertical_tiers(p)
        tier_roles = {di: _tier_role_from_deck_idx(di) for (_z0, _z1, di) in tiers}
    # tiers: list of (z0, z1, deck_idx) covering [0, nz_hull)
    # tier_roles: {deck_idx -> TierRole} for role-keyed zone typing

    # ENGINE_UPPER / MAIN_UPPER share the same upper-hold boundary (acc_min_z).

    # ── 3. Build zones from longitudinal × vertical grid ─────────────
    zones: List[Zone] = []
    zone_id = 0
    n_holds = 0

    # Pre-allocate zone_mask
    zone_mask = np.full((nx, ny, nz_total), -1, dtype=np.int32)

    carve_ranges = _compute_side_carve_x_ranges(bh_x, p.ship_type, rng)
    side_w = (
        side_zone_width_cells(ny, p.ship_type, B=p.B, dy_m=p.dy_m)
        if side_zones_enabled(p.ship_type) else 0
    )
    mirror_counter = [0]
    use_side_carving = side_w > 0 and (
        skin_relative or ny >= 2 * side_w + 1
    )
    # (cx0, cx1, z0, z1) -> (port_band, stbd_band) within block subvolume
    block_side_masks: Dict[Tuple[int, int, int, int], Tuple[np.ndarray, np.ndarray]] = {}

    # Hull zones: centre bands only in carved regions; side zones emitted
    # once per merged carve block after this loop.
    for (x0, x1, region_label, hold_idx) in bh_x:
        if region_label == "main":
            n_holds += 1

        for (z0, z1, deck_idx) in tiers:
            _role = tier_roles.get(deck_idx, _tier_role_from_deck_idx(deck_idx))
            zone_type = _zone_type(region_label, deck_idx, p, tier_role=_role)
            in_carve = (
                use_side_carving
                and region_label in ("engine", "main")
                and _in_side_carve_range(x0, x1, region_label, carve_ranges)
                and _role != TierRole.SS
                and _side_tier_carving_enabled(p.ship_type, _role)
            )
            if in_carve and skin_relative:
                y0, y1 = 0, ny
                centre_mask: Optional[np.ndarray] = None
                block = _carve_block_for_segment(
                    x0, x1, region_label, carve_ranges,
                )
                if block is not None:
                    bx0, bx1 = block
                    bkey = (bx0, bx1, z0, z1)
                    if bkey not in block_side_masks:
                        block_side_masks[bkey] = _compute_skin_relative_side_bands(
                            hm, bx0, bx1, z0, z1, side_w,
                        )
                    port_m, stbd_m = block_side_masks[bkey]
                    xi0, xi1 = x0 - bx0, x1 - bx0
                    sub_occ = hm[x0:x1, :, z0:z1] > EPS_EMPTY
                    centre_mask = sub_occ & ~(port_m[xi0:xi1] | stbd_m[xi0:xi1])
            elif in_carve:
                y0, y1 = side_w, ny - side_w
                centre_mask = None
            else:
                y0, y1 = 0, ny
                centre_mask = None

            if zone_type == ZoneType.ENGINE_UPPER:
                elig_override = _engine_upper_eligible_comps(p.ship_type)
            elif zone_type == ZoneType.MAIN_UPPER:
                elig_override = _main_upper_eligible_comps(p.ship_type)
            else:
                elig_override = None
            zone = _make_zone(
                zone_id=zone_id,
                zone_type=zone_type,
                hold_idx=hold_idx,
                x0=x0, x1=x1,
                y0=y0, y1=y1,
                z0=z0, z1=z1,
                fm=hm,
                p=p,
                deck_idx=deck_idx,
                tier_role=_role,
                side="centre",
                eligible_override=elig_override,
                assign_mask=centre_mask,
            )
            if zone.available_cells == 0:
                continue   # skip degenerate zones with no hull cells
            zones.append(zone)
            if centre_mask is not None:
                zone_mask[x0:x1, :, z0:z1] = np.where(
                    centre_mask,
                    zone_id,
                    zone_mask[x0:x1, :, z0:z1],
                )
            else:
                zone_mask[x0:x1, y0:y1, z0:z1] = np.where(
                    hm[x0:x1, y0:y1, z0:z1] > EPS_EMPTY,
                    zone_id,
                    zone_mask[x0:x1, y0:y1, z0:z1],
                )
            zone_id += 1

    # Side zones: one port/stbd pair per (carve_block × tier), not per hold.
    if use_side_carving and carve_ranges:
        for (cx0, cx1, region_label) in carve_ranges:
            block_hold = -1
            if region_label == "main":
                for x0, x1, lbl, hi in bh_x:
                    if lbl == "main" and x0 >= cx0 and x1 <= cx1:
                        block_hold = hi
                        break
            for (z0, z1, deck_idx) in tiers:
                _srole = tier_roles.get(deck_idx, _tier_role_from_deck_idx(deck_idx))
                if not _side_tier_carving_enabled(p.ship_type, _srole):
                    continue
                centre_type = _zone_type(region_label, deck_idx, p, tier_role=_srole)
                if skin_relative:
                    bkey = (cx0, cx1, z0, z1)
                    if bkey not in block_side_masks:
                        block_side_masks[bkey] = _compute_skin_relative_side_bands(
                            hm, cx0, cx1, z0, z1, side_w,
                        )
                    port_m, stbd_m = block_side_masks[bkey]
                    new_zones, zone_id = _emit_side_zone_pair_skin_relative(
                        zone_id, centre_type, block_hold,
                        cx0, cx1, z0, z1, deck_idx,
                        hm, p, zone_mask, port_m, stbd_m, mirror_counter,
                        tier_role=_srole,
                    )
                else:
                    new_zones, zone_id = _emit_side_zone_pair(
                        zone_id, centre_type, block_hold,
                        cx0, cx1, z0, z1, deck_idx,
                        hm, p, zone_mask, side_w, mirror_counter,
                        tier_role=_srole,
                    )
                zones.extend(new_zones)

    # ── 4. Superstructure zones ───────────────────────────────────────
    # With true-settle SS, voxels can sit at any z (including < nz_hull).
    # We use the ss_only mask directly instead of fixed z-slices.
    if p.nz_ss > 0:
        ss_zones = _build_ss_zones(
            zone_id_start=zone_id,
            sm=sm,
            fm=fm,
            p=p,
            top_deck_z=hull_result.top_deck_z,
        )
        for z in ss_zones:
            zone_mask[z.x0:z.x1, z.y0:z.y1, z.z0:z.z1] = np.where(
                sm[z.x0:z.x1, z.y0:z.y1, z.z0:z.z1] > EPS_EMPTY,
                z.zone_id,
                zone_mask[z.x0:z.x1, z.y0:z.y1, z.z0:z.z1],
            )
        zones.extend(ss_zones)

    # ── 5. Mark cells outside hull+SS as -1 ──────────────────────────
    zone_mask[fm <= EPS_EMPTY] = -1

    # Collect transverse bulkhead x-positions
    bh_xs = sorted({x0 for (x0, x1, lbl, hi) in bh_x} |
                   {x1 for (x0, x1, lbl, hi) in bh_x})
    bh_xs = [x for x in bh_xs if 0 < x < nx]

    layout = BulkheadLayout(
        zones=zones,
        zone_mask=zone_mask,
        transverse_bulkheads_x=bh_xs,
        n_holds=n_holds,
        params=p,
        hull_result=hull_result,
        side_carve_mode=side_carve_mode,
    )

    # One geometric zone ID = one 6-connected voxel component
    from zone_connectivity import normalize_zone_connectivity
    layout = normalize_zone_connectivity(layout)
    return layout


# ─────────────────────────────────────────────────────────────────
# Longitudinal boundary computation
# ─────────────────────────────────────────────────────────────────

def _compute_longitudinal_boundaries(
    p: ShipParameterization,
    rng: np.random.Generator,
) -> List[Tuple[int, int, str, int]]:
    """
    Compute (x0, x1, region_label, hold_idx) tuples covering [0, nx).

    Region labels:
        'aft_peak'  — aft peak tank zone
        'engine'    — engine room / auxiliary machinery
        'main'      — main cargo/accommodation body (one per hold)
        'fwd_peak'  — fore peak tank zone

    hold_idx is the 0-indexed hold number within the main region.
    Non-main regions have hold_idx = -1.

    Domain rules applied
    --------------------
    Anchor positions (from Stage 1 params, type-specific):
      aft_peak_x_end   : afterpeak bulkhead  (~4% L from aft)
      er_max_x         : engine room fwd limit (~30% L from aft)
      fwd_peak_x_start : collision bulkhead   (~7% L from bow, SOLAS)

    Engine region: split into ER proper (aft) + aux machinery (fwd) using
    ER_SPLIT_CONFIG — transport ships have larger ER / smaller aux,
    OSVs and patrols have proportionally more auxiliary systems.

    Hold widths: near-equal spacing with type-specific jitter (HOLD_JITTER_CONFIG)
    and a pull back toward equal spacing (HOLD_UNIFORMITY_CONFIG).  Tankers
    are most regular (grade-separation tanks), OSVs most flexible.
    """
    nx = p.nx

    # Fixed structural boundaries (voxel indices)
    x_aft_pk  = max(1, p.aft_peak_x_end)           # aft peak ends here
    x_er_fwd  = max(x_aft_pk + 1, p.er_max_x)      # engine region ends here
    x_fwd_pk  = min(nx - 1, p.fwd_peak_x_start)    # fwd peak starts here

    # Ensure ordering is sane even for very small ships
    x_aft_pk = min(x_aft_pk, nx // 4)
    x_er_fwd = min(x_er_fwd, x_fwd_pk - 1)
    x_er_fwd = max(x_er_fwd, x_aft_pk + 1)
    x_fwd_pk = max(x_fwd_pk, x_er_fwd + 1)

    regions: List[Tuple[int, int, str, int]] = []

    # Aft peak
    if x_aft_pk > 0:
        regions.append((0, x_aft_pk, "aft_peak", -1))

    # Engine region — may be split into two sub-zones if wide enough
    er_width = x_er_fwd - x_aft_pk
    if er_width >= 4:
        er_lo, er_hi = ER_SPLIT_CONFIG[p.ship_type]
        split = x_aft_pk + max(1, int(er_width * rng.uniform(er_lo, er_hi)))
        split = min(split, x_er_fwd - 1)  # guard: fwd sub-zone must have ≥ 1 voxel
        regions.append((x_aft_pk, split,   "engine", -1))
        regions.append((split,   x_er_fwd, "engine", -1))
    else:
        regions.append((x_aft_pk, x_er_fwd, "engine", -1))

    # Main cargo/accommodation body — divided into n_holds sub-zones
    main_width = x_fwd_pk - x_er_fwd
    n_lo, n_hi = N_HOLDS_CONFIG[p.ship_type]
    # Cap n_holds so each hold has at least 1 voxel width
    n_holds = int(rng.integers(n_lo, n_hi + 1))
    n_holds = min(n_holds, main_width)
    n_holds = max(n_holds, 1)

    hold_boundaries = _sample_hold_boundaries(
        x_start=x_er_fwd,
        x_end=x_fwd_pk,
        n_holds=n_holds,
        rng=rng,
        jitter_frac=HOLD_JITTER_CONFIG[p.ship_type],
        uniformity_bias=HOLD_UNIFORMITY_CONFIG[p.ship_type],
    )
    for hi, (hx0, hx1) in enumerate(hold_boundaries):
        regions.append((hx0, hx1, "main", hi))

    # Fwd peak
    if x_fwd_pk < nx:
        regions.append((x_fwd_pk, nx, "fwd_peak", -1))

    return regions


def _sample_hold_boundaries(
    x_start: int,
    x_end: int,
    n_holds: int,
    rng: np.random.Generator,
    jitter_frac: float = 0.15,
    uniformity_bias: float = 0.0,
) -> List[Tuple[int, int]]:
    """
    Sample n_holds hold widths that sum to (x_end - x_start).

    Parameters
    ----------
    jitter_frac : float
        Maximum displacement of each interior boundary as a fraction of the
        base (equal) hold width.  E.g. 0.12 means ±12% of base width.
        Smaller = more regular spacing (tankers), larger = more variation (OSVs).
    uniformity_bias : float in [0, 1]
        After applying the raw jitter, blend the result back toward the ideal
        equal-spacing position by this fraction.  0.0 = pure jitter result,
        1.0 = no jitter at all.  Tankers/bulkers use a positive bias so that
        even after jitter the holds tend toward equal widths, consistent with
        grade-separation tank design or standardised hatch spacing.

    Method
    ------
    1. Compute ideal (equal) interior boundary positions.
    2. Apply raw jitter ∈ [-jitter_frac, +jitter_frac] × base_width.
    3. Blend raw jittered position back toward ideal by uniformity_bias.
    4. Round to integer voxel and enforce strictly increasing sequence.
    """
    total = x_end - x_start
    if n_holds == 1:
        return [(x_start, x_end)]

    base = total / n_holds
    raw_jitters = rng.uniform(-jitter_frac, jitter_frac, size=n_holds - 1)

    cuts = []
    for i, j in enumerate(raw_jitters, 1):
        ideal    = x_start + i * base
        jittered = ideal + j * base
        # Blend back toward ideal position by uniformity_bias
        blended  = uniformity_bias * ideal + (1.0 - uniformity_bias) * jittered
        # Keep at least 1 voxel gap between consecutive cuts
        lo = x_start + i
        hi = x_end - (n_holds - i)
        cuts.append(int(np.clip(round(blended), lo, hi)))

    # Ensure strictly increasing
    for i in range(1, len(cuts)):
        if cuts[i] <= cuts[i - 1]:
            cuts[i] = cuts[i - 1] + 1

    boundaries = []
    prev = x_start
    for c in cuts:
        boundaries.append((prev, c))
        prev = c
    boundaries.append((prev, x_end))
    return boundaries


# ─────────────────────────────────────────────────────────────────
# Vertical tier computation
# ─────────────────────────────────────────────────────────────────

def _sample_deck_plan(p: ShipParameterization,
                      rng: Optional[np.random.Generator] = None
                      ) -> List[Tuple[int, int, "TierRole"]]:
    """
    Generate a realistic vertical deck plan in METRES, then snap to grid layers.

    Returns list of (z0, z1, tier_role) covering [0, nz_hull): a DB band (role DB)
    plus stacked deck bands (role LOWER below the accommodation floor, UPPER above).
    Each band spans >=1 z-layer; decks are multi-cell at fine dz.

    Steps (the design logic):
      1. DB takes db_layers off the bottom (already dz-aware).
      2. remaining hull height H_r (metres) = (nz_hull - db_layers) * dz.
      3. sample deck clear-heights ~U(h_min, h_max), stack from the tank top.
      4. residual handling: if a small leftover remains, distribute it across
         decks; if a large leftover, add/drop a deck so heights stay in-band.
      5. snap band heights to integer layers via largest-remainder so the layer
         counts sum exactly to (nz_hull - db_layers).
      6. role: bands whose top is at/below the accommodation-floor height -> LOWER,
         above -> UPPER.

    Only used in anisotropic mode (fine dz). Isotropic grids fall back to the
    classic 3-tier split (see _compute_vertical_tiers).
    """
    if rng is None:
        rng = np.random.default_rng(0)
    nz = p.nz_hull
    db = p.db_layers
    dz = p.dz_m

    tiers: List[Tuple[int, int, TierRole]] = []
    if db > 0 and db <= nz:
        tiers.append((0, db, TierRole.DB))

    n_remaining_layers = nz - db
    if n_remaining_layers <= 0:
        if not tiers:
            tiers.append((0, nz, TierRole.UPPER))
        return tiers

    H_r = n_remaining_layers * dz   # metres of hull above the DB

    # ── deck clear-height bounds (metres) ─────────────────────────────
    H_MIN, H_MAX = 2.4, 2.7
    h_target = 0.5 * (H_MIN + H_MAX)

    # number of decks that best fits the remaining height
    n_decks = max(1, int(round(H_r / h_target)))
    # sample per-deck heights, then correct to sum to H_r
    heights = rng.uniform(H_MIN, H_MAX, size=n_decks)
    scale = H_r / heights.sum()
    heights = heights * scale
    # if scaling pushed decks out of band by a lot, adjust deck count
    if heights.max() > H_MAX * 1.25 and n_decks < n_remaining_layers:
        n_decks += 1
        heights = rng.uniform(H_MIN, H_MAX, size=n_decks)
        heights *= H_r / heights.sum()
    elif heights.min() < H_MIN * 0.75 and n_decks > 1:
        n_decks -= 1
        heights = rng.uniform(H_MIN, H_MAX, size=n_decks)
        heights *= H_r / heights.sum()

    # ── snap deck heights to integer layers (largest remainder) ───────
    raw = heights / dz                      # fractional layers per deck
    base = np.floor(raw).astype(int)
    base = np.maximum(base, 1)              # each deck >=1 layer
    # fix the sum to exactly n_remaining_layers
    deficit = n_remaining_layers - int(base.sum())
    if deficit > 0:
        order = np.argsort(-(raw - np.floor(raw)))   # largest remainder first
        for i in range(deficit):
            base[order[i % len(order)]] += 1
    elif deficit < 0:
        # too many layers: remove from decks with smallest remainder, keep >=1
        order = np.argsort(raw - np.floor(raw))
        k = 0
        while deficit < 0 and k < 10 * len(base):
            i = order[k % len(order)]
            if base[i] > 1:
                base[i] -= 1; deficit += 1
            k += 1

    # ── build bands with roles ────────────────────────────────────────
    acc = p.acc_min_z            # accommodation-floor z-index (LOWER/UPPER split)
    z = db
    for nlayers in base:
        z0 = z
        z1 = min(z + int(nlayers), nz)
        if z1 <= z0:
            continue
        # role by where the band sits relative to the accommodation floor
        role = TierRole.LOWER if z1 <= acc else TierRole.UPPER
        tiers.append((z0, z1, role))
        z = z1
    # absorb any rounding gap into the last band
    if tiers and z < nz:
        z0, z1, role = tiers[-1]
        tiers[-1] = (z0, nz, role)

    # ── hard floor: no DECK band below MIN_DECK_M (DB is exempt) ───────
    # Merge any sub-floor deck into the band above (or below if it is the top),
    # so deck clear-heights never fall below 2.0 m other than the double bottom.
    MIN_DECK_M = 2.0
    min_layers = max(1, int(np.ceil(MIN_DECK_M / dz)))
    changed = True
    while changed and len(tiers) > 2:   # keep at least DB + 1 deck
        changed = False
        for i in range(len(tiers)):
            z0, z1, role = tiers[i]
            if role == TierRole.DB:
                continue
            if (z1 - z0) < min_layers:
                # merge into the adjacent deck band (prefer the one above)
                if i + 1 < len(tiers):
                    nz0, nz1, nrole = tiers[i + 1]
                    tiers[i + 1] = (z0, nz1, nrole)
                elif i - 1 >= 0 and tiers[i - 1][2] != TierRole.DB:
                    pz0, pz1, prole = tiers[i - 1]
                    tiers[i - 1] = (pz0, z1, prole)
                else:
                    continue
                del tiers[i]
                changed = True
                break

    if not tiers:
        tiers.append((0, nz, TierRole.UPPER))
    return tiers


def _compute_vertical_tiers(p: ShipParameterization) -> List[Tuple[int, int, int]]:
    """
    Compute vertical tier boundaries for the main hull [0, nz_hull).

    Returns list of (z0, z1, deck_idx) where deck_idx is:
        0 = double-bottom tier
        1 = lower hold (between DB and acc_min_z)
        2 = upper hold / tweendeck (above acc_min_z)

    Tiers are only created if they span at least 1 z-layer.
    At minimum there is always one tier covering the full hull height.
    """
    nz  = p.nz_hull
    db  = p.db_layers      # number of DB layers (z=0..db-1)
    acc = p.acc_min_z      # lowest z ACCOMMODATION is permitted

    tiers = []

    # DB tier: z = 0..db-1
    if db > 0 and db <= nz:
        tiers.append((0, db, 0))

    # Lower hold: z = db..acc-1
    if acc > db:
        z0 = db
        z1 = min(acc, nz)
        if z1 > z0:
            tiers.append((z0, z1, 1))

    # Upper hold: z = acc..nz-1
    if acc < nz:
        z0 = max(acc, db)
        if z0 < nz:
            tiers.append((z0, nz, 2))

    # Edge case: if no tiers were created (e.g. nz=1), create one
    if not tiers:
        tiers.append((0, nz, 2))

    return tiers


# ─────────────────────────────────────────────────────────────────
# Zone type lookup
# ─────────────────────────────────────────────────────────────────

def _zone_type(region_label: str, deck_idx: int,
               p: ShipParameterization,
               tier_role: Optional["TierRole"] = None) -> ZoneType:
    """
    Map (region_label, vertical tier role) → ZoneType.

    Engine strips below ``acc_min_z`` use ENGINE_REGION (DB / lower hold).
    Tiers at or above ``acc_min_z`` use ENGINE_UPPER — same upper-hold
    boundary as :class:`ZoneType.MAIN_UPPER` on the forward holds.
    Eligibility is type-dependent (see :func:`_engine_upper_eligible_comps`).

    The main-body type keys off ``tier_role`` (DB/LOWER/UPPER) so it is correct
    for any number of physical deck bands, not just deck_idx 0/1/2.
    """
    if region_label == "aft_peak":
        return ZoneType.AFT_PEAK
    if region_label == "fwd_peak":
        return ZoneType.FWD_PEAK
    if region_label == "engine":
        role = tier_role if tier_role is not None else _tier_role_from_deck_idx(deck_idx)
        if role == TierRole.UPPER:
            return ZoneType.ENGINE_UPPER
        return ZoneType.ENGINE_REGION
    # main region — keyed by structural role
    role = tier_role if tier_role is not None else _tier_role_from_deck_idx(deck_idx)
    if role == TierRole.DB:
        return ZoneType.MAIN_DB
    if role == TierRole.LOWER:
        return ZoneType.MAIN_LOWER
    return ZoneType.MAIN_UPPER


# ─────────────────────────────────────────────────────────────────
# Zone construction helper
# ─────────────────────────────────────────────────────────────────

def _make_zone(
    zone_id: int,
    zone_type: ZoneType,
    hold_idx: int,
    x0: int, x1: int,
    y0: int, y1: int,
    z0: int, z1: int,
    fm: np.ndarray,
    p: ShipParameterization,
    deck_idx: int,
    tier_role: Optional["TierRole"] = None,
    eligible_override: Optional[FrozenSet[Comp]] = None,
    side: str = "centre",
    mirror_id: int = -1,
    assign_mask: Optional[np.ndarray] = None,
) -> Zone:
    """
    Construct a Zone from its bounding box and the full_mask.
    Computes available_cells, centroid, hull_avail, and node features.

    When ``assign_mask`` is set (skin-relative carving), only voxels where
    assign_mask is True and occupancy > EPS_EMPTY count toward the zone.
    """
    nx, ny, nz_total = fm.shape
    region = fm[x0:x1, y0:y1, z0:z1]
    # Geometric "inside hull" decision: any voxel with occupancy above
    # EPS_EMPTY is kept; below that is treated as empty.
    if assign_mask is not None:
        if assign_mask.shape != region.shape:
            raise ValueError(
                f"assign_mask shape {assign_mask.shape} != region shape {region.shape}"
            )
        inside_mask = assign_mask & (region > EPS_EMPTY)
    else:
        inside_mask = region > EPS_EMPTY
    avail  = int(inside_mask.sum())
    total  = (x1 - x0) * (y1 - y0) * (z1 - z0)

    # Bbox midpoints (normalised) — not voxel-mass centroids. For hull zones,
    # (z0,z1) come from vertical tiers so lower decks always have smaller cz.
    if avail > 0: # use the centroid of the occupied voxels
        occ = np.argwhere(inside_mask)
        cx = (x0 + float(occ[:,0].mean()) + 0.5) / nx
        cy = (y0 + float(occ[:,1].mean()) + 0.5) / ny
        cz = (z0 + float(occ[:,2].mean()) + 0.5) / nz_total
    else:
        cx = ((x0+x1)/2)/nx 
        cy = ((y0+y1)/2)/ny
        cz = ((z0+z1)/2)/nz_total

    # Fractional hull availability: use the sum of fractional occupancies
    # rather than just counting boolean cells. This preserves partial
    # boundary voxels from SDF-based hull masks.
    eff_voxels       = float(region[inside_mask].sum()) if avail > 0 else 0.0
    hull_avail       = eff_voxels / max(1, total)
    zone_width_norm  = (y1 - y0) / ny
    zone_length_norm = (x1 - x0) / nx

    elig = (
        eligible_override
        if eligible_override is not None
        else ZONE_ELIGIBILITY[zone_type]
    )

    return Zone(
        zone_id=zone_id,
        zone_type=zone_type,
        hold_idx=hold_idx,
        x0=x0, x1=x1,
        y0=y0, y1=y1,
        z0=z0, z1=z1,
        available_cells=avail,
        total_cells=total,
        eligible_comps=elig,
        cx_norm=round(cx, 5),
        cy_norm=round(cy, 5),
        cz_norm=round(cz, 5),
        hull_avail=round(hull_avail, 5),
        zone_width_norm=round(zone_width_norm, 5),
        zone_length_norm=round(zone_length_norm, 5),
        deck_idx=deck_idx,
        tier_role=(tier_role if tier_role is not None
                   else _tier_role_from_deck_idx(deck_idx)),
        side=side,
        mirror_id=mirror_id,
    )


def _compute_side_carve_x_ranges(
    bh_x: List[Tuple[int, int, str, int]],
    ship_type: ShipType,
    rng: np.random.Generator,
) -> List[Tuple[int, int, str]]:
    """
    Longitudinal x-ranges that receive transverse side-zone carving.

    1 engine side-zone block + 2–3 cargo-block side-zones snapped to holds.
    """
    if not side_zones_enabled(ship_type):
        return []

    cfg = SIDE_ZONE_CONFIG[ship_type]
    n_blocks = int(cfg.get("n_cargo_blocks", 2))
    n_blocks = int(np.clip(rng.integers(2, 4) if n_blocks < 2 else n_blocks, 2, 3))

    ranges: List[Tuple[int, int, str]] = []

    engine_segs = [(x0, x1) for x0, x1, lbl, _ in bh_x if lbl == "engine"]
    if engine_segs:
        ranges.append((engine_segs[0][0], engine_segs[-1][1], "engine"))

    main_segs = [(x0, x1) for x0, x1, lbl, _ in bh_x if lbl == "main"]
    if main_segs:
        n_holds = len(main_segs)
        n_blocks = min(n_blocks, n_holds)
        holds_per = max(1, n_holds // n_blocks)
        i = 0
        while i < n_holds:
            j = min(i + holds_per, n_holds)
            x0 = main_segs[i][0]
            x1 = main_segs[j - 1][1]
            ranges.append((x0, x1, "main"))
            i = j
            if len([r for r in ranges if r[2] == "main"]) >= n_blocks:
                break
        # merge remainder into last block
        if i < n_holds and ranges and ranges[-1][2] == "main":
            ranges[-1] = (ranges[-1][0], main_segs[-1][1], "main")

    return ranges


def _in_side_carve_range(
    x0: int, x1: int, region_label: str,
    carve_ranges: List[Tuple[int, int, str]],
) -> bool:
    for cx0, cx1, clbl in carve_ranges:
        if clbl != region_label:
            continue
        if x0 >= cx0 and x1 <= cx1:
            return True
        if x0 < cx1 and x1 > cx0 and region_label == "engine":
            return True
    return False


def _carve_block_for_segment(
    x0: int, x1: int,
    region_label: str,
    carve_ranges: List[Tuple[int, int, str]],
) -> Optional[Tuple[int, int]]:
    """Return (cx0, cx1) carve block containing this longitudinal segment."""
    for cx0, cx1, clbl in carve_ranges:
        if clbl != region_label:
            continue
        if region_label == "engine":
            if x0 < cx1 and x1 > cx0:
                return (cx0, cx1)
        elif x0 >= cx0 and x1 <= cx1:
            return (cx0, cx1)
    return None


def _column_port_interval(occ: np.ndarray, w: int) -> Set[int]:
    ys = np.flatnonzero(occ)
    if ys.size == 0:
        return set()
    y_min = int(ys[0])
    out: Set[int] = set()
    for y in range(y_min, min(y_min + w, occ.size)):
        if occ[y]:
            out.add(y)
    return out


def _column_stbd_interval(occ: np.ndarray, w: int) -> Set[int]:
    ys = np.flatnonzero(occ)
    if ys.size == 0:
        return set()
    y_max = int(ys[-1])
    out: Set[int] = set()
    for y in range(max(0, y_max - w + 1), y_max + 1):
        if occ[y]:
            out.add(y)
    return out


def _expand_inward(
    interval: Set[int],
    occ: np.ndarray,
    side: str,
    max_cells: int,
    ny: int,
) -> Set[int]:
    """Expand interval toward centreline by up to max_cells in-hull cells."""
    out = set(interval)
    if not out or max_cells <= 0:
        return out
    centre = ny // 2
    for _ in range(max_cells):
        if side == "port":
            candidates = [
                y + 1 for y in out
                if y + 1 < ny and occ[y + 1] and (y + 1) not in out
            ]
        else:
            candidates = [
                y - 1 for y in out
                if y - 1 >= 0 and occ[y - 1] and (y - 1) not in out
            ]
        if not candidates:
            break
        y_new = min(candidates) if side == "port" else max(candidates)
        if side == "port" and y_new >= centre:
            break
        if side == "stbd" and y_new < centre:
            break
        out.add(y_new)
    return out


def _bridge_side_intervals(
    prev: Set[int],
    curr: Set[int],
    occ_prev: np.ndarray,
    occ_curr: np.ndarray,
    side: str,
    ny: int,
    max_bridge: int = 2,
) -> Tuple[Set[int], Set[int], bool]:
    """
    Expand the wider-hull slice inward to create y overlap across x.
    Returns (prev_out, curr_out, bridged).
    """
    if not prev or not curr:
        return prev, curr, False
    if prev & curr:
        return prev, curr, True

    width_prev = int(occ_prev.sum())
    width_curr = int(occ_curr.sum())
    if width_prev >= width_curr:
        prev_out = _expand_inward(prev, occ_prev, side, max_bridge, ny)
        if prev_out & curr:
            return prev_out, curr, True
        curr_out = _expand_inward(curr, occ_curr, side, max_bridge, ny)
        if prev_out & curr_out:
            return prev_out, curr_out, True
        return prev, curr, False

    curr_out = _expand_inward(curr, occ_curr, side, max_bridge, ny)
    if prev & curr_out:
        return prev, curr_out, True
    prev_out = _expand_inward(prev, occ_prev, side, max_bridge, ny)
    if prev_out & curr_out:
        return prev_out, curr_out, True
    return prev, curr, False


def _compute_skin_relative_side_bands(
    hm: np.ndarray,
    x0: int, x1: int,
    z0: int, z1: int,
    w: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Per-(x,z) transverse side bands with longitudinal face continuity.

    Bands are built per carve-block subvolume, then port/stbd intervals
    are bridged across adjacent x slices (up to 2 cells) when the hull
    skin shifts without y overlap.
    """
    sub = hm[x0:x1, :, z0:z1] > EPS_EMPTY
    nxb, ny, nzb = sub.shape
    port = np.zeros((nxb, ny, nzb), dtype=bool)
    stbd = np.zeros((nxb, ny, nzb), dtype=bool)
    if w <= 0:
        return port, stbd

    for zi in range(nzb):
        prev_port: Set[int] = set()
        prev_stbd: Set[int] = set()
        prev_occ: Optional[np.ndarray] = None
        for xi in range(nxb):
            occ = sub[xi, :, zi]
            if not occ.any():
                prev_port, prev_stbd, prev_occ = set(), set(), None
                continue

            port_ys = _column_port_interval(occ, w)
            stbd_ys = _column_stbd_interval(occ, w)
            overlap = port_ys & stbd_ys
            port_ys -= overlap
            stbd_ys -= overlap

            if xi > 0 and prev_occ is not None:
                prev_port, port_ys, _ = _bridge_side_intervals(
                    prev_port, port_ys, prev_occ, occ, "port", ny,
                )
                for y in prev_port:
                    port[xi - 1, y, zi] = True
                prev_stbd, stbd_ys, _ = _bridge_side_intervals(
                    prev_stbd, stbd_ys, prev_occ, occ, "stbd", ny,
                )
                for y in prev_stbd:
                    stbd[xi - 1, y, zi] = True

            for y in port_ys:
                port[xi, y, zi] = True
            for y in stbd_ys:
                stbd[xi, y, zi] = True
            prev_port, prev_stbd, prev_occ = port_ys, stbd_ys, occ

    return port, stbd


def _emit_side_zone_pair_skin_relative(
    zone_id: int,
    centre_type: ZoneType,
    hold_idx: int,
    x0: int, x1: int,
    z0: int, z1: int,
    deck_idx: int,
    hm: np.ndarray,
    p: ShipParameterization,
    zone_mask: np.ndarray,
    port_mask: np.ndarray,
    stbd_mask: np.ndarray,
    mirror_counter: List[int],
    tier_role: Optional["TierRole"] = None,
) -> Tuple[List[Zone], int]:
    """Emit port + stbd side zones from skin-relative boolean masks."""
    ny = p.ny
    zones: List[Zone] = []
    _role = tier_role if tier_role is not None else _tier_role_from_deck_idx(deck_idx)
    side_type = _side_zone_type_for_tier(deck_idx, tier_role=_role)

    mid = mirror_counter[0]
    mirror_counter[0] += 1
    port_cells = 0
    stbd_cells = 0
    port_zone: Optional[Zone] = None
    stbd_zone: Optional[Zone] = None

    sub_occ = hm[x0:x1, :, z0:z1] > EPS_EMPTY

    for side_name, band_mask in (("port", port_mask), ("stbd", stbd_mask)):
        assign = band_mask & sub_occ
        if not assign.any():
            continue
        elig = _side_eligible_for_centre_zone(
            p.ship_type, side_type, centre_type,
            block_cx=((x0 + x1) / 2.0) / max(p.nx, 1),
        )
        sz = _make_zone(
            zone_id, side_type, hold_idx,
            x0, x1, 0, ny, z0, z1,
            hm, p, deck_idx,
            tier_role=_role,
            eligible_override=elig,
            side=side_name,
            mirror_id=mid,
            assign_mask=assign,
        )
        if sz.available_cells == 0:
            continue
        if side_name == "port":
            port_cells = sz.available_cells
            port_zone = sz
        else:
            stbd_cells = sz.available_cells
            stbd_zone = sz
        zones.append(sz)
        zone_mask[x0:x1, :, z0:z1] = np.where(
            assign,
            zone_id,
            zone_mask[x0:x1, :, z0:z1],
        )
        zone_id += 1

    if port_zone is not None and stbd_zone is None:
        port_zone.mirror_id = -1
    elif stbd_zone is not None and port_zone is None:
        stbd_zone.mirror_id = -1
    elif port_zone is not None and stbd_zone is not None:
        if port_cells == 0 or stbd_cells == 0:
            port_zone.mirror_id = -1
            stbd_zone.mirror_id = -1
        else:
            denom = max(port_cells, stbd_cells, 1)
            if abs(port_cells - stbd_cells) / denom > 0.05:
                port_zone.mirror_id = -1
                stbd_zone.mirror_id = -1

    return zones, zone_id


def _emit_side_zone_pair(
    zone_id: int,
    centre_type: ZoneType,
    hold_idx: int,
    x0: int, x1: int,
    z0: int, z1: int,
    deck_idx: int,
    hm: np.ndarray,
    p: ShipParameterization,
    zone_mask: np.ndarray,
    w: int,
    mirror_counter: List[int],
    tier_role: Optional["TierRole"] = None,
) -> Tuple[List[Zone], int]:
    """Emit port + stbd side zones for one (carve_block × tier)."""
    ny = p.ny
    zones: List[Zone] = []
    _role = tier_role if tier_role is not None else _tier_role_from_deck_idx(deck_idx)
    side_type = _side_zone_type_for_tier(deck_idx, tier_role=_role)
    y_port = (0, w)
    y_stbd = (ny - w, ny)

    mid = mirror_counter[0]
    mirror_counter[0] += 1
    port_cells = 0
    stbd_cells = 0
    port_zone: Optional[Zone] = None
    stbd_zone: Optional[Zone] = None

    for side_name, (ya, yb) in (("port", y_port), ("stbd", y_stbd)):
        elig = _side_eligible_for_centre_zone(
            p.ship_type, side_type, centre_type,
            block_cx=((x0 + x1) / 2.0) / max(p.nx, 1),
        )
        sz = _make_zone(
            zone_id, side_type, hold_idx,
            x0, x1, ya, yb, z0, z1,
            hm, p, deck_idx,
            tier_role=_role,
            eligible_override=elig,
            side=side_name,
            mirror_id=mid,
        )
        if sz.available_cells == 0:
            continue
        if side_name == "port":
            port_cells = sz.available_cells
            port_zone = sz
        else:
            stbd_cells = sz.available_cells
            stbd_zone = sz
        zones.append(sz)
        zone_mask[x0:x1, ya:yb, z0:z1] = np.where(
            hm[x0:x1, ya:yb, z0:z1] > EPS_EMPTY,
            zone_id, zone_mask[x0:x1, ya:yb, z0:z1],
        )
        zone_id += 1

    # If only one twin survived hull taper, drop mirror pairing.
    if port_zone is not None and stbd_zone is None:
        port_zone.mirror_id = -1
    elif stbd_zone is not None and port_zone is None:
        stbd_zone.mirror_id = -1
    elif port_zone is not None and stbd_zone is not None:
        if port_cells == 0 or stbd_cells == 0:
            port_zone.mirror_id = -1
            stbd_zone.mirror_id = -1

    return zones, zone_id


# ─────────────────────────────────────────────────────────────────
# Superstructure zone builder
# ─────────────────────────────────────────────────────────────────

def _build_ss_zones(
    zone_id_start: int,
    sm: np.ndarray,
    fm: np.ndarray,
    p: ShipParameterization,
    top_deck_z: np.ndarray,
) -> List[Zone]:
    """
    Build superstructure zones from the ss_only mask.

    Groups voxels by Tetris **level** (``z - top_deck_z - 1``), then 6-connected
    3D components per level.  ``deck_idx`` is ``3 + logical_deck`` from
    ``ss_layers_per_deck``, so bow and stern cells on the same accommodation
    deck share the same tier even when absolute z differs on sloped hulls.

    ``fm`` is unused but kept for API compatibility with callers.
    """
    del fm  # SS zones use sm only; signature kept for call-site stability
    nz_ss = p.nz_ss
    layers_per = p.ss_layers_per_deck or [max(1, nz_ss)]

    ss_zones: List[Zone] = []
    zid = zone_id_start

    ss_mask = sm > EPS_EMPTY
    if not ss_mask.any():
        return ss_zones

    # Per-voxel Tetris level relative to local hull top: SS layer ``level``
    # sits at ``top_deck_z + 1 + level`` (see place_superstructure), so the
    # level of any SS voxel is ``z - top_deck_z - 1`` everywhere.
    level_field = np.full(sm.shape, -1, dtype=np.int16)
    xs, ys, zs = np.nonzero(ss_mask)
    tdz_at = top_deck_z[xs, ys]
    valid = tdz_at >= 0
    xs, ys, zs = xs[valid], ys[valid], zs[valid]
    tdz_at = tdz_at[valid]
    levels = zs - tdz_at - 1
    level_field[xs, ys, zs] = levels.astype(np.int16)

    struct_3d = generate_binary_structure(3, 1)

    # One zone per logical accommodation deck × 3D connected component
    n_logical = p.ss_n_decks or max(1, len(layers_per))
    for logical in range(n_logical):
        sub = np.zeros_like(ss_mask, dtype=bool)
        for level in range(nz_ss):
            if ss_logical_deck_index(level, layers_per) != logical:
                continue
            sub |= ss_mask & (level_field == level)
        if not sub.any():
            continue
        labeled, n_comp = label(sub, structure=struct_3d)

        for comp_id in range(1, n_comp + 1):
            coords = np.argwhere(labeled == comp_id)
            if coords.size == 0:
                continue

            x0 = int(coords[:, 0].min())
            x1 = int(coords[:, 0].max()) + 1
            y0 = int(coords[:, 1].min())
            y1 = int(coords[:, 1].max()) + 1
            z0 = int(coords[:, 2].min())
            z1 = int(coords[:, 2].max()) + 1

            zone = _make_zone(
                zone_id=zid,
                zone_type=ZoneType.SUPERSTRUCTURE,
                hold_idx=-1,
                x0=x0, x1=x1,
                y0=y0, y1=y1,
                z0=z0, z1=z1,
                fm=sm,
                p=p,
                deck_idx=3 + logical,
            )
            if zone.available_cells == 0:
                continue
            ss_zones.append(zone)
            zid += 1

    return ss_zones


# ─────────────────────────────────────────────────────────────────
# Validation
# ─────────────────────────────────────────────────────────────────

def validate_bulkhead_layout(layout: BulkheadLayout) -> Tuple[bool, List[str]]:
    """
    Sanity checks on a BulkheadLayout.

    Checks
    ------
    1. Every available hull cell is assigned to exactly one zone (no gaps)
    2. No available cell is assigned to zone_id -1 (unassigned)
    3. At least one ENGINE_REGION zone exists
    4. At least one AFT_PEAK zone and one FWD_PEAK zone exist
    5. All zones have available_cells > 0
    6. ENGINE_REGION zones are confined to aft region (cx_norm < er_max_cx + tol)
    7. FWD_PEAK zones are fully fwd of collision bulkhead
    8. Minimum viable graph: at least 4 zones total
    """
    p    = layout.params
    fm   = layout.hull_result.full_mask
    hm   = layout.hull_result.hull_mask
    sm   = layout.hull_result.ss_mask
    zmask = layout.zone_mask
    warns = []

    # Check 1: Every available cell (hull OR SS) assigned
    available   = fm > EPS_EMPTY
    assigned    = zmask >= 0
    unassigned  = available & ~assigned
    if unassigned.sum() > 0:
        warns.append(
            f"  {unassigned.sum()} available cells are unassigned in zone_mask"
        )

    # Check 2: No zone has zero available cells
    zero_avail = [z for z in layout.zones if z.available_cells == 0]
    if zero_avail:
        warns.append(
            f"  {len(zero_avail)} zones have available_cells=0: "
            f"{[z.zone_id for z in zero_avail]}"
        )

    # Check 3: ENGINE_REGION exists
    if not layout.zones_of_type(ZoneType.ENGINE_REGION):
        warns.append("  No ENGINE_REGION zone found")

    # Check 4: Peak zones — use hull-only mask (not full_mask) so SS
    # voxels that settled into the peak region don't inflate the count.
    p_  = layout.params
    hm_ = hm

    aft_pk_cells = int((hm_[:p_.aft_peak_x_end, :, :p_.nz_hull] > EPS_EMPTY).sum())
    if not layout.zones_of_type(ZoneType.AFT_PEAK) and aft_pk_cells > 0:
        warns.append(
            f"  No AFT_PEAK zone found but {aft_pk_cells} cells exist there"
        )

    fwd_pk_cells = int((hm_[p_.fwd_peak_x_start:, :, :p_.nz_hull] > EPS_EMPTY).sum())
    if not layout.zones_of_type(ZoneType.FWD_PEAK) and fwd_pk_cells > 0:
        warns.append(
            f"  No FWD_PEAK zone found but {fwd_pk_cells} cells exist there"
        )

    # Check 5: Minimum graph size
    if layout.n_zones < 4:
        warns.append(
            f"  Only {layout.n_zones} zones — minimum expected is 4"
        )

    # Check 6: ENGINE_REGION confined to aft (type-dependent limit)
    er_cx_limit = ER_MAX_CX_BY_TYPE.get(layout.params.ship_type, SPATIAL_RULES["er_max_cx"])
    er_max_cx = er_cx_limit + 0.05   # small tolerance
    for z in layout.zones_of_type(ZoneType.ENGINE_REGION):
        if z.cx_norm > er_max_cx:
            warns.append(
                f"  ENGINE_REGION zone {z.zone_id} centroid cx={z.cx_norm:.3f} "
                f"> er_max_cx={er_max_cx:.3f}"
            )

    for z in layout.zones_of_type(ZoneType.ENGINE_UPPER):
        if z.cx_norm > er_max_cx:
            warns.append(
                f"  ENGINE_UPPER zone {z.zone_id} centroid cx={z.cx_norm:.3f} "
                f"> er_max_cx={er_max_cx:.3f}"
            )

    # Check 7: FWD_PEAK beyond collision bulkhead
    fwd_pk_min = SPATIAL_RULES["fwd_peak_min_cx"] - 0.05
    for z in layout.zones_of_type(ZoneType.FWD_PEAK):
        if z.cx_norm < fwd_pk_min:
            warns.append(
                f"  FWD_PEAK zone {z.zone_id} centroid cx={z.cx_norm:.3f} "
                f"< fwd_pk_min={fwd_pk_min:.3f}"
            )

    # Check 8: port/stbd mirror geometry (box symmetry about beam centre)
    ny = layout.params.ny
    by_mirror: Dict[int, List[Zone]] = {}
    for z in layout.zones:
        if z.mirror_id >= 0:
            by_mirror.setdefault(z.mirror_id, []).append(z)
    for mid, pair in by_mirror.items():
        if len(pair) != 2:
            warns.append(f"  mirror_id {mid} has {len(pair)} zones, expected 2")
            continue
        port = next((zz for zz in pair if zz.side == "port"), None)
        stbd = next((zz for zz in pair if zz.side == "stbd"), None)
        if port is None or stbd is None:
            warns.append(f"  mirror_id {mid} missing port/stbd pair")
            continue
        denom = max(port.available_cells, stbd.available_cells, 1)
        rel_asym = abs(port.available_cells - stbd.available_cells) / denom
        if rel_asym > 0.05:
            warns.append(
                f"  mirror_id {mid} cell asymmetry: port={port.available_cells} "
                f"stbd={stbd.available_cells} rel={rel_asym:.3f}"
            )
        if layout.side_carve_mode != "skin_relative":
            if port.y1 != ny - stbd.y0 or stbd.y1 != ny - port.y0:
                warns.append(f"  mirror_id {mid} y-extents not symmetric about centre")

    return len(warns) == 0, warns


# ─────────────────────────────────────────────────────────────────
# Visualisation
# ─────────────────────────────────────────────────────────────────

# Colour per ZoneType — used consistently in all panels
