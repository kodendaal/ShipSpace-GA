"""
compartment_assignment.py
=========================
Stage 4: Compartment Assignment
---------------------------------
Assigns one Comp label to every zone in a BulkheadLayout, subject to
three constraint layers: eligibility rules, volume budget satisfaction,
and spatial placement rules.

Strategy: greedy priority fill followed by repair passes
(``assign_compartments``):

  1. Mandatory assignments — structural singletons (peaks, ER seed, SS)
  2. Engine-first prefill — seed ER low, expand upward to meet engine budget
  3. Greedy fill — largest unassigned zones first, comp with highest
     (deficit X spatial_pref X physics_pref) wins; side-zone priors
  4. VOID backfill and VOID reduction — flip VOID zones to budget-deficit
     comps where safe; fuel trim; empty superstructure zones removed
  5. Contiguity repair — ENGINE_ROOM must form a single connected group
  6. Budget rebalancing — targeted reassignments to reduce worst errors
  7. Fuel-above-ER cleanup, final fuel trim, then a budget-preserving
     LCG/KG optimisation (small mixed-integer program, SciPy HiGHS)
  8. Physics computation — mass, LCG, KG, simplified GM
  9. Budget error computation

Inputs:   BulkheadLayout (from Stage 3)
Outputs:  CompartmentAssignment
            .assignments      — Dict[int, Comp], zone_id → Comp label
            .zone_masses      — Dict[int, float], zone_id → mass (tonnes)
            .actual_lcg_frac  — realised LCG / L
            .actual_kg_frac   — realised KG / D
            .actual_gm_t      — simplified transverse GM (metres)
            .budget_errors    — Dict[str, float], per-comp |actual - target|
                               (fractions vs sum(available_cells) over hull + SS)

Coordinate convention (inherited from Stages 1-3):
    x : 0 = aft,   nx-1 = bow
    y : 0 = port,  ny-1 = starboard
    z : 0 = keel,  increasing upward

Dependencies: numpy, scipy, ship_params, bulkhead_placement.
"""

from __future__ import annotations
import contextlib
import os
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, FrozenSet
from collections import defaultdict


@contextlib.contextmanager
def _silence_native_stdio():
    """Mute C/C++ writes to fd 1/2 (HiGHS prints past Python sys.stderr)."""
    devnull = os.open(os.devnull, os.O_WRONLY)
    old_out = os.dup(1)
    old_err = os.dup(2)
    try:
        os.dup2(devnull, 1)
        os.dup2(devnull, 2)
        yield
    finally:
        os.dup2(old_out, 1)
        os.dup2(old_err, 2)
        os.close(old_out)
        os.close(old_err)
        os.close(devnull)

from ship_params import (
    ShipParameterization, ShipType, Comp, density_for, SPATIAL_RULES,
    VOLUME_BUDGETS, WATERFILL_PRIORITY,
)
from bulkhead_placement import (
    BulkheadLayout, Zone, ZoneType, TierRole,
)


# ─────────────────────────────────────────────────────────────────
# Budget key ↔ Comp mapping
# ─────────────────────────────────────────────────────────────────

BUDGET_TO_COMP: Dict[str, Comp] = {
    "engine":           Comp.ENGINE_ROOM,
    "machinery":        Comp.MACHINERY,
    "cargo":            Comp.CARGO,
    "stores":           Comp.STORES,
    "accommodation":    Comp.ACCOMMODATION,
    "fuel_tanks":       Comp.FUEL_TANKS,
    "ballast_tanks":    Comp.BALLAST_TANKS,
    # VOID  — fills remainder, not explicitly budgeted
    # STEERING_GEAR — mandatory aft peak, not budgeted
    # EMPTY — outside hull, never assigned
}

COMP_TO_BUDGET: Dict[Comp, str] = {v: k for k, v in BUDGET_TO_COMP.items()}

# Compartment types that participate in budget tracking.
BUDGETED_COMPS: FrozenSet[Comp] = frozenset(BUDGET_TO_COMP.values())


# ─────────────────────────────────────────────────────────────────
# Result container
# ─────────────────────────────────────────────────────────────────

@dataclass
class CompartmentAssignment:
    """
    Complete compartment assignment for one ship instance.

    assignments   : zone_id -> Comp label (one per zone)
    zone_masses   : zone_id -> mass in tonnes (density x volume)
    actual_lcg_frac : realised LCG / L  (0=aft, 1=bow)
    actual_kg_frac  : realised KG / D   (0=keel, 1=deck)
    actual_gm_t     : simplified transverse GM in metres
    kb_m            : centre of buoyancy height (m) — geometry-only, density-independent
    bm_m            : metacentric radius (m) — geometry-only
    budget_errors   : {comp_budget_key: |actual_frac - target_frac|}
    budget_fracs    : {comp_budget_key: actual volume fraction}
    layout          : BulkheadLayout (back-reference)
    """
    assignments:      Dict[int, Comp]
    zone_masses:      Dict[int, float]
    actual_lcg_frac:  float
    actual_kg_frac:   float
    actual_gm_t:      float
    budget_errors:    Dict[str, float]
    budget_fracs:     Dict[str, float]
    layout:           BulkheadLayout
    kb_m:             float = 0.0
    bm_m:             float = 0.0

    @property
    def params(self) -> ShipParameterization:
        return self.layout.params

    @property
    def total_mass(self) -> float:
        return sum(self.zone_masses.values())

    def comp_zones(self, comp: Comp) -> List[Zone]:
        """All zones assigned to a given Comp type."""
        return [self.layout.zone_by_id[zid]
                for zid, c in self.assignments.items() if c == comp]

    def summary(self) -> str:
        p = self.params
        lines = [
            f"CompartmentAssignment  {p.ship_type.name}  "
            f"L={p.L:.0f}m  {self.layout.n_zones} zones",
            f"",
            f"  {'Zone':>4}  {'Type':15}  {'Comp':20}  "
            f"{'Avail':>6}  {'Mass(t)':>8}  {'cx':>5}  {'cz':>5}",
            f"  {'---' * 25}",
        ]
        for z in self.layout.zones:
            comp = self.assignments[z.zone_id]
            mass = self.zone_masses[z.zone_id]
            lines.append(
                f"  {z.zone_id:>4}  {z.zone_type.name:15}  "
                f"{comp.name:20}  {z.available_cells:>6}  "
                f"{mass:>8.0f}  {z.cx_norm:>5.2f}  {z.cz_norm:>5.2f}"
            )
        lines.append(f"")
        lines.append(f"  Total mass: {self.total_mass:,.0f} t")
        lines.append(
            f"  LCG: actual={self.actual_lcg_frac:.3f}  "
            f"target={p.target_lcg_frac:.3f}  "
            f"error={abs(self.actual_lcg_frac - p.target_lcg_frac):.4f}"
        )
        lines.append(
            f"  KG:  actual={self.actual_kg_frac:.3f}  "
            f"target={p.target_kg_frac:.3f}  "
            f"error={abs(self.actual_kg_frac - p.target_kg_frac):.4f}"
        )
        lines.append(f"  GM_t: {self.actual_gm_t:.2f} m")
        lines.append(f"")
        lines.append(f"  Budget satisfaction:")
        for key in sorted(self.budget_errors.keys()):
            tgt = p.budget.get(key, 0)
            act = self.budget_fracs.get(key, 0)
            err = self.budget_errors[key]
            lines.append(
                f"    {key:20s}  target={tgt:.3f}  "
                f"actual={act:.3f}  error={err:.4f}"
            )
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────

def assign_compartments(
    layout: BulkheadLayout,
    rng: Optional[np.random.Generator] = None,
    seed: Optional[int] = None,
    budget_noise: float = 0.10,
) -> CompartmentAssignment:
    """
    Assign compartment labels to every zone in a BulkheadLayout.

    Parameters
    ----------
    layout : BulkheadLayout
        Output of Stage 3.
    rng : numpy Generator, optional
    seed : int, optional
    budget_noise : float
        Fractional noise (+/-) added to greedy scores for diversity.
        Default 0.10 = +/-10%.

    Returns
    -------
    CompartmentAssignment
    """
    if rng is None:
        rng = np.random.default_rng(seed)

    p = layout.params

    # Capacity pre-filter is enforced in dataset_builder (resample on fail).
    # Callers may invoke _capacity_prefilter() explicitly before assignment.

    # -- 1. Mandatory assignments --
    assignments: Dict[int, Comp] = {}
    _assign_mandatory(layout, assignments)

    # -- 2. Engine-first prefill (before generic greedy) --
    # Small-engine working craft (yacht/OSV) overshoot the engine budget badly
    # because each ER zone is a large fraction of a small hull. Seed less
    # aggressively and let the budget-aware greedy fill complete the ER, so
    # achieved engine volume can reach the small real-ship values.
    engine_floor = 0.5 if p.ship_type in (ShipType.YACHT, ShipType.OSV, ShipType.PATROL) else 1.0
    _prefill_engine_deficit(layout, assignments, p, floor_ratio=engine_floor)
    _prefill_accommodation_deficit(layout, assignments, p)

    # -- 3. Greedy budget-driven fill --
    _greedy_fill(layout, assignments, p, rng, budget_noise)

    # -- 3b. Side-zone domain-prior re-fill --
    # Greedy assigns every zone (VOID included); this pass applies wing
    # priors (fuel abreast ER, ballast low, contrast upper) to side zones
    # greedy left VOID, budget-aware.
    _side_zone_fill_policy(layout, assignments)

    # -- 3c. Drop zone-graph fuel satellites (keep largest connected cluster) --
    _collapse_satellite_fuel_zones(layout, assignments)

    # -- 4. Fill any remaining unassigned zones with VOID --
    for z in layout.zones:
        if z.zone_id not in assignments:
            assignments[z.zone_id] = Comp.VOID

    # -- 4b. Shed excess VOID where a budgeted comp still has deficit --
    _reduce_void_by_deficit(layout, assignments, p)

    # -- 4b1. Fuel trim (budget cap; no filler over target) --
    _trim_excess_fuel_to_budget(layout, assignments, p)

    # -- 4b2. Working-ship accommodation top-up (VOID → ACCOMMODATION) --
    _topup_accommodation_from_void(layout, assignments, p)

    # -- 4c. Excise VOID superstructure zones --
    # Real ships don't have empty boxes on top of accommodation.
    # Remove these zones entirely so they don't appear in the graph,
    # contribute to physics, or consume budget denominator volume.
    # Also clear full_mask (and ss_only) so voxel hull_mask matches graph.
    excised_ids = set()
    for z in layout.zones:
        if (z.zone_type == ZoneType.SUPERSTRUCTURE
                and assignments.get(z.zone_id) == Comp.VOID):
            excised_ids.add(z.zone_id)

    if excised_ids:
        zm = layout.zone_mask
        excised_voxels = np.zeros(zm.shape, dtype=bool)
        for zid in excised_ids:
            excised_voxels |= (zm == zid)

        # Remove from assignments dict
        for zid in excised_ids:
            assignments.pop(zid, None)
        # Remove from zone list
        layout.zones = [z for z in layout.zones if z.zone_id not in excised_ids]
        # Clear from zone_mask so they become unzoned (-1)
        zm[excised_voxels] = -1
        # Shrink hull envelope to match graph: excised SS void is outside layout
        fm = layout.hull_result.full_mask
        fm[excised_voxels] = 0.0
        ss_only = layout.hull_result.ss_only
        if ss_only is not None:
            ss_only[excised_voxels] = 0.0

    # -- 5. Contiguity repair --
    _repair_contiguity(layout, assignments)

    # -- 6. Budget rebalancing --
    # After greedy fill, scan for the worst budget overshoot/undershoot
    # and try targeted single-zone reassignments to improve balance.
    _rebalance_budgets(layout, assignments, p, max_passes=3)

    # -- 6b. Fuel trim (rebalance may reintroduce excess fuel) --
    _trim_excess_fuel_to_budget(layout, assignments, p)

    # -- 7. Physics steering is applied after all semantic cleanup passes. --

    # -- 7b. Final fuel-above-ER cleanup --
    # Despite all guards, the greedy ordering (largest zones first) can
    # assign FUEL_TANKS to an upper tier before ENGINE_ROOM gets assigned
    # to the lower tier in the same x-band.  Fix any remaining violations
    # by reassigning the fuel zone to the best alternative eligible comp.
    _fix_fuel_above_er(layout, assignments)

    # -- 7c. Final fuel trim (the ER fix may reintroduce excess fuel) --
    _trim_excess_fuel_to_budget(layout, assignments, p)

    _collapse_satellite_fuel_zones(layout, assignments)

    # -- 7d. Budget-preserving LCG/KG optimisation --
    # Run last so later fuel trimming cannot turn a permitted small fuel-volume
    # shift into new VOID. The optimiser itself encodes the fuel-above-ER rule.
    _optimise_physics_budget_preserving(layout, assignments, p)

    # -- 8. Physics computation --
    zone_masses, lcg_frac, kg_frac, gm_t, kb_m, bm_m = _compute_physics(
        layout, assignments, p
    )

    # -- 9. Budget error computation --
    budget_errors, budget_fracs = _compute_budget_errors(
        layout, assignments, p
    )

    return CompartmentAssignment(
        assignments=assignments,
        zone_masses=zone_masses,
        actual_lcg_frac=lcg_frac,
        actual_kg_frac=kg_frac,
        actual_gm_t=gm_t,
        kb_m=kb_m,
        bm_m=bm_m,
        budget_errors=budget_errors,
        budget_fracs=budget_fracs,
        layout=layout,
    )


# ─────────────────────────────────────────────────────────────────
# Step 1: Mandatory assignments
# ─────────────────────────────────────────────────────────────────

def _assign_mandatory(
    layout: BulkheadLayout,
    assignments: Dict[int, Comp],
) -> None:
    """
    Apply hard structural assignments that are always the same.

    1. AFT_PEAK  -> STEERING_GEAR (lowest tier), rest BALLAST_TANKS
    2. FWD_PEAK  -> top tier MAY be MACHINERY (if machinery budget >= 0.10);
                    tiers with deck_idx <= 1 -> BALLAST; upper tiers left
                    for greedy (structural void / tanks).
    3. ENGINE_REGION -> DB tier VOID; ENGINE_ROOM seeded on lowest two
       non-DB tiers (double-bottom is not machinery space at 3 m).
    4. SUPERSTRUCTURE -> ACCOMMODATION to budget (deep-cargo types);
       yacht/OSV/patrol left to greedy pool
    5. STORES -> smallest MAIN zone below SS footprint (if stores
       budget >= 0.01); guarantees at least one stores zone that
       would otherwise be starved by large-deficit comps in greedy
    """
    p = layout.params
    # Budget-driven gates for optional mandatory placements.
    # Keep conservative defaults so behavior remains close to baseline unless
    # the target budgets indicate a clear need for these spaces.
    machinery_gate = 0.10
    machinery_target = float(p.budget.get("machinery", 0.0))

    # -- Aft peak: steering gear on lowest tier (real: single tank-top flat) --
    aft_peaks = layout.zones_of_type(ZoneType.AFT_PEAK)
    if aft_peaks:
        steer_zone = min(
            aft_peaks,
            key=lambda z: (z.z0, z.cz_norm, -z.available_cells),
        )
        assignments[steer_zone.zone_id] = Comp.STEERING_GEAR
        for z in aft_peaks:
            if z.zone_id != steer_zone.zone_id:
                assignments[z.zone_id] = Comp.BALLAST_TANKS

    # -- Forward peak: ballast by default, optional top-tier machinery --
    fwd_peaks = layout.zones_of_type(ZoneType.FWD_PEAK)
    if fwd_peaks:
        # Highest deck_idx first, then highest z1, then size.
        fwd_sorted = sorted(
            fwd_peaks,
            key=lambda z: (z.deck_idx, z.z1, z.available_cells),
            reverse=True,
        )
        assigned_machinery = False
        if len(fwd_sorted) >= 2 and machinery_target >= machinery_gate:
            top = fwd_sorted[0]
            if Comp.MACHINERY in top.eligible_comps:
                assignments[top.zone_id] = Comp.MACHINERY
                assigned_machinery = True
        for z in fwd_sorted:
            if assigned_machinery and assignments.get(z.zone_id) == Comp.MACHINERY:
                continue
            if z.tier_role in (TierRole.DB, TierRole.LOWER):
                assignments[z.zone_id] = Comp.BALLAST_TANKS

    # -- Engine region: reserve DB tier, seed ER on lowest non-DB tiers --
    # At 3 m resolution the keel cell is double-bottom, not machinery space.
    # Prefill still expands upward; greedy excludes fuel-above-ER.
    er_zones = layout.zones_of_type(ZoneType.ENGINE_REGION)
    if er_zones:
        er_sorted = sorted(
            er_zones,
            key=lambda z: (z.z0, -z.available_cells),
        )
        for z in er_sorted:
            if z.tier_role == TierRole.DB:
                assignments[z.zone_id] = Comp.VOID
        non_db = [z for z in er_sorted if z.tier_role != TierRole.DB]
        if non_db:
            seed_pool = non_db
            if p.ship_type in (ShipType.YACHT, ShipType.OSV, ShipType.PATROL):
                # Real yacht/OSV/patrol ER sits forward in the machinery block.
                seed_pool = sorted(
                    non_db,
                    key=lambda z: (z.z0, -z.cx_norm, -z.available_cells),
                )
        else:
            # DB-only engine band: seed ER on lowest tiers (including DB).
            seed_pool = er_sorted
        # Small craft: a single seed zone already over-fills a small hull;
        # greedy + prefill grow ER to the (small) budget. Larger types keep
        # the 2-zone seed for a robust contiguous ER block.
        n_seed = 1 if p.ship_type in (ShipType.YACHT, ShipType.OSV, ShipType.PATROL) else 2
        for z in seed_pool[:n_seed]:
            assignments[z.zone_id] = Comp.ENGINE_ROOM

    # -- Superstructure: accommodation for deep-cargo, greedy for others --
    # For yacht/OSV/patrol, skip mandatory SS assignment so the
    # accommodation budget spreads between SS and MAIN_UPPER via the
    # greedy scorer for working/slender types. Deep-cargo MAIN_UPPER is
    # not ACCOM-eligible; SS zone eligibility is {ACCOMMODATION, VOID} so
    # accommodation still goes there naturally, but the budget isn't
    # pre-consumed before hull zones get a chance.
    working_types = {ShipType.YACHT, ShipType.OSV, ShipType.PATROL}
    if layout.params.ship_type not in working_types:
        _fill_ss_accommodation_to_budget(layout, assignments, p)

    # -- Stores: smallest MAIN zone below the superstructure --
    # STORES has a small budget that gets starved by deficit-proportional
    # greedy scoring (CARGO's deficit is ~20X larger).  Reserve one small
    # zone so every ship has at least one provision store.  Real ships
    # place stores on the upper deck directly below the SS (near galley).
    stores_gate = 0.01
    stores_target = float(p.budget.get("stores", 0.0))
    if stores_target >= stores_gate:
        stores_candidates = [
            z for z in layout.zones
            if (z.zone_id not in assignments
                and Comp.STORES in z.eligible_comps
                and z.zone_type in (ZoneType.MAIN_LOWER, ZoneType.MAIN_UPPER))
        ]
        if stores_candidates:
            ss_zones = layout.zones_of_type(ZoneType.SUPERSTRUCTURE)
            if ss_zones:
                ss_x0 = min(z.x0 for z in ss_zones)
                ss_x1 = max(z.x1 for z in ss_zones)
                below_ss = [z for z in stores_candidates
                            if z.x0 < ss_x1 and ss_x0 < z.x1]
                pool = below_ss if below_ss else stores_candidates
            else:
                pool = stores_candidates
            pool.sort(key=lambda z: z.available_cells)
            assignments[pool[0].zone_id] = Comp.STORES

    # Navigation is part of ACCOMMODATION: the bridge is top-forward SS
    # accommodation; derive it at analysis time if needed.


# ─────────────────────────────────────────────────────────────────
# Step 2a: Engine-first prefill
# ─────────────────────────────────────────────────────────────────

def _prefill_engine_deficit(
    layout: BulkheadLayout,
    assignments: Dict[int, Comp],
    p: ShipParameterization,
    floor_ratio: float = 1.0,
) -> int:
    """
    Pre-assign ENGINE_ROOM to eligible unassigned zones until an engine-volume
    floor is reached, expanding contiguously from existing ER zones.

    Rationale:
      Mandatory placement seeds one ER zone (lowest tier). The remaining
      ER band can be under-utilised before generic greedy scoring runs.
      This pass grows ER upward through adjacent ER-eligible zones until the
      engine budget floor is met, preserving contiguity at every step.

    Parameters
    ----------
    floor_ratio : float
        Fraction of engine budget target to satisfy in this prefill pass.
        1.0 = full target; 0.8 would leave margin for later balancing.

    Returns
    -------
    int
        Number of additional zones assigned to ENGINE_ROOM.
    """
    hull_volume = _hull_volume_voxels(layout)
    if hull_volume <= 0:
        return 0

    target_engine_vox = float(p.budget.get("engine", 0.0)) * hull_volume
    target_floor = max(0.0, min(1.0, floor_ratio)) * target_engine_vox
    if target_floor <= 0:
        return 0

    assigned_vox = _count_assigned_voxels(layout, assignments)
    current_engine = float(assigned_vox.get("engine", 0.0))
    if current_engine >= target_floor:
        return 0

    zone_map = layout.zone_by_id
    er_zids = {zid for zid, c in assignments.items() if c == Comp.ENGINE_ROOM}

    n_assigned = 0
    while current_engine < target_floor:
        # Collect unassigned ER-eligible zones adjacent to current ER cluster.
        frontier = [
            z for z in layout.zones
            if (z.zone_id not in assignments
                and Comp.ENGINE_ROOM in z.eligible_comps
                and any(_zones_share_face(z, zone_map[eid]) for eid in er_zids))
        ]
        if not frontier:
            break

        # Prefer higher zones first (expand upward from the low seed).
        # OSV/patrol ER centroid is forward in the machinery block — expand
        # forward; deep-cargo types expand aft.
        if p.ship_type in (ShipType.OSV, ShipType.PATROL):
            frontier.sort(
                key=lambda z: (-z.cz_norm, z.cx_norm, -z.available_cells),
            )
        else:
            frontier.sort(
                key=lambda z: (-z.cz_norm, 1.0 - z.cx_norm, -z.available_cells),
            )
        best = frontier[0]
        assignments[best.zone_id] = Comp.ENGINE_ROOM
        er_zids.add(best.zone_id)
        current_engine += best.available_cells
        n_assigned += 1

    return n_assigned


def _fill_ss_accommodation_to_budget(
    layout: BulkheadLayout,
    assignments: Dict[int, Comp],
    p: ShipParameterization,
) -> None:
    """
    Deep-cargo superstructure fill (replaces force-fill-all-accommodation).

    Fill ACCOMMODATION deck-by-deck from the main-deck level upward until the
    accommodation budget is met (rounded up to a whole deck); leave higher SS
    decks UNASSIGNED. They become VOID in the backfill step and are removed by
    the existing SS-VOID excision, so SS height ends up matching the
    accommodation budget instead of being fixed at n_decks. Always keeps at
    least one accommodation deck (QC requires an accommodation zone).
    """
    ss_zones = layout.zones_of_type(ZoneType.SUPERSTRUCTURE)
    if not ss_zones:
        return

    hull_volume = _hull_volume_voxels(layout)
    accom_target = float(p.budget.get("accommodation", 0.0)) * hull_volume

    decks: Dict[int, list] = {}
    for z in ss_zones:
        decks.setdefault(z.deck_idx, []).append(z)
    ordered = sorted(decks.keys())

    filled = 0.0
    for i, di in enumerate(ordered):
        if i == 0 or filled < accom_target:
            for z in decks[di]:
                assignments[z.zone_id] = Comp.ACCOMMODATION
                filled += z.available_cells
        else:
            break


def _prefill_accommodation_deficit(
    layout: BulkheadLayout,
    assignments: Dict[int, Comp],
    p: ShipParameterization,
    floor_ratio: float = 0.55,
) -> int:
    """
    Seed accommodation on MAIN_UPPER + SS before greedy machinery/cargo
    consume upper-deck volume (patrol real GAs: ~32% accom, nz_ss=1).
    """
    if p.ship_type not in (ShipType.YACHT, ShipType.OSV, ShipType.PATROL):
        return 0

    hull_volume = _hull_volume_voxels(layout)
    if hull_volume <= 0:
        return 0

    target = float(p.budget.get("accommodation", 0.0)) * hull_volume
    if target <= 0:
        return 0

    if p.ship_type == ShipType.PATROL:
        floor_ratio = 0.60

    target_floor = max(0.0, min(1.0, floor_ratio)) * target
    assigned_vox = _count_assigned_voxels(layout, assignments)
    current = float(assigned_vox.get("accommodation", 0.0))
    if current >= target_floor:
        return 0

    ss_zones = layout.zones_of_type(ZoneType.SUPERSTRUCTURE)
    ss_x0 = min((z.x0 for z in ss_zones), default=0)
    ss_x1 = max((z.x1 for z in ss_zones), default=0)

    def _sort_key(z: Zone) -> tuple:
        under_ss = bool(ss_zones and z.x0 < ss_x1 and ss_x0 < z.x1)
        if z.zone_type == ZoneType.SUPERSTRUCTURE:
            tier = 0
        elif under_ss:
            tier = 1
        else:
            tier = 2
        return (tier, -z.cz_norm, -z.available_cells)

    candidates = sorted(
        [
            z for z in layout.zones
            if z.zone_id not in assignments
            and Comp.ACCOMMODATION in z.eligible_comps
            and z.zone_type in (ZoneType.MAIN_UPPER, ZoneType.SUPERSTRUCTURE)
        ],
        key=_sort_key,
    )

    n_assigned = 0
    for z in candidates:
        if current >= target_floor:
            break
        pair_cells = sum(
            layout.zone_by_id[zid].available_cells
            for zid in _mirror_zone_ids(layout, z.zone_id)
            if zid not in assignments
        )
        if pair_cells <= 0:
            continue
        _set_assignment(layout, assignments, z.zone_id, Comp.ACCOMMODATION)
        current += pair_cells
        n_assigned += 1
    return n_assigned


def _topup_accommodation_from_void(
    layout: BulkheadLayout,
    assignments: Dict[int, Comp],
    p: ShipParameterization,
    tol_frac: float = 0.11,
) -> int:
    """
    Post-greedy: flip VOID → ACCOMMODATION on MAIN_UPPER / SS until the
    accommodation budget is within QC tolerance.

    Greedy + machinery/cargo competition systematically undershoots patrol
    accommodation (~24% achieved vs ~30% target) even though eligible
    capacity (~34%) is sufficient.  Generic void-reduction spreads flips
    across all deficits; this pass is accommodation-specific.
    """
    if p.ship_type not in (ShipType.YACHT, ShipType.OSV, ShipType.PATROL):
        return 0

    hull_volume = _hull_volume_voxels(layout)
    if hull_volume <= 0:
        return 0

    target = float(p.budget.get("accommodation", 0.0)) * hull_volume
    if target <= 0:
        return 0

    assigned_vox = _count_assigned_voxels(layout, assignments)
    current = float(assigned_vox.get("accommodation", 0.0))
    min_target = target - tol_frac * hull_volume
    if current >= min_target:
        return 0

    ss_zones = layout.zones_of_type(ZoneType.SUPERSTRUCTURE)
    ss_x0 = min((z.x0 for z in ss_zones), default=0)
    ss_x1 = max((z.x1 for z in ss_zones), default=0)

    def _sort_key(z: Zone) -> tuple:
        under_ss = bool(ss_zones and z.x0 < ss_x1 and ss_x0 < z.x1)
        if z.zone_type == ZoneType.SUPERSTRUCTURE:
            tier = 0
        elif under_ss and z.zone_type == ZoneType.MAIN_UPPER:
            tier = 1
        elif z.zone_type == ZoneType.MAIN_UPPER:
            tier = 2
        else:
            tier = 3
        return (tier, -z.cz_norm, -z.available_cells)

    candidates = sorted(
        [
            z for z in layout.zones
            if assignments.get(z.zone_id) == Comp.VOID
            and Comp.ACCOMMODATION in z.eligible_comps
            and z.zone_type in (ZoneType.MAIN_UPPER, ZoneType.SUPERSTRUCTURE)
        ],
        key=_sort_key,
    )

    n_changed = 0
    for z in candidates:
        if current >= min_target:
            break
        pair_cells = sum(
            layout.zone_by_id[zid].available_cells
            for zid in _mirror_zone_ids(layout, z.zone_id)
            if assignments.get(zid) == Comp.VOID
        )
        if pair_cells <= 0:
            continue
        _set_assignment(layout, assignments, z.zone_id, Comp.ACCOMMODATION)
        current += pair_cells
        n_changed += 1
    return n_changed


# Deep-cargo types with chronically low fuel after void reduction.
_DEEP_CARGO_TYPES = frozenset({ShipType.BULKER, ShipType.TANKER, ShipType.CARGO})
_BALLAST_VOID_PREFERENCE = 1.35
_FUEL_FAR_FROM_ER_FRAC = 0.10  # min fuel deficit (hull frac) to place fuel away from ER

# Fuel is budget-controlled, not filler-controlled.
_FUEL_TRIM_TOL_FRAC = 0.01
_VOID_TRIM_CAP_MARGIN = 0.02
_REPLACEMENT_OVERSHOOT_FRAC = 0.01
# Wing/DB void is a legitimate sink (cofferdams, void DB cells) — not interior slack.
_WING_VOID_ZONE_TYPES = frozenset({
    ZoneType.SIDE_DB, ZoneType.SIDE_LOWER, ZoneType.SIDE_UPPER, ZoneType.MAIN_DB,
})
_WING_VOID_ALLOWANCE_FRAC = 0.12


def _zone_abreast_er(z, er_zids: set, zone_map: dict) -> bool:
    """True if zone shares an x-band with ER at the same deck tier."""
    for er_zid in er_zids:
        er_z = zone_map[er_zid]
        if z.x0 < er_z.x1 and er_z.x0 < z.x1 and z.z0 == er_z.z0:
            return True
    return False


def _fuel_budget_allows_more_fuel(
    layout: BulkheadLayout,
    assignments: Dict[int, Comp],
    p: ShipParameterization,
    extra_vox: float = 0.0,
    *,
    hull_volume: int | None = None,
) -> bool:
    """Fuel gate: FUEL may not be assigned when at/above target + tolerance."""
    hv = hull_volume if hull_volume is not None else _hull_volume_voxels(layout)
    if hv <= 0:
        return False
    assigned = _count_assigned_voxels(layout, assignments)
    fuel_target = float(p.budget.get("fuel_tanks", 0.0)) * hv
    fuel_current = float(assigned.get("fuel_tanks", 0.0))
    cap = fuel_target + _FUEL_TRIM_TOL_FRAC * hv
    return fuel_current + extra_vox <= cap + 1e-9


def _void_fractions_by_region(
    layout: BulkheadLayout,
    assignments: Dict[int, Comp],
) -> Tuple[float, float, float]:
    """Return (interior_void_frac, wing_void_frac, total_void_frac) of hull."""
    hull_total = sum(z.available_cells for z in layout.zones)
    if hull_total <= 0:
        return 0.0, 0.0, 0.0
    interior = wing = 0
    for z in layout.zones:
        if assignments.get(z.zone_id) != Comp.VOID:
            continue
        if z.zone_type in _WING_VOID_ZONE_TYPES:
            wing += z.available_cells
        else:
            interior += z.available_cells
    return interior / hull_total, wing / hull_total, (interior + wing) / hull_total


def _reduce_void_by_deficit(
    layout: BulkheadLayout,
    assignments: Dict[int, Comp],
    p: ShipParameterization,
) -> None:
    """
    Replace VOID with a budgeted compartment when that key still has
    positive volume deficit. Ignores physics scoring. Skips ENGINE_ROOM
    (contiguity) and FUEL_TANKS above ER. Stops near sampled void target.
    """
    hull_volume = _hull_volume_voxels(layout)
    if hull_volume <= 0:
        return

    void_target_frac = float(p.budget.get("void", 0.0))
    tol = 0.012
    zone_map = layout.zone_by_id

    def _void_volume_frac() -> float:
        vox = sum(
            z.available_cells
            for z in layout.zones
            if assignments.get(z.zone_id) == Comp.VOID
        )
        return vox / hull_volume

    def _zones_above_er() -> set:
        er_zids = {zid for zid, c in assignments.items() if c == Comp.ENGINE_ROOM}
        out: set = set()
        for z in layout.zones:
            if z.zone_id in er_zids:
                continue
            for er_zid in er_zids:
                er_z = zone_map[er_zid]
                if _directly_above_er(z, er_z):
                    out.add(z.zone_id)
                    break
        return out

    for _ in range(12):
        if _void_volume_frac() <= void_target_frac + tol:
            break
        z_above = _zones_above_er()
        void_zones = [
            z for z in layout.zones
            if assignments.get(z.zone_id) == Comp.VOID
        ]
        void_zones.sort(
            key=lambda z: (
                int(assignments.get(z.zone_id) == Comp.VOID
                    and _fuel_touches_existing_fuel(layout, assignments, z)),
                z.available_cells,
            ),
            reverse=True,
        )
        assigned_vox = _count_assigned_voxels(layout, assignments)
        changed = False

        for z in void_zones:
            if _void_volume_frac() <= void_target_frac + tol:
                break
            best_comp: Optional[Comp] = None
            best_def = 0.0
            er_zids = {zid for zid, c in assignments.items() if c == Comp.ENGINE_ROOM}
            for comp in z.eligible_comps:
                if comp in (Comp.VOID, Comp.EMPTY, Comp.STEERING_GEAR, Comp.NAVIGATION, Comp.ENGINE_ROOM):
                    continue
                if comp == Comp.FUEL_TANKS and z.zone_id in z_above:
                    if z.zone_type not in (
                        ZoneType.SIDE_DB, ZoneType.SIDE_LOWER, ZoneType.SIDE_UPPER,
                    ):
                        continue
                bkey = COMP_TO_BUDGET.get(comp)
                if not bkey:
                    continue
                target = float(p.budget.get(bkey, 0.0)) * hull_volume
                cur = float(assigned_vox.get(bkey, 0.0))
                deficit = target - cur
                if comp == Comp.FUEL_TANKS:
                    if deficit <= 0:
                        continue
                    if not _fuel_budget_allows_more_fuel(
                        layout, assignments, p, z.available_cells,
                        hull_volume=hull_volume,
                    ):
                        continue
                    near_er = _zone_abreast_er(z, er_zids, zone_map)
                    if not near_er and deficit < _FUEL_FAR_FROM_ER_FRAC * hull_volume:
                        continue
                if comp == Comp.BALLAST_TANKS and deficit > 0 and z.zone_id not in z_above:
                    deficit *= _BALLAST_VOID_PREFERENCE
                if deficit > best_def:
                    best_def = deficit
                    best_comp = comp
            if best_comp is None or best_def <= 0:
                continue
            _set_assignment(layout, assignments, z.zone_id, best_comp)
            for zid in _mirror_zone_ids(layout, z.zone_id):
                zz = layout.zone_by_id[zid]
                bk = COMP_TO_BUDGET[best_comp]
                assigned_vox[bk] = assigned_vox.get(bk, 0.0) + zz.available_cells
            changed = True

        if not changed:
            break


# ─────────────────────────────────────────────────────────────────
# Step 2: Greedy budget-driven fill
# ─────────────────────────────────────────────────────────────────

def _greedy_fill(
    layout: BulkheadLayout,
    assignments: Dict[int, Comp],
    p: ShipParameterization,
    rng: np.random.Generator,
    noise: float = 0.10,
) -> None:
    """
    Zone-first greedy fill with physics-aware scoring.

    Iterates over unassigned zones from largest to smallest.  For each
    zone, scores every eligible comp type by:

        score = deficit_norm x fit_fraction x spatial_pref x physics_pref

    where fit_fraction penalises assigning a comp whose remaining deficit
    is much smaller than the zone (overshoot prevention), and physics_pref
    steers heavy compartments toward LCG/KG targets.

    ``hull_volume`` here is total assignable voxels (hull + superstructure),
    matching Stage 1 budget targets and `_compute_budget_errors`.
    """
    hull_volume = _hull_volume_voxels(layout)
    if hull_volume == 0:
        return

    vs3 = p.cell_volume

    # Current assigned volume per budget key
    assigned_vox = _count_assigned_voxels(layout, assignments)

    # Target voxels per budget key
    target_vox: Dict[str, float] = {}
    for bkey, frac in p.budget.items():
        if bkey == "void":
            continue
        target_vox[bkey] = frac * hull_volume

    # Running mass-weighted sums for physics tracking
    sum_mass = 0.0; sum_mx = 0.0; sum_mz = 0.0
    zone_map = layout.zone_by_id
    for zid, comp in assignments.items():
        z = zone_map[zid]
        d = density_for(comp, p.ship_type)
        m = d * z.available_cells * vs3
        sum_mass += m
        sum_mx += m * z.cx_norm * p.L
        sum_mz += m * z.cz_norm * p.nz_total * p.dz_m

    # ── Precompute fuel-above-ER exclusion ───
    # Identify all zone_ids that sit directly above any ENGINE_ROOM zone.
    # FUEL_TANKS will be hard-excluded from these zones to prevent the
    # recurring fire-safety violation.  This is computed ONCE from the
    # mandatory assignments (which have already placed ENGINE_ROOM) so
    # it doesn't change during the greedy pass.
    er_zone_ids = {zid for zid, c in assignments.items()
                   if c == Comp.ENGINE_ROOM}
    zones_above_er: set = set()
    for z in layout.zones:
        if z.zone_id in er_zone_ids:
            continue
        for er_zid in er_zone_ids:
            er_z = zone_map[er_zid]
            if _directly_above_er(z, er_z):
                zones_above_er.add(z.zone_id)
                break

    # Unassigned zones, sorted by available_cells descending with
    # stochastic perturbation for variant diversity.
    #
    # The zone processing order is the biggest lever for diversity:
    # assigning a large midship zone to CARGO vs FUEL first cascades
    # through all subsequent assignments.  Pure size-sorting is fully
    # deterministic, so variants with different seeds but the same
    # noise level produce identical orderings.
    #
    # Fix: add Gaussian noise to the sort key (proportional to noise
    # parameter).  Zones of very different sizes keep their relative
    # order, but zones of similar size get shuffled.  This creates
    # meaningful diversity without violating the "large zones first"
    # heuristic that prevents pathological fragmentation.
    unassigned = [z for z in layout.zones if z.zone_id not in assignments]


    if noise > 0 and len(unassigned) > 1:
        max_cells = max(z.available_cells for z in unassigned)
        if max_cells > 0:
            # Perturb sort keys: zones within ~noise*max_cells of each
            # other can swap positions.  E.g. noise=0.5, max=100 → zones
            # at 80 and 90 cells can interchange.
            sort_keys = {
                z.zone_id: -(z.available_cells
                             + rng.normal(0, noise * max_cells * 0.3))
                for z in unassigned
            }
            unassigned.sort(key=lambda z: sort_keys[z.zone_id])
        else:
            rng.shuffle(unassigned)
    else:
        unassigned.sort(key=lambda z: z.available_cells, reverse=True)

    for zone in unassigned:
        # Mirror twins are assigned atomically via _set_assignment; skip stbd
        # (or port) when the pair was already filled on the first twin.
        if zone.zone_id in assignments:
            continue

        best_comp = Comp.VOID
        best_score = -1.0

        # Running LCG/KG for physics steering
        if sum_mass > 0:
            running_lcg = (sum_mx / sum_mass) / p.L
            running_kg_m = sum_mz / sum_mass
        else:
            running_lcg = 0.5
            running_kg_m = p.D * 0.5

        lcg_offset = running_lcg - p.target_lcg_frac
        kg_offset = running_kg_m - p.target_kg_frac * p.D
        cx = zone.cx_norm
        cz_m = zone.cz_norm * p.nz_total * p.dz_m
        # Mirror pairs are assigned atomically, so the volume consumed
        # by any assignment is the PAIR total, not the representative twin.
        pair_cells = sum(
            layout.zone_by_id[zid].available_cells
            for zid in _mirror_zone_ids(layout, zone.zone_id)
        )

        for comp in zone.eligible_comps:
            if comp in (Comp.VOID, Comp.EMPTY, Comp.STEERING_GEAR, Comp.NAVIGATION):
                continue

            # Hard constraint: no FUEL directly above ENGINE_ROOM (centerline only).
            if comp == Comp.FUEL_TANKS and zone.zone_id in zones_above_er:
                if zone.zone_type not in (
                    ZoneType.SIDE_DB, ZoneType.SIDE_LOWER, ZoneType.SIDE_UPPER,
                ):
                    continue

            # Fuel budget gate: no new fuel at/above target.
            if comp == Comp.FUEL_TANKS and not _fuel_budget_allows_more_fuel(
                layout, assignments, p, pair_cells, hull_volume=hull_volume,
            ):
                continue

            bkey = COMP_TO_BUDGET.get(comp)
            if bkey is None:
                continue

            target = target_vox.get(bkey, 0)
            current = assigned_vox.get(bkey, 0)
            deficit = target - current

            if deficit <= 0:
                continue

            # Overshoot-aware deficit: penalise if zone >> deficit
            # (pair_cells computed once per zone above).
            fit_frac = min(deficit, pair_cells) / max(pair_cells, 1)
            # Soften: allow moderate overshoot (fit > 0.3) without
            # killing the score entirely
            fit_frac = max(fit_frac, 0.25)
            deficit_norm = (deficit / hull_volume) * fit_frac
            if comp == Comp.ACCOMMODATION and p.ship_type == ShipType.PATROL:
                deficit_norm *= 2.0

            # Spatial preference
            pref = _spatial_preference(zone, comp, p)

            # Physics preference — LCG/KG-aware for ALL mass-carrying comps.
            #
            # Three components:
            #   (a) Positive nudge: heavy comps get a bonus for zones that
            #       would move LCG/KG *toward* the target.
            #   (b) Anti-worsening penalty: ANY comp that would push LCG/KG
            #       further from target gets a multiplicative penalty.
            #   (c) Strength scales with how far off-target we are.
            #
            # This is the single biggest lever for LCG pass rate — the
            # reviewer correctly identified that limiting physics steering
            # to density > 0.8 leaves most of the search space unguided.
            density = density_for(comp, p.ship_type)
            physics_pref = 1.0

            if density > 0.01 and abs(lcg_offset) > 0.01:
                # How far off-target (0 = on target, 1 = badly off)
                lcg_urgency = min(1.0, abs(lcg_offset) / 0.08)

                # Would this placement help or hurt LCG?
                # lcg_offset > 0 means running LCG is too far forward → want
                # heavy stuff aft (low cx) and light stuff forward (high cx).
                # lcg_offset < 0 means running LCG is too far aft → want
                # heavy stuff forward (high cx).
                if lcg_offset > 0:
                    # LCG too fwd — heavy stuff should go aft
                    lcg_alignment = (1.0 - cx)  # 1.0 at aft, 0.0 at bow
                else:
                    # LCG too aft — heavy stuff should go fwd
                    lcg_alignment = cx          # 1.0 at bow, 0.0 at aft

                # Positive nudge: proportional to density (heavy comps steer
                # more) and how aligned this zone is with the needed direction
                nudge_strength = 0.4 * density * lcg_urgency
                physics_pref += nudge_strength * lcg_alignment

                # Anti-worsening penalty: if this comp is heavy AND the zone
                # is in the wrong direction, apply a multiplicative penalty.
                # Light comps get a milder penalty.
                if lcg_alignment < 0.3 and density > 0.5:
                    # Heavy comp in a zone that worsens LCG
                    penalty = 0.5 + 0.5 * (1.0 - lcg_urgency)
                    physics_pref *= penalty

            # KG steering (lighter touch — KG is less frequently the
            # dominant failure and BM provides a natural restoring term)
            if density > 0.01 and abs(kg_offset) > 0.5:
                kg_urgency = min(1.0, abs(kg_offset) / (0.15 * max(p.D, 1)))
                if kg_offset > 0:
                    # KG too high — prefer low-z for heavy comps
                    kg_alignment = 1.0 - cz_m / max(p.D, 1)
                else:
                    # KG too low — prefer high-z for heavy comps
                    kg_alignment = cz_m / max(p.D, 1)
                physics_pref += 0.2 * density * kg_urgency * kg_alignment

            jitter = 1.0 + rng.uniform(-noise, noise)
            score = deficit_norm * pref * physics_pref * jitter
            if comp == Comp.FUEL_TANKS and _fuel_touches_existing_fuel(
                layout, assignments, zone,
            ):
                score *= 1.35

            if score > best_score:
                best_score = score
                best_comp = comp

        _set_assignment(layout, assignments, zone.zone_id, best_comp)

        # Update tracking (include mirror twins)
        for zid in _mirror_zone_ids(layout, zone.zone_id):
            zz = layout.zone_by_id[zid]
            if best_comp in BUDGETED_COMPS:
                bkey = COMP_TO_BUDGET[best_comp]
                assigned_vox[bkey] = assigned_vox.get(bkey, 0.0) + zz.available_cells
            d = density_for(best_comp, p.ship_type)
            m = d * zz.available_cells * vs3
            sum_mass += m
            sum_mx += m * zz.cx_norm * p.L
            sum_mz += m * zz.cz_norm * p.nz_total * p.dz_m

        # If we just assigned ENGINE_ROOM, dynamically extend the
        # fuel-above-ER exclusion set to cover zones above this new ER zone.
        if best_comp == Comp.ENGINE_ROOM:
            for other_z in layout.zones:
                if other_z.zone_id in zones_above_er:
                    continue
                if _directly_above_er(other_z, zone):
                    zones_above_er.add(other_z.zone_id)


# ─────────────────────────────────────────────────────────────────
# Spatial preference scoring
# ─────────────────────────────────────────────────────────────────

def _spatial_preference(
    zone: Zone,
    comp: Comp,
    p: ShipParameterization,
) -> float:
    """
    Return a preference multiplier in [0.1, 2.0] for placing comp in zone.

    Encodes soft spatial rules from naval architecture practice:
    - FUEL_TANKS: prefer aft (near ER), low decks
    - BALLAST_TANKS: prefer double-bottom tier, low decks; penalise high cz in
      FWD_PEAK; mild aft bias in main hull
    - STORES: prefer midship area
    - MACHINERY: prefer aft half (near ER)
    - ACCOMMODATION: prefer upper decks
    - CARGO: OSV/PATROL — aft + upper decks; deep-cargo — MAIN_LOWER / MAIN_DB;
      else mild low-deck preference
    """
    cx = zone.cx_norm
    cz = zone.cz_norm

    if comp == Comp.FUEL_TANKS:
        # Aft clustering (cx < ~0.55); stronger on wing tanks.  Low-z bias
        # is gentle on centre zones but stronger on SIDE_DB / SIDE_LOWER so
        # fuel clusters aft/near ER without hard-bans (controllability).
        fuel_max_cx = SPATIAL_RULES.get("fuel_tanks_max_cx", 0.55)
        on_side = zone.zone_type in (
            ZoneType.SIDE_DB, ZoneType.SIDE_LOWER, ZoneType.SIDE_UPPER,
        )
        if cx <= fuel_max_cx:
            aft_gain = 0.75 if on_side else 0.5
            x_pref = 1.0 + aft_gain * (1.0 - cx / fuel_max_cx)
        else:
            x_pref = max(0.15 if on_side else 0.2,
                         1.0 - (cx - fuel_max_cx) / (1.0 - fuel_max_cx))
        if on_side and zone.tier_role in (TierRole.DB, TierRole.LOWER):
            z_pref = 1.0 + 0.35 * (1.0 - cz)
        else:
            z_pref = 1.0 + 0.1 * (1.0 - cz)
        return float(np.clip(x_pref * z_pref / 1.1, 0.1, 2.0))

    if comp == Comp.BALLAST_TANKS:
        # Strongly prefer DB tier, then lower holds
        if zone.tier_role == TierRole.DB:
            base = 1.8
        elif zone.tier_role == TierRole.LOWER:
            base = 1.3
        else:
            base = float(max(0.3, 1.0 - 0.4 * cz))
        if zone.zone_type == ZoneType.FWD_PEAK:
            base *= max(0.35, 1.0 - 0.55 * cz)
        elif zone.zone_type not in (ZoneType.AFT_PEAK, ZoneType.FWD_PEAK):
            base *= 1.0 + 0.2 * (1.0 - cx)
        return float(np.clip(base, 0.1, 2.0))

    if comp == Comp.STORES:
        # Stores SERVE the accommodation/SS (aft block),
        # preferably upper decks — not midship/forward (visual inspection
        # showed strangely forward side stores).
        aft_factor = max(0.0, 1.0 - max(0.0, cx - 0.40) / 0.40)
        z_factor = 0.8 + 0.4 * cz
        return float(np.clip((0.4 + 1.2 * aft_factor) * z_factor / 1.1,
                             0.1, 2.0))

    if comp == Comp.CARGO:
        # Aft + upper bias only for working ships (incl. ENGINE_UPPER there).
        if p.ship_type in (ShipType.OSV, ShipType.PATROL):
            x_pref = 1.0 + 0.6 * (1.0 - cx)
            z_pref = 1.0 + 0.4 * cz
            return float(np.clip(x_pref * z_pref / 1.3, 0.8, 2.0))

        # Cargo holds should span full height in the main body.
        # On cargo-carrying ship types, strongly prefer MAIN_LOWER
        # (deck_idx=1) so cargo fills from inner bottom upward,
        # not just the upper tier.  Without this, ballast/fuel
        # affinity for low tiers pushes cargo upward, creating
        # inverted arrangements where tanks sit under cargo.
        base = 1.0 + 0.2 * (1.0 - cz)
        if p.ship_type in (ShipType.BULKER, ShipType.TANKER, ShipType.CARGO):
            if zone.tier_role == TierRole.LOWER:   # MAIN_LOWER
                base *= 1.6          # strong preference for lower holds
            elif zone.tier_role == TierRole.DB: # MAIN_DB
                base *= 1.3          # moderate preference for DB cargo
        return float(np.clip(base, 0.8, 2.0))

    if comp == Comp.MACHINERY:
        # Cluster around the ER block; peak cx type-dependent (yacht ER seed ~0.40).
        if p.ship_type == ShipType.YACHT:
            er_peak = 0.40
        else:
            er_peak = 0.20
        prox = max(0.0, 1.0 - abs(cx - er_peak) / 0.30)
        return float(np.clip(0.6 + 1.4 * prox, 0.2, 2.0))

    if comp == Comp.ACCOMMODATION:
        # Type-dependent vertical preference.
        #
        # Real yacht/OSV/patrol ships have accommodation on the main
        # deck (MAIN_UPPER, cz ≈ 0.4-0.5) as well as the superstructure.
        # Deep-cargo ships (bulker/tanker/cargo) have accommodation
        # almost exclusively in the superstructure.
        #
        # The preference curve controls how strongly the greedy scorer
        # pushes accommodation upward. Flatter = more accommodation in
        # hull zones; steeper = more in SS.

        if p.ship_type in (ShipType.YACHT, ShipType.OSV, ShipType.PATROL):
            # Working types: accommodation on MAIN_UPPER + SS (real patrols
            # encode nz_ss=1 but ~32% accom — most volume is main deck).
            # Yacht: main-deck-first — prefer MAIN_UPPER / low SS
            # over high SS so KG/D does not climb with unused upper decks.
            if zone.zone_type == ZoneType.MAIN_UPPER:
                pref = 2.2 if p.ship_type == ShipType.YACHT else 2.0
            elif zone.zone_type == ZoneType.SUPERSTRUCTURE:
                if p.ship_type == ShipType.YACHT:
                    # Logical SS deck 0 is lowest above main deck.
                    ss_level = max(0, int(getattr(zone, "deck_idx", 3)) - 3)
                    pref = 1.55 - 0.35 * ss_level
                else:
                    pref = 1.5
            elif zone.zone_type == ZoneType.ENGINE_UPPER and p.ship_type == ShipType.YACHT:
                # ER deckhouse on yachts (no cargo in ENGINE_UPPER eligibility)
                pref = 2.0
            else:
                pref = 0.5 + 0.5 * cz
        else:
            # Steeper ramp for deep-cargo: accommodation stays in SS.
            # score ≈ 0.6 at cz=0.3, ≈ 0.9 at cz=0.5, ≈ 1.4 at cz=0.9
            pref = 0.4 + 1.0 * cz
            if zone.zone_type == ZoneType.ENGINE_UPPER:
                # Hotel / acc over engine on transport types (no cargo there)
                pref = max(pref, 1.25 + 0.6 * cz)

        return float(np.clip(pref, 0.3, 2.0))

    if comp == Comp.ENGINE_ROOM:
        if p.ship_type in (ShipType.YACHT, ShipType.OSV, ShipType.PATROL):
            return float(np.clip(0.5 + 1.2 * cx, 0.1, 2.0))
        return float(np.clip(1.5 * (1.0 - cx), 0.1, 2.0))

    # Default (VOID, etc.)
    return 0.5


# ─────────────────────────────────────────────────────────────────
# Step 3: Contiguity repair
# ─────────────────────────────────────────────────────────────────

def _repair_contiguity(
    layout: BulkheadLayout,
    assignments: Dict[int, Comp],
) -> None:
    """
    Ensure ENGINE_ROOM zones form a single connected component.

    Two zones are considered connected if they share a face:
      - Adjacent in x (x1_A == x0_B) with overlapping z range
      - Adjacent in z (z1_A == z0_B) with overlapping x range

    If ENGINE_ROOM zones form multiple components, the smaller
    component(s) are reassigned to MACHINERY.
    """
    er_zone_ids = [zid for zid, c in assignments.items()
                   if c == Comp.ENGINE_ROOM]
    if len(er_zone_ids) <= 1:
        return

    zone_map = layout.zone_by_id
    er_zones = [zone_map[zid] for zid in er_zone_ids]

    # Build adjacency among ER zones
    adj: Dict[int, List[int]] = defaultdict(list)
    for i, za in enumerate(er_zones):
        for j, zb in enumerate(er_zones):
            if i >= j:
                continue
            if _zones_share_face(za, zb):
                adj[za.zone_id].append(zb.zone_id)
                adj[zb.zone_id].append(za.zone_id)

    # BFS to find connected components
    visited = set()
    components: List[List[int]] = []
    for zid in er_zone_ids:
        if zid in visited:
            continue
        component = []
        queue = [zid]
        while queue:
            cur = queue.pop(0)
            if cur in visited:
                continue
            visited.add(cur)
            component.append(cur)
            for nb in adj.get(cur, []):
                if nb not in visited:
                    queue.append(nb)
        components.append(component)

    if len(components) <= 1:
        return

    # Keep largest component as ENGINE_ROOM, reassign others to MACHINERY
    components.sort(key=lambda c: sum(zone_map[zid].available_cells
                                      for zid in c), reverse=True)
    for comp in components[1:]:
        for zid in comp:
            assignments[zid] = Comp.MACHINERY


def _zones_share_face(a: Zone, b: Zone) -> bool:
    """Check if two zones share a face (6-connected box adjacency)."""
    x_overlap = a.x0 < b.x1 and b.x0 < a.x1
    y_overlap = a.y0 < b.y1 and b.y0 < a.y1
    z_overlap = a.z0 < b.z1 and b.z0 < a.z1

    x_adjacent = (a.x1 == b.x0 or b.x1 == a.x0) and y_overlap and z_overlap
    y_adjacent = (a.y1 == b.y0 or b.y1 == a.y0) and x_overlap and z_overlap
    z_adjacent = (a.z1 == b.z0 or b.z1 == a.z0) and x_overlap and y_overlap

    return x_adjacent or y_adjacent or z_adjacent


def _fuel_touches_existing_fuel(
    layout: BulkheadLayout,
    assignments: Dict[int, Comp],
    z: Zone,
) -> bool:
    """True if zone shares a face with any zone already labelled FUEL."""
    zone_map = layout.zone_by_id
    for zid, comp in assignments.items():
        if comp != Comp.FUEL_TANKS:
            continue
        if _zones_share_face(z, zone_map[zid]):
            return True
    return False


def _any_fuel_assigned(assignments: Dict[int, Comp]) -> bool:
    return any(c == Comp.FUEL_TANKS for c in assignments.values())


def _collapse_satellite_fuel_zones(
    layout: BulkheadLayout, assignments: Dict[int, Comp],
) -> int:
    return 0


def _mirror_zone_ids(layout: BulkheadLayout, zone_id: int) -> List[int]:
    """Return all zone ids in the same mirror pair (port/stbd)."""
    z = layout.zone_by_id[zone_id]
    if z.mirror_id < 0:
        return [zone_id]
    return [
        zz.zone_id for zz in layout.zones if zz.mirror_id == z.mirror_id
    ]


def _set_assignment(
    layout: BulkheadLayout,
    assignments: Dict[int, Comp],
    zone_id: int,
    comp: Comp,
) -> None:
    """Assign comp to zone and mirror twin(s) if any."""
    for zid in _mirror_zone_ids(layout, zone_id):
        assignments[zid] = comp


def _capacity_prefilter(
    layout: BulkheadLayout,
    p: ShipParameterization,
    tol: float = 0.05,
) -> Optional[str]:
    """
    Necessary-condition budget feasibility: sum of eligible zone volumes
    must cover each budget target.
    """
    capacity: Dict[Comp, float] = {c: 0.0 for c in BUDGETED_COMPS}
    total_assignable = 0.0
    for z in layout.zones:
        vol = float(z.available_cells)
        total_assignable += vol
        for comp in z.eligible_comps:
            if comp in BUDGETED_COMPS:
                capacity[comp] += vol

    if total_assignable <= 0:
        return "zero assignable hull volume"

    for bkey, frac in p.budget.items():
        comp = BUDGET_TO_COMP.get(bkey)
        if comp is None:
            continue
        target = frac
        cap = capacity.get(comp, 0.0) / total_assignable
        if target > cap + tol:
            return (
                f"budget '{bkey}' target {target:.3f} > eligible capacity "
                f"{cap:.3f}+{tol}"
            )
    return None


_VBUDGET_TO_KEY: Dict[str, str] = {
    "v_engine": "engine",
    "v_machinery": "machinery",
    "v_cargo": "cargo",
    "v_stores": "stores",
    "v_accommodation": "accommodation",
    "v_fuel_tanks": "fuel_tanks",
    "v_ballast_tanks": "ballast_tanks",
}


# Max fuel inflate above sampled during Stage-3.5 waterfill (fraction).
_DEEP_CARGO_FUEL_WATERFILL_TOL = 0.01


def _rescale_redistribute_keys(ship_type: ShipType) -> Tuple[str, ...]:
    """Keys allowed to absorb Stage-3.5 clamp slack.

    Deep-cargo: cargo (+ engine for bulker) only — never accommodation,
    machinery, stores, or fuel.
    Working / slender: keep full key set so OSV/patrol/yacht can still
    route slack into accommodation per WATERFILL_PRIORITY intent.
    """
    if ship_type in _DEEP_CARGO_TYPES:
        keys: List[str] = []
        for vkey in WATERFILL_PRIORITY.get(ship_type, ("v_cargo",)):
            bkey = _VBUDGET_TO_KEY.get(vkey)
            # Fuel must not absorb cargo-clamp residual (kills fuel
            # responsiveness and cargo capacity headroom).
            if bkey is not None and bkey != "fuel_tanks":
                keys.append(bkey)
        return tuple(keys) if keys else ("cargo",)
    return tuple(BUDGET_TO_COMP.keys())


def _deep_cargo_accom_effective_cap(p: ShipParameterization) -> float:
    """
    Cap deep-cargo effective accommodation at the sampled/configured band.

    Prevents Stage-3.5 from using accommodation as a sink for lost cargo
    capacity (reviewer: B_eff[accom] <= max(sampled, configured_max)).
    """
    cfg = VOLUME_BUDGETS[p.ship_type]
    configured_hi = float(cfg["v_accommodation"][1])
    sampled = float((getattr(p, "budget_sampled", None) or {}).get(
        "accommodation", p.budget.get("accommodation", 0.0)
    ))
    return max(sampled, configured_hi)


def _deep_cargo_fuel_effective_cap(p: ShipParameterization) -> float:
    """
    Cap deep-cargo effective fuel near the sampled request.

    Stage-3.5 was waterfilling cargo-clamp slack into fuel (sampled ~5–6%
    → effective ~8%), which destroyed fuel responsiveness and crowded cargo.
    Allow a small zone-granularity tolerance only.
    """
    sampled = float((getattr(p, "budget_sampled", None) or {}).get(
        "fuel_tanks", p.budget.get("fuel_tanks", 0.0)
    ))
    return sampled + _DEEP_CARGO_FUEL_WATERFILL_TOL


def rescale_budget_to_capacity(
    layout: BulkheadLayout,
    p: ShipParameterization,
    margin: float = 0.02,
    max_void: float = 0.08,
) -> Dict[str, float]:
    """
    Stage 3.5 — project the sampled Stage-1 budget onto the feasible set of
    the realised Stage-3 geometry (per-key eligible-capacity clamp).

    The sampled budget is a fraction of TOTAL assignable volume, but carving
    and eligibility bound what each comp can actually occupy (e.g. bulker
    cargo cap ~0.35-0.45 of total). This projects each key to
    ``min(target, capacity - margin)`` and redistributes the freed fraction
    to keys with remaining headroom, proportional to headroom — VOID absorbs
    only what no key can take, capped at ``max_void`` against the
    VOLUME_BUDGETS range maxima being exceeded.

    Deep-cargo: accommodation and fuel are excluded from
    redistribution. Accommodation is hard-capped at
    ``max(sampled, configured_max)``; fuel at ``sampled + 0.01`` so residual
    cargo-clamp slack becomes VOID (or cargo/engine) rather than hotel or
    fuel inflation.

    Mutates ``p.budget`` IN PLACE to the effective budget (this is what
    Stage 4, QC, and the graph `cond` vector consume) and stores the original
    draw in ``p.budget_sampled``. Returns the effective budget.

    Idempotent: a second call with the same layout is a no-op (targets
    already within capacity).
    """
    hull_volume = _hull_volume_voxels(layout)
    if hull_volume <= 0:
        return p.budget

    if getattr(p, "budget_sampled", None) is None:
        p.budget_sampled = dict(p.budget)

    caps: Dict[str, float] = {}
    for bkey, comp in BUDGET_TO_COMP.items():
        cap = sum(z.available_cells for z in layout.zones
                  if comp in z.eligible_comps) / hull_volume
        caps[bkey] = max(0.0, cap - margin)

    # 1. Clamp over-capacity keys; collect the freed fraction.
    freed = 0.0
    for bkey in caps:
        tgt = float(p.budget.get(bkey, 0.0))
        if tgt > caps[bkey]:
            freed += tgt - caps[bkey]
            p.budget[bkey] = caps[bkey]

    # 2. Redistribute proportionally to remaining headroom (capacity-bounded),
    #    a few waterfilling passes. VOID only absorbs the remainder.
    #    Deep-cargo: cargo (+ engine); never accom / machinery / stores / fuel.
    redistribute_keys = _rescale_redistribute_keys(p.ship_type)
    cfg = VOLUME_BUDGETS[p.ship_type]
    deep = p.ship_type in _DEEP_CARGO_TYPES
    range_max: Dict[str, float] = {}
    for vkey, bkey in _VBUDGET_TO_KEY.items():
        hi = float(cfg[vkey][1])
        # Working/slender keep the historical 1.4× accom headroom for
        # waterfill; deep-cargo must not inflate beyond configured max.
        if bkey == "accommodation" and not deep:
            hi *= 1.4
        range_max[bkey] = hi
    for _ in range(3):
        if freed <= 1e-9:
            break
        headroom = {
            bkey: max(0.0, min(caps[bkey], range_max.get(bkey, caps[bkey]))
                       - float(p.budget.get(bkey, 0.0)))
            for bkey in redistribute_keys if bkey in caps
        }
        room_total = sum(headroom.values())
        if room_total <= 1e-9:
            break
        take = min(freed, room_total)
        for bkey, room in headroom.items():
            p.budget[bkey] = float(p.budget.get(bkey, 0.0)) + take * room / room_total
        freed -= take

    # 3. Deep-cargo accom hard cap — any excess returns to VOID.
    if deep and "accommodation" in p.budget:
        accom_cap = _deep_cargo_accom_effective_cap(p)
        accom_now = float(p.budget.get("accommodation", 0.0))
        if accom_now > accom_cap + 1e-12:
            freed += accom_now - accom_cap
            p.budget["accommodation"] = accom_cap

    # 4. Deep-cargo fuel hard cap: sampled + small tol only.
    if deep and "fuel_tanks" in p.budget:
        fuel_cap = _deep_cargo_fuel_effective_cap(p)
        fuel_now = float(p.budget.get("fuel_tanks", 0.0))
        if fuel_now > fuel_cap + 1e-12:
            freed += fuel_now - fuel_cap
            p.budget["fuel_tanks"] = fuel_cap

    p.budget["void"] = float(p.budget.get("void", 0.0)) + max(0.0, freed)
    if p.budget["void"] > max_void:
        # Geometry simply cannot absorb this much — leave as-is; the
        # capacity prefilter / QC will surface it rather than hide it.
        pass
    return p.budget


def _adjacent_centre_zone(
    layout: BulkheadLayout,
    side_zone: Zone,
) -> Optional[Zone]:
    """Find the centre zone abreast a side zone (x-overlap, same z-tier)."""
    candidates: List[Zone] = []
    for z in layout.zones:
        if z.side != "centre":
            continue
        if z.z0 != side_zone.z0 or z.z1 != side_zone.z1:
            continue
        if z.x0 < side_zone.x1 and side_zone.x0 < z.x1:
            candidates.append(z)
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda z: min(z.x1, side_zone.x1) - max(z.x0, side_zone.x0),
    )


def _side_fuel_abreast_accommodation(
    layout: BulkheadLayout,
    assignments: Dict[int, Comp],
    z: Zone,
) -> bool:
    """True if SIDE_* fuel is abreast an accommodation centre (Check 13)."""
    if z.zone_type not in (
        ZoneType.SIDE_DB, ZoneType.SIDE_LOWER, ZoneType.SIDE_UPPER,
    ):
        return False
    if assignments.get(z.zone_id) != Comp.FUEL_TANKS:
        return False
    centre = _adjacent_centre_zone(layout, z)
    if centre is None:
        return False
    cc = assignments.get(centre.zone_id)
    return cc in (Comp.ACCOMMODATION, Comp.NAVIGATION)


def _side_zone_fill_policy(
    layout: BulkheadLayout,
    assignments: Dict[int, Comp],
    p: Optional[ShipParameterization] = None,
) -> int:
    """
    Domain-prior fill for SIDE_* zones that the greedy pass left VOID.

    ``_greedy_fill`` assigns EVERY zone (VOID included), so this pass targets
    side zones currently VOID and applies the documented priors with budget
    awareness:

      - SIDE_DB / SIDE_LOWER: FUEL abreast the engine block (if fuel deficit),
        else BALLAST (wing voids are physically ballast; mild overshoot up to
        2% of total volume is tolerated to avoid unrealistic VOID wings).
      - SIDE_UPPER: eligible comp with the largest positive deficit,
        preferring contrast with the abreast centre zone; left VOID if no
        budget key has deficit.

    Returns the number of zones changed (mirror twins counted).
    """
    if p is None:
        p = layout.params
    hull_volume = _hull_volume_voxels(layout)
    if hull_volume <= 0:
        return 0
    assigned_vox = _count_assigned_voxels(layout, assignments)

    def _deficit(comp: Comp) -> float:
        bkey = COMP_TO_BUDGET.get(comp)
        if bkey is None:
            return 0.0
        return (float(p.budget.get(bkey, 0.0)) * hull_volume
                - float(assigned_vox.get(bkey, 0.0)))

    n_changed = 0
    handled: set = set()
    for z in layout.zones:
        if z.zone_id in handled:
            continue
        if z.zone_type not in (
            ZoneType.SIDE_DB, ZoneType.SIDE_LOWER, ZoneType.SIDE_UPPER,
        ):
            continue
        if assignments.get(z.zone_id) != Comp.VOID:
            continue

        pair_ids = _mirror_zone_ids(layout, z.zone_id)
        handled.update(pair_ids)
        pair_vox = sum(layout.zone_by_id[zid].available_cells
                       for zid in pair_ids)

        centre = _adjacent_centre_zone(layout, z)
        centre_type = centre.zone_type if centre else ZoneType.MAIN_LOWER

        comp: Optional[Comp] = None
        overshoot_allow = 0.02 * hull_volume
        pair_vox_f = float(pair_vox)
        if z.zone_type in (ZoneType.SIDE_DB, ZoneType.SIDE_LOWER):
            # Wing ladder: ballast to target → fuel on SIDE_DB only → void sink.
            wing_fuel_tier = z.zone_type == ZoneType.SIDE_DB
            if (Comp.BALLAST_TANKS in z.eligible_comps
                    and _deficit(Comp.BALLAST_TANKS) > 0):
                comp = Comp.BALLAST_TANKS
            elif (wing_fuel_tier
                    and centre_type == ZoneType.ENGINE_REGION
                    and Comp.FUEL_TANKS in z.eligible_comps
                    and _deficit(Comp.FUEL_TANKS) > 0
                    and _fuel_budget_allows_more_fuel(
                        layout, assignments, p, pair_vox_f,
                        hull_volume=hull_volume,
                    )
                    and (
                        not _any_fuel_assigned(assignments)
                        or _fuel_touches_existing_fuel(layout, assignments, z)
                    )):
                comp = Comp.FUEL_TANKS
            elif Comp.VOID in z.eligible_comps:
                comp = Comp.VOID
            else:
                cands = [
                    c for c in z.eligible_comps
                    if c not in (Comp.VOID, Comp.EMPTY, Comp.ENGINE_ROOM,
                                 Comp.STEERING_GEAR, Comp.NAVIGATION,
                                 Comp.FUEL_TANKS)
                    and _deficit(c) > -overshoot_allow
                ]
                if cands:
                    comp = max(cands, key=_deficit)
        else:
            centre_comp = assignments.get(centre.zone_id) if centre else None
            cands = [
                c for c in z.eligible_comps
                if c not in (Comp.VOID, Comp.EMPTY,
                             Comp.ENGINE_ROOM, Comp.STEERING_GEAR,
                             Comp.NAVIGATION)
                and _deficit(c) > -overshoot_allow
                and not (
                    c == Comp.FUEL_TANKS
                    and not _fuel_budget_allows_more_fuel(
                        layout, assignments, p, pair_vox_f,
                        hull_volume=hull_volume,
                    )
                )
            ]
            if cands:
                cands.sort(
                    key=lambda c: (_deficit(c) > 0, c != centre_comp,
                                   _deficit(c)),
                    reverse=True,
                )
                comp = cands[0]
            elif Comp.VOID in z.eligible_comps:
                comp = Comp.VOID

        if comp is None:
            continue
        _set_assignment(layout, assignments, z.zone_id, comp)
        bkey = COMP_TO_BUDGET.get(comp)
        if bkey is not None:
            assigned_vox[bkey] = assigned_vox.get(bkey, 0.0) + pair_vox
        n_changed += len(pair_ids)

    return n_changed


# ─────────────────────────────────────────────────────────────────
# Step 5: Budget rebalancing pass
# ─────────────────────────────────────────────────────────────────

def _rebalance_budgets(
    layout: BulkheadLayout,
    assignments: Dict[int, Comp],
    p: ShipParameterization,
    max_passes: int = 3,
) -> int:
    """
    Targeted single-zone reassignments to reduce worst budget errors.

    After the greedy fill, some comp types may be over-assigned (e.g.
    CARGO got too many large zones) while others are under-assigned
    (e.g. STORES never won a zone).  This pass finds the worst
    overshoot/undershoot pair and tries to fix them.

    Algorithm per pass:
      1. Compute current budget fractions and find the worst overshoot
         (comp with largest positive error) and undershoot (largest
         negative error, excluding VOID).
      2. Scan all zones assigned to the overshoot comp.
      3. For each such zone, check if the undershoot comp is eligible.
      4. Among eligible candidates, pick the one whose reassignment
         produces the best combined budget improvement.
      5. Accept the reassignment only if total budget error improves.

    Safeguards:
      - Never reassigns ENGINE_ROOM, STEERING_GEAR, or SS ACCOMMODATION.
      - Never violates zone eligibility.
      - Stops after max_passes or when no improvement is found.

    Returns the number of accepted reassignments.
    """
    hull_volume = _hull_volume_voxels(layout)
    if hull_volume == 0:
        return 0

    zone_map = layout.zone_by_id
    n_reassigned = 0

    # Comps that cannot be reassigned (structural mandatories)
    PROTECTED = {Comp.ENGINE_ROOM, Comp.STEERING_GEAR, Comp.NAVIGATION}

    # Precompute zones above ER for fuel-above-ER guard
    er_zids = {zid for zid, c in assignments.items()
               if c == Comp.ENGINE_ROOM}
    rebal_zones_above_er: set = set()
    for z in layout.zones:
        for er_zid in er_zids:
            er_z = zone_map[er_zid]
            if _directly_above_er(z, er_z):
                rebal_zones_above_er.add(z.zone_id)
                break

    for _ in range(max_passes):
        # Current budget state
        assigned_vox = _count_assigned_voxels(layout, assignments)

        # Compute signed errors: positive = overshoot, negative = undershoot
        signed_errors: Dict[str, float] = {}
        for bkey, target_frac in p.budget.items():
            if bkey == "void":
                continue
            actual_frac = assigned_vox.get(bkey, 0) / hull_volume
            signed_errors[bkey] = actual_frac - target_frac

        # Find worst overshoot and undershoot
        worst_over_key = max(signed_errors, key=lambda k: signed_errors[k])
        worst_under_key = min(signed_errors, key=lambda k: signed_errors[k])

        worst_over = signed_errors[worst_over_key]
        worst_under = signed_errors[worst_under_key]

        # Only proceed if there's meaningful imbalance
        if worst_over < 0.02 or worst_under > -0.02:
            break

        over_comp = BUDGET_TO_COMP.get(worst_over_key)
        under_comp = BUDGET_TO_COMP.get(worst_under_key)
        if over_comp is None or under_comp is None:
            break

        # Don't touch structural mandatories
        if over_comp in PROTECTED:
            break

        # Find candidate zones: currently assigned to the overshoot comp,
        # where the undershoot comp would be eligible
        candidates = []
        for z in layout.zones:
            if z.zone_type == ZoneType.SUPERSTRUCTURE:
                continue
            if assignments.get(z.zone_id) != over_comp:
                continue
            if under_comp not in z.eligible_comps:
                continue
            pair_vox = sum(
                zone_map[zid].available_cells
                for zid in _mirror_zone_ids(layout, z.zone_id)
            )
            if (under_comp == Comp.FUEL_TANKS
                    and not _fuel_budget_allows_more_fuel(
                        layout, assignments, p, pair_vox,
                        hull_volume=hull_volume,
                    )):
                continue
            # Don't create centerline fuel-above-ER via rebalancing
            if (under_comp == Comp.FUEL_TANKS
                    and z.zone_id in rebal_zones_above_er
                    and z.zone_type not in (
                        ZoneType.SIDE_DB, ZoneType.SIDE_LOWER, ZoneType.SIDE_UPPER,
                    )):
                continue
            candidates.append(z)

        if not candidates:
            break

        # Pick the candidate whose reassignment best improves total error.
        # _set_assignment flips the WHOLE mirror pair, so candidate
        # evaluation must use the pair's combined volume — evaluating with the
        # representative twin alone realises a 2x larger change than predicted
        # and can accept moves that worsen total error.
        best_zone = None
        best_improvement = 0.0
        seen_pairs: set = set()

        for z in candidates:
            pair_ids = tuple(sorted(_mirror_zone_ids(layout, z.zone_id)))
            if pair_ids in seen_pairs:
                continue   # evaluate each mirror pair once
            seen_pairs.add(pair_ids)
            pair_vox = sum(
                zone_map[zid].available_cells for zid in pair_ids
            )

            # How much would this reassignment change the errors?
            over_new = worst_over - pair_vox / hull_volume
            under_new = worst_under + pair_vox / hull_volume

            old_total_err = abs(worst_over) + abs(worst_under)
            new_total_err = abs(over_new) + abs(under_new)
            improvement = old_total_err - new_total_err

            if improvement > best_improvement:
                best_improvement = improvement
                best_zone = z

        if best_zone is None or best_improvement < 0.005:
            break

        # Accept the reassignment (mirror-aware)
        _set_assignment(layout, assignments, best_zone.zone_id, under_comp)
        n_reassigned += 1

    return n_reassigned


# ─────────────────────────────────────────────────────────────────
# Step 6: Budget-preserving optimisation for LCG / KG
# ─────────────────────────────────────────────────────────────────

def _optimise_physics_budget_preserving(
    layout: BulkheadLayout,
    assignments: Dict[int, Comp],
    p: ShipParameterization,
    volume_tol_frac: float = 0.005,
    time_limit_s: float = 1.5,
    lcg_weight: float = 1.0,
    kg_weight: float = 1.0,
    move_penalty: float = 2.0e-4,
) -> int:
    """Reposition semantic labels to improve LCG/KG without creating VOID.

    A pair-swap search can exchange unequal-sized zones; repeated accepted
    swaps then change semantic volumes and, because VOID has no explicit
    budget key, can improve mass-property targets by increasing VOID or
    reducing fuel/ballast.

    This step instead solves a small mixed-integer assignment problem over
    mirror-linked zone groups. It enforces the following invariants:

    * total VOID is exactly preserved; wing and main-interior VOID remain fixed,
      while engine-region and peak VOID may exchange locations;
    * ENGINE_ROOM, STEERING_GEAR, NAVIGATION and superstructure assignments are
      frozen;
    * every movable group receives one eligibility-compatible label;
    * no budgeted class may move farther from its effective target than it was
      before the search (with a small granularity floor);
    * port/starboard mirror groups always move together;
    * centreline fuel directly above an engine-room group is forbidden.

    The LCG and KG residuals are linear in the binary assignment variables:

        sum(m_i * (x_i/L - target_lcg)) = 0
        sum(m_i * (z_i/D - target_kg)) = 0

    Absolute residuals are minimised with two continuous auxiliary variables.
    If SciPy's MILP solver is unavailable, times out, or returns no feasible
    solution, assignments are left unchanged.

    Returns
    -------
    int
        Number of mirror-linked assignment groups whose label changed.
    """
    try:
        from scipy.optimize import milp, LinearConstraint, Bounds
        from scipy.sparse import lil_matrix
    except Exception:
        return 0

    zones = layout.zones
    zone_map = layout.zone_by_id
    if not zones:
        return 0

    hull_volume = float(_hull_volume_voxels(layout))
    if hull_volume <= 0:
        return 0

    # Engine-room geometry is fixed by this optimisation, so the additional
    # fuel-above-ER restriction can be encoded before the solve.
    er_zids = {
        zid for zid, comp in assignments.items()
        if comp == Comp.ENGINE_ROOM
    }
    zones_above_er: set[int] = set()
    for z in zones:
        if z.zone_type in (
            ZoneType.SIDE_DB, ZoneType.SIDE_LOWER, ZoneType.SIDE_UPPER,
        ):
            continue
        if any(_directly_above_er(z, zone_map[er_zid]) for er_zid in er_zids):
            zones_above_er.add(z.zone_id)

    # Build one optimisation unit per centre zone or mirror-linked pair.
    units: List[Tuple[int, ...]] = []
    seen: set[int] = set()
    for z in zones:
        if z.zone_id in seen:
            continue
        ids = tuple(sorted(_mirror_zone_ids(layout, z.zone_id)))
        ids = tuple(zid for zid in ids if zid in zone_map)
        if not ids:
            continue
        units.append(ids)
        seen.update(ids)

    fixed_comps = {
        Comp.ENGINE_ROOM,
        Comp.STEERING_GEAR,
        Comp.NAVIGATION,
    }

    wing_types = {
        ZoneType.SIDE_DB, ZoneType.SIDE_LOWER, ZoneType.SIDE_UPPER,
        ZoneType.MAIN_DB,
    }
    engine_types = {ZoneType.ENGINE_REGION, ZoneType.ENGINE_UPPER}
    peak_types = {ZoneType.AFT_PEAK, ZoneType.FWD_PEAK}

    def _void_region(z: Zone) -> str:
        if z.zone_type in wing_types:
            return "wing"
        if z.zone_type in engine_types or z.zone_type in peak_types:
            return "engine_peak"
        return "main"

    movable_units: List[dict] = []
    for ids in units:
        current = assignments.get(ids[0], Comp.VOID)
        # Mirror-linked zones should already share a label. If not, freeze the
        # group rather than silently changing asymmetrical semantics.
        if any(assignments.get(zid, current) != current for zid in ids):
            continue

        zgroup = [zone_map[zid] for zid in ids]
        if (
            current in fixed_comps
            or any(z.zone_type == ZoneType.SUPERSTRUCTURE for z in zgroup)
        ):
            continue

        eligible = set(zgroup[0].eligible_comps)
        for z in zgroup[1:]:
            eligible.intersection_update(z.eligible_comps)

        eligible.intersection_update((set(BUDGETED_COMPS) - {Comp.ENGINE_ROOM}) | {Comp.VOID})
        if any(z.zone_id in zones_above_er for z in zgroup):
            eligible.discard(Comp.FUEL_TANKS)
        if current not in eligible:
            # The current assignment must remain a feasible fallback.
            continue
        if len(eligible) <= 1:
            continue

        vol_cells = float(sum(z.available_cells for z in zgroup))
        if vol_cells <= 0:
            continue

        # Use volume-weighted positions for a mirror group. Port/starboard
        # twins share x/z, but weighting also handles non-identical boundary
        # occupancy safely.
        cx = sum(z.available_cells * z.cx_norm for z in zgroup) / vol_cells
        z_norm = sum(
            z.available_cells
            * (z.cz_norm * p.nz_total * p.dz_m / max(p.D, 1e-9))
            for z in zgroup
        ) / vol_cells

        movable_units.append({
            "ids": ids,
            "current": current,
            "eligible": tuple(sorted(eligible, key=int)),
            "vol_cells": vol_cells,
            "cx_norm": float(cx),
            "z_norm": float(z_norm),
            "void_region": _void_region(zgroup[0]),
        })

    if not movable_units:
        return 0

    # One binary variable for each allowed unit-label assignment.
    var_records: List[Tuple[int, Comp]] = []
    vars_by_unit: Dict[int, List[int]] = defaultdict(list)
    vars_by_comp: Dict[Comp, List[int]] = defaultdict(list)
    for ui, unit in enumerate(movable_units):
        for comp in unit["eligible"]:
            idx = len(var_records)
            var_records.append((ui, comp))
            vars_by_unit[ui].append(idx)
            vars_by_comp[comp].append(idx)

    n_binary = len(var_records)
    idx_lcg_abs = n_binary
    idx_kg_abs = n_binary + 1
    n_vars = n_binary + 2

    # Reference mass only scales the objective; it does not alter the exact
    # zero-residual target because each residual remains linear in mass.
    ref_mass = 0.0
    for z in zones:
        comp = assignments.get(z.zone_id, Comp.VOID)
        ref_mass += density_for(comp, p.ship_type) * z.available_cells * p.cell_volume
    ref_mass = max(ref_mass, 1e-9)

    objective = np.zeros(n_vars, dtype=float)
    objective[idx_lcg_abs] = float(lcg_weight)
    objective[idx_kg_abs] = float(kg_weight)
    for idx, (ui, comp) in enumerate(var_records):
        unit = movable_units[ui]
        if comp != unit["current"]:
            objective[idx] = (
                move_penalty * unit["vol_cells"] / hull_volume
            )

    lower_bounds = np.zeros(n_vars, dtype=float)
    upper_bounds = np.ones(n_vars, dtype=float)
    upper_bounds[idx_lcg_abs:] = np.inf
    integrality = np.zeros(n_vars, dtype=int)
    integrality[:n_binary] = 1

    rows: List[dict[int, float]] = []
    row_lb: List[float] = []
    row_ub: List[float] = []

    # Exactly one label per movable mirror group.
    for ui in range(len(movable_units)):
        rows.append({idx: 1.0 for idx in vars_by_unit[ui]})
        row_lb.append(1.0)
        row_ub.append(1.0)

    # Keep every class at least as close to its effective budget as before the
    # optimisation. A small floor handles coarse zone granularity without
    # granting a blanket drift allowance to classes that are already off-target.
    current_volumes = _count_assigned_voxels(layout, assignments)
    tol_cells = max(1.0, float(volume_tol_frac) * hull_volume)
    allowed_budget_error_cells: Dict[str, float] = {}
    movable_zone_id_set = {zid for unit in movable_units for zid in unit["ids"]}
    for comp, bkey in COMP_TO_BUDGET.items():
        indices = vars_by_comp.get(comp, [])
        if not indices:
            continue
        fixed_volume = float(sum(
            z.available_cells for z in zones
            if z.zone_id not in movable_zone_id_set
            and assignments.get(z.zone_id) == comp
        ))
        current_volume = float(current_volumes.get(bkey, 0.0))
        effective_target = float(p.budget.get(bkey, 0.0)) * hull_volume
        allowed_error = max(abs(current_volume - effective_target), tol_cells)
        allowed_budget_error_cells[bkey] = allowed_error
        coeffs = {
            idx: movable_units[ui]["vol_cells"]
            for idx, (ui, candidate_comp) in enumerate(var_records)
            if candidate_comp == comp
        }
        rows.append(coeffs)
        row_lb.append(max(0.0, effective_target - allowed_error - fixed_volume))
        row_ub.append(effective_target + allowed_error - fixed_volume)

    # Preserve total and broad regional VOID volume exactly. Moving VOID
    # within a region is permitted; creating additional VOID or shifting it
    # from wing/engine zones into the main interior is not.
    current_void_total = float(sum(
        z.available_cells for z in zones
        if assignments.get(z.zone_id) == Comp.VOID
    ))
    fixed_void_total = float(sum(
        z.available_cells for z in zones
        if z.zone_id not in {zid for unit in movable_units for zid in unit["ids"]}
        and assignments.get(z.zone_id) == Comp.VOID
    ))
    void_indices = [
        idx for idx, (ui, comp) in enumerate(var_records)
        if comp == Comp.VOID
    ]
    if void_indices:
        rows.append({
            idx: movable_units[ui]["vol_cells"]
            for idx, (ui, comp) in enumerate(var_records)
            if comp == Comp.VOID
        })
        movable_void_target = current_void_total - fixed_void_total
        row_lb.append(movable_void_target)
        row_ub.append(movable_void_target)

    for region in ("wing", "engine_peak", "main"):
        current_region_void = float(sum(
            z.available_cells for z in zones
            if assignments.get(z.zone_id) == Comp.VOID
            and _void_region(z) == region
        ))
        fixed_region_void = float(sum(
            z.available_cells for z in zones
            if z.zone_id not in movable_zone_id_set
            and assignments.get(z.zone_id) == Comp.VOID
            and _void_region(z) == region
        ))
        coeffs = {
            idx: movable_units[ui]["vol_cells"]
            for idx, (ui, comp) in enumerate(var_records)
            if comp == Comp.VOID and movable_units[ui]["void_region"] == region
        }
        if coeffs:
            target = current_region_void - fixed_region_void
            rows.append(coeffs)
            row_lb.append(target)
            row_ub.append(target)

    # Absolute normalised LCG residual.
    lcg_coeffs: dict[int, float] = {}
    kg_coeffs: dict[int, float] = {}

    # Fixed-zone residual contributions.
    movable_zone_ids = {
        zid for unit in movable_units for zid in unit["ids"]
    }
    fixed_lcg_residual = 0.0
    fixed_kg_residual = 0.0
    for z in zones:
        if z.zone_id in movable_zone_ids:
            continue
        comp = assignments.get(z.zone_id, Comp.VOID)
        mass = density_for(comp, p.ship_type) * z.available_cells * p.cell_volume
        fixed_lcg_residual += mass * (z.cx_norm - p.target_lcg_frac) / ref_mass
        z_norm = z.cz_norm * p.nz_total * p.dz_m / max(p.D, 1e-9)
        fixed_kg_residual += mass * (z_norm - p.target_kg_frac) / ref_mass

    for idx, (ui, comp) in enumerate(var_records):
        unit = movable_units[ui]
        mass = density_for(comp, p.ship_type) * unit["vol_cells"] * p.cell_volume
        lcg_coeffs[idx] = mass * (unit["cx_norm"] - p.target_lcg_frac) / ref_mass
        kg_coeffs[idx] = mass * (unit["z_norm"] - p.target_kg_frac) / ref_mass

    # residual - abs_var <= 0 and -residual - abs_var <= 0
    row = dict(lcg_coeffs)
    row[idx_lcg_abs] = -1.0
    rows.append(row)
    row_lb.append(-np.inf)
    row_ub.append(-fixed_lcg_residual)

    row = {idx: -coef for idx, coef in lcg_coeffs.items()}
    row[idx_lcg_abs] = -1.0
    rows.append(row)
    row_lb.append(-np.inf)
    row_ub.append(fixed_lcg_residual)

    row = dict(kg_coeffs)
    row[idx_kg_abs] = -1.0
    rows.append(row)
    row_lb.append(-np.inf)
    row_ub.append(-fixed_kg_residual)

    row = {idx: -coef for idx, coef in kg_coeffs.items()}
    row[idx_kg_abs] = -1.0
    rows.append(row)
    row_lb.append(-np.inf)
    row_ub.append(fixed_kg_residual)

    matrix = lil_matrix((len(rows), n_vars), dtype=float)
    for ri, coeffs in enumerate(rows):
        for ci, value in coeffs.items():
            matrix[ri, ci] = value

    try:
        # disp=False already sets HiGHS log_to_console=False, but some builds
        # still emit native lines (e.g. transformNewIntegerFeasibleSolution).
        with _silence_native_stdio():
            result = milp(
                c=objective,
                integrality=integrality,
                bounds=Bounds(lower_bounds, upper_bounds),
                constraints=LinearConstraint(
                    matrix.tocsr(),
                    np.asarray(row_lb, dtype=float),
                    np.asarray(row_ub, dtype=float),
                ),
                options={
                    "time_limit": float(time_limit_s),
                    "mip_rel_gap": 0.01,
                    "presolve": True,
                    "disp": False,
                },
            )
    except Exception:
        return 0

    # HiGHS may return a feasible integer incumbent when the time limit is
    # reached (status=1, success=False). Accept it only after the same hard
    # invariants and an exact post-solve physics check pass below.
    if result.x is None:
        return 0

    original = dict(assignments)
    _, before_lcg, before_kg, _, _, _ = _compute_physics(layout, original, p)
    before_error = (
        lcg_weight * abs(before_lcg - p.target_lcg_frac)
        + kg_weight * abs(before_kg - p.target_kg_frac)
    )
    changed = 0
    for ui, unit in enumerate(movable_units):
        indices = vars_by_unit[ui]
        chosen_idx = max(indices, key=lambda idx: result.x[idx])
        chosen_comp = var_records[chosen_idx][1]
        if chosen_comp != unit["current"]:
            changed += 1
        for zid in unit["ids"]:
            assignments[zid] = chosen_comp

    # Defensive post-checks. The current assignment was feasible, so any
    # numerical or formulation issue should fall back cleanly rather than
    # silently changing the generator distribution.
    after_volumes = _count_assigned_voxels(layout, assignments)
    for bkey in BUDGET_TO_COMP:
        before = float(current_volumes.get(bkey, 0.0))
        after = float(after_volumes.get(bkey, 0.0))
        target = float(p.budget.get(bkey, 0.0)) * hull_volume
        allowed = allowed_budget_error_cells.get(
            bkey, max(abs(before - target), tol_cells)
        )
        if abs(after - target) > allowed + 1e-6:
            assignments.clear(); assignments.update(original)
            return 0

    before_void_by_region = defaultdict(float)
    after_void_by_region = defaultdict(float)
    for z in zones:
        region = _void_region(z)
        if original.get(z.zone_id) == Comp.VOID:
            before_void_by_region[region] += z.available_cells
        if assignments.get(z.zone_id) == Comp.VOID:
            after_void_by_region[region] += z.available_cells
    if any(
        abs(after_void_by_region[r] - before_void_by_region[r]) > 1e-6
        for r in ("wing", "engine_peak", "main")
    ):
        assignments.clear(); assignments.update(original)
        return 0

    # Verify the realised ratio metrics, not only the linearised solver
    # objective. This also rejects a poor time-limit incumbent safely.
    _, after_lcg, after_kg, _, _, _ = _compute_physics(layout, assignments, p)
    after_error = (
        lcg_weight * abs(after_lcg - p.target_lcg_frac)
        + kg_weight * abs(after_kg - p.target_kg_frac)
    )
    if after_error >= before_error - 1e-8:
        assignments.clear(); assignments.update(original)
        return 0

    return changed


# ─────────────────────────────────────────────────────────────────
# Step 6b: Final fuel-above-ER cleanup
# ─────────────────────────────────────────────────────────────────

def _fix_fuel_above_er(
    layout: BulkheadLayout,
    assignments: Dict[int, Comp],
) -> int:
    """
    Find and fix centerline FUEL_TANKS-above-ENGINE_ROOM violations.

    Wing/side-zone fuel directly above the ER footprint is allowed.

    These can occur because the greedy fill processes zones largest-first,
    so an upper-tier zone may get FUEL_TANKS before its lower-tier
    neighbour gets ENGINE_ROOM.

    For each violation, the fuel zone is reassigned to the best alternative
    eligible comp (preference: MACHINERY > CARGO > BALLAST > VOID).
    """
    zone_map = layout.zone_by_id
    er_zids = {zid for zid, c in assignments.items()
               if c == Comp.ENGINE_ROOM}

    ALTERNATIVES = [Comp.MACHINERY, Comp.CARGO, Comp.BALLAST_TANKS, Comp.VOID]
    n_fixed = 0

    for z in layout.zones:
        if assignments.get(z.zone_id) != Comp.FUEL_TANKS:
            continue
        if z.zone_type in (
            ZoneType.SIDE_DB, ZoneType.SIDE_LOWER, ZoneType.SIDE_UPPER,
        ):
            continue
        # Check if this zone is above any ER zone
        above_er = False
        for er_zid in er_zids:
            er_z = zone_map[er_zid]
            if _directly_above_er(z, er_z):
                above_er = True
                break
        if not above_er:
            continue

        # Reassign to best eligible alternative (mirror-aware)
        for alt in ALTERNATIVES:
            if alt in z.eligible_comps:
                _set_assignment(layout, assignments, z.zone_id, alt)
                n_fixed += 1
                break

    return n_fixed


# ─────────────────────────────────────────────────────────────────
# Step 7: Physics computation
# ─────────────────────────────────────────────────────────────────

def _compute_physics(
    layout: BulkheadLayout,
    assignments: Dict[int, Comp],
    p: ShipParameterization,
) -> Tuple[Dict[int, float], float, float, float, float, float]:
    """
    Compute mass distribution, LCG, KG, simplified GM, and hydrostatic primitives.

    Returns (zone_masses, lcg_frac, kg_frac, gm_t, kb_m, bm_m)
    """
    vs3 = p.cell_volume   # volume per voxel in m^3
    L = p.L
    D = p.D

    zone_masses: Dict[int, float] = {}
    sum_mass = 0.0
    sum_mx = 0.0     # sum(mass * x_position)
    sum_mz = 0.0     # sum(mass * z_position)

    for z in layout.zones:
        comp = assignments.get(z.zone_id, Comp.VOID)
        density = density_for(comp, p.ship_type)
        vol_m3 = z.available_cells * vs3
        mass = density * vol_m3
        zone_masses[z.zone_id] = mass

        # Position in metres from aft / keel
        x_m = z.cx_norm * L
        # cz_norm is normalised to nz_total, convert to metres
        z_m = z.cz_norm * p.nz_total * p.dz_m

        sum_mass += mass
        sum_mx += mass * x_m
        sum_mz += mass * z_m

    # LCG and KG as fractions
    if sum_mass > 0:
        lcg_m = sum_mx / sum_mass
        kg_m = sum_mz / sum_mass
        lcg_frac = lcg_m / L if L > 0 else 0.5
        kg_frac = kg_m / D if D > 0 else 0.5
    else:
        lcg_frac = 0.5
        kg_frac = 0.5
        kg_m = 0.0

    kb_m, bm_m, _cw = compute_hydrostatic_primitives(p)
    if sum_mass > 0:
        kg_m_use = kg_m
    else:
        kg_m_use = D * 0.5
    gm_t = kb_m + bm_m - kg_m_use if (p.T > 0 and p.Cb > 0) else 0.0

    return zone_masses, lcg_frac, kg_frac, gm_t, kb_m, bm_m


# ─────────────────────────────────────────────────────────────────
# Budget error computation
# ─────────────────────────────────────────────────────────────────

def _compute_budget_errors(
    layout: BulkheadLayout,
    assignments: Dict[int, Comp],
    p: ShipParameterization,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Compute |actual_fraction - target_fraction| per budget key.

    Budget fractions are relative to total assignable voxel volume
    (sum of ``available_cells`` over hull and superstructure zones).

    Returns (budget_errors, budget_fracs)
    """
    hull_volume = _hull_volume_voxels(layout)
    if hull_volume == 0:
        return {}, {}

    # Per budget key: assigned voxels summed over all zones (hull + SS)
    assigned_vox = _count_assigned_voxels(layout, assignments)

    budget_errors: Dict[str, float] = {}
    budget_fracs: Dict[str, float] = {}

    for bkey, target_frac in p.budget.items():
        if bkey == "void":
            continue
        actual_vox = assigned_vox.get(bkey, 0)
        actual_frac = actual_vox / hull_volume
        budget_fracs[bkey] = actual_frac
        budget_errors[bkey] = abs(actual_frac - target_frac)

    return budget_errors, budget_fracs


def _directly_above_er(z, er_z) -> bool:
    """True if zone z sits DIRECTLY above ER zone er_z.

    Wing fuel in SIDE_* zones directly above the ER footprint is allowed.
    Only centerline zones with x+y overlap and z-adjacency are
    treated as "above ER" for fuel blocking.
    """
    x_overlap = z.x0 < er_z.x1 and er_z.x0 < z.x1
    y_overlap = z.y0 < er_z.y1 and er_z.y0 < z.y1
    return x_overlap and y_overlap and z.z0 == er_z.z1


def _hull_volume_voxels(layout: BulkheadLayout) -> int:
    """Total available_cells across ALL zones (hull + SS)."""
    return sum(z.available_cells for z in layout.zones)

def _count_assigned_voxels(
    layout: BulkheadLayout,
    assignments: Dict[int, Comp],
) -> Dict[str, float]:
    """Count assigned voxels per budget key across ALL zones (hull + SS)."""
    counts: Dict[str, float] = defaultdict(float)
    for z in layout.zones:
        comp = assignments.get(z.zone_id)
        if comp is None:
            continue
        bkey = COMP_TO_BUDGET.get(comp)
        if bkey:
            counts[bkey] += z.available_cells
    return dict(counts)


def _count_assigned_voxels_with_void(
    layout: BulkheadLayout,
    assignments: Dict[int, Comp],
) -> Dict[str, float]:
    """Budget-key counts plus VOID (needed for fuel-trim void cap)."""
    counts: Dict[str, float] = defaultdict(float)
    for z in layout.zones:
        comp = assignments.get(z.zone_id)
        if comp is None:
            continue
        if comp == Comp.VOID:
            counts["void"] += z.available_cells
            continue
        bkey = COMP_TO_BUDGET.get(comp)
        if bkey:
            counts[bkey] += z.available_cells
    return dict(counts)


def _current_pair_volume(
    layout: BulkheadLayout,
    assignments: Dict[int, Comp],
    zone_id: int,
    comp: Comp,
) -> float:
    """Mirror-pair volume currently assigned to comp."""
    vol = 0.0
    for zid in _mirror_zone_ids(layout, zone_id):
        if assignments.get(zid) == comp:
            vol += layout.zone_by_id[zid].available_cells
    return vol


def _fuel_trim_priority(
    layout: BulkheadLayout,
    assignments: Dict[int, Comp],
    z: Zone,
) -> tuple:
    """Higher tuple = trim earlier."""
    er_zids = [zid for zid, c in assignments.items() if c == Comp.ENGINE_ROOM]
    zone_map = layout.zone_by_id
    direct_above_er = False
    for er_zid in er_zids:
        er_z = zone_map[er_zid]
        if _directly_above_er(z, er_z):
            direct_above_er = True
            break
    abreast_accom = _side_fuel_abreast_accommodation(layout, assignments, z)
    return (
        int(direct_above_er),
        int(abreast_accom),
        int(z.cx_norm > 0.55),
        z.cx_norm,
        z.available_cells,
    )


def _best_replacement_for_excess_fuel(
    layout: BulkheadLayout,
    assignments: Dict[int, Comp],
    p: ShipParameterization,
    z: Zone,
    counts: Dict[str, float],
    hull_volume: float,
    pair_vol: float,
    void_cap: float | None = None,
) -> Optional[Comp]:
    """Under-target budgeted comp first; bounded VOID last resort."""
    if pair_vol <= 0:
        return None

    candidates: List[Tuple[float, Comp]] = []
    for comp in z.eligible_comps:
        if comp in (
            Comp.FUEL_TANKS, Comp.ENGINE_ROOM, Comp.STEERING_GEAR,
            Comp.EMPTY, Comp.NAVIGATION, Comp.VOID,
        ):
            continue
        bkey = COMP_TO_BUDGET.get(comp)
        if bkey is None:
            continue
        target = float(p.budget.get(bkey, 0.0)) * hull_volume
        current = float(counts.get(bkey, 0.0))
        deficit = target - current
        if deficit <= 0:
            continue
        allowed_after = target + _REPLACEMENT_OVERSHOOT_FRAC * hull_volume
        if current + pair_vol > allowed_after:
            continue
        score = deficit
        if comp == Comp.BALLAST_TANKS and z.zone_type in (
            ZoneType.SIDE_DB, ZoneType.SIDE_LOWER, ZoneType.SIDE_UPPER,
            ZoneType.MAIN_DB,
        ):
            score *= 1.25
        candidates.append((score, comp))

    if candidates:
        candidates.sort(key=lambda x: x[0], reverse=True)
        return candidates[0][1]

    void_current = float(counts.get("void", 0.0))
    interior_v, wing_v, _ = _void_fractions_by_region(layout, assignments)
    interior_vox = interior_v * hull_volume
    wing_vox = wing_v * hull_volume
    if Comp.VOID in z.eligible_comps:
        if z.zone_type in _WING_VOID_ZONE_TYPES:
            if wing_vox + pair_vol <= _WING_VOID_ALLOWANCE_FRAC * hull_volume:
                return Comp.VOID
        else:
            void_target = float(p.budget.get("void", 0.0)) * hull_volume
            cap = void_target + _VOID_TRIM_CAP_MARGIN * hull_volume
            if interior_vox + pair_vol <= cap:
                return Comp.VOID
    return None


def _trim_excess_fuel_to_budget(
    layout: BulkheadLayout,
    assignments: Dict[int, Comp],
    p: ShipParameterization,
    tol_frac: float = _FUEL_TRIM_TOL_FRAC,
) -> Dict[str, float]:
    """
    Cap FUEL_TANKS at requested budget + tolerance.

    Reassigns excess fuel to under-target eligible compartments first;
    VOID only within target + margin. Does not rerun void reduction.
    """
    hull_volume = float(_hull_volume_voxels(layout))
    if hull_volume <= 0:
        return {"changed": 0}

    counts = _count_assigned_voxels_with_void(layout, assignments)
    fuel_target = float(p.budget.get("fuel_tanks", 0.0)) * hull_volume
    fuel_current = float(counts.get("fuel_tanks", 0.0))
    fuel_cap = fuel_target + tol_frac * hull_volume

    if fuel_current <= fuel_cap:
        frac = fuel_current / hull_volume
        return {
            "changed": 0,
            "fuel_before": frac,
            "fuel_after": frac,
        }

    void_target = float(p.budget.get("void", 0.0)) * hull_volume

    fuel_zones = [
        z for z in layout.zones
        if assignments.get(z.zone_id) == Comp.FUEL_TANKS
    ]
    fuel_zones.sort(
        key=lambda z: _fuel_trim_priority(layout, assignments, z),
        reverse=True,
    )

    fuel_before = fuel_current
    void_before = float(counts.get("void", 0.0))
    changed = 0
    fuel_to_ballast = 0.0
    fuel_to_void = 0.0
    fuel_to_other = 0.0
    untrimmed = 0.0

    for z in fuel_zones:
        counts = _count_assigned_voxels_with_void(layout, assignments)
        fuel_current = float(counts.get("fuel_tanks", 0.0))
        if fuel_current <= fuel_cap:
            break

        pair_vol = _current_pair_volume(
            layout, assignments, z.zone_id, Comp.FUEL_TANKS,
        )
        if pair_vol <= 0:
            continue

        new_comp = _best_replacement_for_excess_fuel(
            layout=layout,
            assignments=assignments,
            p=p,
            z=z,
            counts=counts,
            hull_volume=hull_volume,
            pair_vol=pair_vol,
        )
        if new_comp is None:
            untrimmed += pair_vol
            continue

        _set_assignment(layout, assignments, z.zone_id, new_comp)
        changed += 1
        if new_comp == Comp.BALLAST_TANKS:
            fuel_to_ballast += pair_vol
        elif new_comp == Comp.VOID:
            fuel_to_void += pair_vol
        else:
            fuel_to_other += pair_vol

    final = _count_assigned_voxels_with_void(layout, assignments)
    hv = hull_volume
    return {
        "changed": changed,
        "fuel_before": fuel_before / hv,
        "fuel_after": float(final.get("fuel_tanks", 0.0)) / hv,
        "void_before": void_before / hv,
        "void_after": float(final.get("void", 0.0)) / hv,
        "fuel_to_ballast": fuel_to_ballast / hv,
        "fuel_to_void": fuel_to_void / hv,
        "fuel_to_other": fuel_to_other / hv,
        "untrimmed": untrimmed / hv,
    }


# ─────────────────────────────────────────────────────────────────
# GM feasibility (label convention — not a dataset reject by default)
# ─────────────────────────────────────────────────────────────────

QC_GM_MAX = 10.0
QC_LCG_TOL = 0.10
QC_KG_TOL = 0.10

GM_MIN_BY_TYPE: Dict[ShipType, float] = {
    ShipType.YACHT: 0.10,
    ShipType.OSV: 0.10,
    ShipType.CARGO: 0.10,
}


def gm_min_for_type(ship_type: ShipType, default: float = 0.0) -> float:
    return GM_MIN_BY_TYPE.get(ship_type, default)


def compute_hydrostatic_primitives(
    p: ShipParameterization,
) -> Tuple[float, float, float]:
    """
    Geometry-only KB, BM (m) and Cw estimate used in the proxy GM model.

    GM = KB + BM - KG is recomputable offline for any density convention
    when KG is derived from compartment masses.
    """
    T, B, Cb = p.T, p.B, p.Cb
    if T > 0 and Cb > 0:
        Cw = 0.70 + 0.30 * Cb
        kb_m = T * (5.0 / 6.0 - Cb / (3.0 * Cw))
        bm_m = (Cw / Cb) * B ** 2 / (12.0 * T)
        return float(kb_m), float(bm_m), float(Cw)
    return 0.0, 0.0, 0.70


def gm_feasible_at_convention(
    gm_t: float,
    ship_type: ShipType,
    *,
    gm_min: Optional[float] = None,
) -> bool:
    """Stability feasibility: GM ≥ gm_min_by_type (lower bound only).

    Upper GM is reported as distribution, not a feasibility verdict — loaded
    bulkers naturally read GM 9–16 under the proxy convention.
    """
    lo = gm_min if gm_min is not None else gm_min_for_type(ship_type)
    return gm_t >= lo


def lcg_kg_tracking_metrics(
    actual_lcg_frac: float,
    target_lcg_frac: float,
    actual_kg_frac: float,
    target_kg_frac: float,
    *,
    lcg_tol: float = QC_LCG_TOL,
    kg_tol: float = QC_KG_TOL,
) -> Dict[str, float | bool]:
    """|achieved − sampled_target| vs dataset QC tolerances (fixed densities)."""
    lcg_err = abs(actual_lcg_frac - target_lcg_frac)
    kg_err = abs(actual_kg_frac - target_kg_frac)
    return {
        "lcg_tracking_error": float(lcg_err),
        "kg_tracking_error": float(kg_err),
        "lcg_tracking_within_tol": bool(lcg_err <= lcg_tol),
        "kg_tracking_within_tol": bool(kg_err <= kg_tol),
    }


# ─────────────────────────────────────────────────────────────────
# Validation
# ─────────────────────────────────────────────────────────────────

def validate_assignment(
    result: CompartmentAssignment,
    budget_tol: float = 0.12,
    lcg_tol: float = 0.10,
    kg_tol: float = 0.15,
    gm_min: float = -0.5,
    gm_max: float = 10.0,
    gm_gate: bool = False,
    lcg_kg_gate: bool = False,
    void_tol: float = 0.05,
) -> Tuple[bool, List[str]]:
    """
    QC checks on a CompartmentAssignment.

    Returns (ok, warns). Any failed check sets ok to False (no warnings-only
    pass mode).

    Checks
    ------
    1. Every zone has an assignment
    2. All assignments respect zone eligibility
    3. At least one ENGINE_ROOM zone exists
    4. At least one ACCOMMODATION zone exists
    5. ENGINE_ROOM zones are contiguous
    6. Budget satisfaction within tolerance
    7. LCG error < lcg_tol (only when ``lcg_kg_gate=True``; default off)
    8. KG error < kg_tol (only when ``lcg_kg_gate=True``; default off)
    9. GM_t bounds (only when ``gm_gate=True``; default off — store ``gm_feasible`` label)
    10. STEERING_GEAR in aft peak only
    11. No FUEL_TANKS directly above ENGINE_ROOM (hard)
    15. Achieved VOID fraction <= void target + void_tol (VOID is not
        covered by check 6)
    """
    layout = result.layout
    p = result.params
    warns: List[str] = []

    # Check 1: Every zone assigned
    unassigned = [z.zone_id for z in layout.zones
                  if z.zone_id not in result.assignments]
    if unassigned:
        warns.append(f"  {len(unassigned)} zones unassigned: {unassigned}")

    # Check 2: Eligibility
    for z in layout.zones:
        comp = result.assignments.get(z.zone_id)
        if comp is None:
            continue
        if comp not in z.eligible_comps:
            warns.append(
                f"  Zone {z.zone_id} ({z.zone_type.name}) assigned "
                f"{comp.name} not in eligible set"
            )

    # Check 3: ENGINE_ROOM exists
    er_count = sum(1 for c in result.assignments.values()
                   if c == Comp.ENGINE_ROOM)
    if er_count == 0:
        warns.append("  No ENGINE_ROOM zone assigned")

    # Check 4: ACCOMMODATION exists
    acc_count = sum(1 for c in result.assignments.values()
                    if c == Comp.ACCOMMODATION)
    if acc_count == 0:
        warns.append("  No ACCOMMODATION zone assigned")

    # Check 5: ENGINE_ROOM contiguity
    er_ids = [zid for zid, c in result.assignments.items()
              if c == Comp.ENGINE_ROOM]
    if len(er_ids) > 1:
        zone_map = layout.zone_by_id
        visited = set()
        queue = [er_ids[0]]
        while queue:
            cur = queue.pop(0)
            if cur in visited:
                continue
            visited.add(cur)
            for other_id in er_ids:
                if other_id not in visited:
                    if _zones_share_face(zone_map[cur], zone_map[other_id]):
                        queue.append(other_id)
        if len(visited) < len(er_ids):
            warns.append(
                f"  ENGINE_ROOM not contiguous: "
                f"{len(visited)}/{len(er_ids)} zones connected"
            )

    # Check 6: Budget satisfaction
    # Adaptive tolerance: each main-hull zone is a coarse slice of total assignable
    # volume, so one mis-tagged zone can move a budget key by ~1/n_hull_zones.
    n_hull_zones = sum(1 for z in layout.zones
                       if z.zone_type != ZoneType.SUPERSTRUCTURE)
    effective_budget_tol = max(budget_tol, 2.0 / max(n_hull_zones, 1))
    for bkey, err in result.budget_errors.items():
        if err > effective_budget_tol:
            warns.append(
                f"  Budget '{bkey}' error = {err:.4f} > tol {effective_budget_tol:.3f}"
            )

    # -- 15. VOID overshoot: interior vs wing --
    p_chk = layout.params
    hull_total = sum(z.available_cells for z in layout.zones)
    if hull_total > 0:
        interior_void, wing_void, _ = _void_fractions_by_region(
            layout, result.assignments,
        )
        void_target = float(p_chk.budget.get("void", 0.0))
        effective_void_tol = max(void_tol, 2.0 / max(n_hull_zones, 1))
        if interior_void > void_target + effective_void_tol:
            warns.append(
                f"  Interior VOID = {interior_void:.3f} > target "
                f"{void_target:.3f} + void_tol {effective_void_tol:.2f}"
            )
        if wing_void > _WING_VOID_ALLOWANCE_FRAC:
            warns.append(
                f"  Wing/DB VOID = {wing_void:.3f} > allowance "
                f"{_WING_VOID_ALLOWANCE_FRAC:.2f}"
            )

    # Check 7–8: LCG/KG vs sampled targets (reject only when lcg_kg_gate is on)
    lcg_err = abs(result.actual_lcg_frac - p.target_lcg_frac)
    kg_err = abs(result.actual_kg_frac - p.target_kg_frac)
    if lcg_kg_gate and lcg_err > lcg_tol:
        warns.append(
            f"  LCG error = {lcg_err:.4f} > {lcg_tol:.3f}"
        )
    if lcg_kg_gate and kg_err > kg_tol:
        warns.append(
            f"  KG error = {kg_err:.4f} > {kg_tol:.3f}"
        )

    # Check 9: GM (reject only when gm_gate / SHIP_GM_QC_GATE is on)
    if gm_gate:
        if result.actual_gm_t < gm_min:
            warns.append(
                f"  GM_t = {result.actual_gm_t:.2f} m < min {gm_min:.2f} m"
            )
        if result.actual_gm_t > gm_max:
            warns.append(
                f"  GM_t = {result.actual_gm_t:.2f} m > max {gm_max:.2f} m"
            )

    # Check 10: STEERING_GEAR only in AFT_PEAK
    for z in layout.zones:
        comp = result.assignments.get(z.zone_id)
        if comp == Comp.STEERING_GEAR and z.zone_type != ZoneType.AFT_PEAK:
            warns.append(
                f"  STEERING_GEAR in non-aft-peak zone {z.zone_id}"
            )

    # Check 10b: NAVIGATION only in SUPERSTRUCTURE
    for z in layout.zones:
        comp = result.assignments.get(z.zone_id)
        if comp == Comp.NAVIGATION and z.zone_type != ZoneType.SUPERSTRUCTURE:
            warns.append(
                f"  NAVIGATION in non-SS zone {z.zone_id}"
            )

    # Check 11: No centerline FUEL_TANKS directly above ENGINE_ROOM
    er_zids = {zid for zid, c in result.assignments.items()
               if c == Comp.ENGINE_ROOM}
    for z in layout.zones:
        comp = result.assignments.get(z.zone_id)
        if comp != Comp.FUEL_TANKS:
            continue
        if z.zone_type in (
            ZoneType.SIDE_DB, ZoneType.SIDE_LOWER, ZoneType.SIDE_UPPER,
        ):
            continue
        zone_map = layout.zone_by_id
        for er_zid in er_zids:
            er_z = zone_map[er_zid]
            if _directly_above_er(z, er_z):
                warns.append(
                    f"  FUEL_TANKS zone {z.zone_id} directly above "
                    f"ENGINE_ROOM zone {er_zid} (fire safety)"
                )
                break

    # Check 12: port/stbd mirror symmetry
    by_mirror: Dict[int, List[Zone]] = {}
    for z in layout.zones:
        if z.mirror_id >= 0:
            by_mirror.setdefault(z.mirror_id, []).append(z)
    for mid, pair in by_mirror.items():
        if len(pair) != 2:
            continue
        c0 = result.assignments.get(pair[0].zone_id)
        c1 = result.assignments.get(pair[1].zone_id)
        if c0 != c1:
            warns.append(
                f"  mirror_id {mid}: port/stbd mismatch "
                f"{c0} vs {c1}"
            )

    # Check 13: separation backstop — no fuel abreast cargo/accommodation centre
    for z in layout.zones:
        if z.zone_type not in (ZoneType.SIDE_DB, ZoneType.SIDE_LOWER, ZoneType.SIDE_UPPER):
            continue
        comp = result.assignments.get(z.zone_id)
        if comp != Comp.FUEL_TANKS:
            continue
        centre = _adjacent_centre_zone(layout, z)
        if centre is None:
            continue
        cc = result.assignments.get(centre.zone_id)
        if cc in (Comp.NAVIGATION, Comp.ACCOMMODATION):
            warns.append(
                f"  FUEL in side zone {z.zone_id} abreast {cc.name} centre"
            )

    return len(warns) == 0, warns
