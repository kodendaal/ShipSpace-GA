# Generator specification

This document describes the version 1.0 procedural generator at the level needed to interpret and reproduce the released design space. The Python modules in `src/data_generator/` remain the canonical numerical implementation.

## Coordinate convention

- `x = 0` aft, increasing toward the bow.
- `y = 0` port, increasing toward starboard.
- `z = 0` keel, increasing upward.
- Longitudinal quantities such as `LCG/L` and zone centroids therefore use aft = 0 and bow = 1.

## Vessel families

| Paper family | Code identifier | Structural family |
|---|---|---|
| Bulk carrier | `ShipType.BULKER` | `DEEP_CARGO` |
| Tanker | `ShipType.TANKER` | `DEEP_CARGO` |
| Container vessel | `ShipType.CARGO` | `DEEP_CARGO` |
| Offshore support vessel | `ShipType.OSV` | `WORKING` |
| Patrol vessel | `ShipType.PATROL` | `WORKING` |
| Yacht | `ShipType.YACHT` | `SLENDER_DISP` |

The `CARGO` identifier is retained from the frozen code but corresponds to the container-vessel family in the paper and released dataset.

## Principal-dimension and placement-target ranges

| Family | L (m) | B/L | D/L | LCG/L | KG/D |
|---|---:|---:|---:|---:|---:|
| Bulker | 150–280 | .14–.19 | .058–.081 | .42–.52 | .45–.62 |
| Tanker | 75–150 | .14–.18 | .080–.095 | .42–.52 | .45–.62 |
| Container | 230–300 | .12–.18 | .068–.082 | .42–.53 | .46–.64 |
| OSV | 50–95 | .18–.27 | .084–.113 | .40–.55 | .44–.62 |
| Patrol | 55–85 | .12–.15 | .115–.145 | .40–.52 | .46–.64 |
| Yacht | 50–100 | .13–.22 | .095–.155 | .42–.56 | .46–.63 |

The LCG/KG values are sampled placement targets. They steer assignment/refinement but are not active production QC rejection limits.

## Requested functional programme

Requested fractions refer to assignable internal volume.

| Family | Engine | Machinery | Cargo / mission | Stores | Accommodation | Fuel | Ballast | Programme cap |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Bulker | .05–.08 | .008–.020 | .53–.63 | .008–.018 | .04–.07 | .04–.08 | .13–.20 | .92–.98 |
| Tanker | .05–.08 | .02–.06 | .45–.58 | .02–.04 | .03–.10 | .035–.07 | .11–.20 | .92–.98 |
| Container | .03–.07 | .015–.035 | .55–.70 | .005–.025 | .03–.06 | .04–.08 | .08–.14 | .90–.98 |
| OSV | .04–.08 | .06–.10 | .16–.42 | .02–.07 | .18–.28 | .08–.16 | .10–.19 | .88–.96 |
| Patrol | .08–.12 | .14–.24 | .03–.08 | .10–.15 | .28–.34 | .03–.07 | .04–.08 | .90–.98 |
| Yacht | .045–.085 | .08–.14 | .12–.22 | .04–.10 | .28–.42 | .02–.06 | .04–.09 | .88–.95 |

The unused fraction of the programme cap remains available as residual capacity. Steering gear is mandatory but is not one of the seven requested programme variables.

## Programme states

The generator stores three programme states:

1. `budget_sampled` — original requested programme.
2. `budget_effective` — request after capacity projection against the realised subdivision and eligibility rules.
3. `budget_achieved` — final realised functional fractions after discrete assignment.

The legacy metadata field `program` is an alias for the effective programme. New analyses should use the explicit fields above.

## Grid and hull representation

Production mode uses a `64 × 32 × 24` main-hull grid. Physical cell sizes therefore depend on sampled `L`, `B` and `D`. The stored voxel representation is nearest-neighbour mapped to `64 × 32 × 32`, which accommodates the hull and superstructure in a common tensor shape.

The hull itself is not procedurally generated in production mode. One compatible template is selected from the family pool and independently scaled to the sampled `L`, `B` and `D`. This repository loads the corresponding occupancy masks from `data/hull_cache/`. A sampled superstructure is then added above the local deck profile.

Common geometric rules include:

- aft-peak boundary: `0.04L–0.08L`;
- fore-peak boundary: `0.90L–0.95L`;
- double-bottom height: `clip(0.06B, 1.0 m, 2.5 m)`;
- nominal side-region width: based on `0.06B`, bounded approximately between `0.76 m` and `2.0 m` in the production configuration.

## Superstructure ranges

| Family | Longitudinal centre x/L | Length/L | Width/B | Logical decks | x taper / level | y taper / level |
|---|---:|---:|---:|---:|---:|---:|
| Bulker | .05–.15 | .08–.15 | .70–.90 | 4–6 | .10 | .05 |
| Tanker | .05–.15 | .08–.15 | .70–.90 | 3–4 | .10 | .05 |
| Container | .15–.30 | .05–.08 | .50–.70 | 7–10 | .05 | .05 |
| OSV | .70–.85 | .15–.28 | .75–.95 | 2–3 | .20 | .20 |
| Patrol | .15–.40 | .35–.60 | .65–.85 | 1–2 | .30 | .30 |
| Yacht | .40–.60 | .35–.60 | .75–.85 | 2–3 | .28 | .28 |

Superstructure clear-deck heights are sampled between approximately 2.4 and 2.7 m and are converted to z-layers according to the physical grid spacing.

Canonical implementation: `SUPERSTRUCTURE_CONFIG` and `SS_DECK_CLEAR_H_M` in `src/data_generator/ship_params.py`.

## Subdivision-zone taxonomy and eligibility

| Zone type | Eligible functions |
|---|---|
| Aft peak | Ballast, steering gear, void |
| Engine region | Engine room, machinery, fuel, void |
| Main double bottom | Ballast, fuel, void |
| Main lower | Cargo/mission, fuel, ballast, machinery, stores, void |
| Main upper | Cargo/mission, stores, machinery, accommodation, void; family-dependent restrictions apply |
| Fore peak | Ballast, machinery, void |
| Superstructure | Accommodation, void |
| Engine upper | Machinery, void, with family-dependent cargo/accommodation/stores eligibility |
| Side double bottom | Ballast, fuel, void |
| Side lower | Ballast, fuel, void |
| Side upper | Ballast, fuel, void |

The exact family-dependent eligibility modifications are applied during zone construction. Canonical implementation: `ZONE_ELIGIBILITY`, `_main_upper_eligible_comps` and `_engine_upper_eligible_comps` in `bulkhead_placement.py`.

## Functional mass parameters

Fuel and ballast use representative physical liquid densities. The remaining coefficients are relative mass-intensity values used to establish a common early-stage mass convention; they should not be interpreted as measured physical densities of entire spaces.

| Function | Relative mass intensity (t/m³) |
|---|---:|
| Engine room | 5.00 |
| Steering gear | 3.50 |
| Machinery | 2.50 |
| Cargo / mission | 2.00 |
| Stores | 1.50 |
| Accommodation | 0.50 |
| Void | 0.05 |

| Liquid | Density (t/m³) |
|---|---:|
| Fuel | 0.85 |
| Ballast | 1.025 |

For yachts, the frozen generator applies an accommodation-like coefficient (`0.50`) to the internal `CARGO` class when the yacht-density option is active.

Canonical implementation: `COMP_DENSITY` and `density_for()` in `ship_params.py`.

## Functional assignment

After mandatory and strongly constrained placements, remaining eligible zones are assigned sequentially using a seed-perturbed greedy score. The score combines:

- remaining programme deficit;
- a fit term that discourages excessive overshoot when a zone is much larger than the remaining deficit;
- a bounded spatial-preference multiplier;
- a mass-position preference related to the sampled LCG/KG targets;
- a small seed-controlled perturbation.

The spatial-preference multiplier is bounded and acts as a soft preference rather than a hard rule. The frozen numerical definitions are implemented in `_spatial_preference()` in `compartment_assignment.py`. Examples include aft/low fuel preference, double-bottom/lower ballast preference, machinery preference near the engine region, upper accommodation preference, and family-dependent cargo/mission placement.

## Repair and refinement

Sequential assignment is followed by bounded repair steps that address known inconsistencies, including programme imbalance, avoidable void, engine-room contiguity, mirrored assignments and fuel-placement constraints. Unused void superstructure is removed from the represented envelope.

A later label-refinement step may swap eligible functional labels to improve LCG/KG tracking without changing achieved programme totals. The subdivision geometry itself remains fixed during this refinement. The production pipeline finishes with a budget-preserving mixed-integer LCG/KG optimisation (SciPy HiGHS).

## Canonical code locations

| Quantity / behaviour | Canonical code location |
|---|---|
| Vessel and compartment enums | `ship_params.py` |
| Dimension ranges | `SHIP_DIM_RANGES` in `ship_params.py` |
| Requested programme ranges | `VOLUME_BUDGETS` in `ship_params.py` |
| Programme caps | `BUDGET_CAP_RANGE` in `ship_params.py` |
| LCG/KG placement targets | `PHYSICS_TARGETS` in `ship_params.py` |
| Superstructure ranges | `SUPERSTRUCTURE_CONFIG` in `ship_params.py` |
| Zone taxonomy / base eligibility | `ZoneType`, `ZONE_ELIGIBILITY` in `bulkhead_placement.py` |
| Spatial-preference equations | `_spatial_preference()` in `compartment_assignment.py` |
| Assignment and repair | `assign_compartments()` in `compartment_assignment.py` |
| QC | `validate_assignment()` in `compartment_assignment.py` plus Stage-3 validation |
| Graph schema | `graph_builder.py` |
| Voxel conversion | `representation_converters.py` |
| Compact companions | `companion.py`, `deck_codec.py` |
| Batch generation | `dataset_builder.py` |

## Important interpretation

The generator is a constrained stochastic procedural model, not a global optimiser. The realised subdivision determines the available spatial regions and eligible capacity. The assignment stage then maps functions to those discrete regions. This is why a sampled request can differ from both the effective and achieved programme and why different seeds can produce different layouts for the same broad design intent.
