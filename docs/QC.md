# Production quality-control summary

A generated arrangement enters the released corpus only after passing the active production QC checks. These checks establish consistency with the generator rules; they are not equivalent to complete ship-design validation or regulatory approval.

## Active acceptance checks

| Check | Production acceptance condition | Interpretation |
|---|---|---|
| Subdivision coverage | Non-empty zones cover the occupied envelope and every mapped zone receives a function | Internal generator consistency |
| Functional eligibility | Each assigned function is permitted for its zone type | Encoded early-stage placement constraints |
| Required functions | At least one engine-room and one accommodation region | Minimum generated functional content |
| Engine-room coherence | Engine room remains face-connected | Spatial coherence constraint |
| Symmetry | Mirror-linked port/starboard zones retain matching labels where symmetry is enforced | Generator-specific symmetry rule |
| Steering location | Steering gear remains aft / in the encoded aft-peak region | Generator placement rule |
| Programme compliance | Each effective-to-achieved functional error is at most `max(0.12, 2/N_h)` | Discrete-zone assignment tolerance |
| Interior void | Interior void is limited relative to its target using `max(0.05, 2/N_h)` | Residual-space control |
| Wing / DB void | Wing and double-bottom void is limited to 0.12 of occupied volume | Prevents excessive unused tank-band volume |
| Fuel placement | Encoded fuel-separation restrictions are enforced | Conservative conceptual placement proxies |

`N_h` is the number of main-hull subdivision zones.

## Diagnostic quantities that are not production rejection gates

### LCG / KG target tracking

The sampled `LCG/L` and `KG/D` values guide assignment and label refinement, but production does not reject a sample solely because the achieved values fall outside the legacy target-tracking tolerance. Tracking errors are stored in `aux_physics` for later analysis.

### GM

The generator records a proxy transverse metacentric-height quantity and associated diagnostics, but GM is not an active production dataset QC rejection gate. The more detailed corrected-GM calibration used in the paper's fixed-hull application is a separate evaluation model and should not be conflated with the generator's internal stored proxy quantity.

## Canonical implementation

The principal Stage-4 acceptance function is `validate_assignment()` in `src/data_generator/compartment_assignment.py`. Stage-2/3 hull and subdivision validation is performed before assignment and graph serialization. Dataset assembly retains only candidates that pass the active pipeline checks. Accepted samples also receive an exact compact companion (`companion.py` / `deck_codec.py`); arrangements that fail companion encode/decode checks are rejected.

## Scope

Passing QC means the sample is internally consistent with the procedural generator. It does **not** establish:

- damage stability compliance;
- complete SOLAS/MARPOL compliance;
- structural feasibility;
- systems-routing feasibility;
- access/escape-route compliance;
- production readiness;
- class approval.
