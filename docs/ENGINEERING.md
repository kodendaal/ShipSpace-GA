# P1A engineering evaluator

The physics package recovered from the earlier work is installed as
`src/engineering/`, its original import path. It evaluates **existing native
arrangements** and does not change generation, frozen datasets or QC gates.
It supplies calibrated whole-ship LCG/KG, functional adjacency (`J_adj`), and
optional equilibrium trim and free-surface-corrected GM for Generator v1.0.

## Provenance and integration

- Source: `ShipSpace_GA_Engineering_DropIn_v1.zip`, created 9 October 2026 for
  repository commit `ef38c461f64f44630d9372ab72235a1986e3feb3`.
- Archive SHA-256:
  `a2e361593617b63ec3922fd141b5c79426b59111159f0e76ee69ea04fd04eda6`.
- The archive attributes its numerical cores and calibration to
  `Paper1A_FinalWritingPackage.zip`. Their 11 original file hashes are preserved in
  [`UPSTREAM_SHA256.json`](../src/engineering/UPSTREAM_SHA256.json). The tests
  check the packaged hashes; two comment/docstring naming edits have separate
  hashes recorded alongside the originals. This checks source integrity, not agreement with published
  numerical results.
- The numerical implementation in `_p1a_weight/` and `_p1a_adjacency/`, and
  the family-v4 `fsm_shared` calibration JSON are unchanged. Two files only
  rename the generator to v1.0 in a comment/docstring. The JSON retains its original internal
  `profile_name=family_residual_closure_v4_provisional`; the filename alone is
  not evidence of validation.
- The adapter and CLI come from the drop-in. Integration adds rejection of
  non-integer/out-of-taxonomy node labels before narrowing their dtype,
  invalid node indices, and paired evaluations with differing reference
  draughts. Three regression tests cover these checks.
- The archive's illustrative NPZ and installation-only README are not shipped
  as scientific fixtures. This document replaces those installation notes.

This evaluator is included in [software v1.0.1](https://doi.org/10.5281/zenodo.23913344),
published on 3 November 2026. The original software v1.0.0 archive remains
available as its historical checkpoint. Record the source commit and
calibration hash with any new results; see [RELEASE.md](RELEASE.md).

## Installation and checks

From the repository root:

```bash
pip install -r requirements.txt
python -m unittest discover -s tests -p "test_engineering_dropin.py" -v
```

The evaluator itself needs NumPy, SciPy and NetworkX; PyTorch/PyG are needed
when loading saved PyG graphs. The tests use synthetic volumes, including a
real NPZ round trip. They check kernel hashes, mass closure, identical-pair
deltas, changed labels, missing services, optional equilibrium, adapters and
input integrity. They are not published-ship regression fixtures.

## Native input contract

`EngineeringVolume` accepts these fields (also the CLI NPZ schema):

| Field | Required | Meaning |
|---|---|---|
| `labels` | yes | Integer native `(nx, ny, nz_total)` labels: 0–8 inside, `EMPTY=10` outside |
| `hull_fraction` | yes | Same shape, fractional occupancy in [0, 1], including superstructure; zero exactly where labels are `EMPTY` |
| `family` | yes | `BULKER`, `TANKER`, `CARGO` (container), `OSV`, `PATROL`, or `YACHT` |
| `L`, `B`, `D` | yes | True principal dimensions in metres; `D` is main-hull depth |
| `nz_hull` | yes | Number of native main-hull layers, normally 24 for synthetic production |
| `T` | optional | Sampled/reference draught in metres; supply the actual value (adapter fallback is `0.7D` if omitted) |
| `main_hull_fraction` | strongly preferred | Original hull-only fractional mask, shape `(nx, ny, nz_hull)` |
| `sample_id` | optional | Stable record identifier |
| `reference_labels` | CLI optional | Native reference labels for paired prediction/reference evaluation |
| `model_id`, `seed` | CLI optional | Result provenance fields |

NPZ files must contain numeric/string arrays readable with `allow_pickle=False`.
The evaluator assumes uniform native spacing `(L/nx, B/ny, D/nz_hull)`.
Nonuniform real-reference grids require an appropriate geometry adapter;
do not feed their raw tensors into this API as uniform synthetic grids.

Neither resampled `voxel_labels` nor deck-encoded `compact_labels` can be
treated as a native physical grid merely because their shape is fixed.
Decode with the correct mapping and physical weights first. Do not interpolate
semantic labels to invent missing geometry.

## Python API

For a saved graph and its native companion:

```python
import numpy as np
from src.engineering import from_companion, evaluate_volume

with np.load(companion_path, allow_pickle=False) as companion:
    volume = from_companion(
        graph, companion, main_hull_fraction=original_hull_mask,
        sample_id="candidate",
    )
row = evaluate_volume(volume, equilibrium=True)
```

The graph must supply `ship_type`, `L`, `B`, `D`, `T` and `nz_hull`.
The companion must include `native_zone_id`, `native_zone_labels` and
`native_occupied_fraction`. `labels_by_node` optionally supplies another
assignment in graph-node order. Physical dimensions come from metadata.

For an already-generated live assignment:

```python
from src.engineering import from_live_assignment, evaluate_volume
volume = from_live_assignment(existing_assignment, sample_id="candidate")
row = evaluate_volume(volume)
```

`compare_pair` checks equal family, dimensions, hull layer count, `T`, occupied
fractions and original hull mask availability/content. It returns both metric
sets and prediction-minus-reference deltas for LCG/L, KG/D and valid `J_adj`.

## Command line

```bash
python -m src.engineering.cli --input native_cases --output metrics.csv
python -m src.engineering.cli --input native_case.npz --output metrics.jsonl --equilibrium
```

A directory input reads its immediate `*.npz` files. `--config` selects an
explicit alternative calibration. Input/evaluation exceptions produce
`status=error` rows and a nonzero exit code. Numerical invalidity can still
produce `status=ok`: inspect `weight_valid`, `adjacency_valid`, `trim_status`
and `equilibrium_valid`. Non-finite results become JSON `null` or empty CSV
cells. Report the invalid-case rate rather than silently excluding failures.

## Metric conventions and limits

| Output | Convention |
|---|---|
| `LCG_m`, `KG_m`; `LCG_over_L`, `KG_over_D` | Calibrated whole-ship mass model: compartment density/fill factors plus lightship and residual closure. Different from generator diagnostics. |
| `J_adj` | P1A six-face functional islands and weighted shortest paths via shared interfaces, normalized by L; internal void is traversable. Missing/disconnected required services yield invalid/NaN. This is functional proximity, not an access or routing proof. |
| `trim_m` | Signed forward-minus-aft draught from optional heave/trim solution |
| `trim_pct_L` | Signed `100 * trim_m / L` |
| `J_trim_abs_over_L` | `abs(trim_m)/L`; the paper's percentage objective is **100 times this value**, equivalently `abs(trim_pct_L)` |
| `GM_corrected_m` | Optional `KM - KG - free_surface_correction`; valid equilibrium and calibration provenance must accompany interpretation |

The original hull-only fractional mask is preferred for source-exact
hydrostatics. Without it, the evaluator uses the main-hull slice of final
companion occupancy and records
`main_hull_source=from_native_occupied_fraction`. Zone construction can remove
cells, so this fallback can change displacement and mass closure. Matching a
hull name alone is insufficient; verify the cache/source revision and use the
same physical input for reference and prediction.

The shipped real-reference graphs still contain `NAVIGATION=9`, which this
evaluator rejects. Their taxonomy and native-geometry migration remain separate
work; see [DATASET.md](DATASET.md).

## Validation status

Integration on 24 October 2026 passed 11 unit/regression tests, including the
eight supplied tests and three adapter checks. A live OSV assignment using a
shipped hull mask passed generator QC and produced valid mass/adjacency
outputs; equilibrium correctly reported `overloaded_before_deck_limit`.
CSV/JSONL CLI checks also passed; invalid input was retained as an error row with exit code 1. These checks confirm wiring and failure reporting, not reproduction of a paper case.

The existing tests establish integration, not a rerun of the paper's
500-alternative OSV application. For a numerical comparison with a published
case, use that case's exact native geometry, assignment, dimensions, loading
convention and calibration. The original experiment and calibration-fitting
runners are not included in this restored evaluation package.

See [RELEASE.md](RELEASE.md) for the published software checkpoint and archive correspondence.
