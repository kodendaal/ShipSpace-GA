# Dataset and representation guide

The published 50K synthetic corpus associated with Generator v1.0 has a separate [dataset DOI](https://doi.org/10.5281/zenodo.23197299). This document describes the release identity and serialized sample schema so the data can be used without reverse-engineering the generator.

## Released data

| Corpus | Number of samples | Family balance | Seed | Role |
|---|---:|---:|---:|---|
| Primary generated corpus (`dataset_v1`) | 50K (public corpus name) | Six balanced vessel families | 42 | P1A synthetic corpus |
| Real-reference corpus | 23 | 6 yacht, 5 OSV, 3 each other family | — | Reconstructed real-GA comparison cases |

The public corpus is named **50K**. The default generation request is 8,333 arrangements per family. Exact record and split counts are recorded in the deposited `meta.json`; use that metadata when loading the data. The 23 real-reference graphs are shipped under `data/real_ga/`. This software correction does not replace the dataset or change its split membership.

## Expected archive layout

Generation (`python generate.py`) writes datasets using the following shard-bundle structure (consult the deposit metadata for its exact packaging):

```text
dataset_v1/
├── meta.json
├── train/
│   └── shard_000/
│       ├── original.pt              list of up to 500 graphs
│       ├── companions/000000.npz    companion of graph 0
│       ├── records.json
│       ├── summary.json
│       └── COMMITTED.json           SHA-256 of every file in the shard
├── val/
│   └── ...
└── test/
    └── ...
```

The exact split and archive packaging should be read from the deposited `meta.json`, not inferred from this documentation.

## Sample object

Each generated sample is a `torch_geometric.data.Data` object with a generator-native subdivision graph plus attached voxel fields.

### Core graph tensors

| Field | Shape | Type | Meaning |
|---|---|---|---|
| `x` | `(n_zones, 19)` | float32 | Zone geometry/type input features |
| `y` | `(n_zones,)` | long | Functional class label per subdivision node |
| `edge_index` | `(2, n_directed_edges)` | long | Bidirectional graph connectivity |
| `edge_attr` | `(n_directed_edges, 7)` | float32 | Relative centroid offsets + edge-type one-hot |
| `cond` | `(16,)` | float32 | Ship family, normalized dimensions and programme signal |

The current 19-dimensional node feature count is `8 + 11 zone types`.

### Node-feature layout (`x`)

| Column | Meaning |
|---:|---|
| 0 | normalized longitudinal centroid, aft = 0 and bow = 1 |
| 1 | normalized transverse centroid, port = 0 and starboard = 1 |
| 2 | normalized vertical centroid, keel = 0 |
| 3 | available hull fraction within the zone |
| 4 | normalized longitudinal zone extent |
| 5 | normalized transverse zone extent |
| 6 | normalized vertical zone extent |
| 7 | normalized deck/tier index |
| 8–18 | one-hot `ZoneType` encoding |

### Edge-feature layout (`edge_attr`)

| Column | Meaning |
|---:|---|
| 0 | signed normalized `dx` between zone centroids |
| 1 | signed normalized `dy` between zone centroids |
| 2 | signed normalized `dz` between zone centroids |
| 3 | longitudinal edge indicator |
| 4 | vertical edge indicator |
| 5 | superstructure-to-hull edge indicator |
| 6 | transverse edge indicator |

Edges are stored in both directions.

### Conditioning vector (`cond`)

The production conditioning contract has 16 dimensions:

| Indices | Meaning |
|---|---|
| 0–5 | ship-family one-hot: Bulker, Tanker, Container (`CARGO` in code), OSV, Patrol, Yacht |
| 6 | normalized `L` |
| 7 | normalized `B` |
| 8 | normalized main-hull depth |
| 9–15 | seven programme fractions in order: engine, machinery, cargo, stores, accommodation, fuel, ballast |

The production corpus uses `cond_source = achieved`, so indices 9–15 contain **achieved** programme fractions. Sampled and effective programmes are stored separately as metadata, allowing alternative conditioning to be reconstructed later.

LCG and KG are not included in the 16-dimensional synthetic conditioning vector; targets and achieved generator diagnostics remain available under `aux_physics`. The current source marks this layout as `cond_version=2`. Dimension normalization clips to [0, 1] using L: [51, 280] m, B: [9, 54] m and main-hull D: [6, 36] m.

The legacy real-reference/CC converter uses **18 values**: the same family and dimension fields, LCG and KG at indices 9–10, then the seven programme fractions at 11–17. `validation_constants.parameter_cond_vector()` selects the common 16 fields for comparison. These are two documented formats; this correction does not convert stored graphs.

## Voxel fields

| Field | Shape | Type | Meaning |
|---|---|---|---|
| `voxel_labels` | `(64, 32, 32)` | int8 | Functional class label per stored voxel |
| `voxel_hull_mask` | `(64, 32, 32)` | bool | Represented occupied envelope mask |
| `zone_node_mask` | `(64, 32, 32)` | int16 | Graph node index corresponding to each voxel (`-1` outside zones) |
| `native_grid_shape` | tuple | — | Original grid shape before common-grid mapping |
| `target_grid_shape` | tuple | — | Stored target tensor shape |

The production main-hull grid is `64 × 32 × 24`. Superstructure adds native layers at the same physical spacing. The stored `voxel_labels` view is nearest-neighbour resampled to `64 × 32 × 32`; it is not simply padding. The companion's `compact_labels` uses a separate deck encoding and exact decode checks. Equal tensor shapes do not make these encodings interchangeable.

Use native labels, fractional occupancy, physical dimensions and preferably the original main-hull mask for [engineering evaluation](ENGINEERING.md).

## Functional label IDs

| ID | Label | Notes |
|---:|---|---|
| 0 | `VOID` | residual / structural empty space |
| 1 | `ENGINE_ROOM` | main propulsion machinery |
| 2 | `MACHINERY` | auxiliary machinery / pump / HVAC-like spaces |
| 3 | `CARGO` | cargo or mission-space class, family-dependent interpretation |
| 4 | `ACCOMMODATION` | living / accommodation spaces |
| 5 | `FUEL_TANKS` | fuel |
| 6 | `BALLAST_TANKS` | ballast |
| 7 | `STEERING_GEAR` | steering gear |
| 8 | `STORES` | stores |
| 9 | `NAVIGATION` | legacy real-reference label; not assigned in synthetic production |
| 10 | `EMPTY` | outside the represented hull envelope |

## Zone-type IDs

| ID | Zone type |
|---:|---|
| 0 | `AFT_PEAK` |
| 1 | `ENGINE_REGION` |
| 2 | `MAIN_DB` |
| 3 | `MAIN_LOWER` |
| 4 | `MAIN_UPPER` |
| 5 | `FWD_PEAK` |
| 6 | `SUPERSTRUCTURE` |
| 7 | `ENGINE_UPPER` |
| 8 | `SIDE_DB` |
| 9 | `SIDE_LOWER` |
| 10 | `SIDE_UPPER` |

## Important metadata

Each graph contains additional scalar and structured metadata. The most important fields for downstream use are:

| Field | Meaning |
|---|---|
| `ship_type` | integer `ShipType` identifier |
| `n_zones` | number of subdivision-graph nodes |
| `L`, `B`, `D`, `T`, `Cb` | sampled principal/hydrostatic parameters |
| `hull_source` | parametric or hull-template source identifier |
| `budget_sampled` | requested Stage-1 programme |
| `budget_effective` | capacity-feasible programme after projection |
| `budget_achieved` | final realised programme |
| `program` | legacy alias for `budget_effective` |
| `lcg_actual`, `kg_actual`, `gm_t` | arrangement-dependent mass/stability diagnostics |
| `aux_physics` | targets, achieved physics quantities and tracking diagnostics |
| `bulkhead_x_positions` | longitudinal bulkhead x-indices |
| `n_holds` | longitudinal hold count |
| `db_layers` | double-bottom layer count |
| `acc_min_z` | accommodation-floor threshold index |
| `deck_z_m` | physical vertical tier/deck break heights |
| `zone_bboxes` | `(n_zones, 6)` bounding boxes `[x0,x1,y0,y1,z0,z1]` |
| `deck_tiers` | vertical tier metadata |
| `mirror_ids` | mirror partner per subdivision node |
| `zone_sides` | centre/port/starboard designation |
| `gen_version` | Generator v1.0 producer label on new outputs; archived labels remain unchanged |

Companions written next to each graph are described at the top of `src/data_generator/companion.py`.

## Subdivision graph versus CCGraph

The dataset graph stored directly on each generated sample is the **generator-native subdivision graph**. Nodes correspond to the spatial zones created before functional assignment.

The paper also uses a **Connected-Compartment Graph (CCGraph)** for evaluation. In the CCGraph, adjacent voxels with the same functional class are merged into face-connected functional regions. The CCGraph is therefore a derived analysis representation and should not be confused with the native subdivision graph used by the generator and PyG sample. The `cc_aligned` track of `validate.py` rebuilds this representation from stored layouts.

## Loading a split

```python
import sys
sys.path.insert(0, "src/data_generator")
from dataset_builder import load_split

graphs = load_split("data/dataset_v1", "train")
g = graphs[0]
```

`load_split(..., with_companions=True)` also attaches the companion of each graph.

Only load serialized `.pt` / `.npz` files obtained from a trusted source. PyTorch object deserialization should not be used on untrusted files.

## Dataset caveats

- The generated samples are not independent observations of real ships; they inherit the same procedural rules and sampling assumptions.
- Requested, effective and achieved programme fractions are distinct and should not be silently substituted for one another.
- The 23 real arrangements are comparison references, not a statistical fleet sample.
- `NAVIGATION` is a retained real-reference class ID but is not actively assigned in synthetic production. The engineering evaluator accepts labels 0–8 and outside=10; it does not silently relabel real references.
- `CARGO` is both a functional label and the frozen internal ship-type identifier for the container-vessel family; context distinguishes the two.
