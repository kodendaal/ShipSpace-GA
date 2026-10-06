"""
dataset_builder.py
==================
Stage 6: Dataset Assembly
--------------------------
Runs the full Stage 1-5 pipeline at scale, applies QC filtering, attaches the
zone companion of every accepted graph (``companion.py``) and saves
train/val/test splits as shard bundles.

``generate_dataset_parallel`` splits the per-type quota over workers (each
writes ``work_dir/worker_XX``) and merges them into ``out_dir``;
``generate.py`` calls it with the settings used for the published datasets.
Hulls come from the stored masks in ``data/hull_cache`` (grid 64×32×24).

Output structure:
    dataset/
        meta.json
        train/  shard_000/  shard_001/  ...
        val/    shard_000/  ...
        test/   shard_000/  ...

    shard_NNN/
        original.pt          list of PyG graphs (up to 500)
        companions/NNNNNN.npz   zone companion of graph NNNNNN
        records.json         one row per graph (ids, checks, companion hash)
        summary.json         bundle summary
        COMMITTED.json       SHA-256 of every file in the bundle

``load_split(out_dir, split)`` reads the graphs back.

Hull QC (Stage 2) rejects: set environment variable ``SHIP_LOG_HULL_REJECTS=1``
or pass ``log_hull_rejects=True`` to ``generate_dataset`` for a printed breakdown
of hull fill fraction vs bounding-box grid (see ``hull_mask.hull_bbox_fill_fraction``).
"""

from __future__ import annotations
import copy
import os
import json
import time
import hashlib
import multiprocessing
import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from collections import Counter, defaultdict

from ship_params import (
    ShipParameterizationSampler, ShipType, validate_parameterization,
)
from hull_mask import (
    ParametricHullMask,
    STLHullMask,
    format_hull_qc_reject_log,
    validate_hull_mask,
)
from bulkhead_placement import (
    place_bulkheads, validate_bulkhead_layout,
)
from compartment_assignment import (
    assign_compartments, validate_assignment, _capacity_prefilter,
    rescale_budget_to_capacity,
    gm_min_for_type, QC_GM_MAX, QC_LCG_TOL, QC_KG_TOL,
)
from representation_converters import (
    attach_voxel_and_deck, DEFAULT_TARGET_SHAPE,
)
from graph_builder import (
    build_graph, validate_graph, HAS_PYG, N_COMP_CLASSES, N_NODE_FEATURES,
    N_EDGE_FEATURES, N_COND_DIMS, GENERATOR_VERSION,
)
from companion import CompanionError, attach_companion

if HAS_PYG:
    import torch

try:
    from tqdm import tqdm as _tqdm_factory
except ImportError:  # pragma: no cover - optional dependency
    _tqdm_factory = None  # type: ignore[misc, assignment]


def _resolve_log_hull_rejects(explicit: Optional[bool]) -> bool:
    """
    Whether to print a full hull-QC diagnostic block on Stage-2 reject.

    Set ``log_hull_rejects=True`` on ``generate_dataset``, or set environment
    variable ``SHIP_LOG_HULL_REJECTS=1`` (or ``true`` / ``yes`` / ``on``).
    """
    if explicit is not None:
        return bool(explicit)
    return os.environ.get("SHIP_LOG_HULL_REJECTS", "").strip().lower() in (
        "1", "true", "yes", "on",
    )


def _resolve_gm_qc_gate(explicit: Optional[bool]) -> bool:
    """
    Whether GM bounds reject samples at Stage 4 QC.

    Default **off** (GM stored as ``gm_feasible`` label). Set
    ``gm_qc_gate=True`` on ``generate_dataset``, or ``SHIP_GM_QC_GATE=1``
    to reject samples outside the GM bounds instead.
    """
    if explicit is not None:
        return bool(explicit)
    return os.environ.get("SHIP_GM_QC_GATE", "").strip().lower() in (
        "1", "true", "yes", "on",
    )


def _resolve_lcg_kg_qc_gate(explicit: Optional[bool]) -> bool:
    """
    Whether LCG/KG target-tracking rejects samples at Stage 4 QC.

    Default **off** (store ``lcg_tracking_*`` / ``kg_tracking_*`` labels).
    Set ``lcg_kg_qc_gate=True`` on ``generate_dataset``, or
    ``SHIP_LCG_KG_QC_GATE=1`` to reject samples outside the tolerances instead.
    """
    if explicit is not None:
        return bool(explicit)
    return os.environ.get("SHIP_LCG_KG_QC_GATE", "").strip().lower() in (
        "1", "true", "yes", "on",
    )


def _use_tqdm_bar(use_tqdm: Optional[bool]) -> bool:
    """
    Whether to show an in-loop tqdm bar inside ``generate_dataset``.

    ``use_tqdm=None`` (default): enable only in the main interpreter process
    (not in ``ProcessPoolExecutor`` workers), and only if tqdm is installed.
    """
    if _tqdm_factory is None:
        return False
    if use_tqdm is False:
        return False
    if use_tqdm is True:
        return True
    return multiprocessing.current_process().name == "MainProcess"


def _source_tree_sha256() -> str:
    """Deterministic SHA-256 over the generator Python source tree."""
    root = Path(__file__).resolve().parent
    h = hashlib.sha256()
    for source in sorted(root.glob("*.py"), key=lambda q: q.name):
        h.update(source.name.encode("utf-8"))
        h.update(b"\0")
        h.update(source.read_bytes())
        h.update(b"\0")
    return h.hexdigest()


# ─────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────

SPLIT_RATIOS = {"train": 0.80, "val": 0.10, "test": 0.10}
SHARD_SIZE = 500
ENCODING = "dataset_v1"
BUNDLE_SCHEMA = "v542_enc2_reference_bundle_1"

DEFAULT_TYPE_WEIGHTS = {
    ShipType.BULKER:  2.0,
    ShipType.TANKER:  2.0,
    ShipType.CARGO:   2.0,
    ShipType.OSV:     2.5,
    ShipType.PATROL: 2.0,
    ShipType.YACHT:   2.5,
}


def _split_quota(total: int, n_parts: int) -> List[int]:
    """Split *total* into *n_parts* integers that sum to *total*."""
    if n_parts < 1:
        return [total]
    base, rem = divmod(total, n_parts)
    return [base + (1 if i < rem else 0) for i in range(n_parts)]

from stl_mask_cache import (
    PRODUCTION_GRID, cache_populated, resolve_hull_cache_dir, make_stl_builder,
    ensure_hull_cache,
)


# QC tolerances (moderate tier)
QC_BUDGET_TOL = 0.12
QC_GM_MIN = 0.0    # default GM label threshold (type overrides in GM_MIN_BY_TYPE)


def _normalise_type_weights(weights: np.ndarray) -> np.ndarray:
    """Normalise non-negative type weights to probabilities."""
    weights = np.asarray(weights, dtype=float)
    weights = np.clip(weights, 0.0, None)
    total = weights.sum()
    if total <= 0:
        return np.full(len(weights), 1.0 / max(len(weights), 1), dtype=float)
    return weights / total


def _adaptive_type_weights(
    types: List[ShipType],
    type_counts: Counter,
    smoothing: float = 1.0,
) -> np.ndarray:
    """
    Inverse-frequency sampling weights from accepted sample counts.
    Lower-count ship types get higher probability.
    """
    counts = np.array([type_counts[t.name] for t in types], dtype=float)
    inv = 1.0 / (counts + smoothing)
    return _normalise_type_weights(inv)


def _graph_physics_errors(graph: Any) -> Tuple[float, float]:
    """
    LCG/KG tracking error for dataset stats.

    cond stores budgets at cond[9:16]; CG targets live in aux_physics.
    """
    if HAS_PYG:
        aux = getattr(graph, "aux_physics", None) or {}
        lcg_actual = float(graph.lcg_actual)
        kg_actual = float(graph.kg_actual)
    else:
        aux = graph.get("aux_physics", {}) or {}
        lcg_actual = float(graph["lcg_actual"])
        kg_actual = float(graph["kg_actual"])
    target_lcg = float(aux.get("target_lcg", lcg_actual))
    target_kg = float(aux.get("target_kg", kg_actual))
    return abs(lcg_actual - target_lcg), abs(kg_actual - target_kg)


# ─────────────────────────────────────────────────────────────────
# Single sample generation
# ─────────────────────────────────────────────────────────────────

def generate_one_sample(
    sampler: ShipParameterizationSampler,
    hull_builder: ParametricHullMask,
    ship_type: ShipType,
    seed: int,
    stl_builder: Optional[STLHullMask] = None,
    stl_paths: Optional[Dict[ShipType, List[Path]]] = None,
    rng: Optional[np.random.Generator] = None,
    n_variants: int = 5,
    log_hull_rejects: bool = False,
    target_shape: Tuple[int, int, int] = DEFAULT_TARGET_SHAPE,
    cond_source: str = "achieved",
    gm_qc_gate: Optional[bool] = None,
    lcg_kg_qc_gate: Optional[bool] = None,
) -> Tuple[Optional[List[Any]], str, int]:
    """
    Run Stages 1-5 for one ship instance with n_variants assignment
    variants from the same hull/bulkhead layout.

    Returns (graphs_list, status, n_tried) where n_tried is the number
    of variant assignments actually attempted (0 if failed before S4).

    log_hull_rejects : if True, print ``format_hull_qc_reject_log`` on S2v fail.
    """
    if rng is None:
        rng = np.random.default_rng(seed)

    # Type-dependent GM threshold (used only when the GM gate is on)
    gm_min = gm_min_for_type(ship_type, QC_GM_MIN)
    do_gm_gate = _resolve_gm_qc_gate(gm_qc_gate)
    do_lcg_kg_gate = _resolve_lcg_kg_qc_gate(lcg_kg_qc_gate)

    # Stage 1
    p = sampler.sample(ship_type)
    ok, warns = validate_parameterization(p)
    if not ok:
        return None, f"S1:{warns[0][:50]}", 0
    sampled_params = copy.deepcopy(p)

    # Stage 2
    try:
        if (stl_builder and stl_paths
                and ship_type in stl_paths and stl_paths[ship_type]):
            stl_path = rng.choice(stl_paths[ship_type])
            hull_result = stl_builder.build(p, stl_path)
        else:
            hull_result = hull_builder.build(p)
    except Exception as e:
        return None, f"S2:{str(e)[:50]}", 0

    ok, warns = validate_hull_mask(hull_result)
    if not ok:
        if log_hull_rejects:
            print(
                format_hull_qc_reject_log(ship_type.name, hull_result, warns),
                flush=True,
            )
        return None, f"S2v:{warns[0][:50]}", 0

    # Stage 3
    try:
        layout = place_bulkheads(hull_result, seed=seed)
    except Exception as e:
        return None, f"S3:{str(e)[:50]}", 0

    ok, warns = validate_bulkhead_layout(layout)
    if not ok:
        return None, f"S3v:{warns[0][:50]}", 0

    # Stage 3.5 — project sampled budget onto realised-geometry feasibility.
    # p.budget becomes the EFFECTIVE budget (S4 + QC + cond); the raw draw is
    # kept in p.budget_sampled for analysis.
    rescale_budget_to_capacity(layout, p)

    cap_err = _capacity_prefilter(layout, p)
    if cap_err is not None:
        return None, f"S4cap:{cap_err[:45]}", 0

    # Stage 4 + 5: generate n_variants assignment variants from the
    # same hull/bulkhead layout.  Each variant uses a different seed
    # to produce diverse arrangements for the same geometry.
    #
    # Diversity strategy:
    #   - Variant 0 uses moderate noise (0.25) as the "baseline"
    #   - Variants 1+ use elevated noise (0.50) to explore alternative
    #     arrangements via zone-order shuffling and score perturbation
    #   - Exact-duplicate variants are rejected by fingerprinting zone
    #     assignments (prevents identical local-minimum convergence)
    #   - Each variant gets a deep copy of the base layout so that
    #     in-place mutations (e.g. SS VOID excision) don't leak
    #     between variants.
    hull_id = f"{seed}_{ship_type.name}"
    results = []
    seen_fingerprints: set = set()
    n_tried = 0
    last_fail = "S4v:all_variants_failed"

    for v in range(n_variants):
        n_tried += 1
        variant_seed = seed * 1000 + v

        if ship_type == ShipType.PATROL:
            variant_noise = 0.25
        else:
            variant_noise = 0.25 if v == 0 else 0.50

        # Deep-copy the base layout so assign_compartments can
        # mutate it (SS VOID excision, zone_mask edits) without
        # affecting other variants.
        layout_v = copy.deepcopy(layout)

        try:
            assignment = assign_compartments(
                layout_v, seed=variant_seed, budget_noise=variant_noise,
            )
        except Exception as e:
            last_fail = f"S4x:{type(e).__name__}:{str(e)[:40]}"
            continue

        ok, warns = validate_assignment(
            assignment,
            budget_tol=QC_BUDGET_TOL,
            lcg_tol=QC_LCG_TOL,
            kg_tol=QC_KG_TOL,
            gm_min=gm_min,
            gm_max=QC_GM_MAX,
            gm_gate=do_gm_gate,
            lcg_kg_gate=do_lcg_kg_gate,
        )
        if not ok:
            last_fail = f"S4v:{warns[0].strip()[:50]}"
            continue

        # Deduplication: reject variants with identical zone assignments.
        fp = "|".join(
            f"{zid}:{comp.value}"
            for zid, comp in sorted(assignment.assignments.items())
        )
        if fp in seen_fingerprints:
            last_fail = "S4v:duplicate_variant"
            continue
        seen_fingerprints.add(fp)

        try:
            graph = build_graph(
                assignment,
                cond_source=cond_source,
                lcg_kg_qc_gate=do_lcg_kg_gate,
            )
        except Exception as e:
            last_fail = f"S5x:{type(e).__name__}:{str(e)[:40]}"
            continue

        attach_voxel_and_deck(graph, assignment, target_shape=target_shape)

        ok, warns = validate_graph(graph)
        if ok:
            try:
                attach_companion(graph, assignment, sampled_params, hull_seed=seed)
            except CompanionError as exc:
                last_fail = f"S6:{str(exc)[:50]}"
                continue
            # Attach variant provenance metadata for group-splitting
            # and downstream traceability.
            if HAS_PYG:
                graph.hull_id = hull_id
                graph.variant_id = v
                graph.variant_seed = variant_seed
                graph.variant_noise = variant_noise
                graph.assignment_fp = fp
            else:
                graph["hull_id"] = hull_id
                graph["variant_id"] = v
                graph["variant_seed"] = variant_seed
                graph["variant_noise"] = variant_noise
                graph["assignment_fp"] = fp
            results.append(graph)
        else:
            last_fail = f"S5v:{warns[0].strip()[:50]}"

    if results:
        return results, "OK", n_tried
    else:
        return None, last_fail, n_tried


# ─────────────────────────────────────────────────────────────────
# Main generation loop
# ─────────────────────────────────────────────────────────────────

def generate_dataset(
    n_target: int = 10000,
    out_dir: str = "./dataset",
    seed: int = 42,
    voxel_size: float = 3.0,
    type_weights: Optional[Dict[ShipType, float]] = None,
    stl_paths: Optional[Dict[ShipType, List[Path]]] = None,
    max_attempts_factor: float = 3.0,
    log_every: int = 500,
    log_hull_rejects: Optional[bool] = None,
    use_tqdm: Optional[bool] = None,
    adaptive_reweight: bool = True,
    reweight_every: int = 1000,
    reweight_until: int = 100000,
    reweight_smoothing: float = 1.0,
    n_variants: int = 5,
    fixed_ship_type: Optional[ShipType] = None,
    per_type: Optional[int] = None,
    ship_types: Optional[List[ShipType]] = None,
    grid_counts: Optional[Tuple[int, int, int]] = None,
    cache_dir: Optional[str | Path] = None,
    use_hull_cache: bool = True,
    precompute_hull_cache: bool = True,
    cond_source: str = "achieved",
    gm_qc_gate: Optional[bool] = None,
    lcg_kg_qc_gate: Optional[bool] = None,
) -> Dict:
    """
    Generate a complete dataset of ship layout graphs.

    Parameters
    ----------
    n_target : int
        Number of passing samples to accumulate (ignored when ``per_type`` is set).
    per_type : int, optional
        Generate exactly this many passing graphs per ship type in ``ship_types``
        (default: all six types).  Total target becomes ``per_type × len(types)``.
    out_dir : str
        Output directory for shards and meta.json.
    seed : int
        Master seed for reproducibility.
        When running parallel workers with ``fixed_ship_type``, use distinct
        seeds per worker (e.g. ``seed=42 + i * 100_000``) to avoid
        correlated random states across types.
    voxel_size : float
        Voxel size in metres (default 3.0).
    type_weights : dict, optional
        Sampling weights per ShipType.
    stl_paths : dict, optional
        {ShipType: [Path, ...]} for STL hull masks.
        If None, all samples use parametric hulls.
    max_attempts_factor : float
        Stop if total attempts exceed n_target * factor.
        Consider raising to 5.0+ for single-type runs on harder types
        (e.g. PATROL) which have lower QC pass rates.
    log_every : int
        When tqdm is off, print progress every N attempts.
    log_hull_rejects : bool, optional
        If True, print a detailed block (fill fraction, grid, source) whenever
        Stage-2 hull QC fails.  If None, use environment variable
        ``SHIP_LOG_HULL_REJECTS`` (1 / true / yes / on enables).
    use_tqdm : bool, optional
        If True, show a tqdm bar for graphs collected (requires tqdm).
        If False, never show tqdm (use ``log_every`` prints instead).
        If None (default), show tqdm only in the main process (disabled
        automatically inside ``ProcessPoolExecutor`` workers).
    adaptive_reweight : bool, optional
        If True, update ship-type sampling weights every ``reweight_every``
        accepted samples using inverse frequency weighting.
    reweight_every : int
        Accepted-sample interval between adaptive weight updates.
    reweight_until : int
        Stop adaptive reweighting after this many accepted samples.
    reweight_smoothing : float
        Small value added to counts before inversion.
    fixed_ship_type : ShipType, optional
        If provided, only generate this ship type.  Adaptive reweighting
        is automatically disabled.  When ``stl_paths`` is set but contains
        no entries for the fixed type, parametric hulls are used with a
        printed note.
    grid_counts : tuple, optional
        Anisotropic grid ``(Nx, Ny, Nz_hull)`` for STL hull mode.  Must
        match the grid used when warming ``CachedSTLHullMask`` (default
        ``PRODUCTION_GRID`` = 64×32×24).  Ignored when ``stl_paths`` is
        None (isotropic ``voxel_size`` sampling is used instead).
    cache_dir : path, optional
        Directory for ``CachedSTLHullMask`` disk cache.  Defaults to
        ``data/hull_cache`` under this package when ``use_hull_cache`` is
        True and ``stl_paths`` is set.
    use_hull_cache : bool
        When True and ``stl_paths`` is set, use ``CachedSTLHullMask``
        instead of re-voxelising every sample.
    precompute_hull_cache : bool
        If True, warm the hull cache (once per process) before generation
        when it is empty.

    Returns
    -------
    dict : metadata (also saved as meta.json)
    """
    t_start = time.time()
    if any(Path(out_dir).glob("*/shard_*")):
        raise FileExistsError(f"{out_dir} already holds shards; use a new output folder")
    do_log_hull = _resolve_log_hull_rejects(log_hull_rejects)
    do_gm_gate = _resolve_gm_qc_gate(gm_qc_gate)
    do_lcg_kg_gate = _resolve_lcg_kg_qc_gate(lcg_kg_qc_gate)
    show_pbar = _use_tqdm_bar(use_tqdm)
    # Pool workers: avoid interleaved ``print`` lines on stdout while the driver
    # process shows a tqdm bar (``log_every`` still applies in the main process).
    is_pool_worker = multiprocessing.current_process().name != "MainProcess"
    log_interval = max(log_every, 10**9) if (is_pool_worker and not show_pbar) else log_every

    stratified = per_type is not None and fixed_ship_type is None
    quota_types = ship_types if ship_types is not None else list(ShipType)
    if stratified:
        n_target = per_type * len(quota_types)
        adaptive_reweight = False

    if type_weights is None:
        type_weights = DEFAULT_TYPE_WEIGHTS

    effective_grid = tuple(grid_counts) if grid_counts else PRODUCTION_GRID
    resolved_cache: Optional[Path] = None
    hull_mode = "parametric"

    if stl_paths:
        hull_mode = "stl_cached" if use_hull_cache else "stl"
        sampler = ShipParameterizationSampler(
            grid_counts=effective_grid, rng_seed=seed,
        )
        hull_builder = ParametricHullMask()
        if use_hull_cache:
            resolved_cache = resolve_hull_cache_dir(cache_dir)
            if precompute_hull_cache:
                ensure_hull_cache(stl_paths, cache_dir=resolved_cache, grid_counts=effective_grid)
            elif cache_populated(resolved_cache):
                n_masks = len(list(resolved_cache.glob("*.npy")))
                print(f"Hull cache: {resolved_cache} ({n_masks} masks)", flush=True)
            else:
                print(
                    f"WARNING: hull cache empty at {resolved_cache} "
                    f"and precompute_hull_cache=False",
                    flush=True,
                )
            stl_builder = make_stl_builder(resolved_cache)
        else:
            stl_builder = make_stl_builder(None)
    else:
        sampler = ShipParameterizationSampler(voxel_size=voxel_size, rng_seed=seed)
        hull_builder = ParametricHullMask()
        stl_builder = None

    rng = np.random.default_rng(seed + 7777)

    types = quota_types if stratified else list(ShipType)
    weights = _normalise_type_weights(
        np.array([type_weights.get(t, 1.0) for t in types], dtype=float)
    )
    next_reweight_at = (
        reweight_every
        if adaptive_reweight and reweight_every > 0 and not stratified
        else None
    )

    # ── Fixed ship type override ──────────────────────────────────
    if fixed_ship_type is not None:
        types = [fixed_ship_type]
        weights = np.array([1.0], dtype=float)
        next_reweight_at = None          # reweighting is meaningless for one type
        if stl_paths is not None:
            fixed_stls = stl_paths.get(fixed_ship_type, [])
            if not fixed_stls:
                print(f"  Note: no STL paths for {fixed_ship_type.name}, "
                      f"using parametric hulls only.")

    reweight_history: List[Dict[str, Any]] = []

    graphs: List[Any] = []
    type_counts: Counter = Counter()
    fail_counts: Counter = Counter()
    n_attempts = 0
    attempts_with_success = 0
    variant_trials = 0
    variant_passes = 0
    attempt_factor = max(max_attempts_factor, 5.0) if stratified else max_attempts_factor
    max_attempts = int(n_target * attempt_factor)

    lcg_errors: List[float] = []
    kg_errors: List[float] = []
    zone_counts: List[int] = []

    if stl_paths:
        print(
            f"Generating dataset: target={n_target}  seed={seed}  "
            f"grid={effective_grid}  voxel_target={DEFAULT_TARGET_SHAPE}  "
            f"n_variants={n_variants}",
        )
        cache_note = (
            str(resolved_cache) if resolved_cache is not None else "off (uncached STL)"
        )
        print(f"Hull: {hull_mode}  cache={cache_note}  Output: {out_dir}")
    else:
        print(
            f"Generating dataset: target={n_target}  seed={seed}  "
            f"voxel={voxel_size}m  voxel_target={DEFAULT_TARGET_SHAPE}  "
            f"n_variants={n_variants}",
        )
        print(f"Hull: parametric only  Output: {out_dir}")
    if fixed_ship_type is not None:
        print(f"Fixed ship type: {fixed_ship_type.name}")
    if stratified:
        print(f"Stratified: {per_type} per type × {len(types)} types")
    print(
        f"Hull reject logging: {'ON' if do_log_hull else 'OFF'}  "
        f"(set log_hull_rejects=True or SHIP_LOG_HULL_REJECTS=1)",
    )
    print(
        f"Adaptive reweighting: {'ON' if adaptive_reweight else 'OFF'}  "
        f"(every {reweight_every} accepted until {reweight_until})",
    )
    print(
        f"GM QC gate: {'ON (reject)' if do_gm_gate else 'OFF (gm_feasible label)'}  "
        f"(set gm_qc_gate=True or SHIP_GM_QC_GATE=1 to restore)",
    )
    print(
        f"LCG/KG QC gate: {'ON (reject)' if do_lcg_kg_gate else 'OFF (tracking labels)'}  "
        f"(set lcg_kg_qc_gate=True or SHIP_LCG_KG_QC_GATE=1 to restore)",
    )
    print("-" * 60)

    pbar = None
    prev_n_graphs = 0
    if show_pbar and _tqdm_factory is not None:
        pbar = _tqdm_factory(
            total=n_target,
            desc=f"graphs {Path(out_dir).name}",
            unit="graph",
            dynamic_ncols=True,
        )

    while len(graphs) < n_target and n_attempts < max_attempts:
        if stratified:
            pending = [st for st in types if type_counts[st.name] < per_type]
            if not pending:
                break
            st = pending[rng.integers(len(pending))]
        else:
            st = types[rng.choice(len(types), p=weights)]
        sample_seed = seed + n_attempts

        result, status, n_tried = generate_one_sample(
            sampler=sampler,
            hull_builder=hull_builder,
            ship_type=st,
            seed=sample_seed,
            stl_builder=stl_builder,
            stl_paths=stl_paths,
            rng=rng,
            n_variants=n_variants,
            log_hull_rejects=do_log_hull,
            target_shape=DEFAULT_TARGET_SHAPE,
            cond_source=cond_source,
            gm_qc_gate=do_gm_gate,
            lcg_kg_qc_gate=do_lcg_kg_gate,
        )

        n_attempts += 1
        variant_trials += n_tried

        if result is not None:
            attempts_with_success += 1
            variant_passes += len(result)
            for graph in result:
                if stratified and type_counts[st.name] >= per_type:
                    break
                if len(graphs) >= n_target:
                    break
                graphs.append(graph)
                type_counts[st.name] += 1

                if HAS_PYG:
                    zone_counts.append(graph.x.shape[0])
                    lcg_e, kg_e = _graph_physics_errors(graph)
                    lcg_errors.append(lcg_e)
                    kg_errors.append(kg_e)
                else:
                    zone_counts.append(graph["x"].shape[0])
                    lcg_e, kg_e = _graph_physics_errors(graph)
                    lcg_errors.append(lcg_e)
                    kg_errors.append(kg_e)

                while (
                    next_reweight_at is not None
                    and len(graphs) >= next_reweight_at
                    and len(graphs) <= reweight_until
                ):
                    weights = _adaptive_type_weights(
                        types=types,
                        type_counts=type_counts,
                        smoothing=reweight_smoothing,
                    )
                    reweight_history.append({
                        "n_passed": len(graphs),
                        "n_attempts": n_attempts,
                        "type_counts": {t.name: int(type_counts[t.name]) for t in types},
                        "weights": {t.name: float(w) for t, w in zip(types, weights)},
                    })
                    msg = (
                        f"Reweighted at {len(graphs)} accepted "
                        f"(attempt {n_attempts})"
                    )
                    if pbar is not None:
                        pbar.write(msg)
                    else:
                        print(msg)
                    next_reweight_at += reweight_every
                    if next_reweight_at > reweight_until:
                        next_reweight_at = None
        else:
            fail_counts[status] += 1
            fail_counts[f"_type:{st.name}"] += 1

        n_pass = len(graphs)
        if pbar is not None:
            if n_pass > prev_n_graphs:
                pbar.update(n_pass - prev_n_graphs)
                prev_n_graphs = n_pass
            attempt_success_rate = attempts_with_success / max(n_attempts, 1)
            to_next_reweight = (
                "-"
                if next_reweight_at is None
                else max(next_reweight_at - n_pass, 0)
            )
            pbar.set_postfix(
                att=n_attempts,
                att_cap=max_attempts,
                ok=f"{100 * attempt_success_rate:.1f}%",
                rw_in=to_next_reweight,
                refresh=False,
            )
        elif n_attempts % log_interval == 0 or n_pass == n_target:
            attempt_success_rate = attempts_with_success / max(n_attempts, 1)
            elapsed = time.time() - t_start
            eta = (elapsed / max(n_pass, 1)) * (n_target - n_pass) if n_pass > 0 else 0
            to_next_reweight = (
                "-"
                if next_reweight_at is None
                else max(next_reweight_at - n_pass, 0)
            )
            print(f"  [{n_attempts:>6}] {n_pass:>5}/{n_target}  "
                  f"attempt_pass={100*attempt_success_rate:.1f}%  "
                  f"rw_in={to_next_reweight}  "
                  f"elapsed={elapsed:.0f}s  eta={eta:.0f}s")

    if pbar is not None:
        pbar.close()

    elapsed = time.time() - t_start
    n_pass = len(graphs)

    print("-" * 60)
    print(
        f"Done: {n_pass}/{n_target} collected; "
        f"attempt_pass={100*attempts_with_success/max(n_attempts,1):.1f}% "
        f"({attempts_with_success}/{n_attempts}) in {elapsed:.1f}s"
    )
    print(
        f"Variant pass: {variant_passes}/{variant_trials} "
        f"({100*variant_passes/max(variant_trials,1):.1f}%)"
    )

    if fail_counts:
        print(f"\nTop failures:")
        for reason, cnt in sorted(fail_counts.items(), key=lambda x: -x[1])[:10]:
            print(f"  {cnt:>5}x  {reason}")

    # Split and save
    cache_sha256 = hull_cache_digests(resolved_cache) if resolved_cache else {}
    splits = _stratified_split(graphs, SPLIT_RATIOS, rng=np.random.default_rng(seed))
    _save_splits(splits, out_dir, cache_sha256)

    # Metadata
    meta = {
        "n_target": n_target,
        "n_attempts": n_attempts,
        "n_passed": n_pass,
        "attempts_with_success": attempts_with_success,
        "attempt_success_rate": round(attempts_with_success / max(n_attempts, 1), 4),
        "variant_trials": variant_trials,
        "variant_passes": variant_passes,
        "variant_pass_rate": round(variant_passes / max(variant_trials, 1), 4),
        "yield_per_attempt": round(n_pass / max(n_attempts, 1), 4),
        "time_seconds": round(elapsed, 1),
        "seed": seed,
        "generator_version": GENERATOR_VERSION,
        "encoding": ENCODING,
        "source_tree_sha256": _source_tree_sha256(),
        "voxel_size": voxel_size,
        "hull_mode": hull_mode,
        "grid_counts": list(effective_grid) if stl_paths else None,
        "hull_cache_sha256": cache_sha256,
        "n_variants": n_variants,
        "cond_source": cond_source,
        "gm_qc_gate": do_gm_gate,
        "lcg_kg_qc_gate": do_lcg_kg_gate,
        "qc": {"budget": QC_BUDGET_TOL, "lcg": QC_LCG_TOL,
               "kg": QC_KG_TOL, "gm_min": QC_GM_MIN, "gm_max": QC_GM_MAX,
               "gm_gate": do_gm_gate, "lcg_kg_gate": do_lcg_kg_gate},
        "splits": {k: len(v) for k, v in splits.items()},
        "type_counts": dict(type_counts),
        "adaptive_reweight": {
            "enabled": adaptive_reweight,
            "reweight_every": reweight_every,
            "reweight_until": reweight_until,
            "reweight_smoothing": reweight_smoothing,
            "history": reweight_history,
        },
        "failures": dict(fail_counts),
        "stats": {
            "zones_mean": round(float(np.mean(zone_counts)), 1) if zone_counts else 0,
            "zones_std": round(float(np.std(zone_counts)), 1) if zone_counts else 0,
            "lcg_err_mean": round(float(np.mean(lcg_errors)), 4) if lcg_errors else 0,
            "lcg_err_p95": round(float(np.percentile(lcg_errors, 95)), 4) if lcg_errors else 0,
            "kg_err_mean": round(float(np.mean(kg_errors)), 4) if kg_errors else 0,
            "kg_err_p95": round(float(np.percentile(kg_errors, 95)), 4) if kg_errors else 0,
        },
        "shard_size": SHARD_SIZE,
        "n_node_features": N_NODE_FEATURES,
        "n_edge_features": N_EDGE_FEATURES,
        "n_cond_dims": N_COND_DIMS,
        "n_comp_classes": N_COMP_CLASSES,
        "target_grid_shape": DEFAULT_TARGET_SHAPE,
        "fixed_ship_type": (fixed_ship_type.name if fixed_ship_type is not None else None),
        "per_type": per_type if stratified else None,
        "split_by": "hull_id" if n_variants > 1 else "graph",
        "extra_metadata_fields": [
            "program", "budget_sampled", "budget_effective", "budget_achieved",
            "gen_version", "bulkhead_x_positions", "n_holds",
            "db_layers", "acc_min_z", "nz_hull", "deck_z_m",
            "zone_bboxes", "deck_tiers",
            "hull_id", "variant_id", "variant_seed",
            "variant_noise", "assignment_fp",
        ],
    }

    meta_path = Path(out_dir) / "meta.json"
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\nSaved meta.json -> {meta_path}")

    print(f"\n{'='*60}")
    print(f"Dataset Summary")
    print(f"{'='*60}")
    for k, v in splits.items():
        print(f"  {k:6s}: {len(v):>5} samples")
    print(f"  {'total':6s}: {n_pass:>5} samples")
    print(f"  Zones/graph: {meta['stats']['zones_mean']:.1f} "
          f"+/- {meta['stats']['zones_std']:.1f}")
    print(f"  LCG error:   mean={meta['stats']['lcg_err_mean']:.4f}  "
          f"p95={meta['stats']['lcg_err_p95']:.4f}")
    print(f"  Type distribution:")
    for t in sorted(type_counts):
        print(f"    {t:12s}: {type_counts[t]:>5}")
    print(f"{'='*60}")

    return meta


# ─────────────────────────────────────────────────────────────────
# Stratified splitting
# ─────────────────────────────────────────────────────────────────

def _stratified_split(
    graphs: List[Any],
    ratios: Dict[str, float],
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, List[Any]]:
    """
    Split graphs into train/val/test, stratified by ship_type.

    When graphs carry a ``hull_id`` attribute (set by ``generate_one_sample``
    when ``n_variants > 1``), all variants from the same hull are kept in
    the same split to prevent data leakage.  The split unit is the hull
    group, not the individual graph.

    Falls back to per-graph splitting when ``hull_id`` is absent (backward
    compatible with datasets generated before this change).
    """
    if rng is None:
        rng = np.random.default_rng(0)

    def _get_hull_id(g: Any) -> Optional[str]:
        if HAS_PYG:
            return getattr(g, "hull_id", None)
        return g.get("hull_id", None)

    def _get_ship_type(g: Any) -> int:
        if HAS_PYG:
            return g.ship_type
        return g["ship_type"]

    # Group graphs by hull_id.  Graphs without hull_id each become
    # their own single-element group (backward-compatible).
    hull_groups: Dict[str, List[Any]] = defaultdict(list)
    _anon_counter = 0
    for g in graphs:
        hid = _get_hull_id(g)
        if hid is None:
            hull_groups[f"__anon_{_anon_counter}"] = [g]
            _anon_counter += 1
        else:
            hull_groups[hid].append(g)

    # Organise hull groups by ship type (take type from first graph
    # in each group — all graphs in a hull group share the same type).
    type_to_hull_ids: Dict[int, List[str]] = defaultdict(list)
    for hid, group in hull_groups.items():
        st = _get_ship_type(group[0])
        type_to_hull_ids[st].append(hid)

    splits: Dict[str, List[Any]] = {k: [] for k in ratios}
    ratio_keys = list(ratios.keys())

    for st, hull_ids in type_to_hull_ids.items():
        shuffled = rng.permutation(len(hull_ids))
        n_groups = len(hull_ids)
        cursor = 0
        for i, key in enumerate(ratio_keys):
            if i == len(ratio_keys) - 1:
                size = n_groups - cursor
            else:
                size = int(round(ratios[key] * n_groups))
            for idx in shuffled[cursor:cursor + size]:
                splits[key].extend(hull_groups[hull_ids[idx]])
            cursor += size

    for key in splits:
        rng.shuffle(splits[key])

    return splits


# ─────────────────────────────────────────────────────────────────
# Serialisation
# ─────────────────────────────────────────────────────────────────

def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def hull_cache_digests(cache_dir: Optional[str | Path] = None) -> Dict[str, str]:
    """SHA-256 of every stored hull mask, keyed by file name."""
    root = resolve_hull_cache_dir(cache_dir)
    return {path.name: _sha256(path) for path in sorted(root.glob("*.npy"))}


def _take_companion(graph: Any) -> Tuple[bytes, Dict[str, Any]]:
    """Detach the companion so the saved record holds only the graph attributes."""
    if isinstance(graph, dict):
        payload = graph.pop("companion", None)
    else:
        payload = getattr(graph, "companion", None)
        if payload is not None:
            del graph.companion
    if payload is None:
        raise RuntimeError("accepted sample has no companion")
    return payload


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_bundle(
    bundle: Path,
    graphs: List[Any],
    split: str,
    dataset: str,
    cache_sha256: Dict[str, str],
) -> None:
    """Write one shard bundle: original.pt, companions, records, summary, commit."""
    bundle.mkdir(parents=True)
    companions = [_take_companion(graph) for graph in graphs]
    # torch.save names the archive after the file stem, so the shard is written
    # under its shard name to keep the bytes independent of the bundle layout.
    staged = bundle / f"{bundle.name}.pt"
    torch.save(graphs, str(staged))
    staged.rename(bundle / "original.pt")
    (bundle / "companions").mkdir()

    rows = []
    for index, (graph, (data, record)) in enumerate(zip(graphs, companions)):
        name = f"companions/{index:06d}.npz"
        (bundle / name).write_bytes(data)
        rows.append({
            "index": index,
            "sample_id": f"{dataset}:{split}:{graph.hull_id}:v{int(graph.variant_id)}",
            "source_uid": f"{dataset}/{split}/{bundle.name}/{index:04d}",
            "hull_id": graph.hull_id,
            "variant_id": int(graph.variant_id),
            "locked_split": split,
            "original_attribute_names": sorted(graph.keys()),
            **{key: value for key, value in record.items() if key != "companion_sha256"},
            "companion": name,
            "companion_sha256": record["companion_sha256"],
        })
    _write_json(bundle / "records.json", rows)
    _write_json(bundle / "summary.json", {
        "schema": BUNDLE_SCHEMA,
        "encoding": ENCODING,
        "original_shard": f"{bundle.name}.pt",
        "records": len(rows),
        "locked_split": split,
        "input_sha256": _sha256(bundle / "original.pt"),
        "source_tree_sha256": _source_tree_sha256(),
        "cache_sha256": cache_sha256,
        "shape": [64, 32, 32],
        "functional_exact_records": sum(r["functional_changed_cells"] == 0 for r in rows),
        "native_instance_maps_preserved": len(rows),
        "side_zones_preserved": sum(r["side_zones"] for r in rows),
        "records_with_mixed_compact_instances": sum(r["mixed_instance_columns"] > 0 for r in rows),
        "reordered_or_removed_records": 0,
        "original_bytes": (bundle / "original.pt").stat().st_size,
        "companion_bytes": sum(len(data) for data, _ in companions),
    })
    artifacts = {
        path.relative_to(bundle).as_posix(): _sha256(path)
        for path in sorted(bundle.rglob("*")) if path.is_file()
    }
    _write_json(bundle / "COMMITTED.json", {
        "artifacts": artifacts,
        "input_sha256": artifacts["original.pt"],
    })


def _save_splits(
    splits: Dict[str, List[Any]],
    out_dir: str,
    cache_sha256: Dict[str, str],
) -> None:
    """Save every split as shard bundles with one companion per record."""
    if not HAS_PYG:
        raise RuntimeError("saving shards requires torch and torch_geometric")
    out_path = Path(out_dir)

    for split_name, graphs in splits.items():
        split_dir = out_path / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        if any(split_dir.glob("shard_*")):
            raise FileExistsError(f"{split_dir} already holds shards; use a new output folder")
        if not graphs:
            print(f"  {split_name}: 0 samples")
            continue

        n_shards = max(1, (len(graphs) + SHARD_SIZE - 1) // SHARD_SIZE)
        for shard_idx in range(n_shards):
            start = shard_idx * SHARD_SIZE
            end = min(start + SHARD_SIZE, len(graphs))
            _write_bundle(
                split_dir / f"shard_{shard_idx:03d}",
                graphs[start:end],
                split_name,
                out_path.resolve().name,
                cache_sha256,
            )

        print(f"  {split_name}: {len(graphs)} samples -> {n_shards} shard bundle(s)")


# ─────────────────────────────────────────────────────────────────
# Loading utility
# ─────────────────────────────────────────────────────────────────

COMPANION_RECORD_KEYS = (
    "native_z", "slot_count", "native_zones", "side_zones",
    "mixed_instance_columns", "functional_changed_cells",
    "physics_replay_absolute_difference", "native_zone_sha256", "companion_sha256",
)


def shard_files(out_dir: str | Path, split: str) -> List[Path]:
    """Graph shards of one split, in order (bundle or single-file layout)."""
    split_dir = Path(out_dir) / split
    if not split_dir.exists():
        raise FileNotFoundError(f"Not found: {split_dir}")
    bundles = sorted(split_dir.glob("shard_*/original.pt"))
    return bundles if bundles else sorted(split_dir.glob("shard_*.pt"))


def load_split(out_dir: str, split: str = "train", with_companions: bool = False) -> List[Any]:
    """Load all graphs of one split; optionally re-attach their companions."""
    graphs: List[Any] = []
    for path in shard_files(out_dir, split):
        loaded = torch.load(str(path), weights_only=False)
        if with_companions:
            if path.name != "original.pt":
                raise FileNotFoundError(f"{path} has no companions")
            rows = json.loads((path.parent / "records.json").read_text(encoding="utf-8"))
            if len(rows) != len(loaded):
                raise RuntimeError(f"{path.parent}: {len(rows)} records for {len(loaded)} graphs")
            for graph, row in zip(loaded, rows):
                data = (path.parent / row["companion"]).read_bytes()
                graph.companion = (data, {key: row[key] for key in COMPANION_RECORD_KEYS})
        graphs.extend(loaded)
    return graphs


# ─────────────────────────────────────────────────────────────────
# Merge utility
# ─────────────────────────────────────────────────────────────────

def merge_worker_datasets(
    base_dir: str,
    merged_dir: str = "./dataset_merged",
    seed: int = 42,
) -> Dict:
    """
    Merge parallel worker outputs into one unified dataset with
    proper global stratified splits and a single meta.json.
    """
    base_path = Path(base_dir)
    worker_dirs = sorted(base_path.glob("worker_*"))
    if not worker_dirs:
        raise FileNotFoundError(f"No worker_* dirs in {base_path}")

    all_graphs: List[Any] = []
    agg = {
        "n_attempts": 0, "n_passed": 0,
        "attempts_with_success": 0,
        "variant_trials": 0, "variant_passes": 0,
        "time_seconds": 0.0,
        "type_counts": Counter(),
        "failures": Counter(),
    }

    for wd in worker_dirs:
        meta_file = wd / "meta.json"
        if not meta_file.exists():
            print(f"  Skipping {wd.name}: no meta.json")
            continue

        with open(meta_file) as f:
            m = json.load(f)

        agg["n_attempts"] += m.get("n_attempts", 0)
        agg["n_passed"] += m.get("n_passed", 0)
        agg["attempts_with_success"] += m.get("attempts_with_success", 0)
        agg["variant_trials"] += m.get("variant_trials", 0)
        agg["variant_passes"] += m.get("variant_passes", 0)
        agg["time_seconds"] = max(agg["time_seconds"], m.get("time_seconds", 0))
        for k, v in m.get("type_counts", {}).items():
            agg["type_counts"][k] += v
        for k, v in m.get("failures", {}).items():
            agg["failures"][k] += v

        for split in ["train", "val", "test"]:
            try:
                all_graphs.extend(load_split(str(wd), split, with_companions=True))
            except FileNotFoundError:
                pass

    print(f"Loaded {len(all_graphs)} graphs from {len(worker_dirs)} workers")

    worker_caches = {
        json.dumps(json.loads((wd / "meta.json").read_text(encoding="utf-8")).get("hull_cache_sha256"),
                   sort_keys=True)
        for wd in worker_dirs if (wd / "meta.json").exists()
    }
    if len(worker_caches) != 1:
        raise RuntimeError("workers used different hull masks; regenerate in a new folder")
    cache_sha256 = json.loads(worker_caches.pop())

    rng = np.random.default_rng(seed)
    splits = _stratified_split(all_graphs, SPLIT_RATIOS, rng=rng)
    _save_splits(splits, merged_dir, cache_sha256)

    zone_counts = []
    lcg_errors = []
    kg_errors = []
    for g in all_graphs:
        if HAS_PYG:
            zone_counts.append(g.x.shape[0])
            lcg_e, kg_e = _graph_physics_errors(g)
            lcg_errors.append(lcg_e)
            kg_errors.append(kg_e)
        else:
            zone_counts.append(g["x"].shape[0])
            lcg_e, kg_e = _graph_physics_errors(g)
            lcg_errors.append(lcg_e)
            kg_errors.append(kg_e)

    n_att = max(agg["n_attempts"], 1)
    n_vt = max(agg["variant_trials"], 1)
    first_meta = json.loads((worker_dirs[0] / "meta.json").read_text(encoding="utf-8"))
    meta = {
        "n_target": len(all_graphs),
        "n_attempts": agg["n_attempts"],
        "n_passed": len(all_graphs),
        "attempts_with_success": agg["attempts_with_success"],
        "attempt_success_rate": round(agg["attempts_with_success"] / n_att, 4),
        "variant_trials": agg["variant_trials"],
        "variant_passes": agg["variant_passes"],
        "variant_pass_rate": round(agg["variant_passes"] / n_vt, 4),
        "yield_per_attempt": round(len(all_graphs) / n_att, 4),
        "time_seconds_wall": round(agg["time_seconds"], 1),
        "seed": seed,
        "generator_version": first_meta.get("generator_version", GENERATOR_VERSION),
        "encoding": ENCODING,
        "source_tree_sha256": first_meta.get("source_tree_sha256", _source_tree_sha256()),
        "hull_cache_sha256": cache_sha256,
        "cond_source": first_meta.get("cond_source"),
        "gm_qc_gate": first_meta.get("gm_qc_gate"),
        "lcg_kg_qc_gate": first_meta.get("lcg_kg_qc_gate"),
        "n_variants": first_meta.get("n_variants"),
        "target_grid_shape": first_meta.get("target_grid_shape", list(DEFAULT_TARGET_SHAPE)),
        "grid_counts": first_meta.get("grid_counts"),
        "hull_mode": first_meta.get("hull_mode"),
        "qc": {"budget": QC_BUDGET_TOL, "lcg": QC_LCG_TOL,
               "kg": QC_KG_TOL, "gm_min": QC_GM_MIN, "gm_max": QC_GM_MAX,
               "gm_gate": first_meta.get("gm_qc_gate"),
               "lcg_kg_gate": first_meta.get("lcg_kg_qc_gate")},
        "splits": {k: len(v) for k, v in splits.items()},
        "type_counts": dict(agg["type_counts"]),
        "failures": dict(agg["failures"]),
        "stats": {
            "zones_mean": round(float(np.mean(zone_counts)), 1),
            "zones_std": round(float(np.std(zone_counts)), 1),
            "lcg_err_mean": round(float(np.mean(lcg_errors)), 4),
            "lcg_err_p95": round(float(np.percentile(lcg_errors, 95)), 4),
            "kg_err_mean": round(float(np.mean(kg_errors)), 4),
            "kg_err_p95": round(float(np.percentile(kg_errors, 95)), 4),
        },
        "shard_size": SHARD_SIZE,
        "n_node_features": N_NODE_FEATURES,
        "n_edge_features": N_EDGE_FEATURES,
        "n_cond_dims": N_COND_DIMS,
        "n_comp_classes": N_COMP_CLASSES,
        "n_workers": len(worker_dirs),
    }

    meta_path = Path(merged_dir) / "meta.json"
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n{'='*60}")
    print("Merged Dataset Summary")
    print(f"{'='*60}")
    for k, v in splits.items():
        print(f"  {k:6s}: {len(v):>6} samples")
    print(f"  {'total':6s}: {len(all_graphs):>6} samples")
    print(f"  Zones/graph: {meta['stats']['zones_mean']:.1f} "
          f"+/- {meta['stats']['zones_std']:.1f}")
    print(f"  LCG error:   mean={meta['stats']['lcg_err_mean']:.4f}  "
          f"p95={meta['stats']['lcg_err_p95']:.4f}")
    print(f"  Attempt success rate: "
          f"{100*meta['attempt_success_rate']:.1f}%")
    print(f"  Variant pass rate:    "
          f"{100*meta['variant_pass_rate']:.1f}%")
    print(f"  Type distribution:")
    for t in sorted(agg["type_counts"]):
        print(f"    {t:12s}: {agg['type_counts'][t]:>6}")
    print(f"  Saved -> {merged_dir}")
    print(f"{'='*60}")

    return meta


def _parallel_worker(
    worker_id,
    per_worker,
    work_dir,
    base_seed,
    voxel,
    log_every,
    stl_paths,
    type_weights=None,
    log_hull_rejects=None,
    adaptive_reweight=True,
    reweight_every=5000,
    reweight_until=100000,
    reweight_smoothing=0.5,
    use_tqdm=None,
    n_variants=5,
    fixed_ship_type=None,
    max_attempts_factor=6.0,
    grid_counts=None,
    cache_dir=None,
    use_hull_cache=True,
    precompute_hull_cache=True,
    cond_source: str = "achieved",
    gm_qc_gate: Optional[bool] = None,
    lcg_kg_qc_gate: Optional[bool] = None,
):
    return generate_dataset(
        n_target=per_worker,
        out_dir=f"{work_dir}/worker_{worker_id:02d}",
        seed=base_seed + worker_id * 100_000,
        voxel_size=voxel,
        type_weights=type_weights,
        max_attempts_factor=max_attempts_factor,
        log_every=log_every,
        stl_paths=stl_paths,
        log_hull_rejects=log_hull_rejects,
        adaptive_reweight=adaptive_reweight,
        reweight_every=reweight_every,
        reweight_until=reweight_until,
        reweight_smoothing=reweight_smoothing,
        use_tqdm=use_tqdm,
        n_variants=n_variants,
        fixed_ship_type=fixed_ship_type,
        grid_counts=grid_counts,
        cache_dir=cache_dir,
        use_hull_cache=use_hull_cache,
        precompute_hull_cache=precompute_hull_cache,
        cond_source=cond_source,
        gm_qc_gate=gm_qc_gate,
        lcg_kg_qc_gate=lcg_kg_qc_gate,
    )


def generate_dataset_parallel(
    n_target: int,
    out_dir: str,
    *,
    n_workers: int,
    work_dir: str = "./dataset_workers",
    seed: int = 42,
    voxel_size: float = 3.0,
    type_weights: Optional[Dict[ShipType, float]] = None,
    log_every: int = 500,
    n_variants: int = 5,
    stl_paths=None,
    grid_counts: Optional[Tuple[int, int, int]] = None,
    cache_dir: Optional[str | Path] = None,
    use_hull_cache: bool = True,
    precompute_hull_cache: bool = True,
    adaptive_reweight: bool = True,
    reweight_every: int = 1000,
    reweight_until: int = 100_000,
    reweight_smoothing: float = 1.0,
    fixed_ship_type: Optional[ShipType] = None,
    per_type: Optional[int] = None,
    ship_types: Optional[List[ShipType]] = None,
    max_attempts_factor: float = 6.0,
    cond_source: str = "achieved",
    gm_qc_gate: Optional[bool] = None,
    lcg_kg_qc_gate: Optional[bool] = None,
    n_processes: Optional[int] = None,
) -> Dict:
    """
    Generate ``n_target`` graphs with ``ProcessPoolExecutor``, then merge.

    Each worker writes to ``{work_dir}/worker_XX/``; the merged dataset
    (global stratified split) is written to ``out_dir``.

    When ``per_type`` is set, workers are assigned fixed ship types so each
    type reaches exactly ``per_type`` graphs (total = ``per_type × 6``).

    ``n_workers`` fixes the random streams and therefore the samples;
    ``n_processes`` (default ``n_workers``) only sets how many run at once.
    """
    from concurrent.futures import ProcessPoolExecutor, as_completed
    from functools import partial

    if n_workers < 2 and per_type is None:
        raise ValueError("n_workers must be >= 2 for parallel generation")
    if any(Path(work_dir).glob("worker_*")):
        raise FileExistsError(f"{work_dir} already holds worker output; use a new folder")
    if any(Path(out_dir).glob("*/shard_*")):
        raise FileExistsError(f"{out_dir} already holds shards; use a new output folder")

    types = ship_types if ship_types is not None else list(ShipType)
    if per_type is not None:
        n_target = per_type * len(types)
        adaptive_reweight = False

    effective_grid = grid_counts or PRODUCTION_GRID
    if stl_paths and use_hull_cache and precompute_hull_cache:
        ensure_hull_cache(
            stl_paths,
            cache_dir=cache_dir,
            grid_counts=effective_grid,
        )

    worker_common = dict(
        work_dir=work_dir,
        base_seed=seed,
        voxel=voxel_size,
        type_weights=type_weights,
        log_every=log_every,
        stl_paths=stl_paths,
        adaptive_reweight=adaptive_reweight,
        reweight_every=reweight_every,
        reweight_until=reweight_until,
        reweight_smoothing=reweight_smoothing,
        use_tqdm=False,
        n_variants=n_variants,
        max_attempts_factor=max_attempts_factor,
        grid_counts=effective_grid if stl_paths else None,
        cache_dir=cache_dir,
        use_hull_cache=use_hull_cache,
        precompute_hull_cache=False,
        cond_source=cond_source,
        gm_qc_gate=gm_qc_gate,
        lcg_kg_qc_gate=lcg_kg_qc_gate,
    )
    worker_fn = partial(_parallel_worker, **worker_common)

    if per_type is not None:
        base_wpt, extra_types = divmod(n_workers, len(types))
        tasks: List[Tuple[int, int, ShipType]] = []
        wid = 0
        for ti, st in enumerate(types):
            n_type_workers = base_wpt + (1 if ti < extra_types else 0)
            for sub_target in _split_quota(per_type, n_type_workers):
                if sub_target > 0:
                    tasks.append((wid, sub_target, st))
                    wid += 1
        print(
            f"Parallel stratified: {per_type} per type × {len(types)} types "
            f"= {n_target} graphs  workers={len(tasks)}  "
            f"work_dir={work_dir}  out_dir={out_dir}",
        )
        for worker_id, sub_target, st in tasks:
            print(
                f"  worker_{worker_id:02d}: {st.name} target={sub_target}  "
                f"seed={seed + worker_id * 100_000}",
            )
    else:
        if n_workers < 2:
            raise ValueError("n_workers must be >= 2 for parallel generation")
        per_worker_targets = [
            n_target // n_workers + (1 if i < (n_target % n_workers) else 0)
            for i in range(n_workers)
        ]
        tasks = [
            (i, per_worker_targets[i], fixed_ship_type)
            for i in range(n_workers)
            if per_worker_targets[i] > 0
        ]
        print(
            f"Parallel build: {n_target} graphs  workers={n_workers}  "
            f"work_dir={work_dir}  out_dir={out_dir}",
        )
        for worker_id, sub_target, st in tasks:
            st_label = st.name if st is not None else "mixed"
            print(
                f"  worker_{worker_id:02d}: target={sub_target}  "
                f"type={st_label}  seed={seed + worker_id * 100_000}",
            )

    with ProcessPoolExecutor(max_workers=min(n_processes or n_workers, len(tasks))) as ex:
        futures = [
            ex.submit(worker_fn, worker_id, sub_target, fixed_ship_type=st)
            for worker_id, sub_target, st in tasks
        ]
        if _tqdm_factory is not None:
            pbar = _tqdm_factory(
                total=n_target,
                desc="graphs (all workers)",
                unit="graph",
                dynamic_ncols=True,
            )
            try:
                for fut in as_completed(futures):
                    meta = fut.result()
                    pbar.update(int(meta.get("n_passed", 0)))
            finally:
                pbar.close()
        else:
            for fut in as_completed(futures):
                meta = fut.result()
                print(f"  worker done: {meta['n_passed']} graphs")

    return merge_worker_datasets(work_dir, out_dir, seed=seed)
