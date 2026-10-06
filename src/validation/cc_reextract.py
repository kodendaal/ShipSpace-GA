"""
cc_reextract.py
===============
Connected-component (CC) re-extraction of generated arrangements, plus the
one-call validation runner that computes all metrics, reports and figures.

The real general arrangements are described as connected regions of equal
function (see ``zone_extraction``). ``cc_reextract_stage5_graph`` applies
the same extraction to the stored voxel labels of a generated graph, so both
sides of the comparison share one topology. ``run_full_validation`` compares
the real graphs with either the stored graphs of a dataset (``synth_graphs``)
or their CC re-extraction (``cc_from_dataset_dir``); ``validate.py`` runs both.
"""

from __future__ import annotations
import numpy as np
import torch
import time
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from collections import Counter, defaultdict

from ship_params import ShipType

# Same zone extraction as for the real general arrangements
from zone_extraction import (
    extract_zones, build_node_features, build_node_labels,
    build_edges, build_conditioning, validate_converted_graph, package_graph,
    detect_ss_boundary,
)
from volume_metrics import (
    build_voxel_fields_from_saved_graph,
    compute_budget_fracs_volumetric,
    compute_physics_volumetric,
    hull_band_depth_m,
    refine_zone_centroids_volumetric,
)
from validation_constants import Comp


# ─────────────────────────────────────────────────────────────────
# CC re-extraction of stored graphs
# ─────────────────────────────────────────────────────────────────

def cc_reextract_stage5_graph(
    graph: Any,
    *,
    index: int = 0,
) -> Tuple[Optional[Any], str]:
    """
    CC re-extract a saved Stage-5 graph using its stored ``voxel_labels``.

    Same zone extraction as for the real general arrangements, so validation
    compares ~21-zone topology on both sides while reusing the exact layouts
    of a generated dataset.
    """
    if not hasattr(graph, "voxel_labels"):
        return None, "no voxel_labels"

    try:
        vf = build_voxel_fields_from_saved_graph(graph)
    except (ValueError, TypeError) as exc:
        return None, f"vf:{exc}"

    label_grid = vf.labels.copy()
    dx = float(getattr(graph, "dx", 1.0))
    dy = float(getattr(graph, "dy", 1.0))
    dz = float(getattr(graph, "dz", 1.0))
    nz_hull_native = int(getattr(graph, "nz_hull", label_grid.shape[2]))

    hull_info = {
        "L": vf.L,
        "B": vf.B,
        "D_total": vf.D_total_m,
        "gs": dz,
        "nx": vf.shape[0],
        "ny": vf.shape[1],
        "nz": vf.shape[2],
        "dx_m": dx,
        "dy_m": dy,
        "dz_m": dz,
        "nz_hull_native": nz_hull_native,
    }

    nz_hull, _ = detect_ss_boundary(label_grid, hull_info)
    D_hull_m = hull_band_depth_m(nz_hull, dz, vf)

    zones = extract_zones(label_grid, hull_info, nz_hull)
    if len(zones) < 3:
        return None, f"CC:only {len(zones)} zones"

    refine_zone_centroids_volumetric(zones, vf)

    x = build_node_features(zones, hull_info["nz"])
    y = build_node_labels(zones)
    edge_index, edge_attr = build_edges(zones, label_grid, nz_hull)

    lcg_frac, kg_frac, gm_t = compute_physics_volumetric(vf, nz_hull, dz)
    budget_fracs = compute_budget_fracs_volumetric(vf, nz_hull)
    voxel_volume_m3 = vf.volume_m3.astype(np.float32)

    ship_type = ShipType(int(getattr(graph, "ship_type", 0)))
    ship_name = getattr(graph, "ship_name", None)
    if not ship_name:
        hull_id = getattr(graph, "hull_id", None)
        ship_name = f"cc_{ship_type.name}_{hull_id or index}"

    cond = build_conditioning(
        ship_type=ship_type,
        L=vf.L,
        B=vf.B,
        D_hull=D_hull_m,
        lcg_frac=lcg_frac,
        kg_frac=kg_frac,
        budget_fracs=budget_fracs,
    )
    cc_graph = package_graph(
        x=x, y=y,
        edge_index=edge_index, edge_attr=edge_attr, cond=cond,
        ship_type=int(ship_type), n_zones=len(zones),
        lcg_actual=lcg_frac, kg_actual=kg_frac, gm_t=gm_t,
        ship_name=str(ship_name),
        voxel_labels=label_grid.astype(np.int8),
        voxel_hull_mask=(label_grid != int(Comp.EMPTY)),
        voxel_volume_m3=voxel_volume_m3,
        native_grid_shape=label_grid.shape,
        measurement_basis="native_volume",
    )
    ok, warns = validate_converted_graph(cc_graph)
    if not ok:
        return None, f"CCv:{warns[0][:50]}"

    return cc_graph, "OK"


def cc_reextract_from_dataset(
    dataset_dir: str,
    *,
    per_type: Optional[int] = None,
    seed: int = 42,
    log_every: int = 100,
) -> Tuple[List[Any], dict]:
    """Load Stage-5 graphs from a dataset directory and CC re-extract each."""
    t0 = time.time()
    stage5 = load_dataset_sample(dataset_dir, per_type=per_type, seed=seed)
    graphs: List[Any] = []
    failures: Counter = Counter()
    for i, g in enumerate(stage5):
        cc_g, status = cc_reextract_stage5_graph(g, index=i)
        if cc_g is None:
            failures[status] += 1
            continue
        graphs.append(cc_g)
        if log_every > 0 and (i + 1) % log_every == 0:
            print(f"  CC re-extract: {i + 1}/{len(stage5)} "
                  f"({len(graphs)} ok, {sum(failures.values())} fail)")

    info = {
        "source_dataset_dir": str(Path(dataset_dir).resolve()),
        "per_type": per_type,
        "seed": seed,
        "n_stage5_loaded": len(stage5),
        "n_cc_passed": len(graphs),
        "pass_rate": round(len(graphs) / len(stage5), 4) if stage5 else 0.0,
        "failures": dict(failures),
        "representation": "cc_from_stage5_voxels",
        "time_seconds": round(time.time() - t0, 1),
    }
    return graphs, info


# ─────────────────────────────────────────────────────────────────
# Dataset loader (full or stratified sample, streamed shard by shard)
# ─────────────────────────────────────────────────────────────────

def _iter_dataset_graphs(dataset_dir: str):
    """Yield stored graphs shard by shard (train, val, test)."""
    from dataset_builder import shard_files

    for split in ("train", "val", "test"):
        try:
            paths = shard_files(dataset_dir, split)
        except FileNotFoundError:
            continue
        for p in paths:
            shard = torch.load(str(p), weights_only=False)
            yield from shard
            del shard


def load_dataset_sample(
    dataset_dir: str,
    per_type: Optional[int] = None,
    seed: int = 42,
) -> List[Any]:
    """
    Load Stage-5 graphs from ``dataset_builder`` output.

    If ``per_type`` is set, reservoir-sample up to that many graphs per ship
    type while streaming shards.
    """
    if per_type is None:
        from dataset_builder import load_split

        graphs: List[Any] = []
        for split in ("train", "val", "test"):
            try:
                graphs.extend(load_split(dataset_dir, split))
            except FileNotFoundError:
                pass
        return graphs

    rng = np.random.default_rng(seed)
    reservoirs: Dict[int, List[Any]] = defaultdict(list)
    seen: Dict[int, int] = defaultdict(int)
    for g in _iter_dataset_graphs(dataset_dir):
        st = int(g.ship_type)
        seen[st] += 1
        bucket = reservoirs[st]
        if len(bucket) < per_type:
            bucket.append(g)
        else:
            j = int(rng.integers(0, seen[st]))
            if j < per_type:
                bucket[j] = g
    return [g for st in sorted(reservoirs) for g in reservoirs[st]]


# ─────────────────────────────────────────────────────────────────
# Full validation runner
# ─────────────────────────────────────────────────────────────────

def run_full_validation(
    real_shard: str,
    out_dir: str = "./validation_output",
    *,
    seed: int = 42,
    synth_graphs: Optional[List[Any]] = None,
    cc_from_dataset_dir: Optional[str] = None,
    cc_from_dataset_per_type: Optional[int] = None,
) -> Tuple[Dict[str, Any], List[Any]]:
    """
    Compare the real general arrangements with one set of generated graphs.

    Writes similarity metrics (``similarity_metrics.json`` / ``.csv``), the
    per-family reports (``reports/``) and the comparison figures
    (``cmp1``–``cmp11`` PDF) to ``out_dir``. Provide exactly one source:

      synth_graphs        → the stored graphs of a dataset, as given
      cc_from_dataset_dir → CC re-extraction of the stored graphs of a dataset
                            (``cc_from_dataset_per_type`` graphs per ship type,
                            drawn with ``seed``)

    Returns the metrics and the generated graphs that were compared.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ── Load real GAs ──
    print(f"Loading real GAs from {real_shard} ...")
    real = torch.load(real_shard, weights_only=False)
    if isinstance(real, dict):
        real = real.get("graphs", real.get("real", list(real.values())[0]))
    print(f"  {len(real)} real GA graphs")

    # ── Generated graphs ──
    if synth_graphs is not None:
        synth = synth_graphs
        print(f"\n{len(synth)} stored graphs (no CC re-extraction)")

    elif cc_from_dataset_dir is not None:
        if cc_from_dataset_per_type is not None:
            print(f"\nCC re-extracting from {cc_from_dataset_dir} "
                  f"(≤{cc_from_dataset_per_type}/type, seed={seed}) ...")
        else:
            print(f"\nCC re-extracting from {cc_from_dataset_dir} (full corpus) ...")
        synth, cc_info = cc_reextract_from_dataset(
            cc_from_dataset_dir,
            per_type=cc_from_dataset_per_type,
            seed=seed,
            log_every=max(1, 50),
        )
        print(f"  {len(synth)} CC graphs ({cc_info['pass_rate']:.1%} pass)")
        with open(out / "cc_reextract_info.json", "w", encoding="utf-8") as f:
            json.dump(cc_info, f, indent=2)
    else:
        raise ValueError("Provide synth_graphs or cc_from_dataset_dir")

    # ── Metrics (similarity_metrics.py) ──
    print("\n" + "=" * 60)
    print("SIMILARITY METRICS")
    print("=" * 60)

    from similarity_metrics import (
        compute_label_jsd, compute_ks_tests, compute_mmd_graph_stats,
        compute_adjacency_similarity, compute_centroid_ks,
        compute_coverage, print_summary_table,
        save_results_csv, save_results_json, fig_metrics_summary,
    )

    results: Dict[str, Any] = {}
    print("  1/6  Label frequency JSD ...")
    results.update(compute_label_jsd(real, synth))
    print("  2/6  KS tests ...")
    results.update(compute_ks_tests(real, synth))
    print("  3/6  MMD graph stats ...")
    results.update(compute_mmd_graph_stats(real, synth))
    print("  4/6  Adjacency patterns ...")
    results.update(compute_adjacency_similarity(real, synth))
    print("  5/6  Spatial centroids ...")
    results.update(compute_centroid_ks(real, synth))
    print("  6/6  Coverage ...")
    results.update(compute_coverage(real, synth))

    results["_measurement"] = {
        "measurement_basis": "native_volume",
        "grid_counts": [64, 32, 24],
        "jsd_label_weighting": "volume",
        "ks_gate_features": "budget_physics_lcg_kg_gm",
        "ks_diagnostic_only": ["zones", "edges", "MMD_degree", "MMD_clustering"],
    }

    print_summary_table(results)
    save_results_csv(results, out / "similarity_metrics.csv")
    save_results_json(results, out / "similarity_metrics.json")
    fig_metrics_summary(results, save_path=out / "metrics_summary.pdf")

    # ── Per-family fixed-schema reports (validation_report.py) ──
    print("\n" + "=" * 60)
    print("PER-FAMILY VALIDATION REPORTS")
    print("=" * 60)
    from validation_report import validation_report_all
    reports_dir = out / "reports"
    validation_report_all(real, synth, reports_dir)

    # ── Comparison figures (compare_real_vs_synthetic.py) ──
    print("\n" + "=" * 60)
    print("COMPARISON FIGURES")
    print("=" * 60)

    from compare_real_vs_synthetic import (
        fig_budget_comparison, fig_physics_scatter, fig_spatial_scatter,
        fig_node_edge_dists, fig_label_freq, fig_cond_pca,
        fig_gm_comparison,
        fig_output_space_pca, fig_radar_label_profiles,
        fig_achieved_physics_boxplot,
    )

    fig_budget_comparison(real, synth, save_path=out / "cmp1_budgets.pdf")
    fig_physics_scatter(real, synth,   save_path=out / "cmp2_physics.pdf")
    fig_spatial_scatter(real, synth,   save_path=out / "cmp3_spatial.pdf")
    fig_node_edge_dists(real, synth,   save_path=out / "cmp4_node_edge.pdf")
    fig_label_freq(real, synth,        save_path=out / "cmp5_labels.pdf")
    fig_cond_pca(real, synth,          save_path=out / "cmp6_pca.pdf")
    fig_gm_comparison(real, synth,     save_path=out / "cmp7_gm.pdf")
    fig_output_space_pca(real, synth, save_path=out / "cmp8_output_pca.pdf")
    fig_radar_label_profiles(real, synth, save_path=out / "cmp9_radar_profiles.pdf")
    fig_achieved_physics_boxplot(real, synth, save_path=out / "cmp10_physics_boxplot.pdf")

    # ── Side-by-side (side_by_side_comparison.py) ──
    from side_by_side_comparison import fig_side_by_side_graphs

    try:
        fig_side_by_side_graphs(real, synth, save_path=out / "cmp11_side_by_side_graphs.pdf")
    except (PermissionError, ValueError) as exc:
        print(f"  WARNING: could not write cmp11_side_by_side_graphs.pdf ({exc})")

    print(f"\nAll outputs saved to {out}/")
    print("=" * 60)
    return results, synth
