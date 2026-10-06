"""Validate a generated dataset against the 23 real general arrangements.

Two comparisons, each written to its own folder under --out:

  stage5_conventional/  every stored graph of the dataset, as generated
  cc_aligned/           --per-type graphs per ship type (drawn with --seed),
                        re-extracted as connected regions of equal function,
                        the same zone extraction as used for the real graphs

Each folder holds similarity_metrics.json/.csv, per-family reports and the
comparison figures; comparison_summary.json sets the two side by side.

    python validate.py
    python validate.py --dataset data/test_run --out validation_test
"""
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT / "src" / "data_generator"), str(ROOT / "src" / "validation")]

from cc_reextract import run_full_validation  # noqa: E402
from dataset_builder import shard_files  # noqa: E402

REAL_SHARD = ROOT / "data" / "real_ga" / "real_ga_shard_000.pt"
FAMILIES = ["Bulker", "Tanker", "Cargo", "OSV", "Patrol", "Yacht"]  # ShipType order
NOT_USED_BY_VALIDATION = ("deck_labels", "deck_hull_masks", "zone_node_mask")


def load_stored_graphs(dataset):
    """All stored graphs, with uniform cell volumes for volume-weighted metrics."""
    graphs = []
    for split in ("train", "val", "test"):
        try:
            paths = shard_files(dataset, split)
        except FileNotFoundError:
            continue
        for path in paths:
            for g in torch.load(str(path), weights_only=False):
                for name in NOT_USED_BY_VALIDATION:
                    if hasattr(g, name):
                        delattr(g, name)
                if getattr(g, "voxel_volume_m3", None) is None and hasattr(g, "voxel_labels"):
                    cell = torch.tensor(float(g.dx) * float(g.dy) * float(g.dz), dtype=torch.float32)
                    g.voxel_volume_m3 = cell.expand(tuple(g.voxel_labels.shape))
                    g.measurement_basis = "native_volume"
                graphs.append(g)
    return graphs


def zone_stats(graphs):
    nz = [int(g.n_zones) for g in graphs]
    return {
        "n": len(graphs),
        "zones_median": int(np.median(nz)) if nz else 0,
        "zones_min": int(min(nz)) if nz else 0,
        "zones_max": int(max(nz)) if nz else 0,
    }


def summarise_track(out_dir, name, synth, real):
    reports = json.loads((out_dir / "reports" / "report_index.json").read_text(encoding="utf-8"))
    metrics = json.loads((out_dir / "similarity_metrics.json").read_text(encoding="utf-8"))
    families = {}
    for fam in FAMILIES:
        entry = reports.get(fam, {})
        cont = entry.get("containment", {})
        families[fam] = {
            "n_real": entry.get("n_real"),
            "n_synth": entry.get("n_synth"),
            "JSD": entry.get("JSD"),
            "coverage": entry.get("coverage"),
            "physics": cont.get("physics_status", {}),
            "budget_gap_keys": cont.get("budget_gap_keys", []),
            "budget_partial_keys": cont.get("budget_partial_keys", []),
            "budget_contained": cont.get("budget_contained"),
        }
    kg = {}
    for st, fam in enumerate(FAMILIES):
        vals = [float(g.kg_actual) for g in synth if int(g.ship_type) == st]
        rvals = [float(g.kg_actual) for g in real if int(g.ship_type) == st]
        kg[fam] = {
            "synth_kg_mean": round(float(np.mean(vals)), 4) if vals else None,
            "synth_kg_p5": round(float(np.percentile(vals, 5)), 4) if vals else None,
            "synth_kg_p95": round(float(np.percentile(vals, 95)), 4) if vals else None,
            "real_kg_mean": round(float(np.mean(rvals)), 4) if rvals else None,
        }
    return {
        "track": name,
        "out_dir": str(out_dir),
        "synth_zone_stats": zone_stats(synth),
        "real_zone_stats": zone_stats(real),
        "coverage_global": metrics.get("Coverage_global"),
        "jsd_global": metrics.get("JSD_global"),
        "measurement": metrics.get("_measurement"),
        "families": families,
        "kg_by_family": kg,
    }


def family_contrast(stage5, cc):
    contrast = {}
    for fam in FAMILIES:
        s5, c = stage5["families"][fam], cc["families"][fam]
        contrast[fam] = {
            "KG_status": {"stage5": s5["physics"].get("KG"), "cc": c["physics"].get("KG")},
            "budget_gaps": {"stage5": s5["budget_gap_keys"], "cc": c["budget_gap_keys"]},
            "JSD": {"stage5": s5["JSD"], "cc": c["JSD"]},
            "coverage": {"stage5": s5["coverage"], "cc": c["coverage"]},
        }
    return contrast


def main():
    ap = argparse.ArgumentParser(description="Validate a generated dataset against the real general arrangements.")
    ap.add_argument("--dataset", default="data/dataset_v1", help="dataset folder (from generate.py or Zenodo)")
    ap.add_argument("--out", default="validation_output", help="output folder")
    ap.add_argument("--per-type", type=int, default=833, help="graphs per ship type for the cc_aligned comparison")
    ap.add_argument("--seed", type=int, default=44, help="seed for drawing the cc_aligned graphs")
    args = ap.parse_args()

    dataset = Path(args.dataset).resolve()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    real = torch.load(str(REAL_SHARD), weights_only=False)

    stored = load_stored_graphs(dataset)
    if not stored:
        raise SystemExit(f"No dataset shards found in {dataset}")
    print(f"\n=== stage5_conventional: {len(stored)} stored graphs ===")
    s5_dir = out / "stage5_conventional"
    run_full_validation(str(REAL_SHARD), str(s5_dir), seed=args.seed, synth_graphs=stored)
    stage5 = summarise_track(s5_dir, "stage5_conventional", stored, real)
    del stored

    print(f"\n=== cc_aligned: {args.per_type} graphs per ship type, seed {args.seed} ===")
    cc_dir = out / "cc_aligned"
    _, cc_graphs = run_full_validation(
        str(REAL_SHARD), str(cc_dir), seed=args.seed,
        cc_from_dataset_dir=str(dataset), cc_from_dataset_per_type=args.per_type,
    )
    cc = summarise_track(cc_dir, "cc_aligned", cc_graphs, real)

    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config": {
            "synth_dir": str(dataset),
            "per_type": args.per_type,
            "cc_n_synth": len(cc_graphs),
            "seed": args.seed,
            "real_n": len(real),
        },
        "tracks": {"stage5_conventional": stage5, "cc_aligned": cc},
        "family_contrast": family_contrast(stage5, cc),
    }
    path = out / "comparison_summary.json"
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nWrote {path}")


if __name__ == "__main__":
    main()
