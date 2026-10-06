"""Generate the synthetic general-arrangement dataset.

The defaults reproduce the published dataset: 8,333 arrangements per ship type
(49,998 in total), seed 42, 16 workers. Each worker has its own random stream,
so --workers changes the samples; --processes only sets how many workers run
at the same time.

    python generate.py
    python generate.py --out data/test_run --per-type 10 --workers 6
"""
import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src" / "data_generator"))

from dataset_builder import generate_dataset_parallel  # noqa: E402
from hull_mask import SHIPTYPE_TO_STL_LIB  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="Generate the synthetic general-arrangement dataset.")
    ap.add_argument("--out", default="data/dataset_v1",
                    help="output folder (must not contain a dataset yet)")
    ap.add_argument("--per-type", type=int, default=8333, help="arrangements per ship type")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=16, help="number of random streams")
    ap.add_argument("--processes", type=int, default=None,
                    help="workers running at the same time (default: all)")
    args = ap.parse_args()

    out = Path(args.out).resolve()
    work_dir = out.parent / f"{out.name}_workers"
    generate_dataset_parallel(
        n_target=6 * args.per_type,
        out_dir=str(out),
        n_workers=args.workers,
        work_dir=str(work_dir),
        seed=args.seed,
        per_type=args.per_type,
        stl_paths=SHIPTYPE_TO_STL_LIB,
        grid_counts=(64, 32, 24),
        use_hull_cache=True,
        precompute_hull_cache=False,
        n_processes=args.processes,
    )
    shutil.rmtree(work_dir)
    print(f"Dataset written to {out}")


if __name__ == "__main__":
    main()
