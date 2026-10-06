"""
validation_report.py
====================
Emit a fixed-schema validation package per ship family for blind diagnosis.

    report = validation_report("Yacht", real_graphs, synth_graphs, out_dir)

Writes ``report_yacht.json`` plus 2–3 PNG figures.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from validation_constants import (
    ACTIVE_LABEL_INDICES, BUDGET_KEYS, COMP_SHORT, CONTAINMENT_FEATURE_KEYS,
    FAMILY_NAME_TO_TYPE, KS_FEATURE_KEYS, METRIC_THRESHOLDS, TYPE_NAMES,
)
from similarity_metrics import (
    compute_adjacency_similarity,
    compute_centroid_ks,
    compute_coverage,
    compute_ks_tests,
    compute_label_jsd,
    compute_mmd_graph_stats,
    jensen_shannon_divergence,
    mean_label_freq,
    _budget_fraction,
)


def _get(g, attr):
    v = getattr(g, attr) if hasattr(g, attr) else g[attr]
    return v.numpy() if isinstance(v, torch.Tensor) else v


def _filter_family(graphs: Sequence, ship_type: int) -> List:
    return [g for g in graphs if int(_get(g, "ship_type")) == ship_type]


def _distribution_summary(values: Sequence[float]) -> Dict[str, Any]:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return {
            "n": 0, "mean": None, "std": None,
            "min": None, "max": None, "deciles": [],
        }
    deciles = np.percentile(arr, np.arange(0, 101, 10)).tolist()
    return {
        "n": int(arr.size),
        "mean": round(float(arr.mean()), 6),
        "std": round(float(arr.std()), 6),
        "min": round(float(arr.min()), 6),
        "max": round(float(arr.max()), 6),
        "deciles": [round(float(v), 6) for v in deciles],
    }


def _label_freq_vector(graphs: Sequence) -> Tuple[List[float], str]:
    freq, basis = mean_label_freq(graphs)
    return [round(float(x), 6) for x in freq], basis


def _centroid_stats(graphs: Sequence, comp_val: int) -> Dict[str, Optional[float]]:
    cx, cz = [], []
    for g in graphs:
        x = _get(g, "x")
        y_lbl = _get(g, "y").astype(int)
        mask = y_lbl == comp_val
        if mask.any():
            cx.extend(x[mask, 0].tolist())
            cz.extend(x[mask, 2].tolist())
    if not cx:
        return {
            "cx_mean": None, "cx_std": None,
            "cz_mean": None, "cz_std": None, "n_zones": 0,
        }
    cx_a, cz_a = np.asarray(cx), np.asarray(cz)
    return {
        "cx_mean": round(float(cx_a.mean()), 6),
        "cx_std": round(float(cx_a.std()), 6),
        "cz_mean": round(float(cz_a.mean()), 6),
        "cz_std": round(float(cz_a.std()), 6),
        "n_zones": int(len(cx)),
    }


def _feature_values(graphs: Sequence, feature: str) -> List[float]:
    vals: List[float] = []
    for g in graphs:
        if feature == "zones":
            vals.append(float(_get(g, "n_zones")))
        elif feature == "edges":
            ei = _get(g, "edge_index")
            vals.append(float(ei.shape[1] // 2))
        elif feature == "LCG":
            vals.append(float(_get(g, "lcg_actual")))
        elif feature == "KG":
            vals.append(float(_get(g, "kg_actual")))
        elif feature == "GM":
            vals.append(float(_get(g, "gm_t")))
        elif feature in BUDGET_KEYS:
            ki = BUDGET_KEYS.index(feature)
            vals.append(_budget_fraction(g, ki))
    return vals


def _collect_metrics(real: List, synth: List, family: str) -> Dict[str, Any]:
    metrics: Dict[str, Any] = {}
    prefix = family

    jsd = compute_label_jsd(real, synth)
    metrics["JSD_label"] = jsd.get(f"JSD_label_{family}")

    ks = compute_ks_tests(real, synth)
    for feat in KS_FEATURE_KEYS:
        key = f"KS_{feat}_{prefix}"
        if key in ks:
            stat, p = ks[key]
            metrics[key] = {
                "statistic": round(float(stat), 6),
                "p_value": round(float(p), 6),
                "pass": bool(p > METRIC_THRESHOLDS["KS_pvalue_pass"]),
            }

    mmd = compute_mmd_graph_stats(real, synth)
    for mk in (f"MMD_degree_{prefix}", f"MMD_clustering_{prefix}"):
        if mk in mmd:
            metrics[mk] = round(float(mmd[mk]), 6)

    adj = compute_adjacency_similarity(real, synth)
    adj_key = f"AdjFrob_{prefix}"
    if adj_key in adj:
        metrics[adj_key] = round(float(adj[adj_key]), 6)

    cov = compute_coverage(real, synth)
    cov_key = f"Coverage_{prefix}"
    if cov_key in cov:
        metrics[cov_key] = round(float(cov[cov_key]), 6)

  # Centroid KS (global keys, not per-family suffixed)
    cks = compute_centroid_ks(real, synth)
    for ck, (stat, p) in cks.items():
        metrics[ck] = {
            "statistic": round(float(stat), 6),
            "p_value": round(float(p), 6),
            "pass": bool(p > METRIC_THRESHOLDS["KS_pvalue_pass"]),
        }

    return metrics


def _ks_failure_details(
    real: List,
    synth: List,
    metrics: Dict[str, Any],
    family: str,
) -> Dict[str, Any]:
    failures: Dict[str, Any] = {}
    prefix = family
    for feat in KS_FEATURE_KEYS:
        key = f"KS_{feat}_{prefix}"
        entry = metrics.get(key)
        if not entry or entry.get("pass", True):
            continue
        failures[feat] = {
            "real": _distribution_summary(_feature_values(real, feat)),
            "synth": _distribution_summary(_feature_values(synth, feat)),
        }
    return failures


def _threshold_pass_summary(metrics: Dict[str, Any]) -> Dict[str, Any]:
    jsd = metrics.get("JSD_label")
    jsd_thr = METRIC_THRESHOLDS["JSD_label"]
    jsd_pass = None
    if jsd is not None:
        jsd_pass = {
            "excellent": jsd < jsd_thr["excellent"],
            "good": jsd < jsd_thr["good"],
            "acceptable": jsd < jsd_thr["acceptable"],
        }

    ks_keys = [k for k in metrics if k.startswith("KS_") and isinstance(metrics[k], dict)]
    ks_pass_count = sum(1 for k in ks_keys if metrics[k].get("pass"))
    return {
        "frozen_thresholds": METRIC_THRESHOLDS,
        "JSD_label": jsd_pass,
        "KS_pass_rate": f"{ks_pass_count}/{len(ks_keys)}" if ks_keys else "0/0",
    }


# ──────────────────────────────────────────────────────────────────────
# Containment scorecard
# ──────────────────────────────────────────────────────────────────────
# Goal: the synthetic envelope should be broad, plausible, and *contain*
# the real operating points — not match the real distribution shape. The
# headline becomes: for each budget/physics dimension, does the synthetic
# range bracket the real range, with enough diversity and no implausible
# tail? KS is kept only as a one-sided bias diagnostic.

def _robust_stats(values: Sequence[float]) -> Dict[str, Any]:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return {
            "n": 0, "min": None, "p5": None, "p50": None,
            "p95": None, "max": None, "mean": None, "std": None,
        }
    lo, hi = METRIC_THRESHOLDS["containment"]["robust_band"]
    p5, p50, p95 = np.percentile(arr, [lo, 50.0, hi])
    return {
        "n": int(arr.size),
        "min": round(float(arr.min()), 6),
        "p5": round(float(p5), 6),
        "p50": round(float(p50), 6),
        "p95": round(float(p95), 6),
        "max": round(float(arr.max()), 6),
        "mean": round(float(arr.mean()), 6),
        "std": round(float(arr.std()), 6),
    }


def _containment_for_feature(
    real_vals: Sequence[float],
    synth_vals: Sequence[float],
) -> Optional[Dict[str, Any]]:
    r = _robust_stats(real_vals)
    s = _robust_stats(synth_vals)
    if not r["n"] or not s["n"]:
        return None

    cfg = METRIC_THRESHOLDS["containment"]
    r_lo, r_hi = r["min"], r["max"]
    r_span = max(r_hi - r_lo, 1e-9)

    # CONTAIN: robust synth band [p5,p95] brackets the full real range.
    contain_robust = (s["p5"] <= r_lo) and (s["p95"] >= r_hi)
    # COVERS_FULL: full synth support brackets real (maybe only in tails).
    contain_full = (s["min"] <= r_lo) and (s["max"] >= r_hi)
    # Overlap of full synth support with real range.
    overlap_lo = max(s["min"], r_lo)
    overlap_hi = min(s["max"], r_hi)
    overlap = max(0.0, overlap_hi - overlap_lo)
    coverage_frac = round(overlap / r_span, 4)

    if contain_robust:
        status = "CONTAIN"
    elif contain_full:
        status = "COVERS_FULL"
    elif overlap > 0:
        status = "PARTIAL"
    else:
        status = "GAP"

    diversity_ratio = (
        round(s["std"] / r["std"], 4) if r["std"] and r["std"] > 1e-9 else None
    )
    diversity_ok = (
        diversity_ratio is None or diversity_ratio >= cfg["diversity_min_ratio"]
    )

    median_shift = round(s["p50"] - r["p50"], 6)
    if status in ("CONTAIN", "COVERS_FULL"):
        bias = "ok"
    elif median_shift > 0:
        bias = "high"   # synth sits above real (real low end uncovered)
    else:
        bias = "low"

    implausible_high = bool(
        r_hi > 0 and s["max"] > cfg["implausible_tail_ratio"] * r_hi
    )

    return {
        "real": r,
        "synth": s,
        "status": status,
        "real_in_synth_p5_p95": bool(contain_robust),
        "real_in_synth_minmax": bool(contain_full),
        "coverage_frac": coverage_frac,
        "diversity_ratio": diversity_ratio,
        "diversity_ok": bool(diversity_ok),
        "median_shift": median_shift,
        "bias": bias,
        "implausible_high_tail": implausible_high,
    }


def _containment_details(real: List, synth: List) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for feat in CONTAINMENT_FEATURE_KEYS:
        entry = _containment_for_feature(
            _feature_values(real, feat), _feature_values(synth, feat),
        )
        if entry is not None:
            out[feat] = entry
    return out


def _containment_summary(containment: Dict[str, Any]) -> Dict[str, Any]:
    budget = {k: v for k, v in containment.items() if k in BUDGET_KEYS}
    n_budget = len(budget)
    contained = [k for k, v in budget.items() if v["status"] == "CONTAIN"]
    covered = [k for k, v in budget.items()
               if v["status"] in ("CONTAIN", "COVERS_FULL")]
    partial = [k for k, v in budget.items() if v["status"] == "PARTIAL"]
    gaps = [k for k, v in budget.items() if v["status"] == "GAP"]
    low_diversity = [k for k, v in containment.items() if not v["diversity_ok"]]
    implausible = [k for k, v in containment.items() if v["implausible_high_tail"]]

    physics = {
        k: containment[k]["status"]
        for k in ("LCG", "KG", "GM") if k in containment
    }

    return {
        "n_budget_keys": n_budget,
        "budget_contained": f"{len(contained)}/{n_budget}" if n_budget else "0/0",
        "budget_covered_or_contained": f"{len(covered)}/{n_budget}" if n_budget else "0/0",
        "budget_partial_keys": partial,
        "budget_gap_keys": gaps,
        "physics_status": physics,
        "low_diversity_keys": low_diversity,
        "implausible_tail_keys": implausible,
    }


def _fig_containment(
    containment: Dict[str, Any],
    family: str,
    save_path: Path,
) -> None:
    """Horizontal envelope plot: real [min,max] vs synth [p5,p95] & [min,max]."""
    feats = list(containment.keys())
    if not feats:
        return
    fig, ax = plt.subplots(
        figsize=(9, 0.55 * len(feats) + 1.5), facecolor="white",
    )
    ax.set_facecolor("white")
    status_color = {
        "CONTAIN": "#2ecc71",
        "COVERS_FULL": "#3498db",
        "PARTIAL": "#f39c12",
        "GAP": "#e74c3c",
    }
    for i, feat in enumerate(feats):
        e = containment[feat]
        r, s = e["real"], e["synth"]
        col = status_color.get(e["status"], "#888888")
        # synth full support (thin), robust band (thick), real range (marker bar)
        ax.plot([s["min"], s["max"]], [i, i], color=col, lw=1.2, alpha=0.5, zorder=1)
        ax.plot([s["p5"], s["p95"]], [i, i], color=col, lw=7, alpha=0.35, zorder=2)
        ax.plot([r["min"], r["max"]], [i + 0.16, i + 0.16], color="#222222",
                lw=3, zorder=3)
        ax.plot([r["p50"]], [i + 0.16], marker="|", color="#222222",
                markersize=10, zorder=4)
    ax.set_yticks(range(len(feats)))
    ax.set_yticklabels(feats, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("fraction / physics value")
    ax.set_title(
        f"{family}: containment — synth band (colour) vs real range (black)"
    )
    handles = [
        plt.Line2D([0], [0], color=c, lw=7, alpha=0.5, label=st)
        for st, c in status_color.items()
    ]
    handles.append(plt.Line2D([0], [0], color="#222222", lw=3, label="real [min,max]"))
    ax.legend(handles=handles, loc="lower right", fontsize=7, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def _fig_centroid_scatter(
    real: List,
    synth: List,
    family: str,
    save_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 6), facecolor="white")
    ax.set_facecolor("white")
    for g, color, alpha, label in (
        (real, "#1f77b4", 0.85, "real"),
        (synth, "#ff7f0e", 0.45, "synth"),
    ):
        cx, cz, comps = [], [], []
        for gr in g:
            x = _get(gr, "x")
            y_lbl = _get(gr, "y").astype(int)
            for c in ACTIVE_LABEL_INDICES:
                m = y_lbl == c
                if m.any():
                    cx.extend(x[m, 0].tolist())
                    cz.extend(x[m, 2].tolist())
                    comps.extend([c] * int(m.sum()))
        if cx:
            ax.scatter(cx, cz, s=12, c=color, alpha=alpha, label=label, edgecolors="none")
    ax.set_xlabel("cx (norm)")
    ax.set_ylabel("cz (norm)")
    ax.set_title(f"{family}: compartment centroids (cx–cz)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def _fig_label_radar(
    real_freq: List[float],
    synth_freq: List[float],
    family: str,
    save_path: Path,
    weighting: str = "volume",
) -> None:
    labels = [COMP_SHORT[c] for c in ACTIVE_LABEL_INDICES]
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False)
    angles = np.concatenate([angles, angles[:1]])
    r = np.array(real_freq + [real_freq[0]])
    s = np.array(synth_freq + [synth_freq[0]])

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True), facecolor="white")
    ax.plot(angles, r, "o-", color="#1f77b4", label="real")
    ax.fill(angles, r, alpha=0.15, color="#1f77b4")
    ax.plot(angles, s, "o-", color="#ff7f0e", label="synth")
    ax.fill(angles, s, alpha=0.15, color="#ff7f0e")
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, size=8)
    ax.set_title(f"{family}: label mix ({weighting})", y=1.08)
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.1))
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def _fig_ks_failures(
    ks_failures: Dict[str, Any],
    family: str,
    save_path: Path,
) -> None:
    if not ks_failures:
        return
    feats = list(ks_failures.keys())
    fig, axes = plt.subplots(1, len(feats), figsize=(4 * len(feats), 4), facecolor="white")
    if len(feats) == 1:
        axes = [axes]
    for ax, feat in zip(axes, feats):
        ax.set_facecolor("white")
        entry = ks_failures[feat]
        for side, color in (("real", "#1f77b4"), ("synth", "#ff7f0e")):
            d = entry[side]
            if d["n"] > 0:
                ax.bar(
                    [f"{feat}\n{side}"],
                    [d["mean"]],
                    yerr=[d["std"]],
                    color=color,
                    alpha=0.7,
                    capsize=4,
                )
        ax.set_title(feat)
    fig.suptitle(f"{family}: KS-failing features (mean ± std)")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def validation_report(
    family: str,
    real_graphs: Sequence,
    synth_graphs: Sequence,
    out_dir: Union[str, Path],
    *,
    min_real: int = 2,
    min_synth: int = 5,
) -> Dict[str, Any]:
    """
    Build the fixed-schema validation package for one ship family.

    Parameters
    ----------
    family
        One of ``Bulker``, ``Tanker``, ``Cargo``, ``OSV``, ``Patrol``, ``Yacht``.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if family not in FAMILY_NAME_TO_TYPE:
        raise ValueError(f"Unknown family {family!r}; expected one of {list(FAMILY_NAME_TO_TYPE)}")

    ship_type = int(FAMILY_NAME_TO_TYPE[family])
    real = _filter_family(real_graphs, ship_type)
    synth = _filter_family(synth_graphs, ship_type)

    class_names = [COMP_SHORT[c] for c in ACTIVE_LABEL_INDICES]
    real_freq, freq_basis = _label_freq_vector(real)
    synth_freq, _ = _label_freq_vector(synth)

    centroids: Dict[str, Any] = {}
    for c in ACTIVE_LABEL_INDICES:
        cname = COMP_SHORT[c]
        r_st = _centroid_stats(real, c)
        s_st = _centroid_stats(synth, c)
        delta: Dict[str, Optional[float]] = {}
        if r_st["cx_mean"] is not None and s_st["cx_mean"] is not None:
            delta["cx"] = round(s_st["cx_mean"] - r_st["cx_mean"], 6)
            delta["cz"] = round(s_st["cz_mean"] - r_st["cz_mean"], 6)
        centroids[cname] = {"real": r_st, "synth": s_st, "delta": delta}

    report: Dict[str, Any] = {
        "family": family,
        "ship_type": ship_type,
        "n_real": len(real),
        "n_synth": len(synth),
        "thresholds": METRIC_THRESHOLDS,
        "label_freq": {
            "weighting": freq_basis,
            "class_names": class_names,
            "real": real_freq,
            "synth": synth_freq,
            "JSD": (
                round(jensen_shannon_divergence(
                    np.array(real_freq), np.array(synth_freq),
                ), 6)
                if real and synth else None
            ),
        },
        "centroids": centroids,
        "metrics": {},
        "ks_failures": {},
        "containment": {},
        "containment_summary": {},
        "pass_summary": {},
        "figures": [],
    }

    if len(real) >= min_real and len(synth) >= min_synth:
        report["metrics"] = _collect_metrics(real, synth, family)
        report["ks_failures"] = _ks_failure_details(real, synth, report["metrics"], family)
        report["containment"] = _containment_details(real, synth)
        report["containment_summary"] = _containment_summary(report["containment"])
        report["pass_summary"] = _threshold_pass_summary(report["metrics"])
        report["pass_summary"]["containment"] = report["containment_summary"]
        cov_key = f"Coverage_{family}"
        if cov_key in report["metrics"]:
            report["pass_summary"]["coverage"] = report["metrics"][cov_key]
            report["pass_summary"]["coverage_pass"] = (
                report["metrics"][cov_key] >= METRIC_THRESHOLDS["coverage_target"]
            )
        else:
            report["pass_summary"]["coverage_pass"] = None
    else:
        report["pass_summary"] = {
            "skipped": True,
            "reason": f"need >={min_real} real and >={min_synth} synth "
                      f"(got {len(real)}, {len(synth)})",
        }

    slug = family.lower()
    json_path = out_dir / f"report_{slug}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    if len(real) >= min_real and len(synth) >= min_synth:
        p1 = out_dir / f"report_{slug}_centroids.png"
        p2 = out_dir / f"report_{slug}_label_radar.png"
        _fig_centroid_scatter(real, synth, family, p1)
        _fig_label_radar(real_freq, synth_freq, family, p2, weighting=freq_basis)
        report["figures"].extend([str(p1.name), str(p2.name)])

        if report["ks_failures"]:
            p3 = out_dir / f"report_{slug}_ks_failures.png"
            _fig_ks_failures(report["ks_failures"], family, p3)
            report["figures"].append(str(p3.name))

        if report["containment"]:
            p4 = out_dir / f"report_{slug}_containment.png"
            _fig_containment(report["containment"], family, p4)
            report["figures"].append(str(p4.name))

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

    print(f"Wrote {json_path}  ({len(real)} real, {len(synth)} synth)")
    return report


def validation_report_all(
    real_graphs: Sequence,
    synth_graphs: Sequence,
    out_dir: Union[str, Path],
) -> Dict[str, Dict[str, Any]]:
    """Run ``validation_report`` for every comparable family."""
    out_dir = Path(out_dir)
    reports = {}
    for family in TYPE_NAMES.values():
        reports[family] = validation_report(
            family, real_graphs, synth_graphs, out_dir,
        )
    summary_path = out_dir / "report_index.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(
            {k: {"n_real": v["n_real"], "n_synth": v["n_synth"],
                  "JSD": v["label_freq"].get("JSD"),
                  "coverage": v.get("pass_summary", {}).get("coverage"),
                  "containment": v.get("containment_summary", {}),
                  "figures": v.get("figures", [])}
             for k, v in reports.items()},
            f, indent=2,
        )
    print(f"Wrote {summary_path}")
    _print_containment_rollup(reports)
    return reports


def _print_containment_rollup(reports: Dict[str, Dict[str, Any]]) -> None:
    """Console headline: containment is the lead metric."""
    print("\n" + "=" * 60)
    print("  CONTAINMENT SCORECARD (budget envelope brackets real?)")
    print("=" * 60)
    for fam, rep in reports.items():
        cs = rep.get("containment_summary") or {}
        if not cs:
            print(f"  {fam:8} (skipped — insufficient samples)")
            continue
        gaps = cs.get("budget_gap_keys", [])
        partial = cs.get("budget_partial_keys", [])
        impl = cs.get("implausible_tail_keys", [])
        line = (
            f"  {fam:8} budget_contained={cs.get('budget_contained')}  "
            f"covered={cs.get('budget_covered_or_contained')}"
        )
        if gaps:
            line += f"  GAP={gaps}"
        if partial:
            line += f"  PARTIAL={partial}"
        if impl:
            line += f"  implausible_tail={impl}"
        print(line)
    print("=" * 60)
