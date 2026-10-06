"""
similarity_metrics.py
=====================
Quantitative comparison of real GA validation graphs vs CC-extracted
synthetic graphs using well-established distance/similarity measures.

Metrics implemented (grouped by what they measure):

1. DISTRIBUTIONAL — do real and synthetic share the same marginal
   distributions of scalar features?

   · Jensen-Shannon Divergence (JSD) on label frequencies
     - Symmetric KL divergence, bounded [0, 1], 0 = identical
     - Reference: Lin (1991), "Divergence measures based on the
       Shannon entropy"

   · Kolmogorov-Smirnov (KS) test on continuous variables
     - Two-sample non-parametric test, returns (statistic, p-value)
     - p > 0.05 → cannot reject that distributions are the same
     - Reference: Standard scipy.stats.ks_2samp

2. GRAPH STRUCTURAL — do the graphs have similar topology?

   · MMD on degree distributions
   · MMD on clustering coefficient distributions
     - Maximum Mean Discrepancy with RBF kernel
     - Standard GGM evaluation metric from GraphRNN (You et al. 2018)
       and GRAN (Liao et al. 2019)
     - Lower = more similar, 0 = identical distributions
     - Reference: Gretton et al. (2012), "A kernel two-sample test"

3. SPATIAL/DOMAIN — do compartments occupy similar positions?

   · Adjacency pattern similarity
     - Build a Comp×Comp adjacency probability matrix for each dataset
     - Frobenius norm of the difference
     - Lower = more similar adjacency patterns

   · Centroid distribution overlap (per compartment class)
     - KS test on cx and cz distributions for each compartment type

4. COVERAGE — do real samples fall within the synthetic distribution?

   · Coverage metric (Naeem et al. 2020)
     - Fraction of real samples that have at least one synthetic
       neighbour within a threshold distance
     - Computed in 16-d parameter conditioning space
       (type + L/B/D + budgets; LCG/KG excluded)
     - 1.0 = every real ship has a nearby synthetic counterpart
     - Reference: Naeem et al. (2020), "Reliable fidelity and diversity
       metrics for generative models"

All metrics are computed per-family where sample size permits, and
globally across all comparable types.

Output: summary table (console + CSV) and diagnostic PDF figures.
"""

from __future__ import annotations
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict
from scipy import stats
from scipy.spatial.distance import cdist
from validation_constants import (
    ACTIVE_LABEL_INDICES, BUDGET_KEYS, COMP_SHORT, COMPARABLE_TYPES,
    TYPE_NAMES, Comp, parameter_cond_vector,
)

from volume_metrics import label_freq_from_graph_voxels

ACTIVE_COMPS = ACTIVE_LABEL_INDICES  # backward-compatible alias

from typing import Any, Dict, List, Tuple
import csv
import json


def _get(g, attr):
    v = getattr(g, attr) if hasattr(g, attr) else g[attr]
    return v.numpy() if isinstance(v, torch.Tensor) else v


def _get_parameter_cond(g) -> np.ndarray:
    """16-d parameter block from graph ``cond`` (18-d layout also accepted)."""
    return parameter_cond_vector(_get(g, "cond"))


def _budget_fraction(g, ki: int) -> float:
    """Achieved volumetric budget frac when voxels exist; else cond slot."""
    from compare_real_vs_synthetic import _synthetic_actual_budget_fracs

    bf = _synthetic_actual_budget_fracs(g)
    if not np.isnan(bf[ki]):
        return float(bf[ki])
    return float(_get_parameter_cond(g)[9 + ki])
def _group_by_type(graphs):
    d = defaultdict(list)
    for g in graphs:
        d[_get(g, "ship_type")].append(g)
    return d


# ═══════════════════════════════════════════════════════════════
# 1. DISTRIBUTIONAL METRICS
# ═══════════════════════════════════════════════════════════════

def jensen_shannon_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """
    JSD between two probability distributions.

    JSD(P||Q) = 0.5 * KL(P||M) + 0.5 * KL(Q||M)  where M = 0.5*(P+Q)

    Bounded [0, 1] (when using log base 2) or [0, ln(2)] (natural log).
    We use log2 so the result is in [0, 1].
    """
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    # Normalise
    p = p / (p.sum() + 1e-12)
    q = q / (q.sum() + 1e-12)
    # Add small epsilon to avoid log(0)
    eps = 1e-12
    p = p + eps
    q = q + eps
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)
    kl_pm = np.sum(p * np.log2(p / m))
    kl_qm = np.sum(q * np.log2(q / m))
    return float(0.5 * kl_pm + 0.5 * kl_qm)


def mean_label_freq(graphs) -> Tuple[np.ndarray, str]:
    """
    Mean label mix across graphs.

    Returns (freq_vector, basis) where basis is ``volume`` (in-hull m³) or
    ``zone_count`` (fallback when voxel tensors are absent).
    """
    vol_freqs = []
    for g in graphs:
        vf = label_freq_from_graph_voxels(g, tuple(ACTIVE_COMPS), _get=_get)
        if vf is not None:
            vol_freqs.append(vf)
    if vol_freqs:
        return np.mean(vol_freqs, axis=0), "volume"
    all_labels = []
    for g in graphs:
        all_labels.extend(_get(g, "y").tolist())
    total = len(all_labels)
    return np.array([
        sum(1 for l in all_labels if l == c) / max(total, 1)
        for c in ACTIVE_COMPS
    ]), "zone_count"


def compute_label_jsd(real_graphs, synth_graphs) -> Dict[str, float]:
    """
    JSD on compartment label frequency distributions.

    Uses volume-weighted mix when ``voxel_volume_m3`` is attached to graphs;
    otherwise falls back to zone-count frequencies.
    """
    results = {}

    def _label_freq(graphs):
        freq, _ = mean_label_freq(graphs)
        return freq

    # Per family
    real_by = _group_by_type(real_graphs)
    synth_by = _group_by_type(synth_graphs)

    for st in COMPARABLE_TYPES:
        if st in real_by and st in synth_by:
            p = _label_freq(real_by[st])
            q = _label_freq(synth_by[st])
            results[f"JSD_label_{TYPE_NAMES[st]}"] = jensen_shannon_divergence(p, q)

    # Global (all comparable types pooled)
    real_comp = [g for st in COMPARABLE_TYPES for g in real_by.get(st, [])]
    synth_comp = [g for st in COMPARABLE_TYPES for g in synth_by.get(st, [])]
    if real_comp and synth_comp:
        p = _label_freq(real_comp)
        q = _label_freq(synth_comp)
        results["JSD_label_global"] = jensen_shannon_divergence(p, q)

    return results


def compute_ks_tests(real_graphs, synth_graphs) -> Dict[str, Tuple[float, float]]:
    """
    Two-sample KS tests on continuous graph-level features.

    Tests: zone count, edge count, LCG, KG, GM, and each budget key.
    Returns {metric_name: (ks_statistic, p_value)}.

    Interpretation: p > 0.05 means we cannot reject that the real
    and synthetic values come from the same distribution.
    """
    results = {}
    real_by = _group_by_type(real_graphs)
    synth_by = _group_by_type(synth_graphs)

    for st in COMPARABLE_TYPES:
        rg = real_by.get(st, [])
        sg = synth_by.get(st, [])
        if len(rg) < 2 or len(sg) < 2:
            continue

        prefix = TYPE_NAMES[st]

        # Zone count
        r_nz = [_get(g, "n_zones") for g in rg]
        s_nz = [_get(g, "n_zones") for g in sg]
        ks, p = stats.ks_2samp(r_nz, s_nz)
        results[f"KS_zones_{prefix}"] = (ks, p)

        # Edge count
        r_ne = [_get(g, "edge_index").shape[1] // 2 for g in rg]
        s_ne = [_get(g, "edge_index").shape[1] // 2 for g in sg]
        ks, p = stats.ks_2samp(r_ne, s_ne)
        results[f"KS_edges_{prefix}"] = (ks, p)

        # LCG
        r_lcg = [_get(g, "lcg_actual") for g in rg]
        s_lcg = [_get(g, "lcg_actual") for g in sg]
        ks, p = stats.ks_2samp(r_lcg, s_lcg)
        results[f"KS_LCG_{prefix}"] = (ks, p)

        # KG
        r_kg = [_get(g, "kg_actual") for g in rg]
        s_kg = [_get(g, "kg_actual") for g in sg]
        ks, p = stats.ks_2samp(r_kg, s_kg)
        results[f"KS_KG_{prefix}"] = (ks, p)

        # GM
        r_gm = [_get(g, "gm_t") for g in rg]
        s_gm = [_get(g, "gm_t") for g in sg]
        ks, p = stats.ks_2samp(r_gm, s_gm)
        results[f"KS_GM_{prefix}"] = (ks, p)

        # Budget keys (volumetric when voxels attached; cond indices otherwise)
        for ki, bkey in enumerate(BUDGET_KEYS):
            r_vals = [_budget_fraction(g, ki) for g in rg]
            s_vals = [_budget_fraction(g, ki) for g in sg]
            ks, p = stats.ks_2samp(r_vals, s_vals)
            results[f"KS_{bkey}_{prefix}"] = (ks, p)

    return results


# ═══════════════════════════════════════════════════════════════
# 2. GRAPH STRUCTURAL METRICS (MMD)
# ═══════════════════════════════════════════════════════════════

def _degree_distribution(g) -> np.ndarray:
    """Normalised degree histogram for one graph."""
    ei = _get(g, "edge_index")
    n = _get(g, "n_zones")
    if ei.size == 0:
        return np.array([1.0])
    degrees = np.bincount(ei[0], minlength=n)
    # Normalised histogram with fixed bins 0..max_reasonable_degree
    max_deg = min(int(degrees.max()) + 1, 30)
    hist, _ = np.histogram(degrees, bins=np.arange(0, max_deg + 1),
                           density=True)
    return hist


def _clustering_coefficients(g) -> np.ndarray:
    """Clustering coefficient distribution for one graph."""
    import networkx as nx
    ei = _get(g, "edge_index")
    n = _get(g, "n_zones")
    G = nx.Graph()
    G.add_nodes_from(range(n))
    if ei.size > 0:
        edges = set()
        for k in range(ei.shape[1]):
            i, j = int(ei[0, k]), int(ei[1, k])
            if i < j:
                edges.add((i, j))
        G.add_edges_from(edges)
    cc = list(nx.clustering(G).values())
    return np.array(cc) if cc else np.array([0.0])


def mmd_rbf(X: np.ndarray, Y: np.ndarray, gamma: float = 1.0) -> float:
    """
    Maximum Mean Discrepancy with RBF (Gaussian) kernel.

    MMD²(X, Y) = E[k(x,x')] + E[k(y,y')] - 2E[k(x,y)]

    where k(a,b) = exp(-gamma * ||a-b||²)

    Uses the unbiased estimator. Lower = more similar.

    Reference: Gretton et al. (2012)
    """
    n = len(X)
    m = len(Y)
    if n == 0 or m == 0:
        return float("nan")

    XX = cdist(X.reshape(n, -1), X.reshape(n, -1), "sqeuclidean")
    YY = cdist(Y.reshape(m, -1), Y.reshape(m, -1), "sqeuclidean")
    XY = cdist(X.reshape(n, -1), Y.reshape(m, -1), "sqeuclidean")

    kXX = np.exp(-gamma * XX)
    kYY = np.exp(-gamma * YY)
    kXY = np.exp(-gamma * XY)

    # Unbiased estimator (exclude diagonal)
    np.fill_diagonal(kXX, 0)
    np.fill_diagonal(kYY, 0)

    mmd2 = (kXX.sum() / max(n * (n - 1), 1)
            + kYY.sum() / max(m * (m - 1), 1)
            - 2 * kXY.mean())
    return float(max(mmd2, 0.0))


def _pad_histograms(hists: List[np.ndarray]) -> np.ndarray:
    """Pad histograms to same length and stack."""
    max_len = max(len(h) for h in hists)
    padded = np.zeros((len(hists), max_len))
    for i, h in enumerate(hists):
        padded[i, :len(h)] = h
    return padded


def compute_mmd_graph_stats(
    real_graphs, synth_graphs
) -> Dict[str, float]:
    """
    MMD on degree and clustering coefficient distributions.

    For each graph, compute a normalised histogram of node degrees
    (or clustering coefficients). Then compute MMD between the two
    sets of histograms using RBF kernel with median heuristic bandwidth.
    """
    results = {}
    real_by = _group_by_type(real_graphs)
    synth_by = _group_by_type(synth_graphs)

    for st in COMPARABLE_TYPES:
        rg = real_by.get(st, [])
        sg = synth_by.get(st, [])
        if len(rg) < 2 or len(sg) < 5:
            continue
        prefix = TYPE_NAMES[st]

        # Degree MMD
        r_deg = [_degree_distribution(g) for g in rg]
        s_deg = [_degree_distribution(g) for g in sg]
        all_deg = r_deg + s_deg
        all_padded = _pad_histograms(all_deg)
        R = all_padded[:len(r_deg)]
        S = all_padded[len(r_deg):]
        # Median heuristic for gamma
        all_pts = np.vstack([R, S])
        dists = cdist(all_pts, all_pts, "sqeuclidean")
        median_dist = np.median(dists[dists > 0])
        gamma = 1.0 / max(median_dist, 1e-6)
        results[f"MMD_degree_{prefix}"] = mmd_rbf(R, S, gamma)

        # Clustering coefficient MMD
        r_cc = [_clustering_coefficients(g) for g in rg]
        s_cc = [_clustering_coefficients(g) for g in sg]
        # Use mean + std as 2D summary per graph
        R_cc = np.array([[c.mean(), c.std()] for c in r_cc])
        S_cc = np.array([[c.mean(), c.std()] for c in s_cc])
        all_cc = np.vstack([R_cc, S_cc])
        dists_cc = cdist(all_cc, all_cc, "sqeuclidean")
        median_cc = np.median(dists_cc[dists_cc > 0])
        gamma_cc = 1.0 / max(median_cc, 1e-6)
        results[f"MMD_clustering_{prefix}"] = mmd_rbf(R_cc, S_cc, gamma_cc)

    return results


# ═══════════════════════════════════════════════════════════════
# 3. SPATIAL / DOMAIN METRICS
# ═══════════════════════════════════════════════════════════════

def compute_adjacency_similarity(
    real_graphs, synth_graphs
) -> Dict[str, float]:
    """
    Compare compartment adjacency patterns.

    For each dataset, build a 9×9 matrix where entry (i,j) = probability
    that compartment class i is adjacent to class j. Compare via
    Frobenius norm of the difference.
    """
    results = {}
    real_by = _group_by_type(real_graphs)
    synth_by = _group_by_type(synth_graphs)

    n_classes = len(ACTIVE_COMPS)
    comp_to_idx = {c: i for i, c in enumerate(ACTIVE_COMPS)}

    for st in COMPARABLE_TYPES:
        rg = real_by.get(st, [])
        sg = synth_by.get(st, [])
        if not rg or not sg:
            continue

        def _adj_matrix(graphs):
            mat = np.zeros((n_classes, n_classes))
            for g in graphs:
                y_lbl = _get(g, "y")
                ei = _get(g, "edge_index")
                if ei.size == 0:
                    continue
                for k in range(0, ei.shape[1], 2):
                    i, j = int(ei[0, k]), int(ei[1, k])
                    ci = int(y_lbl[i])
                    cj = int(y_lbl[j])
                    if ci in comp_to_idx and cj in comp_to_idx:
                        ii, jj = comp_to_idx[ci], comp_to_idx[cj]
                        mat[ii, jj] += 1
                        mat[jj, ii] += 1
            # Normalise rows to probabilities
            row_sums = mat.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1
            return mat / row_sums

        R_adj = _adj_matrix(rg)
        S_adj = _adj_matrix(sg)
        frob = np.linalg.norm(R_adj - S_adj, "fro")
        results[f"AdjFrob_{TYPE_NAMES[st]}"] = float(frob)

    return results


def compute_centroid_ks(real_graphs, synth_graphs) -> Dict[str, Tuple[float, float]]:
    """
    Per-compartment KS test on centroid positions (cx, cy, cz).

    Tests whether the spatial placement of each compartment type
    follows similar distributions in real vs synthetic.
    """
    results = {}

    for comp_val in ACTIVE_COMPS:
        r_cx, r_cy, r_cz = [], [], []
        s_cx, s_cy, s_cz = [], [], []

        for g in real_graphs:
            x = _get(g, "x"); y_lbl = _get(g, "y")
            mask = y_lbl == comp_val
            if mask.any():
                r_cx.extend(x[mask, 0].tolist())
                r_cy.extend(x[mask, 1].tolist())
                r_cz.extend(x[mask, 2].tolist())

        for g in synth_graphs:
            x = _get(g, "x"); y_lbl = _get(g, "y")
            mask = y_lbl == comp_val
            if mask.any():
                s_cx.extend(x[mask, 0].tolist())
                s_cy.extend(x[mask, 1].tolist())
                s_cz.extend(x[mask, 2].tolist())

        cname = COMP_SHORT[comp_val]
        if len(r_cx) >= 2 and len(s_cx) >= 2:
            ks_cx, p_cx = stats.ks_2samp(r_cx, s_cx)
            ks_cy, p_cy = stats.ks_2samp(r_cy, s_cy)
            ks_cz, p_cz = stats.ks_2samp(r_cz, s_cz)
            results[f"KS_cx_{cname}"] = (ks_cx, p_cx)
            results[f"KS_cy_{cname}"] = (ks_cy, p_cy)
            results[f"KS_cz_{cname}"] = (ks_cz, p_cz)

    return results


# ═══════════════════════════════════════════════════════════════
# 4. COVERAGE METRIC
# ═══════════════════════════════════════════════════════════════

def _fit_cond_scaler(*graph_lists) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fit a per-dimension (mean, std) standardiser on the POOLED conditioning
    vectors of the supplied graph lists (real + synth of the comparison
    being made).

    The 16-d parameter conditioning vector (type + L/B/D + budgets; LCG/KG
    excluded) mixes incommensurate dimensions. On the raw
    scale a single high-variance dimension dominates the Euclidean
    distance in :func:`compute_coverage`, so a 0.0 can mean "off-manifold
    in an unscaled space" rather than "no real–synth overlap". Z-scoring
    on the shared pool puts every dimension on equal footing.

    Zero-variance dimensions (e.g. a constant budget key, or the ship-type
    one-hot inside a single-type pool) get std=1 so they contribute 0 to
    pairwise distances instead of blowing up.
    """
    pool = [_get_parameter_cond(g) for gl in graph_lists for g in gl]
    pool = np.asarray(pool, dtype=np.float64)
    mu = pool.mean(axis=0)
    sigma = pool.std(axis=0)
    sigma[sigma < 1e-12] = 1.0
    return mu, sigma


def _coverage_fraction(R: np.ndarray, S: np.ndarray, k: int) -> float:
    """Naeem-2020 coverage: fraction of real points inside any synthetic
    point's k-th-NN ball, in whatever (standardised) space R, S are given."""
    SS_dists = cdist(S, S, "euclidean")
    np.fill_diagonal(SS_dists, np.inf)
    thresholds = np.sort(SS_dists, axis=1)[:, k - 1]  # k-th NN dist per synth
    RS_dists = cdist(R, S, "euclidean")
    covered = sum(1 for i in range(len(R)) if np.any(RS_dists[i] <= thresholds))
    return covered / len(R)


def compute_coverage(
    real_graphs, synth_graphs, k: int = 5,
) -> Dict[str, float]:
    """
    Coverage: fraction of real samples with a synthetic nearest
    neighbour within a threshold.

    Uses the **16-d parameter** conditioning block (type +
    L/B/D + budgets; LCG/KG excluded), **z-scored per dimension** on the
    pooled real+synth set (see :func:`_fit_cond_scaler`). The scaler is fit
    on the pool that is actually being compared:

      * per-family coverage  → standardised on that family's real+synth
        pool (so within-family L/B/D/budget variation is balanced, and the
        result is identical whether called on the full set or a single
        family — which keeps ``similarity_metrics.json`` and the per-family
        ``report_*.json`` consistent);
      * global coverage      → standardised on the full real+synth pool.

    Threshold = distance to k-th nearest synthetic neighbour
    (adaptive per the method of Naeem et al. 2020).

    Coverage = 1.0 means every real ship has at least one synthetic
    counterpart nearby in (standardised) parameter space.
    """
    results = {}
    real_by = _group_by_type(real_graphs)
    synth_by = _group_by_type(synth_graphs)

    for st in COMPARABLE_TYPES:
        rg = real_by.get(st, [])
        sg = synth_by.get(st, [])
        if not rg or len(sg) < k + 1:
            continue

        mu, sigma = _fit_cond_scaler(rg, sg)  # per-family scaler
        R = (np.array([_get_parameter_cond(g) for g in rg], dtype=np.float64) - mu) / sigma
        S = (np.array([_get_parameter_cond(g) for g in sg], dtype=np.float64) - mu) / sigma
        results[f"Coverage_{TYPE_NAMES[st]}"] = _coverage_fraction(R, S, k)

    # Global — standardised on the full pool (ship-type one-hot now varies)
    all_r = [g for st in COMPARABLE_TYPES for g in real_by.get(st, [])]
    all_s = [g for st in COMPARABLE_TYPES for g in synth_by.get(st, [])]
    if all_r and len(all_s) > k:
        mu, sigma = _fit_cond_scaler(all_r, all_s)  # global scaler
        R = (np.array([_get_parameter_cond(g) for g in all_r], dtype=np.float64) - mu) / sigma
        S = (np.array([_get_parameter_cond(g) for g in all_s], dtype=np.float64) - mu) / sigma
        results["Coverage_global"] = _coverage_fraction(R, S, k)

    return results


# ═══════════════════════════════════════════════════════════════
# REPORTING
# ═══════════════════════════════════════════════════════════════

def _box(title_lines: list[str]) -> None:
    """ASCII banner (cp1252-safe; box-drawing chars break Windows consoles)."""
    width = 61
    print("\n+" + "-" * width + "+")
    for line in title_lines:
        print("|  " + line.ljust(width - 2) + "|")
    print("+" + "-" * width + "+")


def print_summary_table(all_results: Dict[str, Any]):
    """Pretty-print all metrics with interpretations."""

    print("\n" + "=" * 80)
    print("  SIMILARITY METRICS: Real GA Validation vs CC-Extracted Synthetic")
    print("=" * 80)

    # ── JSD ──
    _box([
        "1. LABEL FREQUENCY - Jensen-Shannon Divergence",
        "   JSD in [0, 1]:  0 = identical, <0.05 excellent,",
        "                  <0.10 good, <0.20 acceptable",
    ])
    for key, val in sorted(all_results.items()):
        if key.startswith("JSD"):
            quality = "excellent" if val < 0.05 else "good" if val < 0.10 else "acceptable" if val < 0.20 else "poor"
            print(f"  {key:30s}  {val:.4f}  ({quality})")

    # ── KS tests ──
    _box([
        "2. CONTINUOUS FEATURES - Kolmogorov-Smirnov test",
        "   p > 0.05: cannot reject same distribution (pass)",
        "   p < 0.05: distributions differ significantly (fail)",
    ])

    ks_results = {k: v for k, v in all_results.items() if k.startswith("KS_") and not k.startswith("KS_cx") and not k.startswith("KS_cz")}

    # Group by family
    for st in COMPARABLE_TYPES:
        prefix = TYPE_NAMES[st]
        fam_results = {k: v for k, v in ks_results.items() if k.endswith(f"_{prefix}")}
        if not fam_results:
            continue
        n_pass = sum(1 for v in fam_results.values() if _ks_p_value(v) > 0.05)
        print(f"\n  {prefix} ({n_pass}/{len(fam_results)} pass):")
        for key, val in sorted(fam_results.items()):
            ks_stat = val["statistic"] if isinstance(val, dict) else val[0]
            p_val = _ks_p_value(val)
            short_key = key.replace(f"_{prefix}", "").replace("KS_", "")
            status = "pass" if p_val > 0.05 else "FAIL"
            print(f"    {short_key:18s}  D={ks_stat:.3f}  p={p_val:.3f}  {status}")

    # ── MMD ──
    _box([
        "3. GRAPH STRUCTURE - MMD (RBF kernel, median heuristic)",
        "   Lower = more similar. Values are relative.",
    ])
    for key, val in sorted(all_results.items()):
        if key.startswith("MMD"):
            print(f"  {key:30s}  {val:.6f}")

    # ── Adjacency ──
    _box([
        "4. ADJACENCY PATTERNS - Frobenius norm of difference",
        "   Lower = more similar compartment adjacency patterns.",
    ])
    for key, val in sorted(all_results.items()):
        if key.startswith("AdjFrob"):
            print(f"  {key:30s}  {val:.4f}")

    # ── Centroid KS ──
    _box([
        "5. SPATIAL CENTROIDS - KS test per compartment (cx, cz)",
        "   p > 0.05 = compartment occupies similar positions.",
    ])
    centroid_results = {k: v for k, v in all_results.items()
                        if k.startswith("KS_cx") or k.startswith("KS_cz")}
    for key, (ks_stat, p_val) in sorted(centroid_results.items()):
        status = "pass" if p_val > 0.05 else "FAIL"
        print(f"  {key:20s}  D={ks_stat:.3f}  p={p_val:.3f}  {status}")

    # ── Coverage ──
    _box([
        "6. COVERAGE - fraction of real ships with synthetic",
        "   neighbour in conditioning space (1.0 = full coverage)",
    ])
    for key, val in sorted(all_results.items()):
        if key.startswith("Coverage"):
            quality = "full" if val >= 1.0 else "good" if val >= 0.8 else "partial"
            print(f"  {key:30s}  {val:.3f}  ({quality})")


def save_results_csv(all_results: Dict[str, Any], path: Path):
    """Save metrics to CSV for paper tables."""
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Metric", "Value", "Detail"])
        for key, val in sorted(all_results.items()):
            if key.startswith("_"):
                continue
            if isinstance(val, tuple):
                writer.writerow([key, f"{val[0]:.4f}", f"p={val[1]:.4f}"])
            else:
                writer.writerow([key, f"{val:.6f}", ""])
    print(f"Saved CSV: {path}")


def _ks_p_value(v) -> float:
    """Extract p-value from KS result (tuple or JSON dict)."""
    if isinstance(v, dict):
        return float(v.get("p_value", 0.0))
    if isinstance(v, (tuple, list)) and len(v) >= 2:
        return float(v[1])
    return float(v)


def save_results_json(all_results: Dict[str, Any], path: Path):
    """Save metrics to JSON (includes nested metadata dicts)."""

    def _convert(obj):
        if isinstance(obj, dict):
            return {k: _convert(v) for k, v in obj.items()}
        if isinstance(obj, tuple):
            return {"statistic": round(float(obj[0]), 6), "p_value": round(float(obj[1]), 6)}
        if isinstance(obj, (np.floating, np.integer)):
            return round(float(obj), 6)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, list):
            return [_convert(v) for v in obj]
        if isinstance(obj, (int, float, str, bool)) or obj is None:
            return obj
        return str(obj)

    serialisable = _convert(all_results)
    with open(path, "w") as f:
        json.dump(serialisable, f, indent=2)
    print(f"Saved JSON: {path}")


# ═══════════════════════════════════════════════════════════════
# SUMMARY FIGURE
# ═══════════════════════════════════════════════════════════════

def fig_metrics_summary(all_results: Dict[str, Any], save_path=None):
    """Visual summary of key metrics across families."""
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), facecolor="white")

    family_colors = {0: "#FF7F0E", 1: "#D62728", 2: "#8C564B", 3: "#1F77B4", 4: "#9467BD", 5: "#2CA02C"}

    # ── Panel 1: JSD per family ──
    ax = axes[0][0]
    ax.set_facecolor("white")
    families = [TYPE_NAMES[st] for st in COMPARABLE_TYPES]
    jsd_vals = [all_results.get(f"JSD_label_{f}", 0) for f in families]
    colors = [family_colors[st] for st in COMPARABLE_TYPES]
    bars = ax.bar(families, jsd_vals, color=colors, edgecolor="white", linewidth=0.5)
    ax.axhline(0.05, color="#2CA02C", lw=0.8, ls="--", alpha=0.6, label="Excellent (<0.05)")
    ax.axhline(0.10, color="#FF7F0E", lw=0.8, ls="--", alpha=0.6, label="Good (<0.10)")
    ax.set_ylabel("JSD", fontsize=9, color="#333")
    ax.set_title("Label frequency JSD", fontsize=10, color="#222")
    ax.legend(fontsize=6, frameon=True, edgecolor="#CCC")
    for sp in ax.spines.values(): sp.set_edgecolor("#CCC"); sp.set_linewidth(0.5)
    ax.tick_params(colors="#555", labelsize=7)
    ax.set_xticks(np.arange(len(families)))
    ax.set_xticklabels(families, rotation=30, ha="right")

    # ── Panel 2: Coverage per family ──
    ax = axes[0][1]
    ax.set_facecolor("white")
    cov_vals = [all_results.get(f"Coverage_{f}", 0) for f in families]
    ax.bar(families, cov_vals, color=colors, edgecolor="white", linewidth=0.5)
    ax.axhline(1.0, color="#2CA02C", lw=0.8, ls="--", alpha=0.6)
    ax.set_ylabel("Coverage", fontsize=9, color="#333")
    ax.set_title("Parameter space coverage", fontsize=10, color="#222")
    ax.set_ylim(0, 1.15)
    for sp in ax.spines.values(): sp.set_edgecolor("#CCC"); sp.set_linewidth(0.5)
    ax.tick_params(colors="#555", labelsize=7)

    # ── Panel 3: Adjacency Frobenius per family ──
    ax = axes[1][0]
    ax.set_facecolor("white")
    adj_vals = [all_results.get(f"AdjFrob_{f}", 0) for f in families]
    ax.bar(families, adj_vals, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_ylabel("Frobenius norm", fontsize=9, color="#333")
    ax.set_title("Adjacency pattern difference", fontsize=10, color="#222")
    for sp in ax.spines.values(): sp.set_edgecolor("#CCC"); sp.set_linewidth(0.5)
    ax.tick_params(colors="#555", labelsize=7)

    # ── Panel 4: KS pass rate per family ──
    ax = axes[1][1]
    ax.set_facecolor("white")
    pass_rates = []
    for st in COMPARABLE_TYPES:
        prefix = TYPE_NAMES[st]
        ks_fam = {k: v for k, v in all_results.items()
                  if k.startswith("KS_") and k.endswith(f"_{prefix}")
                  and not k.startswith("KS_cx") and not k.startswith("KS_cz")}
        if ks_fam:
            n_pass = sum(1 for v in ks_fam.values() if _ks_p_value(v) > 0.05)
            pass_rates.append(n_pass / len(ks_fam))
        else:
            pass_rates.append(0)
    ax.bar(families, pass_rates, color=colors, edgecolor="white", linewidth=0.5)
    ax.axhline(0.8, color="#2CA02C", lw=0.8, ls="--", alpha=0.6, label="80% target")
    ax.set_ylabel("Pass rate", fontsize=9, color="#333")
    ax.set_title("KS test pass rate (p > 0.05)", fontsize=10, color="#222")
    ax.set_ylim(0, 1.15)
    ax.legend(fontsize=6, frameon=True, edgecolor="#CCC")
    for sp in ax.spines.values(): sp.set_edgecolor("#CCC"); sp.set_linewidth(0.5)
    ax.tick_params(colors="#555", labelsize=7)

    fig.suptitle("Similarity metrics summary — real GA vs CC-extracted synthetic",
                 fontsize=12, color="#222", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"Saved: {save_path}")
    plt.close(fig)
