"""
compare_real_vs_synthetic.py
============================
Publication-quality comparison of real GA vs synthetic graphs.

Figures produced:
  1. Budget fraction comparison (grouped bars per family)
  2. Physics scatter (LCG vs KG, coloured by source)
  3. Spatial heatmaps (cx vs cz per comp class, real vs synth overlay)
  4. Node/edge count distributions
  5. Label frequency stacked bars (real vs synth per family)
  6. Conditioning vector PCA (2D projection, real vs synth)
  7. GM distribution comparison
  8. Output / similarity PCA (achieved LCG, KG, GM + LCG/KG targets from cond,
     label mix, spatial centroids, L/B/D)
  9. Polar compartment label profiles (real vs synthetic per type)
  10. Achieved physics — synthetic boxplots vs real (stars), per type

All: white bg, vector PDF, journal-ready.
"""

from __future__ import annotations
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from collections import defaultdict
from validation_constants import (
    Comp, COMP_SHORT, BUDGET_KEYS, TYPE_NAMES, COMPARABLE_TYPES,
    ACTIVE_LABEL_INDICES, COMP_PALETTE, PLOT_ASSIGNED_COMPS,
    parameter_cond_vector,
)
from sklearn.decomposition import PCA

FAMILY_COLORS = {0:"#FF7F0E", 1:"#D62728", 2:"#8C564B", 3:"#1F77B4", 4:"#9467BD", 5:"#2CA02C"}

OUTPUT_ACTIVE_COMPS = ACTIVE_LABEL_INDICES

# Polar subplots: yacht & OSV first
RADAR_TYPE_ORDER = [5, 3, 0, 1, 2, 4]


def _get(g, attr):
    v = getattr(g, attr) if hasattr(g, attr) else g[attr]
    return v.numpy() if isinstance(v, torch.Tensor) else v


def _clean_ax(ax, xlabel=None, ylabel=None):
    ax.set_facecolor("white")
    for sp in ax.spines.values():
        sp.set_edgecolor("#CCCCCC"); sp.set_linewidth(0.5)
    ax.tick_params(colors="#555", labelsize=7, width=0.5, length=3)
    ax.grid(True, color="#E8E8E8", lw=0.4, alpha=0.8, zorder=0)
    if xlabel: ax.set_xlabel(xlabel, fontsize=8, color="#333")
    if ylabel: ax.set_ylabel(ylabel, fontsize=8, color="#333")


def _group_by_type(graphs):
    d = defaultdict(list)
    for g in graphs:
        d[_get(g, "ship_type")].append(g)
    return d


def _cond_budget_fracs(g) -> np.ndarray:
    """7 budget slots from cond (16-d or 18-d layout)."""
    return np.asarray(parameter_cond_vector(_get(g, "cond"))[9:16], dtype=np.float32)

def _synthetic_actual_budget_fracs(g) -> np.ndarray:
    """
    Achieved budget fractions from stored voxel tensors.

    Uses ``voxel_volume_m3`` when present (native volume-weighted); otherwise
    falls back to uniform cell counts.
    """
    has_vox = hasattr(g, "voxel_labels") or (
        isinstance(g, dict) and "voxel_labels" in g
    )
    if not has_vox:
        return np.full(len(BUDGET_KEYS), np.nan, dtype=np.float32)

    vl = _get(g, "voxel_labels")
    hm = _get(g, "voxel_hull_mask").astype(bool)
    vol = _get(g, "voxel_volume_m3") if hasattr(g, "voxel_volume_m3") or (
        isinstance(g, dict) and "voxel_volume_m3" in g
    ) else None

    in_hull = hm & (vl != int(Comp.EMPTY))
    if vol is not None:
        vol = np.asarray(vol, dtype=np.float64)
        denom = float(vol[in_hull].sum())
        if denom <= 0:
            return np.full(len(BUDGET_KEYS), np.nan, dtype=np.float32)
        budget_comps = [
            Comp.ENGINE_ROOM, Comp.MACHINERY, Comp.CARGO, Comp.STORES,
            Comp.ACCOMMODATION, Comp.FUEL_TANKS, Comp.BALLAST_TANKS,
        ]
        return np.asarray(
            [float(vol[(vl == int(c)) & in_hull].sum()) / denom for c in budget_comps],
            dtype=np.float32,
        )

    total = int(hm.sum())
    if total <= 0:
        return np.full(len(BUDGET_KEYS), np.nan, dtype=np.float32)

    budget_comps = [
        Comp.ENGINE_ROOM, Comp.MACHINERY, Comp.CARGO, Comp.STORES,
        Comp.ACCOMMODATION, Comp.FUEL_TANKS, Comp.BALLAST_TANKS,
    ]
    return np.asarray(
        [float(np.sum((vl == int(c)) & hm)) / total for c in budget_comps],
        dtype=np.float32,
    )

# ──────────────────────────────────────────────
# Fig 1: Budget fractions — real vs synthetic
# ──────────────────────────────────────────────

def fig_budget_comparison(real, synth, save_path=None):
    real_by = _group_by_type(real)
    synth_by = _group_by_type(synth)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5), facecolor="white")
    axes = axes.ravel()

    for idx, st in enumerate(COMPARABLE_TYPES):
        ax = axes[idx]
        _clean_ax(ax, ylabel="Volume fraction")

        r_gs = real_by.get(st, [])
        s_gs = synth_by.get(st, [])

        x_pos = np.arange(len(BUDGET_KEYS))
        w = 0.24

        # Existing synthetic target bars (from cond)
        if s_gs:
            s_targets = np.array([_cond_budget_fracs(g) for g in s_gs], dtype=np.float32)
            s_t_mean = s_targets.mean(axis=0)
            s_t_std = s_targets.std(axis=0)
            ax.bar(
                x_pos - w, s_t_mean, w, yerr=s_t_std, capsize=2,
                color=FAMILY_COLORS[st], alpha=0.25, edgecolor=FAMILY_COLORS[st],
                linewidth=0.8, label=f"Synth target (n={len(s_gs)})",
                error_kw={"lw": 0.5}
            )

        # Synthetic achieved bars (from voxel_labels + voxel_hull_mask)
        if s_gs:
            s_actual = np.array([_synthetic_actual_budget_fracs(g) for g in s_gs], dtype=np.float32)
            s_actual = s_actual[~np.isnan(s_actual).any(axis=1)]
            if len(s_actual) > 0:
                s_a_mean = s_actual.mean(axis=0)
                s_a_std = s_actual.std(axis=0)
                ax.bar(
                    x_pos, s_a_mean, w, yerr=s_a_std, capsize=2,
                    color=FAMILY_COLORS[st], alpha=0.55, edgecolor="white",
                    linewidth=0.5, label=f"Synth actual (n={len(s_actual)})",
                    error_kw={"lw": 0.5}
                )

        # Keep your current real bars unchanged
        if r_gs:
            r_budgets = np.array([_cond_budget_fracs(g) for g in r_gs], dtype=np.float32)
            r_mean = r_budgets.mean(axis=0)
            r_std = r_budgets.std(axis=0) if len(r_gs) > 1 else np.zeros(7)
            ax.bar(
                x_pos + w, r_mean, w, yerr=r_std, capsize=2,
                color=FAMILY_COLORS[st], edgecolor="#333", linewidth=0.5,
                label=f"Real (n={len(r_gs)})", error_kw={"lw": 0.5}
            )

        ax.set_xticks(x_pos)
        ax.set_xticklabels([k.replace("_", "\n") for k in BUDGET_KEYS], fontsize=6)
        ax.set_title(f"{TYPE_NAMES[st]}", fontsize=10, color="#222")
        ax.legend(fontsize=6.5, frameon=True, edgecolor="#CCC")

    fig.suptitle(
        "Volume budget fractions — synthetic target vs synthetic actual vs real",
        fontsize=12, color="#222", y=0.98
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"Saved: {save_path}")
    plt.close(fig)

# ──────────────────────────────────────────────
# Fig 2: Physics scatter — LCG vs KG
# ──────────────────────────────────────────────
def fig_physics_scatter(real, synth, save_path=None):
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), facecolor="white")
    axes = axes.ravel()

    real_by = _group_by_type(real)
    synth_by = _group_by_type(synth)

    for idx, st in enumerate(COMPARABLE_TYPES):
        ax = axes[idx]
        _clean_ax(ax, xlabel="LCG / L (aft=0, bow=1)", ylabel="KG / D (keel=0, top=1)")

        s_gs = synth_by.get(st, [])
        r_gs = real_by.get(st, [])

        if s_gs:
            s_lcg = [_get(g, "lcg_actual") for g in s_gs]
            s_kg = [_get(g, "kg_actual") for g in s_gs]
            ax.scatter(s_lcg, s_kg, s=20, c=FAMILY_COLORS[st], alpha=0.3,
                       edgecolors="none", label=f"Synthetic (n={len(s_gs)})", zorder=2)

        if r_gs:
            r_lcg = [_get(g, "lcg_actual") for g in r_gs]
            r_kg = [_get(g, "kg_actual") for g in r_gs]
            ax.scatter(r_lcg, r_kg, s=80, c=FAMILY_COLORS[st], marker="*",
                       edgecolors="#333", linewidths=0.5,
                       label=f"Real (n={len(r_gs)})", zorder=3)

        ax.set_title(f"{TYPE_NAMES[st]}", fontsize=10, color="#222")
        ax.legend(fontsize=7, frameon=True, edgecolor="#CCC")
        ax.set_xlim(0.3, 0.7)
        ax.set_ylim(0.3, 0.9)

    fig.suptitle("Physics: LCG vs KG — real vs synthetic", fontsize=12, color="#222", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"Saved: {save_path}")
    plt.close(fig)


# ──────────────────────────────────────────────
# Fig 3: Spatial centroid scatter — all comps
# ──────────────────────────────────────────────
def fig_spatial_scatter(real, synth, save_path=None):
    active_comps = PLOT_ASSIGNED_COMPS
    n_comps = len(active_comps)
    cols = 3
    rows = (n_comps + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(11, 3.3 * rows), facecolor="white")
    axes = axes.ravel()

    for ci, comp_val in enumerate(active_comps):
        ax = axes[ci]
        _clean_ax(ax)

        # Synthetic centroids for this comp
        for g in synth:
            x = _get(g, "x"); y_lbl = _get(g, "y")
            mask = y_lbl == comp_val
            if mask.any():
                ax.scatter(x[mask, 0], x[mask, 2], s=8, c=COMP_PALETTE[comp_val],
                           alpha=0.15, edgecolors="none", zorder=2)

        # Real centroids
        for g in real:
            x = _get(g, "x"); y_lbl = _get(g, "y")
            mask = y_lbl == comp_val
            if mask.any():
                ax.scatter(x[mask, 0], x[mask, 2], s=50, c=COMP_PALETTE[comp_val],
                           marker="*", edgecolors="#333", linewidths=0.5, zorder=3)

        ax.set_xlim(-0.05, 1.05); ax.set_ylim(-0.05, 1.05)
        ax.set_title(f"{COMP_SHORT[comp_val]}", fontsize=9, color="#333")
        if ci >= (rows - 1) * cols:
            ax.set_xlabel("cx (aft→bow)", fontsize=7, color="#555")
        if ci % cols == 0:
            ax.set_ylabel("cz (keel→top)", fontsize=7, color="#555")

    for i in range(n_comps, len(axes)):
        axes[i].set_visible(False)

    legend_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#888", markersize=5,
               label="Synthetic", linewidth=0),
        Line2D([0], [0], marker="*", color="w", markerfacecolor="#888",
               markeredgecolor="#333", markersize=9, label="Real", linewidth=0),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=2, fontsize=8,
               frameon=True, edgecolor="#CCC", bbox_to_anchor=(0.5, -0.01))

    fig.suptitle("Zone centroid positions by compartment class — real (stars) vs synthetic (dots)",
                 fontsize=11, color="#222", y=1.0)
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"Saved: {save_path}")
    plt.close(fig)


# ──────────────────────────────────────────────
# Fig 4: Node/edge count distributions
# ──────────────────────────────────────────────
def fig_node_edge_dists(real, synth, save_path=None):
    fig, axes = plt.subplots(2, 3, figsize=(14, 7), facecolor="white")

    real_by = _group_by_type(real)
    synth_by = _group_by_type(synth)

    for idx, st in enumerate(COMPARABLE_TYPES):
        ax = axes[idx // 3][idx % 3]
        _clean_ax(ax, xlabel="Count")

        s_gs = synth_by.get(st, [])
        r_gs = real_by.get(st, [])

        if s_gs:
            s_nz = [_get(g, "n_zones") for g in s_gs]
            s_ne = [_get(g, "edge_index").shape[1] // 2 for g in s_gs]
            ax.hist(s_nz, bins=15, alpha=0.4, color=FAMILY_COLORS[st],
                    label=f"Synth zones (n={len(s_gs)})", edgecolor="white", linewidth=0.5)
            ax.hist(s_ne, bins=15, alpha=0.25, color=FAMILY_COLORS[st],
                    hatch="//", label="Synth edges", edgecolor=FAMILY_COLORS[st], linewidth=0.5)

        if r_gs:
            r_nz = [_get(g, "n_zones") for g in r_gs]
            r_ne = [_get(g, "edge_index").shape[1] // 2 for g in r_gs]
            for v in r_nz:
                ax.axvline(v, color="#333", lw=1.2, ls="-", alpha=0.7)
            for v in r_ne:
                ax.axvline(v, color="#333", lw=0.8, ls="--", alpha=0.5)

            ax.plot([], [], color="#333", lw=1.2, ls="-", label=f"Real zones (n={len(r_gs)})")
            ax.plot([], [], color="#333", lw=0.8, ls="--", label="Real edges")

        ax.set_title(f"{TYPE_NAMES[st]}", fontsize=10, color="#222")
        ax.legend(fontsize=6, frameon=True, edgecolor="#CCC")
        ax.set_ylabel("Count", fontsize=8, color="#333")

    fig.suptitle("Node and edge count distributions — real vs synthetic",
                 fontsize=12, color="#222", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"Saved: {save_path}")
    plt.close(fig)


# ──────────────────────────────────────────────
# Fig 5: Label frequency comparison
# ──────────────────────────────────────────────
def fig_label_freq(real, synth, save_path=None):
    from similarity_metrics import mean_label_freq

    real_by = _group_by_type(real)
    synth_by = _group_by_type(synth)

    fig, axes = plt.subplots(2, 3, figsize=(14, 8), facecolor="white")
    axes = axes.ravel()
    active_comps = ACTIVE_LABEL_INDICES
    ylabel = "Fraction of in-hull volume"

    for idx, st in enumerate(COMPARABLE_TYPES):
        ax = axes[idx]
        _clean_ax(ax)

        x_pos = np.arange(len(active_comps))
        w = 0.35

        s_gs = synth_by.get(st, [])
        if s_gs:
            s_fracs, s_basis = mean_label_freq(s_gs)
            if s_basis == "zone_count":
                ylabel = "Fraction of zones"
            ax.bar(x_pos - w/2, s_fracs, w, color=[COMP_PALETTE[c] for c in active_comps],
                   alpha=0.4, edgecolor="white", linewidth=0.5, label=f"Synthetic (n={len(s_gs)})")

        r_gs = real_by.get(st, [])
        if r_gs:
            r_fracs, r_basis = mean_label_freq(r_gs)
            if r_basis == "zone_count":
                ylabel = "Fraction of zones"
            ax.bar(x_pos + w/2, r_fracs, w, color=[COMP_PALETTE[c] for c in active_comps],
                   edgecolor="#333", linewidth=0.5, label=f"Real (n={len(r_gs)})")

        ax.set_xticks(x_pos)
        ax.set_xticklabels([COMP_SHORT[c] for c in active_comps], fontsize=6, rotation=45)
        ax.set_ylabel(ylabel, fontsize=8, color="#333")
        ax.set_title(f"{TYPE_NAMES[st]}", fontsize=10, color="#222")
        ax.legend(fontsize=7, frameon=True, edgecolor="#CCC")

    fig.suptitle("Label mix (volume-weighted) — real vs synthetic",
                 fontsize=12, color="#222", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"Saved: {save_path}")
    plt.close(fig)


# ──────────────────────────────────────────────
# Fig 6: Conditioning vector PCA
# ──────────────────────────────────────────────
def fig_cond_pca(real, synth, save_path=None):
    fig, ax = plt.subplots(1, 1, figsize=(8, 6), facecolor="white")
    _clean_ax(ax, xlabel="PC1", ylabel="PC2")

    # Collect conditioning vectors
    all_cond = []
    all_types = []
    all_source = []

    for g in synth:
        all_cond.append(parameter_cond_vector(_get(g, "cond")))
        all_types.append(_get(g, "ship_type"))
        all_source.append("synth")

    for g in real:
        all_cond.append(parameter_cond_vector(_get(g, "cond")))
        all_types.append(_get(g, "ship_type"))
        all_source.append("real")

    X = np.array(all_cond)
    pca = PCA(n_components=2)
    Z = pca.fit_transform(X)

    # Plot synthetic
    for st in sorted(set(all_types)):
        mask = [(t == st and s == "synth") for t, s in zip(all_types, all_source)]
        if any(mask):
            pts = Z[mask]
            ax.scatter(pts[:, 0], pts[:, 1], s=15, c=FAMILY_COLORS.get(st, "#888"),
                       alpha=0.3, edgecolors="none", zorder=2)

    # Plot real (on top, with borders)
    for st in sorted(set(all_types)):
        mask = [(t == st and s == "real") for t, s in zip(all_types, all_source)]
        if any(mask):
            pts = Z[mask]
            ax.scatter(pts[:, 0], pts[:, 1], s=120, c=FAMILY_COLORS.get(st, "#888"),
                       marker="*", edgecolors="#333", linewidths=0.5, zorder=3)

    # Legends
    type_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=FAMILY_COLORS.get(st, "#888"),
               markersize=7, label=TYPE_NAMES.get(st, str(st)), linewidth=0)
        for st in sorted(set(all_types))
    ]
    leg1 = ax.legend(handles=type_handles, loc="upper left", fontsize=7,
                     frameon=True, edgecolor="#CCC", title="Ship type", title_fontsize=8)

    source_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#888",
               markersize=5, label="Synthetic", linewidth=0),
        Line2D([0], [0], marker="*", color="w", markerfacecolor="#888",
               markeredgecolor="#333", markersize=9, label="Real", linewidth=0),
    ]
    ax.legend(handles=source_handles, loc="lower right", fontsize=7,
              frameon=True, edgecolor="#CCC", title="Source", title_fontsize=8)
    ax.add_artist(leg1)

    var = pca.explained_variance_ratio_
    ax.set_xlabel(f"PC1 ({var[0]*100:.1f}% var)", fontsize=9, color="#333")
    ax.set_ylabel(f"PC2 ({var[1]*100:.1f}% var)", fontsize=9, color="#333")
    ax.set_title("Conditioning vector PCA — real (stars) vs synthetic (dots)",
                 fontsize=10, color="#222")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"Saved: {save_path}")
    plt.close(fig)


# ──────────────────────────────────────────────
# Fig 7: GM distribution comparison
# ──────────────────────────────────────────────
def fig_gm_comparison(real, synth, save_path=None):
    fig, axes = plt.subplots(2, 3, figsize=(14, 7), facecolor="white")
    axes = axes.ravel()

    real_by = _group_by_type(real)
    synth_by = _group_by_type(synth)

    for idx, st in enumerate(COMPARABLE_TYPES):
        ax = axes[idx]
        _clean_ax(ax, xlabel="GM_t (m)", ylabel="Count")

        s_gs = synth_by.get(st, [])
        r_gs = real_by.get(st, [])

        if s_gs:
            s_gm = [_get(g, "gm_t") for g in s_gs]
            ax.hist(s_gm, bins=20, alpha=0.4, color=FAMILY_COLORS[st],
                    edgecolor="white", linewidth=0.5,
                    label=f"Synthetic (n={len(s_gs)})")

        if r_gs:
            r_gm = [_get(g, "gm_t") for g in r_gs]
            for v in r_gm:
                ax.axvline(v, color="#333", lw=1.5, ls="-", alpha=0.8)
            ax.plot([], [], color="#333", lw=1.5, label=f"Real (n={len(r_gs)})")

        ax.set_title(f"{TYPE_NAMES[st]}", fontsize=10, color="#222")
        ax.legend(fontsize=7, frameon=True, edgecolor="#CCC")

    fig.suptitle("GM transverse distribution — real vs synthetic",
                 fontsize=12, color="#222", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"Saved: {save_path}")
    plt.close(fig)


# ──────────────────────────────────────────────
# Fig 8: Output / similarity PCA (achieved physics + targets + layout stats)
# ──────────────────────────────────────────────
def build_output_embedding_vector(g):
    """
    Fixed-length vector for comparing graphs in "outcome + intent + hull" space:
    achieved LCG/KG/GM, LCG/KG targets (cond), per-class zone fractions and mean
    (cx, cz) centroids, normalised L/B/D from cond.
    """
    x = _get(g, "x")
    y = _get(g, "y")
    cond = _get(g, "cond")

    lcg = float(_get(g, "lcg_actual"))
    kg = float(_get(g, "kg_actual"))
    gm = float(_get(g, "gm_t"))
    gm_norm = float(np.clip(gm / 15.0, 0.0, 1.0))
    lcg_tgt = float(cond[9])
    kg_tgt = float(cond[10])

    n = max(len(y), 1)
    label_freq = np.array(
        [float(np.sum(y == c)) / n for c in OUTPUT_ACTIVE_COMPS], dtype=np.float64
    )

    spatial: list[float] = []
    for c in OUTPUT_ACTIVE_COMPS:
        mask = y == c
        if mask.any():
            spatial.extend([
                float(x[mask, 0].mean()),
                float(x[mask, 1].mean()),
                float(x[mask, 2].mean()),
            ])
        else:
            spatial.extend([0.0, 0.0, 0.0])

    l_n, b_n, d_n = float(cond[6]), float(cond[7]), float(cond[8])
    return np.concatenate(
        [[lcg, kg, gm_norm, lcg_tgt, kg_tgt], label_freq, spatial, [l_n, b_n, d_n]]
    )


def _pooled_zone_label_fracs(graphs, comp_ids):
    if not graphs:
        return np.zeros(len(comp_ids), dtype=np.float64)
    ys = [_get(g, "y") for g in graphs]
    all_y = np.concatenate(ys)
    n = max(len(all_y), 1)
    return np.array([float(np.sum(all_y == c)) / n for c in comp_ids], dtype=np.float64)


def fig_output_space_pca(real, synth, save_path=None):
    all_vecs, all_types, all_source = [], [], []
    for g in synth:
        all_vecs.append(build_output_embedding_vector(g))
        all_types.append(_get(g, "ship_type"))
        all_source.append("synth")
    for g in real:
        all_vecs.append(build_output_embedding_vector(g))
        all_types.append(_get(g, "ship_type"))
        all_source.append("real")

    X = np.asarray(all_vecs, dtype=np.float64)
    pca = PCA(n_components=2)
    Z = pca.fit_transform(X)

    fig, ax = plt.subplots(1, 1, figsize=(8, 6), facecolor="white")
    _clean_ax(ax)

    for st in sorted(set(all_types)):
        mask = [(t == st and s == "synth") for t, s in zip(all_types, all_source)]
        if any(mask):
            ax.scatter(
                Z[mask, 0], Z[mask, 1], s=15, c=FAMILY_COLORS.get(st, "#888"),
                alpha=0.3, edgecolors="none", zorder=2,
            )
    for st in sorted(set(all_types)):
        mask = [(t == st and s == "real") for t, s in zip(all_types, all_source)]
        if any(mask):
            ax.scatter(
                Z[mask, 0], Z[mask, 1], s=120, c=FAMILY_COLORS.get(st, "#888"),
                marker="*", edgecolors="#333", linewidths=0.5, zorder=3,
            )

    var = pca.explained_variance_ratio_
    ax.set_xlabel(f"PC1 ({var[0]*100:.1f}% var)", fontsize=9, color="#333")
    ax.set_ylabel(f"PC2 ({var[1]*100:.1f}% var)", fontsize=9, color="#333")
    ax.set_title(
        "Similarity PCA — achieved physics + LCG/KG targets + labels + centroids + L/B/D",
        fontsize=10, color="#222",
    )

    type_handles = [
        Line2D(
            [0], [0], marker="o", color="w", markerfacecolor=FAMILY_COLORS.get(st, "#888"),
            markersize=7, label=TYPE_NAMES.get(st, str(st)), linewidth=0,
        )
        for st in sorted(set(all_types))
    ]
    leg1 = ax.legend(
        handles=type_handles, loc="upper left", fontsize=7, frameon=True,
        edgecolor="#CCC", title="Ship type", title_fontsize=8,
    )
    source_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#888",
               markersize=5, label="Synthetic", linewidth=0),
        Line2D([0], [0], marker="*", color="w", markerfacecolor="#888",
               markeredgecolor="#333", markersize=9, label="Real", linewidth=0),
    ]
    ax.legend(handles=source_handles, loc="lower right", fontsize=7,
              frameon=True, edgecolor="#CCC", title="Source", title_fontsize=8)
    ax.add_artist(leg1)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"Saved: {save_path}")
    plt.close(fig)


# ──────────────────────────────────────────────
# Fig 9: Polar label profiles per ship type
# ──────────────────────────────────────────────
def fig_radar_label_profiles(real, synth, save_path=None):
    real_by = _group_by_type(real)
    synth_by = _group_by_type(synth)

    fig, axes = plt.subplots(
        2, 3, figsize=(14, 8), facecolor="white", subplot_kw=dict(projection="polar"),
    )

    comp_labels = [COMP_SHORT[c] for c in OUTPUT_ACTIVE_COMPS]
    n_ax = len(OUTPUT_ACTIVE_COMPS)
    angles = np.linspace(0, 2 * np.pi, n_ax, endpoint=False).tolist()
    angles = angles + angles[:1]

    for idx, st in enumerate(RADAR_TYPE_ORDER):
        ax = axes[idx // 3][idx % 3]
        rg = real_by.get(st, [])
        sg = synth_by.get(st, [])

        ymax = 1e-9
        if sg:
            sf = _pooled_zone_label_fracs(sg, OUTPUT_ACTIVE_COMPS)
            s_closed = np.concatenate([sf, sf[:1]])
            ax.fill(angles, s_closed, alpha=0.15, color=FAMILY_COLORS[st])
            ax.plot(angles, s_closed, lw=1.5, color=FAMILY_COLORS[st], alpha=0.5, label="Synthetic")
            ymax = max(ymax, float(sf.max()))
        if rg:
            rf = _pooled_zone_label_fracs(rg, OUTPUT_ACTIVE_COMPS)
            r_closed = np.concatenate([rf, rf[:1]])
            ax.plot(angles, r_closed, lw=2, color=FAMILY_COLORS[st], label="Real")
            ax.scatter(
                angles[:-1], rf, s=25, c=FAMILY_COLORS[st],
                edgecolors="#333", linewidths=0.5, zorder=3,
            )
            ymax = max(ymax, float(rf.max()))

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(comp_labels, fontsize=6)
        ax.set_title(TYPE_NAMES[st], fontsize=10, color="#222", pad=12)
        ax.set_ylim(0, max(ymax * 1.2, 0.01))
        ax.tick_params(labelsize=5)
        ax.legend(fontsize=6, loc="upper right")

    fig.suptitle(
        "Compartment label profiles — real (line + markers) vs synthetic (fill)",
        fontsize=12, color="#222", y=1.02,
    )
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"Saved: {save_path}")
    plt.close(fig)


# ──────────────────────────────────────────────
# Fig 10: Achieved physics — synthetic boxplots vs real stars
# ──────────────────────────────────────────────
def fig_achieved_physics_boxplot(real, synth, save_path=None):
    real_by = _group_by_type(real)
    synth_by = _group_by_type(synth)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), facecolor="white")
    metrics = [("lcg_actual", "LCG / L"), ("kg_actual", "KG / D"), ("gm_t", "GM (m)")]

    for mi, (attr, label) in enumerate(metrics):
        ax = axes[mi]
        _clean_ax(ax)

        for ti, st in enumerate(RADAR_TYPE_ORDER):
            sg = synth_by.get(st, [])
            rg = real_by.get(st, [])
            if sg:
                s_vals = [float(_get(g, attr)) for g in sg]
                bp = ax.boxplot(
                    [s_vals], positions=[ti], widths=0.5, patch_artist=True,
                    showfliers=False, medianprops=dict(color="#333", lw=1),
                )
                bp["boxes"][0].set_facecolor(FAMILY_COLORS[st])
                bp["boxes"][0].set_alpha(0.3)
            if rg:
                r_vals = [float(_get(g, attr)) for g in rg]
                ax.scatter(
                    [ti] * len(r_vals), r_vals, s=60, c=FAMILY_COLORS[st],
                    marker="*", edgecolors="#333", linewidths=0.5, zorder=3,
                )

        ax.set_xticks(range(len(RADAR_TYPE_ORDER)))
        ax.set_xticklabels([TYPE_NAMES[st] for st in RADAR_TYPE_ORDER], fontsize=7, rotation=30, ha="right")
        ax.set_ylabel(label, fontsize=9, color="#333")

    fig.suptitle("Achieved physics — synthetic (boxplot) vs real (stars)", fontsize=11, color="#222")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"Saved: {save_path}")
    plt.close(fig)
