"""
side_by_side_comparison.py
==========================
Matched-pair visualisation: real GA vs closest synthetic ship.

One row per real ship (up to three per family): the real graph next to the
generated graph closest in normalised (L, B, D) conditioning space, drawn as
cx vs cz with compartment-coloured nodes and typed edges.
"""

from __future__ import annotations

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection
from collections import defaultdict
from typing import Dict, List

from validation_constants import COMP_SHORT, COMP_PALETTE, GRAPH_LEGEND_EXCLUDE

EDGE_COLORS = {0: "#888888", 1: "#CC6666", 2: "#66AA66", 3: "#6699CC"}

FAMILIES = [
    ("Yacht", 5), ("OSV", 3), ("Bulker", 0),
    ("Tanker", 1), ("Cargo", 2), ("Patrol", 4),
]

LEGEND_EXCLUDE = GRAPH_LEGEND_EXCLUDE
MAX_PER_FAMILY_GRAPH = 3


# ─────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────

def _get(g, attr):
    """Retrieve graph attribute, converting tensors to numpy."""
    v = getattr(g, attr) if hasattr(g, attr) else g[attr]
    return v.numpy() if isinstance(v, torch.Tensor) else v


def _ship_name_str(g) -> str:
    """`ship_name` when set (real GA / CC synth); Stage‑5 graphs omit it — use ``getattr`` only."""
    v = getattr(g, "ship_name", "")
    return v if isinstance(v, str) else ""


def _style_ax(ax, *, grid: bool = True):
    """Apply consistent white-background styling to an axis."""
    ax.set_facecolor("white")
    for sp in ax.spines.values():
        sp.set_edgecolor("#CCC")
        sp.set_linewidth(0.5)
    ax.tick_params(colors="#555", labelsize=6, width=0.5, length=3)
    if grid:
        ax.grid(True, color="#E8E8E8", lw=0.4, alpha=0.8, zorder=0)
    else:
        ax.grid(False)


def _comp_legend(fig):
    """Add shared compartment-colour legend at the bottom of a figure."""
    patches = [
        mpatches.Patch(facecolor=COMP_PALETTE[k], edgecolor="#555",
                       linewidth=0.5, label=COMP_SHORT[k])
        for k in sorted(COMP_PALETTE) if k not in LEGEND_EXCLUDE
    ]
    fig.legend(handles=patches, loc="lower center", ncol=len(patches),
               fontsize=6.5, frameon=True, edgecolor="#CCC",
               facecolor="white", bbox_to_anchor=(0.5, -0.01))


def _volume_fraction(x: np.ndarray) -> np.ndarray:
    """Per-node volume fraction: hull_avail × length × width × height."""
    vf = x[:, 3] * x[:, 4] * x[:, 5] * x[:, 6]
    return vf / max(vf.max(), 1e-6)


def _group_by_type(graphs) -> Dict[int, List]:
    """Group graphs by ship_type, sorted by ship_name within each type."""
    by_type = defaultdict(list)
    for g in graphs:
        by_type[_get(g, "ship_type")].append(g)
    for st in by_type:
        by_type[st].sort(key=_ship_name_str)
    return by_type


def _graph_title(g, prefix=""):
    """Two-line annotation: name + physics summary."""
    nz = _get(g, "n_zones")
    ne = _get(g, "edge_index").shape[1] // 2
    lcg = _get(g, "lcg_actual")
    kg = _get(g, "kg_actual")
    gm = _get(g, "gm_t")
    return f"{prefix}\n{nz}z {ne}e  LCG={lcg:.2f} KG={kg:.2f} GM={gm:.1f}m"


# ─────────────────────────────────────────────────────────────────
# Matching
# ─────────────────────────────────────────────────────────────────

def find_best_match(real_graph, synth_graphs):
    """Find the synthetic graph closest in normalised (L, B, D) space."""
    r_lbd = _get(real_graph, "cond")[6:9]
    best_dist, best = float("inf"), None
    for sg in synth_graphs:
        dist = np.linalg.norm(r_lbd - _get(sg, "cond")[6:9])
        if dist < best_dist:
            best_dist, best = dist, sg
    return best, best_dist


# ─────────────────────────────────────────────────────────────────
# Drawing: graph
# ─────────────────────────────────────────────────────────────────

def _draw_graph(ax, g, title=None, show_labels=True, node_scale=1.0):
    """Draw graph: cx vs cz with comp-coloured nodes and typed edges."""
    x = _get(g, "x")
    y_lbl = _get(g, "y")
    ei = _get(g, "edge_index")
    ea = _get(g, "edge_attr")
    cx, cz = x[:, 0], x[:, 2]
    vol_norm = _volume_fraction(x)
    n, ne = len(cx), (ei.shape[1] if ei.size > 0 else 0)

    # Edges (undirected — step by 2); type one-hot at cols 3–6
    for etype, colour in EDGE_COLORS.items():
        segs = []
        for idx in range(0, ne, 2):
            if ea[idx, 3 + etype] > 0.5:
                i, j = ei[0, idx], ei[1, idx]
                segs.append([(cx[i], cz[i]), (cx[j], cz[j])])
        if segs:
            ax.add_collection(LineCollection(
                segs, colors=colour, linewidths=0.8, alpha=0.45, zorder=1))

    # Nodes
    for i in range(n):
        size = (30 + 250 * vol_norm[i]) * node_scale
        ax.scatter(cx[i], cz[i], s=size,
                   c=COMP_PALETTE.get(y_lbl[i], "#888"),
                   edgecolors="#333", linewidths=0.4, alpha=0.9, zorder=3)
        if show_labels and n <= 22:
            ax.annotate(COMP_SHORT.get(y_lbl[i], "?"), (cx[i], cz[i]),
                        fontsize=4.5, ha="center", va="bottom",
                        xytext=(0, 3.5), textcoords="offset points",
                        color="#333", zorder=5)

    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    if title:
        ax.set_title(title, fontsize=8, color="#222", pad=4)


# ─────────────────────────────────────────────────────────────────
# Figure: side-by-side graph comparison
# ─────────────────────────────────────────────────────────────────

def fig_side_by_side_graphs(real, synth, save_path=None):
    """One row per real ship: left = real, right = matched synthetic."""
    synth_by_type = _group_by_type(synth)
    real_by_type = _group_by_type(real)

    pairs = []
    for fam_name, fam_id in FAMILIES:
        for rg in real_by_type.get(fam_id, [])[:MAX_PER_FAMILY_GRAPH]:
            sg, dist = find_best_match(rg, synth_by_type.get(fam_id, []))
            if sg is not None:
                pairs.append((fam_name, rg, sg, dist))

    n = len(pairs)
    fig, axes = plt.subplots(n, 2, figsize=(9, 3.2 * n),
                             facecolor="white", squeeze=False)

    for row, (fam_name, rg, sg, dist) in enumerate(pairs):
        ax_r, ax_s = axes[row]
        _style_ax(ax_r)
        _style_ax(ax_s)

        _draw_graph(ax_r, rg,
                    title=_graph_title(rg, f"REAL: {_ship_name_str(rg)}"),
                    node_scale=0.9)
        _draw_graph(ax_s, sg,
                    title=_graph_title(sg, f"SYNTHETIC (d={dist:.3f})"),
                    node_scale=0.9)

        ax_r.set_ylabel(f"{fam_name}\ncz (keel→top)",
                        fontsize=7, color="#333", fontweight="bold")
        if row == n - 1:
            ax_r.set_xlabel("cx (aft→bow)", fontsize=7, color="#555")
            ax_s.set_xlabel("cx (aft→bow)", fontsize=7, color="#555")

    _comp_legend(fig)
    fig.suptitle("Real GA vs matched synthetic — graph layout comparison",
                 fontsize=12, color="#222", y=1.0)
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    if save_path:
        fig.savefig(
            save_path, dpi=300, bbox_inches="tight",
            facecolor="white", edgecolor="white",
        )
        print(f"Saved: {save_path}")
    plt.close(fig)
