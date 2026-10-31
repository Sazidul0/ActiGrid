"""Generate the result figures of the paper from the released result files.

Usage (from the repository root):
    python scripts/make_paper_figures.py

Every plotted number is read from results/; nothing is retrained or hand-typed.
Run scripts/verify_results.py first; it recomputes every file read here from
its per-row inputs. Paper Fig. 1 (the method diagram) is drawn by
scripts/make_method_diagram.py.

Sources
-------
results/next_addon/all_same_backbone_pareto_points.csv
    Seed-42 DQN lambda points, every Greedy-VOI threshold, and the calibrated
    entropy settings that are Pareto optimal on validation (the candidates
    behind the paper's Pareto counts; nothing is filtered on test results).
results/four_issue_closure/<ds>_split42_static_corrected.csv
    Static HGB subset models on partition 42 (= primary split) from the
    corrected rerun; the best subset of each size on the selection half.
results/four_issue_closure/<ds>_split42_sequential_corrected.csv
    Entropy stopping over the per-subset HGB models of partition 42; the
    settings that are Pareto optimal on the policy half of validation.
results/reviewer_hardening/TU_Delft_2026_CPPS_dqn_branching_seed42.csv
results/reviewer_hardening/<ds>_mask_calibration_seed42.csv

Writes (into figures/):
    fig_frontier.pdf / .png      Macro-F1 vs. normalized cost, primary split (Fig. 2)
    fig_calibration.pdf / .png   mask-wise test ECE before/after scaling (Fig. 3)
    fig_branching.pdf / .png     decision after resource telemetry, TU Delft (Fig. 4)
"""

import csv
import os
import tempfile
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "actigrid-matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
ADDON = RESULTS / "next_addon"
HARD = RESULTS / "reviewer_hardening"
CLOSURE = RESULTS / "four_issue_closure"
OUT = ROOT / "figures"
OUT.mkdir(exist_ok=True)

TU, MSU = "TU_Delft_2026_CPPS", "MSU_ORNL_Power"
SEED = "42"  # partition 42 (= primary split) and initialization seed 42

# Fixed categorical colors used consistently across panels.
BLUE, BRICK, OCHRE, GREEN = "#2d68a6", "#b8472f", "#c4951a", "#3b8752"
INK, MUTED, RULE = "#222222", "#6b6b6b", "#d4d4d4"

COL_W = 3.5  # IEEE single column width in inches

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Nimbus Roman", "Times New Roman", "STIXGeneral"],
    "mathtext.fontset": "stix",
    "font.size": 8,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "axes.edgecolor": MUTED,
    "axes.linewidth": 0.6,
    "axes.labelcolor": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.labelcolor": INK,
    "ytick.labelcolor": INK,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.dpi": 300,
})


def read(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def best_per_k(rows):
    """Validation-selected subset for each size k (ties: cheaper, then name)."""
    best = {}
    for r in sorted(rows, key=lambda r: (-float(r["selection_macro_f1"]),
                                         float(r["normalized_cost"]), r["subset"])):
        best.setdefault(r["k"], r)
    return best


def grid(ax, axis="y"):
    ax.grid(axis=axis, color=RULE, linewidth=0.4)
    ax.set_axisbelow(True)


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight", pad_inches=0.02,
                metadata={"CreationDate": None, "ModDate": None})
    fig.savefig(OUT / f"{name}.png", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print("wrote", OUT / f"{name}.pdf")


# --------------------------------------------------------------------------
# Paper Fig. 2: Macro-F1 against normalized cost on the primary split.
#
# The sequential points are exactly the candidates behind the paper's Pareto
# counts (all_same_backbone_pareto_points.csv): every DQN lambda, every
# Greedy-VOI threshold, and the calibrated entropy (order, threshold) settings
# that are Pareto optimal on validation. Nothing is filtered on test results.
# --------------------------------------------------------------------------
def frontier_data(ds):
    pts = defaultdict(set)
    for r in read(ADDON / "all_same_backbone_pareto_points.csv"):
        if r["dataset"] == ds and r["seed"] == SEED:
            pts[r["method_family"]].add((float(r["normalized_cost"]), float(r["macro_f1"])))
    # Tree curves come from the corrected rerun (training-only MSU medians, no
    # HGB early stopping); the neural curves above come from the hardening run.
    static = read(CLOSURE / f"{ds}_split{SEED}_static_corrected.csv")
    hgb = sorted((float(r["normalized_cost"]), float(r["test_macro_f1"]))
                 for r in best_per_k(static).values())
    # Entropy stopping over the per subset HGB models of split 42 (the primary
    # split): the calibrated settings that are Pareto optimal on the policy
    # half of validation, shown at their test scores.
    hgb_seq = sorted({(float(r["test_cost"]), float(r["test_macro_f1"]))
                      for r in read(CLOSURE / f"{ds}_split{SEED}_sequential_corrected.csv")
                      if r["validation_pareto"] == "True"})
    # One threshold sweep, so cost order is threshold order and a line is meaningful.
    voi = sorted(pts["Greedy-VOI"])
    return sorted(pts["Calibrated-OrderEntropy"]), voi, sorted(pts["DQN"]), hgb, hgb_seq


def fig_frontier():
    fig, axes = plt.subplots(1, 2, figsize=(COL_W, 1.85))
    panels = [(TU, "(a) TU Delft", (0.5, 0.9)), (MSU, "(b) MSU/ORNL", (0.25, 0.46))]
    for ax, (ds, label, ylim) in zip(axes, panels):
        ent, voi, dqn, hgb, hgb_seq = frontier_data(ds)
        ax.plot(*zip(*voi), color=OCHRE, lw=0.9, marker="^", ms=2.4, zorder=3)
        # Entropy settings come from different orders, so they are not one curve.
        # Hollow markers are validation-chosen settings of a sequential rule.
        ax.scatter(*zip(*ent), s=7, marker="o", facecolor="white", edgecolor=BLUE,
                   linewidth=0.7, zorder=4)
        ax.scatter(*zip(*hgb_seq), s=5, marker="s", facecolor="white", edgecolor=GREEN,
                   linewidth=0.6, zorder=4)
        ax.plot(*zip(*hgb), color=GREEN, lw=1.1, marker="s", ms=3.4,
                mec="white", mew=0.6, zorder=5)
        # On MSU/ORNL two lambdas land 1 pt apart; no outline, so they merge cleanly.
        ax.scatter(*zip(*dqn), s=12, marker="D", color=BRICK, linewidth=0, zorder=5)
        ax.set_xlim(0.15, 1.04)
        ax.set_ylim(*ylim)
        ax.set_xticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_xlabel("Normalized cost")
        ax.set_title(label, loc="left", fontsize=8, color=INK, pad=3)
        grid(ax, "both")
    axes[0].set_ylabel("Macro-F1")
    ent_h = Line2D([], [], lw=0, marker="o", ms=2.9, mfc="white", mec=BLUE, mew=0.7,
                   label="Calibrated entropy")
    voi_h = Line2D([], [], color=OCHRE, lw=0.9, marker="^", ms=2.4, label="Greedy-VOI")
    dqn_h = Line2D([], [], color=BRICK, lw=0, marker="D", ms=3.4, mec="white", label="DQN")
    hgb_h = Line2D([], [], color=GREEN, lw=1.1, marker="s", ms=3.4, mec="white", label="Static HGB")
    seq_h = Line2D([], [], lw=0, marker="s", ms=2.6, mfc="white", mec=GREEN, mew=0.6,
                   label="Sequential HGB")
    # The legend fills column by column; this order reads row by row.
    handles = [ent_h, hgb_h, voi_h, seq_h, dqn_h]
    fig.legend(handles=handles, loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.97),
               handlelength=1.6, columnspacing=1.0, handletextpad=0.4)
    fig.tight_layout(w_pad=1.0)
    save(fig, "fig_frontier")


# --------------------------------------------------------------------------
# Paper Fig. 4: decision after the first acquisition, TU Delft DQN seed 42.
# --------------------------------------------------------------------------
CLASS_NAMES = {
    "resource_exhaustion": "Resource exh.",
    "reconnaissance": "Recon.",
    "normal": "Normal",
    "physical_fault": "Phys. fault",
    "goose_injection": "GOOSE inj.",
    "sv_injection": "SV inj.",
}


def fig_branching():
    rows = read(HARD / f"{TU}_dqn_branching_seed{SEED}.csv")
    share, n = defaultdict(dict), {}
    for r in rows:
        share[r["class_name"]][r["decision_after_first"]] = float(r["proportion_mean"])
        n[r["class_name"]] = int(r["n_test"])
    order = sorted(share, key=lambda c: share[c].get("STOP", 0.0))  # most STOP on top
    actions = [("STOP", "Stop", BLUE), ("network", "Network", OCHRE),
               ("physical", "Physical", GREEN)]

    fig, ax = plt.subplots(figsize=(COL_W, 1.6))
    for y, cls in enumerate(order):
        left = 0.0
        for key, _, color in actions:
            w = share[cls].get(key, 0.0)
            if w <= 0:
                continue
            ax.barh(y, w, left=left, height=0.66, color=color,
                    edgecolor="white", linewidth=1.0)
            if w >= 0.14:  # label only segments wide enough to hold the text
                ax.text(left + w / 2, y, f"{100 * w:.0f}", ha="center", va="center",
                        fontsize=6.5, color=INK if color == OCHRE else "white")
            left += w
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([f"{CLASS_NAMES[c]} ({n[c]})" for c in order])
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["0", "25", "50", "75", "100"])
    ax.set_xlabel("Test samples (%)")
    ax.spines["left"].set_visible(False)
    handles = [Patch(color=c, label=l) for _, l, c in actions]
    # The caption names the step; a legend title would repeat it.
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.42, 1.0), ncol=3,
              handlelength=1.0, handleheight=0.8, columnspacing=1.2)
    fig.tight_layout()
    save(fig, "fig_branching")


# --------------------------------------------------------------------------
# Paper Fig. 3: mask-wise test ECE before and after temperature scaling.
# --------------------------------------------------------------------------
def fig_calibration():
    fig, axes = plt.subplots(1, 2, figsize=(COL_W, 1.75), sharey=True,
                             gridspec_kw={"width_ratios": [3, 5]})
    for ax, ds, label in [(axes[0], TU, "(a) TU Delft"), (axes[1], MSU, "(b) MSU/ORNL")]:
        rows = read(HARD / f"{ds}_mask_calibration_seed{SEED}.csv")
        by_k = defaultdict(list)
        for r in rows:
            by_k[int(r["k"])].append((float(r["test_ece_raw"]), float(r["test_ece_temp_scaled"])))
        for k, vals in by_k.items():
            vals.sort(reverse=True)
            m = len(vals)
            span = min(0.56, 0.11 * m)
            for i, (raw, cal) in enumerate(vals):
                x = k if m == 1 else k - span / 2 + span * i / (m - 1)
                ax.plot([x, x], [cal, raw], color="#b5b5b5", lw=0.6, zorder=2)
                ax.scatter(x, raw, s=11, facecolor="white", edgecolor=MUTED,
                           linewidth=0.7, zorder=3)
                ax.scatter(x, cal, s=11, color=BLUE, edgecolor="white",
                           linewidth=0.4, zorder=4)
        ks = sorted(by_k)
        ax.set_xticks(ks)
        ax.set_xlim(ks[0] - 0.5, ks[-1] + 0.5)
        ax.set_xlabel("Modalities observed ($k$)")
        ax.set_title(label, loc="left", fontsize=8, color=INK, pad=3)
        grid(ax, "y")
    axes[0].set_ylabel("Test ECE")
    axes[0].set_ylim(0, 0.3)
    handles = [
        Line2D([], [], lw=0, marker="o", ms=3.6, mfc="white", mec=MUTED, label="Raw"),
        Line2D([], [], lw=0, marker="o", ms=3.6, color=BLUE, mec="white",
               label="Temperature scaled"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, bbox_to_anchor=(0.5, 0.97),
               handletextpad=0.3, columnspacing=1.4)
    fig.tight_layout(w_pad=0.8)
    save(fig, "fig_calibration")


if __name__ == "__main__":
    fig_frontier()
    fig_branching()
    fig_calibration()
