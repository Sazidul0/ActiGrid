"""Generate the figures of the revised paper from the released result files.

Usage (from the repository root):
    python scripts/make_paper_figures.py

Every plotted number is read from results/; nothing is retrained or hand-typed.

Sources
-------
results/controlled_same_backbone/controlled_summary.csv
results/controlled_same_backbone/controlled_paired_effects.csv
    Final same-backbone ten-seed run (extracted from
    results/logs/controlled_same_backbone.log by
    scripts/extract_controlled_results.py). Supplies ActiGrid-DQN, full fusion,
    best-order entropy, best fixed subsets, and all paired DQN-minus-comparator
    effects.
results/reviewer_clearance/<ds>_exhaustive_all_subsets.csv
    Test Macro-F1 of every nonempty fixed subset for every seed.
results/reviewer_clearance/<ds>_fixed_order_entropy_budget_curve_summary.csv
    Validation-selected fixed-order entropy-stopping curve over budgets.
The reviewer-clearance subset/entropy results use the same per-seed fusion
backbones as the controlled run; the script asserts that both sources agree.

Writes (into figures/):
    fig_framework.pdf / .png         ActiGrid acquisition loop (Fig. 1)
    fig_tradeoff.pdf / .png          Macro-F1 vs. normalized acquisition cost (Fig. 2)
    fig_paired_effects.pdf / .png    paired DQN-minus-comparator effects (Fig. 3)
"""
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle

ROOT = Path(__file__).resolve().parents[1]
CTL = ROOT / "results" / "controlled_same_backbone"
RC = ROOT / "results" / "reviewer_clearance"
OUT = ROOT / "figures"
DATASETS = [("TU_Delft_2026_CPPS", "TU Delft"), ("MSU_ORNL_Power", "MSU/ORNL")]
SHORT = {  # compact modality names used in subset labels
    "physical": "P", "resource": "R", "network": "N",
    "pmu_r1": "R1", "pmu_r2": "R2", "pmu_r3": "R3", "pmu_r4": "R4", "cyber_logs": "L",
}

# Reference categorical palette (validated: blue/orange pass all-pairs CVD checks);
# marker shape is the secondary encoding, so identity never relies on color alone.
C_DQN, C_ENT = "#eb6834", "#2a78d6"
C_FULL, C_SUB, C_SUBDOT = "#0b0b0b", "#52514e", "#b5b4ae"
C_INK, C_MUTED, C_GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "TeX Gyre Termes", "Nimbus Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.5,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 6.8,
    "axes.edgecolor": C_MUTED, "axes.linewidth": 0.6,
    "xtick.color": C_MUTED, "ytick.color": C_MUTED,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "axes.labelcolor": C_INK, "text.color": C_INK,
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "savefig.dpi": 600, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})


# ---------------------------------------------------------------- data loading
TEXT_COLS = {"dataset", "method", "focal", "comparator"}


def load_controlled():
    def load(name, key):
        out = {}
        for r in read_csv(CTL / name):
            r = {k: (v if k in TEXT_COLS else float(v)) for k, v in r.items()}
            out.setdefault(r["dataset"], {})[r[key]] = r
        return out
    summary = load("controlled_summary.csv", "method")
    paired = load("controlled_paired_effects.csv", "comparator")
    assert set(summary) == set(paired) == {d for d, _ in DATASETS}
    return summary, paired


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def subset_means(ds):
    """Per-subset mean/std of test Macro-F1 over seeds."""
    by = defaultdict(list)
    cost = {}
    for r in read_csv(RC / f"{ds}_exhaustive_all_subsets.csv"):
        by[r["subset"]].append(float(r["test_macro_f1"]))
        cost[r["subset"]] = float(r["normalized_cost"])
    out = []
    for s, v in by.items():
        lab = "+".join(SHORT.get(t, t) for t in s.split("+"))
        out.append((s, lab, cost[s], float(np.mean(v)), float(np.std(v, ddof=1)), len(v)))
    return out


def entropy_curve(ds):
    rows = read_csv(RC / f"{ds}_fixed_order_entropy_budget_curve_summary.csv")
    rows.sort(key=lambda r: float(r["test_cost_mean"]))
    g = lambda k: np.array([float(r[k]) for r in rows])
    return g("test_cost_mean"), g("test_macro_f1_mean"), g("test_macro_f1_std")


def check_consistency(summary):
    """Reviewer-clearance subset/entropy numbers must equal the controlled run."""
    for ds, _ in DATASETS:
        for r in read_csv(RC / f"{ds}_best_subset_per_k_summary.csv"):
            ctl = summary[ds][f"BestFixedSubset-k{r['k']}"]["macro_f1_mean"]
            assert abs(float(r["test_macro_f1_mean"]) - ctl) < 5e-6, (ds, r["k"])
        r = read_csv(RC / f"{ds}_best_order_entropy_summary.csv")[0]
        ctl = summary[ds]["BestOrder-EntropyStop"]
        assert abs(float(r["test_macro_f1_mean"]) - ctl["macro_f1_mean"]) < 5e-6, ds
        assert abs(float(r["test_cost_mean"]) - ctl["normalized_cost_mean"]) < 5e-6, ds


def save(fig, stem):
    OUT.mkdir(exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{stem}.{ext}")
    plt.close(fig)
    print("wrote", OUT / f"{stem}.pdf")


def style_axes(ax):
    ax.grid(True, color=C_GRID, linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


# ------------------------------------------------------------- figure 1: loop
def fig_framework():
    fig, ax = plt.subplots(figsize=(3.5, 2.05))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6.05)
    ax.axis("off")

    def box(x, y, w, h, text, fc="#ffffff", ec=C_MUTED, fs=7, bold=False, lw=0.7):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12",
                                    fc=fc, ec=ec, lw=lw, zorder=2))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
                fontweight="bold" if bold else "normal", zorder=3, linespacing=1.15)

    def arrow(p, q, color=C_MUTED, style="-|>", rad=0.0, lw=0.7, ls="-"):
        ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=6, color=color,
                                     lw=lw, ls=ls, connectionstyle=f"arc3,rad={rad}",
                                     shrinkA=0, shrinkB=0, zorder=1))

    rows = [(4.55, r"$x_1$", r"$f_1$", r"$a_1$"),
            (3.45, r"$x_2$", r"$f_2$", r"$a_2$"),
            (1.85, r"$x_M$", r"$f_M$", r"$a_M$")]
    ax.text(0.65, 1.0, "evidence sources\n(cost $c_m$ each)", ha="center", va="top",
            fontsize=6.3, color=C_MUTED)
    for y, xl, fl, al in rows:
        box(0.1, y, 1.1, 0.75, xl, fc="#f3f2ef")
        box(1.75, y, 1.05, 0.75, fl)
        arrow((1.2, y + 0.375), (1.75, y + 0.375))
        ax.add_patch(Circle((3.45, y + 0.375), 0.27, fc="#ffffff", ec=C_INK, lw=0.7, zorder=2))
        ax.text(3.45, y + 0.375, r"$\times$", ha="center", va="center", fontsize=8, zorder=3)
        ax.text(3.78, y + 0.62, al, ha="left", va="bottom", fontsize=6.5, color=C_DQN)
        arrow((2.8, y + 0.375), (3.18, y + 0.375))
        arrow((3.72, y + 0.375), (4.3, y + 0.375))
    for x in (0.65, 2.27, 3.45):
        ax.text(x, 3.0, r"$\vdots$", ha="center", va="center", fontsize=8, color=C_MUTED)
    ax.text(2.27, 5.62, "encoders", ha="center", fontsize=6.3, color=C_MUTED)
    ax.text(3.45, 5.62, "mask", ha="center", fontsize=6.3, color=C_MUTED)

    box(4.3, 1.85, 1.75, 3.45, "Mask-aware\nfusion\nclassifier\n" + r"$[a_m z_m \Vert a]$"
        + "\n\n(frozen)", fs=6.8)
    box(6.55, 4.35, 3.35, 0.95, r"posterior $p$, entropy $H$", fs=6.8)
    arrow((6.05, 4.82), (6.55, 4.82))
    box(6.55, 2.55, 3.35, 1.3, "Double-DQN policy\n" + r"state $[z(a), p, H, a, b]$", fs=6.8,
        ec=C_DQN, lw=1.0)
    arrow((8.22, 4.35), (8.22, 3.85))
    box(7.75, 0.75, 2.15, 1.0, "STOP:\ndiagnosis $\\hat{y}$", fs=6.8, fc="#f3f2ef")
    arrow((8.82, 2.55), (8.82, 1.75))
    # Acquire action closes the loop back onto the mask gates.
    ax.plot([7.3, 7.3, 3.45], [2.55, 0.3, 0.3], color=C_DQN, lw=0.9, zorder=1)
    arrow((3.45, 0.3), (3.45, 1.95), color=C_DQN, lw=0.9)
    ax.text(5.4, 0.42, r"acquire $m$: $a_m\!\leftarrow\!1$, pay $\lambda c_m$",
            ha="center", va="bottom", fontsize=6.3, color=C_INK)
    save(fig, "fig_framework")


# ------------------------------------------------------- figure 2: trade-off
def fig_tradeoff(summary):
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.55))
    for ax, (ds, title) in zip(axes, DATASETS):
        S = summary[ds]
        M = 3 if ds.startswith("TU") else 5
        subs = subset_means(ds)

        # every fixed subset (mean over seeds)
        c = [s[2] for s in subs]
        f = [s[3] for s in subs]
        ax.scatter(c, f, s=12, color=C_SUBDOT, edgecolors="none", zorder=2,
                   label="Every fixed subset")
        if M == 3:  # few enough to name all of them; stagger labels at equal cost
            by_cost = defaultdict(list)
            for _, lab, cc, ff, _, _ in subs:
                if cc < 0.99:
                    by_cost[round(cc, 3)].append((ff, lab, cc))
            for group in by_cost.values():
                ly = np.inf
                for ff, lab, cc in sorted(group, reverse=True):
                    ly = min(ff, ly - 0.03)
                    ax.annotate(lab, (cc, ff), xytext=(cc - 0.025, ly), textcoords="data",
                                ha="right", va="center", fontsize=6, color=C_MUTED,
                                arrowprops=dict(arrowstyle="-", color=C_SUBDOT, lw=0.4))

        # validation-selected best subset per size k
        ks = range(1, M + 1)
        kc = [k / M for k in ks]
        kf = [S[f"BestFixedSubset-k{k}"]["macro_f1_mean"] for k in ks]
        ke = [S[f"BestFixedSubset-k{k}"]["macro_f1_std"] for k in ks]
        ax.errorbar(kc, kf, yerr=ke, color=C_SUB, lw=1.0, ls="--", marker="o", ms=3.5,
                    mfc="#ffffff", mec=C_SUB, mew=0.9, capsize=1.8, elinewidth=0.6, zorder=3,
                    label="Best fixed subset of size $k$ (val.-selected)")

        # fixed-order entropy-stopping curve
        ec, ef, es = entropy_curve(ds)
        ax.fill_between(ec, ef - es, ef + es, color=C_ENT, alpha=0.12, lw=0, zorder=1)
        ax.plot(ec, ef, color=C_ENT, lw=1.0, zorder=3, label="Fixed-order entropy curve")

        def pt(method, color, marker, ms, label):
            r = S[method]
            ax.errorbar(r["normalized_cost_mean"], r["macro_f1_mean"],
                        xerr=r["normalized_cost_std"], yerr=r["macro_f1_std"],
                        fmt=marker, ms=ms, color=color, mec="#ffffff", mew=0.6,
                        elinewidth=0.8, capsize=2, zorder=5, label=label)

        pt("BestOrder-EntropyStop", C_ENT, "D", 5, "Best-order entropy (cost-matched)")
        pt("FullFusion", C_FULL, "s", 5, "Full fusion")
        pt("ActiGrid-DQN", C_DQN, "*", 10, "ActiGrid-DQN")

        ax.set_title(f"({'ab'[DATASETS.index((ds, title))]}) {title}", loc="left")
        ax.set_xlabel("Normalized acquisition cost")
        ax.set_xlim(0.15 if M == 5 else 0.28, 1.04)
        ax.set_xticks(kc)
        ax.set_xticklabels([f"{k}/{M}" if M == 3 else f"{k / M:.1f}" for k in ks])
        style_axes(ax)
    axes[0].set_ylabel("Test Macro-F1")
    h, l = axes[0].get_legend_handles_labels()
    order = [l.index(x) for x in ("ActiGrid-DQN", "Best-order entropy (cost-matched)",
                                  "Fixed-order entropy curve", "Full fusion",
                                  "Best fixed subset of size $k$ (val.-selected)",
                                  "Every fixed subset")]
    fig.legend([h[i] for i in order], [l[i] for i in order], loc="lower center", ncol=6,
               frameon=False, bbox_to_anchor=(0.5, -0.07), handletextpad=0.3,
               columnspacing=1.0)
    fig.tight_layout(rect=(0, 0.06, 1, 1), w_pad=1.5)
    save(fig, "fig_tradeoff")


# ------------------------------------------------- figure 3: paired effects
# Label placement (offset points, alignment) chosen so labels do not collide.
LABEL_POS = {
    ("TU", "FullFusion"): (6, -4, "left", "top"),
    ("MS", "FullFusion"): (0, 7, "center", "bottom"),
    ("MS", "BestOrder-EntropyStop"): (0, 15, "center", "bottom"),
    ("MS", "BestFixedSubset-k4"): (-6, -10, "right", "top"),
    ("MS", "BestFixedSubset-k3"): (0, -9, "center", "top"),
    ("MS", "BestFixedSubset-k2"): (0, 7, "center", "bottom"),
    ("MS", "BestFixedSubset-k1"): (-6, 0, "right", "center"),
}


def fig_paired(paired):
    names = {"FullFusion": "Full", "BestOrder-EntropyStop": "Order+entropy"}
    fig, axes = plt.subplots(2, 1, figsize=(3.5, 4.1))
    for ax, (ds, title) in zip(axes, DATASETS):
        M = 3 if ds.startswith("TU") else 5
        P = paired[ds]
        comps = ["FullFusion"] + [f"BestFixedSubset-k{k}" for k in range(1, M)] \
            + ["BestOrder-EntropyStop"]  # k=M is identical to full fusion
        ax.axhline(0, color=C_MUTED, lw=0.6, zorder=1)
        ax.axvline(0, color=C_MUTED, lw=0.6, zorder=1)
        for cmp_ in comps:
            r = P[cmp_]
            x, y = r["dqn_minus_comparator_cost"], r["dqn_minus_comparator_f1"]
            xl, xh = r["cost_diff_ci95_low"], r["cost_diff_ci95_high"]
            yl, yh = r["f1_diff_ci95_low"], r["f1_diff_ci95_high"]
            seq = cmp_ == "BestOrder-EntropyStop"
            color = C_ENT if seq else (C_FULL if cmp_ == "FullFusion" else C_SUB)
            marker = "D" if seq else ("s" if cmp_ == "FullFusion" else "o")
            ax.errorbar(x, y, xerr=[[x - xl], [xh - x]], yerr=[[y - yl], [yh - y]],
                        fmt=marker, ms=4.5, color=color, mec="#ffffff", mew=0.5,
                        elinewidth=0.8, capsize=1.8, zorder=4)
            lab = names.get(cmp_, f"Fixed $k$={cmp_[-1]}")
            p = r["wilcoxon_p_f1"]
            dx, dy, ha, va = LABEL_POS.get((ds[:2], cmp_), (5, 3, "left", "bottom"))
            ax.annotate(f"{lab} ($p$={p:.3f})", (x, y), xytext=(dx, dy), textcoords="offset points",
                        ha=ha, va=va, fontsize=6.2, color=C_INK)
        ax.set_title(f"({'ab'[DATASETS.index((ds, title))]}) {title}", loc="left")
        ax.set_ylabel(r"$\Delta$Macro-F1 (DQN $-$ comp.)")
        lo, hi = ax.get_ylim()
        ax.set_ylim(lo - 0.15 * (hi - lo), hi + 0.15 * (hi - lo))
        ax.set_xlim(-0.36, 0.63)
        lo, hi = ax.get_ylim()
        ax.text(-0.35, hi, "DQN better,\ncheaper", ha="left", va="top", fontsize=6,
                color=C_MUTED, style="italic")
        ax.text(0.62, lo, "DQN worse,\ncostlier", ha="right", va="bottom", fontsize=6,
                color=C_MUTED, style="italic")
        style_axes(ax)
    axes[-1].set_xlabel(r"$\Delta$normalized cost (DQN $-$ comparator)")
    fig.tight_layout(h_pad=0.8)
    save(fig, "fig_paired_effects")


if __name__ == "__main__":
    summary, paired = load_controlled()
    check_consistency(summary)
    fig_framework()
    fig_tradeoff(summary)
    fig_paired(paired)
