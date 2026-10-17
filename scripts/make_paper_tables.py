"""Rebuild the numbers in the paper's Tables I-II from the result files.

Usage (from the repository root):
    python scripts/make_paper_tables.py

Reads results/controlled_same_backbone/ (the final same-backbone ten-seed run)
and prints LaTeX table rows plus the derived numbers quoted in the text;
nothing is retrained.
"""
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CTL = ROOT / "results" / "controlled_same_backbone"

# (dataset key, short label, number of modalities M)
DATASETS = [("TU_Delft_2026_CPPS", "TU Delft", 3), ("MSU_ORNL_Power", "MSU/ORNL", 5)]


def read(name, key):
    out = {}
    with open(CTL / name, newline="") as f:
        for r in csv.DictReader(f):
            out[(r["dataset"], r[key])] = r
    return out


def methods_for(ds):
    """Rows shown in Table I: full, DQN, entropy, then the informative fixed subsets."""
    rows = [("FullFusion", "Full fusion"), ("ActiGrid-DQN", "ActiGrid-DQN"),
            ("BestOrder-EntropyStop", "Best-order entropy")]
    ks = (2,) if ds.startswith("TU") else (2, 3)
    return rows + [(f"BestFixedSubset-k{k}", f"Best fixed, $k={k}$") for k in ks]


def table_main(S):
    print("% Table I: controlled ten-seed comparison (mean +- s.d.)")
    for ds, label, M in DATASETS:
        for method, name in methods_for(ds):
            r = S[(ds, method)]
            print(f"{label:9s} & {name:22s} & ${float(r['macro_f1_mean']):.4f}\\pm"
                  f"{float(r['macro_f1_std']):.4f}$ & {float(r['normalized_cost_mean']):.4f} \\\\")
        dqn, full = S[(ds, "ActiGrid-DQN")], S[(ds, "FullFusion")]
        cost = float(dqn["normalized_cost_mean"])
        print(f"%   {label}: DQN retains "
              f"{100 * float(dqn['macro_f1_mean']) / float(full['macro_f1_mean']):.1f}% of "
              f"full-fusion Macro-F1, avoids {100 * (1 - cost):.1f}% of modality units, "
              f"mean {cost * M:.2f} of {M} modalities")


def table_paired(P):
    print("\n% Table II: paired effects, DQN minus comparator (95% t CI, Wilcoxon p)")
    names = {"FullFusion": "Full", "BestOrder-EntropyStop": "Order+entropy"}
    for ds, label, M in DATASETS:
        for comp in ["FullFusion"] + [f"BestFixedSubset-k{k}" for k in range(1, M)] \
                + ["BestOrder-EntropyStop"]:
            r = P[(ds, comp)]
            name = names.get(comp, f"Fixed $k={comp[-1]}$")
            print(f"{label:9s} & {name:16s} & ${float(r['dqn_minus_comparator_f1']):+.4f}$ & "
                  f"$[{float(r['f1_diff_ci95_low']):.4f},{float(r['f1_diff_ci95_high']):.4f}]$ & "
                  f"{float(r['wilcoxon_p_f1']):.4f} \\\\  "
                  f"% dCost {float(r['dqn_minus_comparator_cost']):+.4f} "
                  f"[{float(r['cost_diff_ci95_low']):.4f},{float(r['cost_diff_ci95_high']):.4f}] "
                  f"p={float(r['wilcoxon_p_cost']):.4f}")


if __name__ == "__main__":
    table_main(read("controlled_summary.csv", "method"))
    table_paired(read("controlled_paired_effects.csv", "comparator"))
