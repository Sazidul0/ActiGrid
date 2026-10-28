"""Rebuild the numbers in the paper's Tables I-III from the result files.

Usage (from the repository root):
    python scripts/make_paper_tables.py

Reads results/controlled_same_backbone/ (ten seeds, one shared backbone per
seed; Table I and the paired effects quoted in the text) and
results/reviewer_hardening/ (three group splits and validation-selected HGB
subsets, Table II; seed-42 calibration and corruption, Table III). Prints
LaTeX table rows; nothing is retrained.
"""
import csv
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CTL = ROOT / "results" / "controlled_same_backbone"
HARD = ROOT / "results" / "reviewer_hardening"

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
    rows = [("FullFusion", "Full fusion"), ("ActiGrid-DQN", "DQN ($\\lambda=0.5$)"),
            ("BestOrder-EntropyStop", "Order+entropy")]
    ks = (2,) if ds.startswith("TU") else (2, 3)
    return rows + [(f"BestFixedSubset-k{k}", f"Fixed subset, $k={k}$") for k in ks]


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
    print("\n% Paired effects quoted in Sec. V-A: DQN minus comparator (95% t CI, Wilcoxon p)")
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


def rows(name):
    with open(HARD / name, newline="") as f:
        return list(csv.DictReader(f))


def table_hardening():
    print("\n% Table II: three group splits (seeds 42/123/456), mean +- s.d. over splits")
    hgb_k = {"TU_Delft_2026_CPPS": (1, 2, 3), "MSU_ORNL_Power": (2, 3, 5)}
    for ds, label, M in DATASETS:
        summ = {r["method"]: r for r in rows(f"{ds}_repeated_group_split_summary.csv")}
        for method, name in (("DQN", "DQN"), ("OrderEntropy", "Order+entropy")):
            r = summ[method]
            print(f"{label:9s} & {name:14s} & ${float(r['macro_f1_mean']):.4f}\\pm"
                  f"{float(r['macro_f1_sd_split']):.4f}$ & {float(r['cost_mean']):.3f} \\\\")
        sel = rows(f"{ds}_strong_subset_validation_selected.csv")
        for k in hgb_k[ds]:
            picked = [r for r in sel if r["model"] == "HGB" and int(r["k"]) == k]
            f1 = [float(r["test_macro_f1"]) for r in picked]
            print(f"{label:9s} & HGB, $k={k}$     & ${statistics.mean(f1):.4f}\\pm{statistics.stdev(f1):.4f}$ & "
                  f"{float(picked[0]['normalized_cost']):.3f} \\\\  % subsets: "
                  + ", ".join(r["subset"] for r in picked))


def table_calibration_corruption():
    print("\n% Table III: mask-wise test ECE (mean over masks) and first-modality corruption, seed-42 DQN")
    for ds, label, M in DATASETS:
        cal = rows(f"{ds}_mask_calibration_seed42.csv")
        raw = statistics.mean(float(r["test_ece_raw"]) for r in cal)
        scaled = statistics.mean(float(r["test_ece_temp_scaled"]) for r in cal)
        better = sum(float(r["test_ece_temp_scaled"]) < float(r["test_ece_raw"]) for r in cal)
        corr = {r["attack"]: r for r in rows(f"{ds}_premature_stop_corruption_stress.csv")}
        print(f"{label:9s} & ${raw:.3f}\\rightarrow{scaled:.3f}$ & ${float(corr['zero']['delta_macro_f1']):.3f}$ & "
              f"${float(corr['noise']['delta_macro_f1']):.3f}$ \\\\  % ECE lower on {better}/{len(cal)} masks; "
              f"clean F1 {float(corr['zero']['clean_macro_f1']):.4f}, cost {float(corr['zero']['clean_cost']):.3f}"
              f" -> {float(corr['zero']['adv_cost']):.3f} (zero), {float(corr['noise']['adv_cost']):.3f} (noise)")


if __name__ == "__main__":
    table_main(read("controlled_summary.csv", "method"))
    table_paired(read("controlled_paired_effects.csv", "comparator"))
    table_hardening()
    table_calibration_corruption()
