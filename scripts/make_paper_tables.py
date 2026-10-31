"""Rebuild the numbers in the paper's Tables I-VIII from the result files.

Usage (from the repository root):
    python scripts/make_paper_tables.py

Reads results/controlled_same_backbone/ (ten seeds on one shared backbone per
seed; Table I and the paired effects quoted in the text),
results/final_reviewer_clearance/ (policy holdout, MSU sanity check and
mimicry; Tables II, IV and VII), results/four_issue_closure/ (five DQN and ten
HGB partitions, common budgets and static HGB corruption; Tables III, V and
VIII) and results/reviewer_hardening/ (seed-42 calibration and corruption,
Table VI). Prints LaTeX table rows; nothing is retrained.
"""
import csv
import glob
import itertools
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CTL = ROOT / "results" / "controlled_same_backbone"
HARD = ROOT / "results" / "reviewer_hardening"
FINAL = ROOT / "results" / "final_reviewer_clearance"
CLOSURE = ROOT / "results" / "four_issue_closure"

# (dataset key, short label, number of modalities M)
DATASETS = [("TU_Delft_2026_CPPS", "TU Delft", 3), ("MSU_ORNL_Power", "MSU/ORNL", 5)]


def read(name, key):
    out = {}
    with open(CTL / name, newline="") as f:
        for r in csv.DictReader(f):
            out[(r["dataset"], r[key])] = r
    return out


def rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def mean_sd(values):
    return f"${statistics.mean(values):.4f}\\pm{statistics.stdev(values):.4f}$"


def wilcoxon_exact(d):
    """Exact two-sided Wilcoxon signed-rank p-value; zero differences dropped."""
    d = [x for x in d if abs(x) > 1e-12]
    order = sorted(range(len(d)), key=lambda i: abs(d[i]))
    rank = {i: k + 1 for k, i in enumerate(order)}
    w = sum(rank[i] for i in range(len(d)) if d[i] > 0)
    mu = len(d) * (len(d) + 1) / 4
    sums = [sum(rank[i] for i in range(len(d)) if s[i]) for s in itertools.product((0, 1), repeat=len(d))]
    return sum(abs(x - mu) >= abs(w - mu) - 1e-9 for x in sums) / len(sums)


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
                  f"{float(r['macro_f1_std']):.4f}$ & {float(r['normalized_cost_mean']):.3f} \\\\")
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


def table_holdout():
    print("\n% Table II: policy holdout controls, three fits on the primary test split")
    for ds, label, M in DATASETS:
        hold = rows(FINAL / f"{ds}_backbone_policy_holdout_control.csv")
        for method, name in (("DQN", "DQN"), ("OrderEntropy", "Order+entropy"), ("FullFusion", "Full fusion")):
            sel = [r for r in hold if r["method"] == f"PolicyHoldout-{method}"]
            print(f"{label:9s} & {name:14s} & {mean_sd([float(r['test_macro_f1']) for r in sel])} & "
                  f"{statistics.mean(float(r['test_cost']) for r in sel):.3f} \\\\")
        voi = rows(CLOSURE / f"{ds}_GreedyVOI_holdout_ALL.csv")
        print(f"%   {label} Greedy-VOI holdout (separate backbones): "
              f"{mean_sd([float(r['test_macro_f1']) for r in voi])} at cost "
              f"{statistics.mean(float(r['test_cost']) for r in voi):.3f}; holdout unit "
              f"{voi[0]['backbone_policy_holdout_unit']}")


def best_static(static, seed, k):
    """Validation-selected subset of size k on one partition (ties: cheaper, then name)."""
    return min((r for r in static if r["split_seed"] == seed and int(r["k"]) == k),
               key=lambda r: (-float(r["selection_macro_f1"]), float(r["normalized_cost"]), r["subset"]))


def table_partitions():
    print("\n% Table III: five DQN and ten HGB group-disjoint partitions (mean +- s.d.)")
    hgb_k = {"TU_Delft_2026_CPPS": (1, 2, 3), "MSU_ORNL_Power": (2, 3, 5)}
    for ds, label, M in DATASETS:
        dqn = [r for f in sorted(glob.glob(str(CLOSURE / f"{ds}_DQN_split*.csv"))) for r in rows(f)]
        by = {(r["split_seed"], r["method"]): r for r in dqn}
        seeds = sorted({r["split_seed"] for r in dqn}, key=int)
        for method, name in (("DQN", "DQN"), ("OrderEntropy", "Order+entropy"), ("FullFusion", "Full fusion")):
            sel = [by[(s, method)] for s in seeds]
            print(f"{label:9s} & {name:14s} & {len(sel)} & {mean_sd([float(r['macro_f1']) for r in sel])} & "
                  f"{statistics.mean(float(r['cost']) for r in sel):.3f} \\\\")
        d = [float(by[(s, "DQN")]["macro_f1"]) - float(by[(s, "OrderEntropy")]["macro_f1"]) for s in seeds]
        print(f"%   {label}: DQN minus order+entropy {statistics.mean(d):+.4f}, DQN wins "
              f"{sum(x > 0 for x in d)}/{len(d)}, Wilcoxon p={wilcoxon_exact(d):.3f}")
        static = rows(CLOSURE / f"{ds}_10split_static_corrected_ALL.csv")
        split_seeds = sorted({r["split_seed"] for r in static}, key=int)
        for k in hgb_k[ds]:
            picked = [best_static(static, s, k) for s in split_seeds]
            print(f"{label:9s} & HGB, $k={k}$     & {len(picked)} & "
                  f"{mean_sd([float(r['test_macro_f1']) for r in picked])} & "
                  f"{float(picked[0]['normalized_cost']):.3f} \\\\  % subsets: "
                  + ", ".join(r["subset"] for r in picked))


def table_msu_sanity():
    print("\n% Table IV: MSU full-HGB preprocessing and split sanity check")
    for r in rows(FINAL / "MSU_HGB_PREPROCESSING_AND_SPLIT_SANITY.csv"):
        print(f"{r['protocol']}, {r['preprocessing']} & {float(r['accuracy']):.4f} & "
              f"{float(r['balanced_accuracy']):.4f} & {float(r['macro_f1']):.4f} \\\\")


def table_budgets():
    print("\n% Table V: static versus calibrated sequential HGB at common validation budgets, ten partitions")
    for ds, label, M in DATASETS:
        match = rows(CLOSURE / f"{ds}_10split_matched_corrected_ALL.csv")
        for b in sorted({round(float(r["budget"]), 6) for r in match}):
            sel = [r for r in match if abs(float(r["budget"]) - b) < 1e-6]
            d = [float(r["delta_f1_seq_minus_static"]) for r in sel]
            seq_cost = [float(r["seq_test_cost"]) for r in sel]
            print(f"{label:9s} & {b:.3f} & {statistics.mean(float(r['static_test_f1']) for r in sel):.4f} & "
                  f"{statistics.mean(float(r['seq_test_f1']) for r in sel):.4f} & {statistics.mean(d):+.4f} & "
                  f"{sum(x > 0 for x in d)} & {statistics.mean(seq_cost):.3f} \\\\  % delta sd "
                  f"{statistics.stdev(d):.4f}, Wilcoxon p={wilcoxon_exact(d):.3f}, max test cost {max(seq_cost):.3f}")


def table_calibration_corruption():
    print("\n% Table VI: mask-wise test ECE (mean over masks) and first-modality corruption, seed-42 DQN")
    for ds, label, M in DATASETS:
        cal = rows(HARD / f"{ds}_mask_calibration_seed42.csv")
        raw = statistics.mean(float(r["test_ece_raw"]) for r in cal)
        scaled = statistics.mean(float(r["test_ece_temp_scaled"]) for r in cal)
        better = sum(float(r["test_ece_temp_scaled"]) < float(r["test_ece_raw"]) for r in cal)
        corr = {r["attack"]: r for r in rows(HARD / f"{ds}_premature_stop_corruption_stress.csv")}
        print(f"{label:9s} & ${raw:.3f}\\rightarrow{scaled:.3f}$ & ${float(corr['zero']['delta_macro_f1']):.3f}$ & "
              f"${float(corr['noise']['delta_macro_f1']):.3f}$ \\\\  % ECE lower on {better}/{len(cal)} masks; "
              f"clean F1 {float(corr['zero']['clean_macro_f1']):.4f}, cost {float(corr['zero']['clean_cost']):.3f}"
              f" -> {float(corr['zero']['adv_cost']):.3f} (zero), {float(corr['noise']['adv_cost']):.3f} (noise)")


def table_mimicry():
    print("\n% Table VII: mimicry of each DQN's first source, policy holdout seed 42")
    for ds, label, M in DATASETS:
        corr = rows(FINAL / f"{ds}_holdout_policy_corruption_and_mimicry.csv")
        first = next(r["modality"] for r in corr if r["corruption"] == "mimic_normal"
                     and r["method"] == "DQN" and r["is_method_first_modality"] == "True")
        for method, name in (("DQN", "DQN"), ("OrderEntropy", "Entropy"), ("FullFusion", "Full")):
            r = next(x for x in corr if x["corruption"] == "mimic_normal" and x["modality"] == first
                     and x["method"] == method)
            stop = f"{float(r['early_stop_rate_attack']):.3f}" if r["early_stop_rate_attack"] else "N/A"
            print(f"{label:9s} & {name:7s} & {float(r['macro_f1']):.4f} & {float(r['cost']):.3f} & "
                  f"{1 - float(r['attack_detection_recall']):.3f} & {stop} \\\\  % mimicked source: {first}")


def table_static_corruption():
    print("\n% Table VIII: static HGB under single-source corruption, mean of ten partitions")
    print("% The SequentialHGB rows of *_HGB_corruption_ALL.csv are not used: see the README.")
    data = {ds: [r for r in rows(CLOSURE / f"{ds}_10split_HGB_corruption_ALL.csv") if r["method"] == "StaticHGB"]
            for ds, label, M in DATASETS}
    for mode, name in (("clean", "Clean"), ("mean_replace", "Mean replacement"),
                       ("noise", "Gaussian noise"), ("mimic_normal", "Normal mimicry")):
        cells = []
        for ds, label, M in DATASETS:
            sel = [r for r in data[ds] if r["corruption"] == mode]
            cells += [f"{statistics.mean(float(r['macro_f1']) for r in sel):.4f}",
                      f"{statistics.mean(float(r['attack_detection_recall']) for r in sel):.3f}"]
        print(f"{name:16s} & " + " & ".join(cells) + " \\\\")


if __name__ == "__main__":
    table_main(read("controlled_summary.csv", "method"))
    table_paired(read("controlled_paired_effects.csv", "comparator"))
    table_holdout()
    table_partitions()
    table_msu_sanity()
    table_budgets()
    table_calibration_corruption()
    table_mimicry()
    table_static_corruption()
