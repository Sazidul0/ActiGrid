"""Verify the released hardening and Greedy-VOI results and the numbers in the paper.

Cross-checks three sources against each other:
  * results/reviewer_hardening/  (--stage hardening; log results/logs/reviewer_hardening.log)
  * results/next_addon/          (--stage greedy-voi; log results/logs/greedy_voi.log)
  * results/hgb_sequential/      (--stage hgb-sequential; log results/logs/hgb_sequential.log)
  * the two Kaggle run logs and the stage code in actigrid/

It recomputes every derived file from its per-row inputs (summaries, selections,
Pareto flags, AUCs, branching shares, path metrics), matches the numbers
printed in the logs, and finally checks the values quoted in the paper.
Nothing is retrained; no dataset is needed.

Usage (from the repository root):
    python scripts/verify_results.py
Needs:  numpy
"""
import csv
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
HARD = ROOT / "results" / "reviewer_hardening"
ADDON = ROOT / "results" / "next_addon"
HGBSEQ = ROOT / "results" / "hgb_sequential"
LOGS = ROOT / "results" / "logs"
CODE = ROOT / "actigrid"
TU, MSU = "TU_Delft_2026_CPPS", "MSU_ORNL_Power"
DATASETS = (TU, MSU)
SEEDS = ("42", "123", "456")
EPS = 1e-9

failures, notes = [], []


def check(ok, msg):
    print(("  OK   " if ok else "  FAIL ") + msg)
    if not ok:
        failures.append(msg)


def note(msg):
    print("  NOTE " + msg)
    notes.append(msg)


def section(title):
    print(f"\n== {title}")


def read(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def close(a, b, tol=1e-9):
    return abs(float(a) - float(b)) <= tol


def same_rows(a, b, keys, tol=1e-9):
    """Rows equal on `keys` (numbers to tol, everything else exactly), order-free."""
    def key(r):
        out = []
        for k in keys:
            try:
                out.append(round(float(r[k]) / tol) if r[k] not in ("True", "False") else r[k])
            except ValueError:
                out.append(r[k])
        return tuple(out)
    return Counter(map(key, a)) == Counter(map(key, b))


def pareto_mask(cost, score):
    """Same rule as pareto_mask() in the add-on script."""
    cost, score = np.asarray(cost, float), np.asarray(score, float)
    keep = np.ones(len(cost), bool)
    for i in range(len(cost)):
        dom = (cost <= cost[i] + 1e-12) & (score >= score[i] - 1e-12) & (
            (cost < cost[i] - 1e-12) | (score > score[i] + 1e-12))
        dom[i] = False
        keep[i] = not dom.any()
    return keep


def macro_f1(y, p):
    scores = []
    for c in np.unique(np.concatenate([y, p])):
        tp = np.sum((y == c) & (p == c))
        fp = np.sum((y != c) & (p == c))
        fn = np.sum((y == c) & (p != c))
        scores.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(scores))


def observed_auc(points):
    """Same rule as observed_auc() in the add-on: sort by cost, keep last per cost."""
    d = {}
    for c, f in sorted(points, key=lambda t: t[0]):
        d[c] = f
    xx, yy = np.array(list(d)), np.array(list(d.values()))
    if len(xx) < 2:
        return float("nan"), float("nan"), float("nan"), float("nan")
    auc = float(np.sum((xx[1:] - xx[:-1]) * (yy[1:] + yy[:-1]) / 2))
    span = xx[-1] - xx[0]
    return auc, xx[0], xx[-1], auc / span if span > 0 else float("nan")


def short_lines(path):
    """Log lines without the embedded base64 payloads."""
    return [ln.rstrip("\n") for ln in open(path, errors="replace") if len(ln) < 400]


# ============================================================================
section("1. Released fusion checkpoints")
for ds in DATASETS:
    for kind, seeds in (("hardening_fusion_seed", SEEDS), ("splitrobust_fusion_split", SEEDS)):
        missing = [s for s in seeds if not (HARD / "cache" / f"{ds}_{kind}{s}.pt").exists()]
        check(not missing, f"{ds} cache/*_{kind}{{42,123,456}}.pt present")

# ============================================================================
section("2. Run logs (results/logs/) and stage code")
log1 = short_lines(LOGS / "reviewer_hardening.log")
log2 = short_lines(LOGS / "greedy_voi.log")
hard_code = (CODE / "hardening.py").read_text()
voi_code = (CODE / "greedy_voi.py").read_text()
text1 = "\n".join(log1)

check("Profile: conference | datasets: ['tu', 'msu'] | device: cpu" in text1,
      "hardening run used the conference profile on CPU")
check(sum("REVIEW HARDENING COMPLETE" == ln.strip() for ln in log1) >= 1, "hardening run completed")
check("P.CFG.final_multiseed_dqn_episodes = 20000" in hard_code, "hardening DQNs use 20,000 episodes")
check(any("NEXT ADD-ON COMPLETE" == ln.strip() for ln in log2), "Greedy-VOI add-on completed")
check(any("GPU: Tesla T4" in ln for ln in log2), "Greedy-VOI add-on ran on a Tesla T4")
fit = Counter(re.findall(r"VOI policy-fit sample: (\d+) / (\d+)", "\n".join(log2)))
check(fit == Counter({("3246", "3246"): 3, ("7000", "47471"): 3}),
      f"Greedy-VOI fit sizes 3,246/3,246 (TU) and 7,000/47,471 (MSU) per seed: {dict(fit)}")
check("split = P.group_split(bundle, seed=42)" in voi_code
      and "_hardening_fusion_seed{seed}.pt" in voi_code,
      "Greedy-VOI reuses the hardening checkpoints on group_split(seed=42)")

# The split printed just before each split-robust fusion model is that split seed's split.
PRIMARY = {TU: ("3246", "1276", "1316"), MSU: ("47471", "15272", "15634")}
last_split = {}
robust = defaultdict(set)
for ln in log1:
    m = re.search(r"Split (\S+): train=(\d+) val=(\d+) test=(\d+)", ln)
    if m:
        last_split[m.group(1)] = m.group(2, 3, 4)
    m = re.search(r"Training fusion classifier \[splitrobust_(\S+)_(\d+)\]", ln)
    if m:
        robust[(m.group(1), m.group(2))].add(last_split[m.group(1)])
for ds in DATASETS:
    if (ds, "42") in robust:
        check(robust[(ds, "42")] == {PRIMARY[ds]},
              f"{ds} repeated split seed 42 reproduces the primary split {'/'.join(PRIMARY[ds])}")
        check(all(robust[(ds, s)] != {PRIMARY[ds]} for s in ("123", "456") if (ds, s) in robust),
              f"{ds} partition seeds 123 and 456 differ from the primary split")
    else:
        note(f"{ds}: split-robust training lines not in this log (run was resumed past them)")
test_counts = {}
for i, ln in enumerate(log1):
    m = re.search(r"Split (\S+): train=\d+ val=\d+ test=(\d+)", ln)
    if m and m.group(1) not in test_counts:
        tc = re.search(r"Test  classes: (.*)", log1[i + 3])
        test_counts[m.group(1)] = {k: int(v) for k, v in re.findall(r"(\w+)=(\d+)", tc.group(1))}

# Per-subset tree scores printed during the hardening run.
logged = {}
for ds, split, model, subset, val in re.findall(
        r"\[checkpoint subset\] (\S+) (\d+) (\w+) (\S+) ([0-9.]+)", text1):
    logged[(ds, split, model, subset)] = float(val)
for ds in DATASETS:
    rows = read(HARD / f"{ds}_strong_subset_classifier_frontier.csv")
    hits = [(r, logged[(ds, r["split_seed"], r["model"], r["subset"])]) for r in rows
            if (ds, r["split_seed"], r["model"], r["subset"]) in logged]
    which = "test_macro_f1" if all(close(r["test_macro_f1"], v, 1e-12) for r, v in hits) else \
            "val_macro_f1" if all(close(r["val_macro_f1"], v, 1e-12) for r, v in hits) else None
    check(which is not None,
          f"{ds}: {len(hits)}/{len(rows)} logged subset scores match the frontier CSV ({which})")
    if len(hits) < len(rows):
        note(f"{ds}: {len(rows) - len(hits)} frontier rows have no printed log line "
             f"(log.txt is a resumed run; earlier rows were restored from checkpoints)")

# ============================================================================
section("3. reviewer_hardening/ derived files")
for ds in DATASETS:
    audit = read(HARD / f"{ds}_repeated_group_split_policy_audit.csv")
    summ = {r["method"]: r for r in read(HARD / f"{ds}_repeated_group_split_summary.csv")}
    for method, s in summ.items():
        f = np.array([float(r["macro_f1"]) for r in audit if r["method"] == method])
        c = np.array([float(r["cost"]) for r in audit if r["method"] == method])
        check(close(s["macro_f1_mean"], f.mean()) and close(s["macro_f1_sd_split"], f.std(ddof=1))
              and close(s["cost_mean"], c.mean()) and int(s["n_splits"]) == len(f),
              f"{ds} repeated-split summary {method} = mean/SD of the 3 audit rows")

    lam = read(HARD / f"{ds}_dqn_lambda_frontier.csv")
    sel = [r for r in read(HARD / "hardening_dqn_selected_summary.csv") if r["dataset"] == ds][0]
    f = np.array([float(r["macro_f1"]) for r in lam if float(r["lambda"]) == 0.5])
    c = np.array([float(r["normalized_cost"]) for r in lam if float(r["lambda"]) == 0.5])
    check(close(sel["macro_f1_mean"], f.mean()) and close(sel["macro_f1_sd"], f.std(ddof=1))
          and close(sel["cost_mean"], c.mean()),
          f"{ds} hardening_dqn_selected_summary = lambda 0.5 rows of the lambda frontier")

    front = read(HARD / f"{ds}_strong_subset_classifier_frontier.csv")
    chosen = read(HARD / f"{ds}_strong_subset_validation_selected.csv")
    groups = defaultdict(list)
    for r in front:
        groups[(r["split_seed"], r["model"], r["k"])].append(r)
    picks = {g: max(rs, key=lambda r: float(r["val_macro_f1"]))["subset"] for g, rs in groups.items()}
    check(len(chosen) == len(groups) and
          all(picks[(r["split_seed"], r["model"], r["k"])] == r["subset"] for r in chosen),
          f"{ds} validation-selected subsets are the val-best per (split, model, k): {len(chosen)} rows")

    paths = read(HARD / f"{ds}_dqn_paths_seed42.csv")
    lam42 = [r for r in lam if r["seed"] == "42" and float(r["lambda"]) == 0.5][0]
    pc = np.array([float(r["cost"]) for r in paths])
    acc = np.mean([r["correct"] in ("1", "True", "true") for r in paths])
    check(close(pc.mean(), lam42["normalized_cost"], 1e-9) and close(acc, lam42["accuracy"], 1e-9),
          f"{ds} seed-42 DQN paths reproduce the lambda 0.5 cost {pc.mean():.4f} and accuracy {acc:.4f}")
    corr = read(HARD / f"{ds}_premature_stop_corruption_stress.csv")
    check(all(close(r["clean_macro_f1"], lam42["macro_f1"]) and close(r["clean_cost"], lam42["normalized_cost"])
              for r in corr),
          f"{ds} corruption 'clean' row is the seed-42 lambda 0.5 DQN")
    check(all(close(r["delta_macro_f1"], float(r["adv_macro_f1"]) - float(r["clean_macro_f1"]))
              for r in corr), f"{ds} corruption deltas = adv - clean")

    by_class = defaultdict(list)
    for r in paths:
        by_class[r["class_name"]].append(r)
    branch = read(HARD / f"{ds}_dqn_branching_seed42.csv")
    ok = all(close(r["proportion_mean"],
                   np.mean([p["second_action_or_stop"] == r["decision_after_first"]
                            for p in by_class[r["class_name"]]]), 1e-12)
             and int(r["n_test"]) == len(by_class[r["class_name"]]) for r in branch)
    check(ok, f"{ds} branching shares recomputed from {len(paths)} DQN paths")
    sums = defaultdict(float)
    for r in branch:
        sums[r["class_name"]] += float(r["proportion_mean"])
    check(all(close(v, 1.0, 1e-9) for v in sums.values()), f"{ds} branching shares sum to 1 per class")
    check(test_counts.get(ds) == {c: len(v) for c, v in by_class.items()},
          f"{ds} path class counts match the logged test split")
    pcc = read(HARD / f"{ds}_per_class_acquisition_cost_seed42.csv")
    check(all(close(r["mean_cost"], np.mean([float(p["cost"]) for p in by_class[r["class_name"]]]))
              and int(r["n"]) == len(by_class[r["class_name"]]) for r in pcc),
          f"{ds} per-class acquisition cost recomputed from paths")

    cal = read(HARD / f"{ds}_mask_calibration_seed42.csv")
    check(all(int(r["k"]) == r["subset"].count("+") + 1 for r in cal)
          and len(cal) == {TU: 7, MSU: 31}[ds] and len({r["subset"] for r in cal}) == len(cal),
          f"{ds} mask calibration covers all {len(cal)} nonempty masks once")

# ============================================================================
section("4. next_addon/ derived files")
for ds in DATASETS:
    lam = read(HARD / f"{ds}_dqn_lambda_frontier.csv")  # the add-on reads this from the hardening run
    for kind in ("calibrated_order_entropy_frontier", "greedy_voi_frontier"):
        per_seed = [r for s in SEEDS for r in read(ADDON / f"{ds}_{kind}_seed{s}.csv")]
        combined = read(ADDON / f"{ds}_{kind}.csv")
        check(same_rows(per_seed, combined, list(combined[0].keys())),
              f"{ds} {kind}.csv = concat of the 3 seed files ({len(combined)} rows)")

    ent_all, voi_all = [], []
    for s in SEEDS:
        ent = read(ADDON / f"{ds}_calibrated_order_entropy_frontier_seed{s}.csv")
        keep = pareto_mask([r["val_cost"] for r in ent], [r["val_macro_f1"] for r in ent])
        check(all((r["val_pareto"] == "True") == k for r, k in zip(ent, keep)),
              f"{ds} seed {s} entropy val_pareto flags recomputed ({int(keep.sum())}/{len(ent)})")
        ent_all += [r for r in ent if r["val_pareto"] == "True"]
        voi = read(ADDON / f"{ds}_greedy_voi_frontier_seed{s}.csv")
        check(len(voi) == 17, f"{ds} seed {s} Greedy-VOI frontier has 17 thresholds (15 quantiles + 2 extremes)")
        keep = pareto_mask([r["normalized_cost"] for r in voi], [r["macro_f1"] for r in voi])
        check(all((r["pareto_within_greedy"] == "True") == k for r, k in zip(voi, keep)),
              f"{ds} seed {s} Greedy-VOI pareto_within_greedy flags recomputed (test-set flag)")
        voi_all += voi

        tr = read(ADDON / f"{ds}_greedy_voi_paths_seed{s}.csv")
        t = float(tr[0]["threshold"])
        row = [r for r in voi if close(r["threshold"], t, 1e-12)]
        y = np.array([int(r["y_true"]) for r in tr])
        p = np.array([int(r["y_pred"]) for r in tr])
        cost = np.mean([float(r["normalized_cost"]) for r in tr])
        check(len(row) == 1 and close(macro_f1(y, p), row[0]["macro_f1"], 1e-9)
              and close(cost, row[0]["normalized_cost"], 1e-9),
              f"{ds} seed {s} Greedy-VOI paths reproduce frontier row at threshold {t:.4f} "
              f"(F1 {macro_f1(y, p):.4f}, cost {cost:.4f})")

        temps = {int(r["mask_int"]): float(r["temperature"])
                 for r in read(ADDON / f"{ds}_mask_temperatures_seed{s}.csv")}
        check(len(temps) == {TU: 7, MSU: 31}[ds] and 0 not in temps,
              f"{ds} seed {s} has one temperature per nonempty mask")

    # Rebuild the same-backbone comparison exactly as build_same_backbone_comparison().
    rebuilt = []
    for r in lam:
        rebuilt.append({"seed": r["seed"], "macro_f1": r["macro_f1"], "normalized_cost": r["normalized_cost"],
                        "method_family": "DQN"})
    for r in voi_all:
        rebuilt.append({"seed": r["seed"], "macro_f1": r["macro_f1"], "normalized_cost": r["normalized_cost"],
                        "method_family": "Greedy-VOI"})
    for r in ent_all:
        rebuilt.append({"seed": r["seed"], "macro_f1": r["macro_f1"], "normalized_cost": r["normalized_cost"],
                        "method_family": "Calibrated-OrderEntropy"})
    for s in SEEDS:
        sub = [r for r in rebuilt if r["seed"] == s]
        for r, k in zip(sub, pareto_mask([r["normalized_cost"] for r in sub], [r["macro_f1"] for r in sub])):
            r["pareto"] = str(bool(k))
    comp = read(ADDON / f"{ds}_DQN_vs_GreedyVOI_same_backbone.csv")
    check(same_rows(rebuilt, comp, ["seed", "macro_f1", "normalized_cost", "method_family", "pareto"]),
          f"{ds} DQN_vs_GreedyVOI_same_backbone rebuilt with identical Pareto flags ({len(comp)} rows)")

    # pandas sort_values is not stable, so which of several tied-cost points the add-on
    # kept is arbitrary; that moves the AUC in the 7th decimal, hence the tolerance.
    auc = read(ADDON / f"{ds}_observed_range_auc.csv")
    ok = True
    for r in auc:
        pts = [(float(x["normalized_cost"]), float(x["macro_f1"])) for x in comp
               if x["seed"] == r["seed"] and x["method_family"] == r["method_family"]]
        a, lo, hi, avg = observed_auc(pts)
        ok &= close(a, r["observed_auc"], 1e-6) and close(lo, r["cost_min"]) and close(hi, r["cost_max"]) \
            and int(r["n_points"]) == len(pts)
    check(ok, f"{ds} observed-range AUC recomputed to 1e-6 ({len(auc)} rows)")

allpts = read(ADDON / "all_same_backbone_pareto_points.csv")
parts = [r for ds in DATASETS for r in read(ADDON / f"{ds}_DQN_vs_GreedyVOI_same_backbone.csv")]
check(same_rows(parts, allpts, list(allpts[0].keys())), "all_same_backbone_pareto_points = concat of both datasets")
allauc = read(ADDON / "all_observed_range_auc.csv")
parts = [r for ds in DATASETS for r in read(ADDON / f"{ds}_observed_range_auc.csv")]
check(same_rows(parts, allauc, list(allauc[0].keys())), "all_observed_range_auc = concat of both datasets")

for r in read(ADDON / "calibration_summary.csv"):
    cal = read(HARD / f"{r['dataset']}_mask_calibration_seed42.csv")
    test_raw = np.mean([float(x["test_ece_raw"]) for x in cal])
    val_raw = np.mean([float(x["val_ece_raw"]) for x in cal])
    test_cal = np.mean([float(x["test_ece_temp_scaled"]) for x in cal])
    check(close(r["mean_calibrated_ece"], test_cal, 1e-12),
          f"{r['dataset']} calibration_summary mean calibrated ECE = test mean {test_cal:.4f}")
    check(close(r["mean_raw_ece"], test_raw, 1e-12),
          f"{r['dataset']} calibration_summary mean raw ECE = test mean {test_raw:.4f} "
          f"(the Kaggle file had the validation mean {val_raw:.4f}; see README)")

# ============================================================================
section("5. hgb_sequential/ derived files")
log3 = "\n".join(short_lines(LOGS / "hgb_sequential.log"))
check(log3.count("max |new HGB test F1 - rev-hard HGB test F1| = 0.000000000000") == 6,
      "run log: static HGB reproduces the hardening frontier on all six dataset/partition runs")
check("SEQUENTIAL-OVER-HGB AUDIT COMPLETE" in log3, "hgb-sequential run completed")
rep = read(HGBSEQ / "HGB_SEQUENTIAL_REPRESENTATIVE_RESULTS.csv")
matched_all = []
for ds in DATASETS:
    front = [r for r in read(HARD / f"{ds}_strong_subset_classifier_frontier.csv") if r["model"] == "HGB"]
    for s_ in SEEDS:
        static = read(HGBSEQ / f"{ds}_split{s_}_static_hgb.csv")
        old = {r["subset"]: float(r["test_macro_f1"]) for r in front if r["split_seed"] == s_}
        check(len(static) == len(old) and all(close(r["test_macro_f1"], old[r["subset"]], 1e-12) for r in static),
              f"{ds} partition {s_}: {len(static)} static HGB scores equal the hardening frontier")
        for mode in ("raw", "calibrated"):
            q = read(HGBSEQ / f"{ds}_split{s_}_sequential_hgb_{mode}.csv")
            keep = pareto_mask([r["selection_cost"] for r in q], [r["selection_macro_f1"] for r in q])
            vp = [r for r, k in zip(q, keep) if k]
            flags_ok = all((r["validation_pareto"] == "True") == k for r, k in zip(q, keep))
            best = sorted(vp, key=lambda r: (-float(r["selection_macro_f1"]), float(r["selection_cost"])))[0]
            want = [r for r in rep if r["dataset"] == ds and r["split_seed"] == s_
                    and r["method"] == f"Sequential HGB {mode}"][0]
            check(flags_ok and close(best["test_macro_f1"], want["test_macro_f1"], 1e-12)
                  and close(best["test_cost"], want["test_cost"], 1e-12),
                  f"{ds} partition {s_} {mode}: Pareto flags and chosen policy recomputed from the policy half")
        pr = read(HGBSEQ / f"{ds}_split{s_}_representative_seq_predictions.csv")
        y = np.array([int(r["y_true"]) for r in pr]); yp = np.array([int(r["y_pred_seq_hgb"]) for r in pr])
        cost = np.mean([float(r["cost_seq_hgb"]) for r in pr])
        want = [r for r in rep if r["dataset"] == ds and r["split_seed"] == s_
                and r["method"] == "Sequential HGB calibrated"][0]
        check(close(macro_f1(y, yp), want["test_macro_f1"], 1e-9) and close(cost, want["test_cost"], 1e-9),
              f"{ds} partition {s_}: calibrated policy predictions give F1 {macro_f1(y, yp):.4f}, cost {cost:.3f}")
        matched_all += read(HGBSEQ / f"{ds}_split{s_}_matched_budget.csv")
hgb_summ = read(HGBSEQ / "HGB_SEQUENTIAL_SPLIT_SUMMARY.csv")
ok = True
for r in hgb_summ:
    v = [float(x["test_macro_f1"]) for x in rep if x["dataset"] == r["dataset"] and x["method"] == r["method"]]
    c = [float(x["test_cost"]) for x in rep if x["dataset"] == r["dataset"] and x["method"] == r["method"]]
    ok &= close(r["mean_test_macro_f1"], np.mean(v)) and close(r["sd_test_macro_f1"], np.std(v, ddof=1)) \
        and close(r["mean_test_cost"], np.mean(c))
check(ok and len(hgb_summ) == 6, "split summary = mean/SD of the representative results")
ok = True
for r in read(HGBSEQ / "HGB_SEQUENTIAL_MATCHED_BUDGET_SUMMARY.csv"):
    d = [float(x["delta_f1_seq_minus_static"]) for x in matched_all if x["dataset"] == r["dataset"]
         and x["sequential_method"] == r["sequential_method"] and close(x["budget"], r["budget"], 1e-9)]
    ok &= len(d) == 3 and close(r["delta_f1_mean"], np.mean(d))
check(ok, "matched-budget summary = mean over the three per-partition tables")
note("HGB_CALIBRATION_SUMMARY.csv reports ECE on the calibration half that fits the temperatures "
     "(in-sample); the paper does not use it.")

# ============================================================================
section("6. Values quoted in the paper")


def rd(ds, name):
    return read(HARD / f"{ds}_{name}.csv")


sel = {ds: rd(ds, "strong_subset_validation_selected") for ds in DATASETS}


def hgb(ds, k):
    f = np.array([float(r["test_macro_f1"]) for r in sel[ds] if r["model"] == "HGB" and r["k"] == str(k)])
    return f.mean(), f.std(ddof=1)


def summ(ds, method):
    return [r for r in rd(ds, "repeated_group_split_summary") if r["method"] == method][0]


def ece(ds, col):
    return np.mean([float(r[col]) for r in rd(ds, "mask_calibration_seed42")])


def corr(ds, attack, col):
    return float([r for r in rd(ds, "premature_stop_corruption_stress") if r["attack"] == attack][0][col])


def boot(ds, col):
    return float(rd(ds, "cluster_bootstrap_seed42")[0][col])


def share(cls, act):
    return float([r for r in rd(TU, "dqn_branching_seed42")
                  if r["class_name"] == cls and r["decision_after_first"] == act][0]["proportion_mean"])


def n_dqn_pareto(ds, seeds):
    return sum(r["pareto"] == "True" for r in allpts
               if r["dataset"] == ds and r["method_family"] == "DQN" and r["seed"] in seeds)


paper = [  # (quoted value, recomputed value, tolerance, where)
    (0.8045, float(summ(TU, "DQN")["macro_f1_mean"]), 5e-5, "Table II TU DQN mean"),
    (0.0165, float(summ(TU, "DQN")["macro_f1_sd_split"]), 5e-5, "Table II TU DQN SD"),
    (0.706, float(summ(TU, "DQN")["cost_mean"]), 5e-4, "Table II TU DQN cost"),
    (0.8095, float(summ(TU, "OrderEntropy")["macro_f1_mean"]), 5e-5, "Table II TU order+entropy"),
    (0.732, float(summ(TU, "OrderEntropy")["cost_mean"]), 5e-4, "Table II TU order+entropy cost"),
    (0.3632, float(summ(MSU, "DQN")["macro_f1_mean"]), 5e-5, "Table II MSU DQN mean"),
    (0.728, float(summ(MSU, "DQN")["cost_mean"]), 5e-4, "Table II MSU DQN cost"),
    (0.3800, float(summ(MSU, "OrderEntropy")["macro_f1_mean"]), 5e-5, "Table II MSU order+entropy"),
    (0.751, float(summ(MSU, "OrderEntropy")["cost_mean"]), 5e-4, "Table II MSU order+entropy cost"),
    (0.8673, hgb(TU, 1)[0], 5e-5, "Table II TU HGB k=1"), (0.0272, hgb(TU, 1)[1], 5e-5, "Table II TU HGB k=1 SD"),
    (0.8779, hgb(TU, 2)[0], 5e-5, "Table II TU HGB k=2"), (0.9077, hgb(TU, 3)[0], 5e-5, "Table II TU HGB k=3"),
    (0.4299, hgb(MSU, 2)[0], 5e-5, "Table II MSU HGB k=2"), (0.4389, hgb(MSU, 3)[0], 5e-5, "Table II MSU HGB k=3"),
    (0.4402, hgb(MSU, 5)[0], 5e-5, "Table II MSU HGB k=5"),
    (0.162, ece(TU, "test_ece_raw"), 5e-4, "TU mean raw test ECE"),
    (0.100, ece(TU, "test_ece_temp_scaled"), 5e-4, "TU mean calibrated test ECE"),
    (0.163, ece(MSU, "test_ece_raw"), 5e-4, "MSU mean raw test ECE"),
    (0.059, ece(MSU, "test_ece_temp_scaled"), 5e-4, "MSU mean calibrated test ECE"),
    (0.7656, corr(TU, "zero", "clean_macro_f1"), 5e-5, "TU clean F1 before corruption"),
    (0.3951, corr(TU, "zero", "adv_macro_f1"), 5e-5, "TU F1 after zeroing"),
    (0.6585, corr(TU, "noise", "adv_macro_f1"), 5e-5, "TU F1 after noise"),
    (-0.0329, corr(MSU, "zero", "delta_macro_f1"), 5e-5, "MSU zeroing delta"),
    (-0.1344, corr(MSU, "noise", "delta_macro_f1"), 5e-5, "MSU noise delta"),
    (0.761, corr(MSU, "noise", "clean_cost"), 5e-4, "MSU clean cost"),
    (0.620, corr(MSU, "noise", "adv_cost"), 5e-4, "MSU cost under noise"),
    (-0.0135, boot(TU, "mean_diff"), 5e-5, "TU bootstrap difference"),
    (-0.0419, boot(TU, "ci95_low"), 5e-5, "TU bootstrap low"), (0.0142, boot(TU, "ci95_high"), 5e-5, "TU bootstrap high"),
    (-0.0150, boot(MSU, "mean_diff"), 5e-5, "MSU bootstrap difference"),
    (-0.0300, boot(MSU, "ci95_low"), 5e-5, "MSU bootstrap low"), (-0.0039, boot(MSU, "ci95_high"), 5e-5, "MSU bootstrap high"),
    (0.987, share("resource_exhaustion", "STOP"), 5e-4, "resource exhaustion STOP share"),
    (0.023, share("goose_injection", "STOP"), 5e-4, "GOOSE injection STOP share"),
    (0.900, share("sv_injection", "physical"), 5e-4, "SV injection physical share"),
    (0, n_dqn_pareto(TU, ("42", "123")), 0, "TU DQN Pareto points, seeds 42 and 123"),
    (3, n_dqn_pareto(TU, ("456",)), 0, "TU DQN Pareto points, seed 456"),
    (0, n_dqn_pareto(MSU, SEEDS), 0, "MSU DQN Pareto points, all seeds"),
]
hs = {(r["dataset"], r["method"]): r for r in hgb_summ}
for ds, key, f1, sd, cost in [
        (TU, "Static HGB (best validation F1)", 0.8972, 0.0580, 0.778),
        (TU, "Sequential HGB raw", 0.9254, 0.0443, 0.520),
        (TU, "Sequential HGB calibrated", 0.9282, 0.0378, 0.571),
        (MSU, "Static HGB (best validation F1)", 0.4376, 0.0077, 0.933),
        (MSU, "Sequential HGB raw", 0.4395, 0.0057, 0.879),
        (MSU, "Sequential HGB calibrated", 0.4400, 0.0066, 0.861)]:
    r = hs[(ds, key)]
    paper += [(f1, float(r["mean_test_macro_f1"]), 5e-5, f"Table III {ds} {key} mean"),
              (sd, float(r["sd_test_macro_f1"]), 5e-5, f"Table III {ds} {key} SD"),
              (cost, float(r["mean_test_cost"]), 5e-4, f"Table III {ds} {key} cost")]
tu_mid = [float(x["delta_f1_seq_minus_static"]) for x in matched_all if x["dataset"] == TU
          and x["sequential_method"] == "HGB-seq-calibrated" and close(x["budget"], 2 / 3, 1e-6)]
paper += [(0.039, min(tu_mid), 5e-4, "TU sequential minus static at budget 2/3, smallest"),
          (0.100, max(tu_mid), 5e-4, "TU sequential minus static at budget 2/3, largest")]
for quoted, value, tol, where in paper:
    check(close(quoted, value, tol), f"{where}: paper {quoted} vs data {value:.4f}")
for ds, n in ((TU, 6), (MSU, 31)):
    cal = rd(ds, "mask_calibration_seed42")
    k = sum(float(r["test_ece_temp_scaled"]) < float(r["test_ece_raw"]) for r in cal)
    check(k == n, f"{ds} masks improved by temperature scaling: paper {n}, data {k}/{len(cal)}")

print(f"\n{len(failures)} failure(s), {len(notes)} note(s)")
for f in failures:
    print("  FAIL", f)
raise SystemExit(1 if failures else 0)
