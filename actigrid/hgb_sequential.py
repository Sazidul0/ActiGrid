"""Sequential stopping over subset-specific HGB models (paper Table III, Fig. 2).

The shared-backbone comparison sets neural sequential policies against static
tree models, so part of the gap could come from the classifier family. This
stage removes that confound by putting static and sequential acquisition on
the SAME class-balanced HistGradientBoostingClassifier family. For each dataset
and partition seed (42, 123, 456):
  1. rebuild the dataset bundle and the group-disjoint split,
  2. train one HGB model for every nonempty modality subset (7 on TU Delft,
     31 on MSU/ORNL) with the hardening-stage hyperparameters,
  3. split the validation rows into a stratified calibration half (one
     temperature per subset) and a policy half (order, threshold, and the
     static comparator are all chosen there),
  4. evaluate every fixed acquisition order with entropy stopping on raw and
     temperature-calibrated posteriors, and compare with static subsets at
     matched validation budgets.
The test partition is only scored. No fusion, DQN, RF or XGBoost model is
trained.

Outputs go to <work_dir>/hgb_sequential/. The released copy is in
results/hgb_sequential/ and the run log is results/logs/hgb_sequential.log.
The static HGB scores are cross-checked against the hardening stage's frontier
(<hardening_dir>/*_strong_subset_classifier_frontier.csv) when it is present.

Code is the standalone Kaggle script. For the release it reads the pipeline
from actigrid.pipeline instead of an embedded copy (that copy is byte-identical
to the hardening run's script), loads datasets through actigrid.hardening, and
takes its folders and options as arguments of hgb_sequential_main(). The
analysis logic is unchanged. HGB_CALIBRATION_SUMMARY.csv reports ECE on the
calibration half that fits the temperatures, so it is in-sample.
"""
import itertools
import json
import shutil
from typing import Dict, Iterable, List, Optional, Tuple

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, log_loss
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.utils.class_weight import compute_sample_weight

from actigrid import hardening
from actigrid import pipeline as P
from actigrid.pipeline import Path, gc, np, pd

SPLIT_SEEDS = [42, 123, 456]
HGB_MAX_ITER = 250
HGB_LR = 0.08
HGB_MAX_LEAVES = 31
HGB_L2 = 1e-4
VAL_CALIBRATION_FRAC = 0.50
N_THRESHOLD_QUANTILES = 31
TEMPERATURE_GRID = np.exp(np.linspace(np.log(0.20), np.log(5.0), 121))
EPS = 1e-12

# Set by hgb_sequential_main() after pipeline.setup().
OUT: Path = None
CACHE: Path = None
MODEL_CACHE: Path = None
PROBA_CACHE: Path = None
FIG_DIR: Path = None
HARDEN_DIR: Path = None
FORCE = False


def _discover_hardening_dir(hardening_dir: Optional[str] = None) -> Path:
    """Folder with *_strong_subset_classifier_frontier.csv: argument, work dir, or repo results/."""
    candidates = [hardening_dir, P.WORK / "reviewer_hardening",
                  Path(__file__).resolve().parents[1] / "results" / "reviewer_hardening"]
    for c in candidates:
        if c and any(Path(c).glob("*_strong_subset_classifier_frontier.csv")):
            return Path(c)
    return Path(candidates[1])  # absent: the reproduction cross-check is skipped


def load_bundle(key: str):
    """Load a dataset bundle once and cache the processed bundle in OUT/cache."""
    cache_path = CACHE / f"{key}_dataset_bundle.joblib"
    if cache_path.exists() and not FORCE:
        print(f"[resume] loading processed {key} bundle from {cache_path}")
        return joblib.load(cache_path)
    bundle = hardening.load_bundle(key)
    joblib.dump(bundle, cache_path, compress=3)
    print(f"[checkpoint] saved processed {key} bundle -> {cache_path}")
    return bundle


# -----------------------------------------------------------------------------
# Common metrics/probability helpers
# -----------------------------------------------------------------------------
def metrics(y_true, pred) -> Dict[str, float]:
    return {
        "macro_f1": float(f1_score(y_true, pred, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "accuracy": float(accuracy_score(y_true, pred)),
    }


def entropy(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, float), EPS, 1.0)
    p = p / p.sum(axis=1, keepdims=True)
    return -np.sum(p * np.log(p), axis=1)


def temperature_scale_probs(p: np.ndarray, T: float) -> np.ndarray:
    # For a probability vector p, softmax(log(p)/T) is a valid scalar-temperature
    # transform. It preserves argmax class and changes confidence only.
    logp = np.log(np.clip(p, EPS, 1.0)) / float(T)
    logp -= logp.max(axis=1, keepdims=True)
    q = np.exp(logp)
    q /= q.sum(axis=1, keepdims=True)
    return q


def fit_temperature(p_cal: np.ndarray, y_cal: np.ndarray) -> Tuple[float, float]:
    classes = np.arange(p_cal.shape[1])
    best_T, best_nll = 1.0, float("inf")
    for T in TEMPERATURE_GRID:
        q = temperature_scale_probs(p_cal, float(T))
        try:
            nll = float(log_loss(y_cal, q, labels=classes))
        except Exception:
            nll = float("inf")
        if nll < best_nll:
            best_T, best_nll = float(T), nll
    return best_T, best_nll


def expected_calibration_error(p: np.ndarray, y: np.ndarray, bins: int = 15) -> float:
    conf = p.max(axis=1)
    pred = p.argmax(axis=1)
    correct = (pred == y).astype(float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    n = len(y)
    for i in range(bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (conf >= lo) & (conf < hi if i < bins - 1 else conf <= hi)
        if mask.any():
            ece += mask.mean() * abs(correct[mask].mean() - conf[mask].mean())
    return float(ece)


def subset_key(subset: Iterable[str], modalities: List[str]) -> str:
    s = set(subset)
    return "+".join([m for m in modalities if m in s])


def subset_cost(subset: Iterable[str], costs: Dict[str, float]) -> float:
    return float(sum(costs[m] for m in subset))


def X_subset(bundle, subset: Tuple[str, ...], idx: np.ndarray) -> np.ndarray:
    return np.concatenate([bundle.X[m][idx] for m in subset], axis=1)


def all_nonempty_subsets(modalities: List[str]):
    for k in range(1, len(modalities) + 1):
        yield from itertools.combinations(modalities, k)


def split_validation_indices(y: np.ndarray, val_idx: np.ndarray, seed: int):
    """Split original validation rows into calibration and policy-selection halves."""
    val_idx = np.asarray(val_idx, int)
    yv = y[val_idx]
    sss = StratifiedShuffleSplit(
        n_splits=1,
        test_size=1.0 - VAL_CALIBRATION_FRAC,
        random_state=70000 + int(seed),
    )
    cal_local, sel_local = next(sss.split(np.zeros(len(val_idx)), yv))
    cal_idx, sel_idx = val_idx[cal_local], val_idx[sel_local]
    return cal_idx, sel_idx


# -----------------------------------------------------------------------------
# HGB training / resumable probability cache
# -----------------------------------------------------------------------------
def model_path(dataset: str, split_seed: int, subset_name: str) -> Path:
    safe = subset_name.replace("+", "__")
    return MODEL_CACHE / f"{dataset}_split{split_seed}_{safe}.joblib"


def proba_path(dataset: str, split_seed: int, subset_name: str) -> Path:
    safe = subset_name.replace("+", "__")
    return PROBA_CACHE / f"{dataset}_split{split_seed}_{safe}.npz"


def train_hgb(Xtr, ytr, seed: int):
    sw = compute_sample_weight(class_weight="balanced", y=ytr)
    model = HistGradientBoostingClassifier(
        max_iter=HGB_MAX_ITER,
        learning_rate=HGB_LR,
        max_leaf_nodes=HGB_MAX_LEAVES,
        l2_regularization=HGB_L2,
        random_state=10000 + int(seed),
    )
    model.fit(Xtr, ytr, sample_weight=sw)
    return model


def aligned_predict_proba(model, X: np.ndarray, n_classes: int) -> np.ndarray:
    raw = model.predict_proba(X)
    out = np.full((len(X), n_classes), EPS, dtype=np.float64)
    for j, cls in enumerate(model.classes_):
        out[:, int(cls)] = raw[:, j]
    out /= out.sum(axis=1, keepdims=True)
    return out


def fit_all_subset_hgb(bundle, split, split_seed: int, cal_idx, sel_idx):
    modalities = list(bundle.X.keys())
    y = bundle.y
    n_classes = len(bundle.label_names)
    records = []
    probs = {}
    temp_records = []

    for subset in all_nonempty_subsets(modalities):
        name = subset_key(subset, modalities)
        mp = model_path(bundle.name, split_seed, name)
        pp = proba_path(bundle.name, split_seed, name)

        if mp.exists() and not FORCE:
            model = joblib.load(mp)
            print(f"[resume HGB] {bundle.name} split={split_seed} subset={name}")
        else:
            print(f"[fit HGB] {bundle.name} split={split_seed} subset={name}")
            Xtr = X_subset(bundle, subset, split.train_idx)
            model = train_hgb(Xtr, y[split.train_idx], split_seed)
            joblib.dump(model, mp, compress=3)
            del Xtr

        if pp.exists() and not FORCE:
            z = np.load(pp)
            p_cal = z["p_cal"]
            p_sel = z["p_sel"]
            p_test = z["p_test"]
        else:
            p_cal = aligned_predict_proba(model, X_subset(bundle, subset, cal_idx), n_classes)
            p_sel = aligned_predict_proba(model, X_subset(bundle, subset, sel_idx), n_classes)
            p_test = aligned_predict_proba(model, X_subset(bundle, subset, split.test_idx), n_classes)
            np.savez_compressed(pp, p_cal=p_cal, p_sel=p_sel, p_test=p_test)

        T, cal_nll = fit_temperature(p_cal, y[cal_idx])
        q_cal = temperature_scale_probs(p_cal, T)
        q_sel = temperature_scale_probs(p_sel, T)
        q_test = temperature_scale_probs(p_test, T)

        pred_sel = p_sel.argmax(axis=1)
        pred_test = p_test.argmax(axis=1)
        ms = metrics(y[sel_idx], pred_sel)
        mt = metrics(y[split.test_idx], pred_test)
        cost = subset_cost(subset, bundle.modality_costs)

        records.append({
            "dataset": bundle.name,
            "split_seed": split_seed,
            "subset": name,
            "k": len(subset),
            "normalized_cost": cost,
            "selection_macro_f1": ms["macro_f1"],
            "selection_balanced_accuracy": ms["balanced_accuracy"],
            "test_macro_f1": mt["macro_f1"],
            "test_balanced_accuracy": mt["balanced_accuracy"],
            "test_accuracy": mt["accuracy"],
        })
        temp_records.append({
            "dataset": bundle.name,
            "split_seed": split_seed,
            "subset": name,
            "temperature": T,
            "calibration_nll": cal_nll,
            "calibration_ece_raw": expected_calibration_error(p_cal, y[cal_idx]),
            "calibration_ece_scaled": expected_calibration_error(q_cal, y[cal_idx]),
        })
        probs[name] = {
            "raw_sel": p_sel,
            "raw_test": p_test,
            "cal_sel": q_sel,
            "cal_test": q_test,
        }
        del model, p_cal, p_sel, p_test, q_cal
        gc.collect()

        # checkpoint after every subset
        pd.DataFrame(records).to_csv(
            OUT / f"{bundle.name}_split{split_seed}_static_hgb.partial.csv", index=False
        )
        pd.DataFrame(temp_records).to_csv(
            OUT / f"{bundle.name}_split{split_seed}_temperatures.partial.csv", index=False
        )

    return pd.DataFrame(records), pd.DataFrame(temp_records), probs


# -----------------------------------------------------------------------------
# Sequential HGB entropy stopping
# -----------------------------------------------------------------------------
def make_threshold_grid(probs: Dict[str, dict], mode: str, y_len: int) -> np.ndarray:
    vals = []
    key = "cal_sel" if mode == "calibrated" else "raw_sel"
    for d in probs.values():
        vals.append(entropy(d[key]))
    all_e = np.concatenate(vals)
    q = np.quantile(all_e, np.linspace(0.0, 1.0, N_THRESHOLD_QUANTILES))
    # below minimum -> never early stop; above maximum -> stop after first acquisition
    grid = np.unique(np.r_[all_e.min() - 1e-9, q, all_e.max() + 1e-9])
    return grid.astype(float)


def policy_predictions_for_order(
    order: Tuple[str, ...],
    threshold: float,
    probs: Dict[str, dict],
    modalities: List[str],
    costs: Dict[str, float],
    split_name: str,
    mode: str,
):
    pkey = ("cal_" if mode == "calibrated" else "raw_") + split_name
    N = len(next(iter(probs.values()))[pkey])
    M = len(order)
    E = np.empty((N, M), dtype=np.float64)
    P = np.empty((N, M), dtype=np.int16)
    C = np.empty(M, dtype=np.float64)
    prefix = []
    for j, m in enumerate(order):
        prefix.append(m)
        sk = subset_key(prefix, modalities)
        p = probs[sk][pkey]
        E[:, j] = entropy(p)
        P[:, j] = p.argmax(axis=1)
        C[j] = subset_cost(prefix, costs)

    stop = E <= float(threshold)
    stop[:, -1] = True  # force diagnosis after all modalities
    stage = stop.argmax(axis=1)
    rows = np.arange(N)
    pred = P[rows, stage]
    cost = C[stage]
    return pred, cost, stage + 1


def evaluate_all_sequential_policies(bundle, split, split_seed, sel_idx, probs, mode: str):
    modalities = list(bundle.X.keys())
    costs = bundle.modality_costs
    y_sel = bundle.y[sel_idx]
    y_test = bundle.y[split.test_idx]
    thresholds = make_threshold_grid(probs, mode, len(sel_idx))
    rows = []

    orders = list(itertools.permutations(modalities))
    print(f"[sequential-{mode}] {bundle.name} split={split_seed}: {len(orders)} orders x {len(thresholds)} thresholds")
    for oi, order in enumerate(orders, 1):
        order_name = ">".join(order)
        for t in thresholds:
            pred_s, cost_s, steps_s = policy_predictions_for_order(
                order, t, probs, modalities, costs, "sel", mode
            )
            pred_t, cost_t, steps_t = policy_predictions_for_order(
                order, t, probs, modalities, costs, "test", mode
            )
            ms = metrics(y_sel, pred_s)
            mt = metrics(y_test, pred_t)
            rows.append({
                "dataset": bundle.name,
                "split_seed": split_seed,
                "mode": mode,
                "order": order_name,
                "threshold": float(t),
                "selection_macro_f1": ms["macro_f1"],
                "selection_balanced_accuracy": ms["balanced_accuracy"],
                "selection_cost": float(np.mean(cost_s)),
                "selection_mean_steps": float(np.mean(steps_s)),
                "test_macro_f1": mt["macro_f1"],
                "test_balanced_accuracy": mt["balanced_accuracy"],
                "test_accuracy": mt["accuracy"],
                "test_cost": float(np.mean(cost_t)),
                "test_mean_steps": float(np.mean(steps_t)),
            })
        if oi % 20 == 0 or oi == len(orders):
            print(f"  evaluated {oi}/{len(orders)} orders")

    df = pd.DataFrame(rows)
    df["validation_pareto"] = pareto_mask(df["selection_cost"].to_numpy(), df["selection_macro_f1"].to_numpy())
    return df


def pareto_mask(cost: np.ndarray, score: np.ndarray) -> np.ndarray:
    cost = np.asarray(cost, float)
    score = np.asarray(score, float)
    keep = np.ones(len(cost), dtype=bool)
    for i in range(len(cost)):
        # j dominates i if j is no more costly, no worse, and strictly better in one dimension.
        dom = (cost <= cost[i] + 1e-12) & (score >= score[i] - 1e-12) & (
            (cost < cost[i] - 1e-12) | (score > score[i] + 1e-12)
        )
        dom[i] = False
        if dom.any():
            keep[i] = False
    return keep


def best_static_by_budget(static_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for budget in sorted(static_df.normalized_cost.unique()):
        cand = static_df[static_df.normalized_cost <= budget + 1e-12]
        # selection based only on policy-selection validation half
        r = cand.sort_values(
            ["selection_macro_f1", "normalized_cost", "subset"],
            ascending=[False, True, True],
        ).iloc[0]
        z = r.to_dict()
        z["budget"] = float(budget)
        rows.append(z)
    return pd.DataFrame(rows)


def best_sequential_by_budget(seq_df: pd.DataFrame, budgets: Iterable[float]) -> pd.DataFrame:
    rows = []
    for budget in budgets:
        cand = seq_df[seq_df.selection_cost <= float(budget) + 1e-12]
        if len(cand) == 0:
            continue
        r = cand.sort_values(
            ["selection_macro_f1", "selection_cost", "order", "threshold"],
            ascending=[False, True, True, True],
        ).iloc[0]
        z = r.to_dict()
        z["budget"] = float(budget)
        rows.append(z)
    return pd.DataFrame(rows)


def build_matched_budget_table(static_df, raw_df, cal_df):
    stat = best_static_by_budget(static_df)
    budgets = stat.budget.tolist()
    raw = best_sequential_by_budget(raw_df, budgets)
    cal = best_sequential_by_budget(cal_df, budgets)
    rows = []
    for budget in budgets:
        s = stat[np.isclose(stat.budget, budget)].iloc[0]
        for label, df in [("HGB-seq-raw", raw), ("HGB-seq-calibrated", cal)]:
            q = df[np.isclose(df.budget, budget)]
            if len(q) == 0:
                continue
            r = q.iloc[0]
            rows.append({
                "dataset": s.dataset,
                "split_seed": int(s.split_seed),
                "budget": float(budget),
                "static_subset": s.subset,
                "static_test_f1": float(s.test_macro_f1),
                "static_cost": float(s.normalized_cost),
                "sequential_method": label,
                "sequential_order": r.order,
                "sequential_threshold": float(r.threshold),
                "sequential_test_f1": float(r.test_macro_f1),
                "sequential_test_cost": float(r.test_cost),
                "delta_f1_seq_minus_static": float(r.test_macro_f1 - s.test_macro_f1),
                "delta_cost_seq_minus_static": float(r.test_cost - s.normalized_cost),
            })
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Reproduction cross-check against existing rev-hard HGB frontier
# -----------------------------------------------------------------------------
def reproduction_check(bundle_name: str, split_seed: int, static_df: pd.DataFrame):
    p = HARDEN_DIR / f"{bundle_name}_strong_subset_classifier_frontier.csv"
    if not p.exists():
        return pd.DataFrame()
    old = pd.read_csv(p)
    old = old[(old.model == "HGB") & (old.split_seed.astype(int) == int(split_seed))].copy()
    if len(old) == 0:
        return pd.DataFrame()
    cur = static_df[["subset", "test_macro_f1"]].rename(columns={"test_macro_f1": "new_test_macro_f1"})
    chk = old[["subset", "test_macro_f1"]].rename(columns={"test_macro_f1": "revhard_test_macro_f1"}).merge(cur, on="subset")
    chk["abs_diff"] = (chk.revhard_test_macro_f1 - chk.new_test_macro_f1).abs()
    chk["dataset"] = bundle_name
    chk["split_seed"] = split_seed
    return chk


# -----------------------------------------------------------------------------
# Compact plots (default matplotlib colors only)
# -----------------------------------------------------------------------------
def plot_split_frontier(bundle_name, split_seed, static_df, raw_df, cal_df):
    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    ax.scatter(static_df.normalized_cost, static_df.test_macro_f1, marker="o", label="Static HGB subsets")
    rp = raw_df[raw_df.validation_pareto]
    cp = cal_df[cal_df.validation_pareto]
    ax.scatter(rp.test_cost, rp.test_macro_f1, marker="^", label="Sequential HGB, raw entropy")
    ax.scatter(cp.test_cost, cp.test_macro_f1, marker="s", label="Sequential HGB, calibrated entropy")
    ax.set_xlabel("Normalized acquisition cost")
    ax.set_ylabel("Test Macro-F1")
    ax.set_title(f"{bundle_name} — split seed {split_seed}")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = FIG_DIR / f"{bundle_name}_split{split_seed}_HGB_sequential_frontier.png"
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out


# -----------------------------------------------------------------------------
# Dataset/split analysis
# -----------------------------------------------------------------------------
def analyze_dataset_split(bundle, split_seed: int):
    print("\n" + "=" * 96)
    print(f"HGB SEQUENTIAL AUDIT | dataset={bundle.name} | split_seed={split_seed}")
    print("=" * 96)
    split = P.group_split(bundle, seed=split_seed)
    cal_idx, sel_idx = split_validation_indices(bundle.y, split.val_idx, split_seed)
    print(
        f"train={len(split.train_idx)} | validation calibration={len(cal_idx)} | "
        f"validation policy={len(sel_idx)} | test={len(split.test_idx)}"
    )

    static_df, temp_df, probs = fit_all_subset_hgb(bundle, split, split_seed, cal_idx, sel_idx)
    static_path = OUT / f"{bundle.name}_split{split_seed}_static_hgb.csv"
    temp_path = OUT / f"{bundle.name}_split{split_seed}_hgb_temperatures.csv"
    static_df.to_csv(static_path, index=False)
    temp_df.to_csv(temp_path, index=False)

    raw_df = evaluate_all_sequential_policies(bundle, split, split_seed, sel_idx, probs, "raw")
    cal_df = evaluate_all_sequential_policies(bundle, split, split_seed, sel_idx, probs, "calibrated")
    raw_df.to_csv(OUT / f"{bundle.name}_split{split_seed}_sequential_hgb_raw.csv", index=False)
    cal_df.to_csv(OUT / f"{bundle.name}_split{split_seed}_sequential_hgb_calibrated.csv", index=False)

    matched = build_matched_budget_table(static_df, raw_df, cal_df)
    matched.to_csv(OUT / f"{bundle.name}_split{split_seed}_matched_budget.csv", index=False)

    chk = reproduction_check(bundle.name, split_seed, static_df)
    if len(chk):
        chk.to_csv(OUT / f"{bundle.name}_split{split_seed}_revhard_reproduction_check.csv", index=False)
        print(f"[cross-check] max |new HGB test F1 - rev-hard HGB test F1| = {chk.abs_diff.max():.12f}")

    plot_split_frontier(bundle.name, split_seed, static_df, raw_df, cal_df)

    # Small summary rows: best validation-selected static and validation-Pareto sequential points.
    summaries = []
    best_static = static_df.sort_values(
        ["selection_macro_f1", "normalized_cost", "subset"], ascending=[False, True, True]
    ).iloc[0]
    summaries.append({
        "dataset": bundle.name, "split_seed": split_seed, "method": "Static HGB (best validation F1)",
        "descriptor": best_static.subset, "test_macro_f1": best_static.test_macro_f1,
        "test_cost": best_static.normalized_cost,
    })
    for label, df in [("Sequential HGB raw", raw_df), ("Sequential HGB calibrated", cal_df)]:
        vp = df[df.validation_pareto].copy()
        # representative: highest validation F1, then lower selection cost
        r = vp.sort_values(["selection_macro_f1", "selection_cost"], ascending=[False, True]).iloc[0]
        summaries.append({
            "dataset": bundle.name, "split_seed": split_seed, "method": label,
            "descriptor": f"{r.order}; tau={r.threshold:.6g}",
            "test_macro_f1": r.test_macro_f1, "test_cost": r.test_cost,
        })

    # Save test predictions for the representative calibrated policy and its matched static comparator.
    # This supports later bootstrap/significance analysis without retraining.
    cal_rep = cal_df[cal_df.validation_pareto].sort_values(
        ["selection_macro_f1", "selection_cost"], ascending=[False, True]
    ).iloc[0]
    order = tuple(cal_rep.order.split(">"))
    pred_seq, cost_seq, steps_seq = policy_predictions_for_order(
        order, float(cal_rep.threshold), probs, list(bundle.X.keys()), bundle.modality_costs, "test", "calibrated"
    )
    pred_df = pd.DataFrame({
        "row_index": split.test_idx,
        "group": bundle.groups[split.test_idx].astype(str),
        "y_true": bundle.y[split.test_idx],
        "y_pred_seq_hgb": pred_seq,
        "cost_seq_hgb": cost_seq,
        "steps_seq_hgb": steps_seq,
    })
    pred_df.to_csv(OUT / f"{bundle.name}_split{split_seed}_representative_seq_predictions.csv", index=False)

    return pd.DataFrame(summaries), matched, temp_df, chk


# -----------------------------------------------------------------------------
# Aggregate reporting
# -----------------------------------------------------------------------------
def aggregate_results(summary_df: pd.DataFrame, matched_df: pd.DataFrame, temp_df: pd.DataFrame):
    agg = (
        summary_df.groupby(["dataset", "method"], as_index=False)
        .agg(
            mean_test_macro_f1=("test_macro_f1", "mean"),
            sd_test_macro_f1=("test_macro_f1", "std"),
            mean_test_cost=("test_cost", "mean"),
            sd_test_cost=("test_cost", "std"),
        )
    )
    agg.to_csv(OUT / "HGB_SEQUENTIAL_SPLIT_SUMMARY.csv", index=False)

    # Budget-level mean across the three group splits.
    if len(matched_df):
        m = (
            matched_df.groupby(["dataset", "budget", "sequential_method"], as_index=False)
            .agg(
                static_test_f1_mean=("static_test_f1", "mean"),
                static_test_f1_sd=("static_test_f1", "std"),
                sequential_test_f1_mean=("sequential_test_f1", "mean"),
                sequential_test_f1_sd=("sequential_test_f1", "std"),
                sequential_test_cost_mean=("sequential_test_cost", "mean"),
                delta_f1_mean=("delta_f1_seq_minus_static", "mean"),
                delta_f1_sd=("delta_f1_seq_minus_static", "std"),
            )
        )
        m.to_csv(OUT / "HGB_SEQUENTIAL_MATCHED_BUDGET_SUMMARY.csv", index=False)
    else:
        m = pd.DataFrame()

    if len(temp_df):
        t = (
            temp_df.groupby("dataset", as_index=False)
            .agg(
                mean_temperature=("temperature", "mean"),
                mean_calibration_ece_raw=("calibration_ece_raw", "mean"),
                mean_calibration_ece_scaled=("calibration_ece_scaled", "mean"),
            )
        )
        t.to_csv(OUT / "HGB_CALIBRATION_SUMMARY.csv", index=False)
    else:
        t = pd.DataFrame()

    return agg, m, t


def make_paper_ready_markdown(agg, matched, cal, checks):
    lines = [
        "# ActiGrid — sequential-over-HGB reviewer hardening",
        "",
        "## Protocol",
        "",
        "- Same subset-specific HGB classifier family is used for both static and sequential policies.",
        "- HGB hyperparameters match the reviewer-hardening frontier: 250 boosting iterations, learning rate 0.08, 31 max leaves, L2=1e-4, class-balanced sample weights.",
        "- Original validation groups are split into a stratified calibration half and policy-selection half.",
        "- Per-subset temperature scaling is fitted only on the calibration half.",
        "- Acquisition order and entropy threshold are chosen only from the policy-selection half.",
        "- Test data are used only for final evaluation.",
        "- Group splits 42, 123, and 456 are evaluated independently.",
        "",
        "## Split-level representative results",
        "```",
        agg.to_string(index=False),
        "```",
        "",
    ]
    if len(matched):
        lines += ["## Matched-budget summary", "```", matched.to_string(index=False), "```", ""]
    if len(cal):
        lines += ["## HGB calibration summary", "```", cal.to_string(index=False), "```", ""]
    if len(checks):
        mx = float(pd.concat(checks).abs_diff.max())
        lines += [f"Static-HGB reproduction check maximum absolute Macro-F1 difference versus rev-hard: {mx:.12g}.", ""]
    lines += [
        "## Interpretation template",
        "",
        "Use the matched-budget table to decide the Reviewer-#2 claim. If calibrated sequential HGB improves Macro-F1 at the same or lower cost across splits, sequential acquisition remains useful once classifier-family confounding is removed. If the static HGB frontier still dominates, the appropriate conclusion is that the small modality spaces do not justify sequential acquisition on these benchmarks, even with a strong classifier family.",
        "",
        "Do not compare only the single highest-F1 sequential point with a cheaper static subset; use matched budgets or the Pareto frontier.",
        "",
        "## MSU/ORNL reporting note",
        "",
        "The lower MSU/ORNL scores should be described as performance under strict whole-file group-disjoint generalization, not as evidence that the dataset is intrinsically difficult or that earlier work leaked data. A literature citation is required before contrasting this protocol with prior random/row-level splits.",
    ]
    (OUT / "HGB_SEQUENTIAL_PAPER_READY.md").write_text("\n".join(lines))


def hgb_sequential_main(hardening_dir: Optional[str] = None, force: bool = False, fast: bool = False):
    """Run sequential-over-HGB. Call pipeline.setup() first.

    hardening_dir  folder with the hardening stage's subset frontier, used only for the
                   reproduction cross-check (default: <work_dir>/reviewer_hardening,
                   else results/reviewer_hardening).
    force          refit models and recompute cached probabilities.
    fast           smoke test: partition seed 42 only. Not for results.
    """
    global OUT, CACHE, MODEL_CACHE, PROBA_CACHE, FIG_DIR, HARDEN_DIR, FORCE, SPLIT_SEEDS
    OUT = P.WORK / "hgb_sequential"
    CACHE = OUT / "cache"
    MODEL_CACHE = CACHE / "models"
    PROBA_CACHE = CACHE / "probabilities"
    FIG_DIR = OUT / "figures"
    for p in [OUT, CACHE, MODEL_CACHE, PROBA_CACHE, FIG_DIR]:
        p.mkdir(parents=True, exist_ok=True)
    HARDEN_DIR = _discover_hardening_dir(hardening_dir)
    FORCE = force
    if fast:
        SPLIT_SEEDS = [42]
    if HARDEN_DIR.exists():
        print("[ok] rev-hard :", HARDEN_DIR)
    else:
        print("[note] rev-hard not found; the new analysis will still run, but reproduction cross-checks are skipped")
    print("[info] This run trains ONLY subset-specific HGB models and policy-selection logic.")
    print("[info] No DQN/fusion/RF/XGBoost training is executed.\n")

    summary_parts, matched_parts, temp_parts, check_parts = [], [], [], []
    keys = [k for k, on in (("tu", P.CFG.run_tudelft), ("msu", P.CFG.run_msu)) if on]
    for key in keys:
        bundle = load_bundle(key)
        for split_seed in SPLIT_SEEDS:
            s, m, t, c = analyze_dataset_split(bundle, split_seed)
            summary_parts.append(s)
            matched_parts.append(m)
            temp_parts.append(t)
            if len(c):
                check_parts.append(c)

    summary_df = pd.concat(summary_parts, ignore_index=True)
    matched_df = pd.concat(matched_parts, ignore_index=True)
    temp_df = pd.concat(temp_parts, ignore_index=True)
    summary_df.to_csv(OUT / "HGB_SEQUENTIAL_REPRESENTATIVE_RESULTS.csv", index=False)
    matched_df.to_csv(OUT / "HGB_SEQUENTIAL_ALL_MATCHED_BUDGETS.csv", index=False)
    temp_df.to_csv(OUT / "HGB_SEQUENTIAL_ALL_TEMPERATURES.csv", index=False)

    agg, matched_agg, cal_agg = aggregate_results(summary_df, matched_df, temp_df)
    make_paper_ready_markdown(agg, matched_agg, cal_agg, check_parts)

    manifest = {
        "datasets": ["TU_Delft_2026_CPPS", "MSU_ORNL_Power"],
        "split_seeds": SPLIT_SEEDS,
        "hgb": {
            "max_iter": HGB_MAX_ITER,
            "learning_rate": HGB_LR,
            "max_leaf_nodes": HGB_MAX_LEAVES,
            "l2_regularization": HGB_L2,
            "class_weighting": "balanced sample weights",
        },
        "validation_calibration_fraction": VAL_CALIBRATION_FRAC,
        "temperature_grid_min": float(TEMPERATURE_GRID.min()),
        "temperature_grid_max": float(TEMPERATURE_GRID.max()),
        "temperature_grid_n": int(len(TEMPERATURE_GRID)),
        "threshold_quantiles": N_THRESHOLD_QUANTILES,
        "selection_rule": "validation-calibration for T; disjoint validation-policy half for order/threshold; test final only",
    }
    (OUT / "run_manifest.json").write_text(json.dumps(manifest, indent=2))

    zip_path = shutil.make_archive(str(OUT), "zip", root_dir=str(OUT))
    print("\n" + "=" * 96)
    print("SEQUENTIAL-OVER-HGB AUDIT COMPLETE")
    print("Output:", OUT)
    print("ZIP   :", zip_path)
    print("=" * 96)
