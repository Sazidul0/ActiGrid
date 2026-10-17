"""Reviewer-clearance analyses used in the revised paper (PES GM 2027).

Runs on top of actigrid.pipeline: for every evaluation seed it retrains the
fusion classifier with the same seed (deterministically reproducing the
per-seed backbone) and then evaluates, without any policy training,

  A) every nonempty fixed subset (7 on TU Delft, 31 on MSU/ORNL) with
     validation-selected best subset per size k,
  B) all M! fixed acquisition orders with entropy stopping, selected on
     validation within +/-0.02 of the DQN validation cost, plus a
     validation-selected order/threshold curve over budgets,
  C) a post-hoc static subset mixture at the observed DQN test cost
     (descriptive only),
  D) Holm-corrected p-values, and
  E) TU resource-exhaustion sensitivity and seed-42 first-action summaries.

Outputs go to <work_dir>/reviewer_clearance/. The released copy is in
results/reviewer_clearance/.

Code is the reviewer-clearance section of the original Kaggle notebook. For
the release it was moved into its own module; CFG/WORK/LOG are read from
actigrid.pipeline at call time (they are set by pipeline.setup()), and the
frozen-v7 results folder is passed explicitly instead of searched for under
/kaggle. Analysis logic is unchanged.
"""
from itertools import combinations, permutations

from actigrid import pipeline as P
from actigrid.pipeline import (
    Any, Dict, List, Optional, Path, Tuple,
    DatasetBundle, SplitBundle,
    discover_dataset_root, f1_score, gc, group_split, json, load_msu_bundle,
    load_tudelft_bundle, macro_metrics, math, maybe_extract_archives, np, os, pd,
    seed_everything, shutil, torch, train_fusion_classifier,
)

# ============================================================
# Purpose:
#   Address the strongest reviewer questions WITHOUT retraining the DQN:
#   A) exhaustive validation-selected fixed subsets (3->7 / 5->31 masks),
#   B) exhaustive fixed acquisition orders + entropy stopping,
#   C) a transparent post-hoc static mixture at the observed DQN cost,
#   D) Holm corrections with F1 and cost treated as separate families,
#   E) TU resource-exhaustion sensitivity and seed-42 action-order summary.
#
# Scientific guardrails:
#   * Subsets/orders are SELECTED ON VALIDATION ONLY, then frozen on test.
#   * The post-hoc cost-matched mixture uses the observed DQN TEST COST only to
#     match resource consumption, never labels/predictions. It is diagnostic,
#     not a primary confirmatory comparison.
#   * Existing v7 DQN results remain frozen. This file retrains only the fusion
#     classifier for the same seeds because only seed 42 was persisted.
# ============================================================

# Set by reviewer_clearance_main() after pipeline.setup(); the paper used the
# same ten seeds as the v7 run (P.CFG.final_seeds).
REVIEW_DIR: Path = None
REVIEW_SEEDS: Tuple[int, ...] = ()
REVIEW_BATCH = 4096
REVIEW_ENTROPY_QUANTILES = 31
REVIEW_TARGET_TOL = 0.02


def _discover_prior_results_dir(prior_results: Optional[str] = None) -> Path:
    """Find the directory that contains the frozen v7 result CSVs.

    Order: explicit argument, $ACTIGRID_RESULTS, the current work dir, then the
    repository's results/ folder (which ships the frozen v7 files).
    """
    candidates = [prior_results, os.environ.get("ACTIGRID_RESULTS"), P.WORK,
                  Path(__file__).resolve().parents[1] / "results"]
    for c in candidates:
        if c and (Path(c) / "v7_all_multiseed_raw.csv").exists():
            return Path(c)
    raise FileNotFoundError(
        "Could not locate v7_all_multiseed_raw.csv. Pass --prior-results or set ACTIGRID_RESULTS."
    )


PRIOR_RESULTS = None


def _mask_key(mask: np.ndarray) -> str:
    return "".join(str(int(x)) for x in mask.astype(int).tolist())


def _mask_from_subset(modalities: List[str], subset: Tuple[str, ...]) -> np.ndarray:
    s = set(subset)
    return np.asarray([1.0 if m in s else 0.0 for m in modalities], dtype=np.float32)


@torch.no_grad()
def _predict_fixed_mask_batched(model, bundle: DatasetBundle, split: SplitBundle,
                                idx: np.ndarray, mask_np: np.ndarray,
                                batch_size: int = REVIEW_BATCH) -> Dict[str, Any]:
    """Fast predictions for one fixed evidence mask."""
    modalities = list(bundle.X.keys())
    idx = np.asarray(idx, dtype=int)
    y_true, y_pred, entropy = [], [], []
    model.eval()
    for start in range(0, len(idx), batch_size):
        ids = idx[start:start + batch_size]
        xs = [torch.from_numpy(split.X_scaled[m][ids]).float().to(P.CFG.device) for m in modalities]
        mask = torch.from_numpy(mask_np).float().to(P.CFG.device).unsqueeze(0).repeat(len(ids), 1)
        logits, _ = model(xs, mask)
        probs = torch.softmax(logits, dim=1)
        pred = probs.argmax(1)
        ent = -(probs * torch.log(probs.clamp_min(1e-12))).sum(1)
        y_true.append(bundle.y[ids].astype(int))
        y_pred.append(pred.cpu().numpy().astype(int))
        entropy.append(ent.cpu().numpy().astype(float))
    yt = np.concatenate(y_true)
    yp = np.concatenate(y_pred)
    en = np.concatenate(entropy)
    met = macro_metrics(yt, yp, len(bundle.label_names))
    cost = float(sum(bundle.modality_costs[m] for j, m in enumerate(modalities) if mask_np[j] > 0.5))
    return {
        "idx": idx,
        "y_true": yt,
        "y_pred": yp,
        "entropy": en,
        "metrics": met,
        "normalized_cost": cost,
        "mean_steps": int(mask_np.sum()),
    }


def _build_all_mask_cache(model, bundle: DatasetBundle, split: SplitBundle, idx: np.ndarray) -> Dict[str, Dict[str, Any]]:
    modalities = list(bundle.X.keys())
    M = len(modalities)
    cache = {}
    for k in range(1, M + 1):
        for subset in combinations(modalities, k):
            mask = _mask_from_subset(modalities, subset)
            key = _mask_key(mask)
            r = _predict_fixed_mask_batched(model, bundle, split, idx, mask)
            r["subset"] = tuple(subset)
            cache[key] = r
    return cache


def _exhaustive_subset_rows(bundle: DatasetBundle, seed: int,
                            val_cache: Dict[str, Dict[str, Any]],
                            test_cache: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    modalities = list(bundle.X.keys())
    rows = []
    for key, vr in val_cache.items():
        tr = test_cache[key]
        subset = list(vr["subset"])
        rows.append({
            "dataset": bundle.name,
            "seed": int(seed),
            "subset": "+".join(subset),
            "k": len(subset),
            "normalized_cost": vr["normalized_cost"],
            "val_macro_f1": vr["metrics"]["macro_f1"],
            "val_balanced_accuracy": vr["metrics"]["balanced_accuracy"],
            "test_macro_f1": tr["metrics"]["macro_f1"],
            "test_balanced_accuracy": tr["metrics"]["balanced_accuracy"],
        })
    return pd.DataFrame(rows)


def _select_best_subset_per_k(exhaustive: pd.DataFrame) -> pd.DataFrame:
    """Validation-only selection, then report frozen test outcome."""
    chosen = []
    for (dataset, seed, k), g in exhaustive.groupby(["dataset", "seed", "k"], sort=False):
        g = g.sort_values(["val_macro_f1", "val_balanced_accuracy", "subset"],
                          ascending=[False, False, True])
        chosen.append(g.iloc[0])
    return pd.DataFrame(chosen).reset_index(drop=True)


def _simulate_order(cache: Dict[str, Dict[str, Any]], modalities: List[str], costs: Dict[str, float],
                    order: Tuple[str, ...], threshold: float) -> Dict[str, Any]:
    """Vectorized fixed-order + entropy-stop policy using cached model outputs."""
    n = len(next(iter(cache.values()))["y_true"])
    yt = next(iter(cache.values()))["y_true"]
    final_pred = np.zeros(n, dtype=int)
    final_ent = np.full(n, np.nan, dtype=float)
    total_cost = np.zeros(n, dtype=float)
    steps = np.zeros(n, dtype=int)
    active = np.ones(n, dtype=bool)
    acquired = []

    for step, m in enumerate(order, start=1):
        acquired.append(m)
        mask = np.asarray([1.0 if x in acquired else 0.0 for x in modalities], dtype=np.float32)
        r = cache[_mask_key(mask)]
        use = active.copy()
        final_pred[use] = r["y_pred"][use]
        final_ent[use] = r["entropy"][use]
        total_cost[use] += float(costs[m])
        steps[use] += 1
        if step == len(order):
            active[:] = False
        else:
            stop = active & (r["entropy"] <= threshold)
            active[stop] = False
        if not active.any():
            break

    met = macro_metrics(yt, final_pred, int(max(np.max(yt), np.max(final_pred))) + 1)
    return {
        "macro_f1": float(met["macro_f1"]),
        "balanced_accuracy": float(met["balanced_accuracy"]),
        "accuracy": float(met["accuracy"]),
        "normalized_cost": float(total_cost.mean()),
        "mean_steps": float(steps.mean()),
        "y_true": yt,
        "y_pred": final_pred,
    }


def _entropy_threshold_grid(val_cache: Dict[str, Dict[str, Any]], n_quantiles: int = REVIEW_ENTROPY_QUANTILES) -> np.ndarray:
    all_ent = np.concatenate([r["entropy"] for r in val_cache.values()])
    qs = np.linspace(0.0, 1.0, int(n_quantiles))
    vals = np.quantile(all_ent, qs)
    # Include extremes: -inf forces all modalities; +inf stops after first.
    return np.unique(np.concatenate(([-np.inf], vals, [np.inf]))).astype(float)


def _search_fixed_order_entropy(bundle: DatasetBundle,
                                val_cache: Dict[str, Dict[str, Any]],
                                test_cache: Dict[str, Dict[str, Any]],
                                target_cost: float,
                                seed: int) -> Tuple[pd.DataFrame, pd.Series]:
    """Exhaust all M! orders and validation-select entropy stopping."""
    modalities = list(bundle.X.keys())
    thresholds = _entropy_threshold_grid(val_cache)
    candidates = []
    for order in permutations(modalities):
        for thr in thresholds:
            vr = _simulate_order(val_cache, modalities, bundle.modality_costs, order, float(thr))
            candidates.append({
                "dataset": bundle.name,
                "seed": int(seed),
                "order": " -> ".join(order),
                "threshold": float(thr),
                "val_macro_f1": vr["macro_f1"],
                "val_balanced_accuracy": vr["balanced_accuracy"],
                "val_cost": vr["normalized_cost"],
                "val_steps": vr["mean_steps"],
            })
    cand = pd.DataFrame(candidates)

    # Primary selection is validation-only. Prefer candidates within a small
    # cost band of the original DQN validation target, then maximize F1.
    near = cand[np.abs(cand.val_cost - float(target_cost)) <= REVIEW_TARGET_TOL].copy()
    if len(near) == 0:
        cand["cost_gap"] = np.abs(cand.val_cost - float(target_cost))
        min_gap = cand.cost_gap.min()
        near = cand[cand.cost_gap <= min_gap + 1e-12].copy()
    near = near.sort_values(["val_macro_f1", "val_cost", "order"], ascending=[False, True, True])
    best = near.iloc[0].copy()
    order = tuple(x.strip() for x in str(best.order).split("->"))
    tr = _simulate_order(test_cache, modalities, bundle.modality_costs, order, float(best.threshold))
    best["test_macro_f1"] = tr["macro_f1"]
    best["test_balanced_accuracy"] = tr["balanced_accuracy"]
    best["test_cost"] = tr["normalized_cost"]
    best["test_steps"] = tr["mean_steps"]
    best["target_cost"] = float(target_cost)
    return cand, best



def _validation_selected_order_curve(bundle: DatasetBundle, candidates: pd.DataFrame,
                                     test_cache: Dict[str, Dict[str, Any]], seed: int) -> pd.DataFrame:
    """Validation-selected fixed-order entropy curve at predeclared common budgets."""
    modalities = list(bundle.X.keys())
    M = len(modalities)
    # Predeclared grid based only on number of modalities, not test outcomes.
    targets = np.linspace(1.0 / M, 1.0, 9)
    rows = []
    for target in targets:
        feasible = candidates[candidates.val_cost <= float(target) + 0.01].copy()
        if len(feasible) == 0:
            feasible = candidates.iloc[[int(np.argmin(np.abs(candidates.val_cost.to_numpy(float) - float(target))))]].copy()
        best = feasible.sort_values(["val_macro_f1", "val_cost", "order"], ascending=[False, True, True]).iloc[0]
        order = tuple(x.strip() for x in str(best.order).split("->"))
        tr = _simulate_order(test_cache, modalities, bundle.modality_costs, order, float(best.threshold))
        rows.append({
            "dataset": bundle.name, "seed": int(seed), "target_budget": float(target),
            "order": best.order, "threshold": float(best.threshold),
            "val_macro_f1": float(best.val_macro_f1), "val_cost": float(best.val_cost),
            "test_macro_f1": float(tr["macro_f1"]),
            "test_balanced_accuracy": float(tr["balanced_accuracy"]),
            "test_cost": float(tr["normalized_cost"]), "test_steps": float(tr["mean_steps"]),
        })
    return pd.DataFrame(rows)


def _select_network_first_entropy(bundle: DatasetBundle, candidates: pd.DataFrame,
                                  val_cache: Dict[str, Dict[str, Any]], test_cache: Dict[str, Dict[str, Any]],
                                  target_cost: float) -> Optional[pd.Series]:
    network_names = [m for m in bundle.X if "network" in m.lower() or "pcap" in m.lower()]
    if not network_names:
        return None
    network = network_names[0]
    c = candidates[candidates.order.str.startswith(network + " ->") | (candidates.order == network)].copy()
    if len(c) == 0:
        return None
    near = c[np.abs(c.val_cost - float(target_cost)) <= REVIEW_TARGET_TOL].copy()
    if len(near) == 0:
        c["cost_gap"] = np.abs(c.val_cost - float(target_cost))
        near = c[c.cost_gap <= c.cost_gap.min() + 1e-12].copy()
    best = near.sort_values(["val_macro_f1", "val_cost", "order"], ascending=[False, True, True]).iloc[0].copy()
    order = tuple(x.strip() for x in str(best.order).split("->"))
    tr = _simulate_order(test_cache, list(bundle.X.keys()), bundle.modality_costs, order, float(best.threshold))
    best["test_macro_f1"] = tr["macro_f1"]
    best["test_balanced_accuracy"] = tr["balanced_accuracy"]
    best["test_cost"] = tr["normalized_cost"]
    best["test_steps"] = tr["mean_steps"]
    best["target_cost"] = float(target_cost)
    best["method"] = "NetworkFirst-EntropyStop"
    return best


def _best_subset_row(best_per_k: pd.DataFrame, dataset: str, seed: int, k: int) -> pd.Series:
    g = best_per_k[(best_per_k.dataset == dataset) & (best_per_k.seed == seed) & (best_per_k.k == k)]
    if len(g) != 1:
        raise RuntimeError(f"Expected one best subset row for {dataset}/seed={seed}/k={k}, got {len(g)}")
    return g.iloc[0]


def _posthoc_static_mix_at_cost(bundle: DatasetBundle, seed: int, target_cost: float,
                                best_per_k: pd.DataFrame,
                                test_cache: Dict[str, Dict[str, Any]],
                                mix_repeats: int = 20) -> Dict[str, Any]:
    """Diagnostic comparator at the observed DQN cost.

    IMPORTANT: target_cost may be the observed DQN test cost. No test labels are
    used to choose subsets (those are validation-selected), but the target itself
    is post-hoc. Do not present this as a preregistered confirmatory test.
    """
    modalities = list(bundle.X.keys())
    M = len(modalities)
    target_steps = float(np.clip(target_cost * M, 1.0, float(M)))
    lo = int(np.floor(target_steps + 1e-12))
    hi = int(np.ceil(target_steps - 1e-12))
    lo = max(1, min(M, lo)); hi = max(1, min(M, hi))
    if lo == hi:
        q = 0.0
    else:
        q = (target_steps - lo) / (hi - lo)

    rlo = _best_subset_row(best_per_k, bundle.name, seed, lo)
    slo = tuple(str(rlo.subset).split("+"))
    klo = _mask_key(_mask_from_subset(modalities, slo))
    plo = test_cache[klo]["y_pred"]
    yt = test_cache[klo]["y_true"]

    if hi == lo:
        phi = plo
        shi = slo
        khi = klo
    else:
        rhi = _best_subset_row(best_per_k, bundle.name, seed, hi)
        shi = tuple(str(rhi.subset).split("+"))
        khi = _mask_key(_mask_from_subset(modalities, shi))
        phi = test_cache[khi]["y_pred"]

    f1s, bas, costs = [], [], []
    for rep in range(mix_repeats):
        rng = np.random.default_rng(int(seed) * 100003 + rep * 97 + 17)
        use_hi = rng.random(len(yt)) < q if hi != lo else np.zeros(len(yt), dtype=bool)
        yp = np.where(use_hi, phi, plo)
        met = macro_metrics(yt, yp, len(bundle.label_names))
        actual_steps = lo + use_hi.astype(float) * (hi - lo)
        f1s.append(met["macro_f1"]); bas.append(met["balanced_accuracy"])
        costs.append(float(actual_steps.mean() / M))
    return {
        "dataset": bundle.name,
        "seed": int(seed),
        "method": "POSTHOC-CostMatchedStaticMix",
        "target_dqn_cost": float(target_cost),
        "low_k": lo,
        "high_k": hi,
        "mix_high_probability": float(q),
        "low_subset": "+".join(slo),
        "high_subset": "+".join(shi),
        "macro_f1": float(np.mean(f1s)),
        "macro_f1_mix_sd": float(np.std(f1s, ddof=1)) if len(f1s) > 1 else 0.0,
        "balanced_accuracy": float(np.mean(bas)),
        "normalized_cost": float(np.mean(costs)),
    }


def _holm_adjust(pvals: np.ndarray) -> np.ndarray:
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    order = np.argsort(p)
    ps = p[order]
    adj_sorted = np.empty(m, dtype=float)
    running = 0.0
    for i, pv in enumerate(ps):
        val = (m - i) * pv
        running = max(running, val)
        adj_sorted[i] = min(1.0, running)
    out = np.empty(m, dtype=float)
    out[order] = adj_sorted
    return out


def make_corrected_pvalue_table(prior_results: Path) -> pd.DataFrame:
    p = prior_results / "v7_all_paired_tests.csv"
    df = pd.read_csv(p)
    # Support v7 column names and the paper-output names.
    fcol = "wilcoxon_p_macro_f1" if "wilcoxon_p_macro_f1" in df.columns else "p_macro_f1"
    ccol = "wilcoxon_p_cost" if "wilcoxon_p_cost" in df.columns else "p_cost"
    parts = []
    for dataset, g in df.groupby("dataset", sort=False):
        z = g.copy()
        z["holm_p_macro_f1"] = _holm_adjust(z[fcol].to_numpy(float))
        z["holm_p_cost"] = _holm_adjust(z[ccol].to_numpy(float))
        z["holm_family_note"] = "F1 and cost corrected as separate 5-comparison families"
        parts.append(z)
    out = pd.concat(parts, ignore_index=True)
    out.to_csv(REVIEW_DIR / "reviewer_holm_corrected_tests.csv", index=False)
    return out


def _resource_exhaustion_sensitivity_from_samples(prior_results: Path, bundle: DatasetBundle) -> pd.DataFrame:
    if "TU" not in bundle.name:
        return pd.DataFrame()
    names = [x.lower().replace(" ", "_") for x in bundle.label_names]
    try:
        rid = names.index("resource_exhaustion")
    except ValueError:
        matches = [i for i, x in enumerate(names) if "resource" in x and "exhaust" in x]
        if not matches:
            return pd.DataFrame()
        rid = matches[0]
    keep_labels = [i for i in range(len(bundle.label_names)) if i != rid]
    rows = []
    for method, suffix in [("ActiGrid-DQN-seed42", "dqn_samples.csv"),
                           ("Entropy-seed42", "entropy_samples.csv")]:
        files = list(prior_results.glob(f"{bundle.name}_v7_seed42_*{suffix}"))
        if not files:
            # Exact known naming pattern fallback
            q = prior_results / f"{bundle.name}_v7_seed42_{suffix}"
            files = [q] if q.exists() else []
        if not files:
            continue
        df = pd.read_csv(files[0])
        s = df[df.y_true.astype(int) != rid].copy()
        f1 = f1_score(s.y_true.astype(int), s.y_pred.astype(int), labels=keep_labels,
                      average="macro", zero_division=0)
        ba = np.mean([recall_score_safe(s.y_true.astype(int), s.y_pred.astype(int), lab) for lab in keep_labels])
        rows.append({"dataset": bundle.name, "method": method, "excluded_class": bundle.label_names[rid],
                     "n_remaining": len(s), "macro_f1_5class": float(f1), "balanced_accuracy_5class": float(ba)})
    return pd.DataFrame(rows)


def recall_score_safe(y_true, y_pred, label: int) -> float:
    mask = np.asarray(y_true) == int(label)
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(np.asarray(y_pred)[mask] == int(label)))


def _seed42_action_order(prior_results: Path, bundle: DatasetBundle) -> pd.DataFrame:
    p = prior_results / f"{bundle.name}_v7_seed42_dqn_samples.csv"
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_csv(p)
    if "acquired" not in df.columns:
        return pd.DataFrame()
    def first_action(s):
        s = str(s)
        return s.split(" -> ")[0].strip() if s and s.lower() != "nan" else "<none>"
    df["first_action"] = df.acquired.map(first_action)
    df["class_name"] = df.y_true.astype(int).map(lambda i: bundle.label_names[i])
    out = (df.groupby(["class_name", "first_action"]).size().rename("n").reset_index())
    totals = out.groupby("class_name").n.transform("sum")
    out["percentage"] = 100.0 * out.n / totals
    out.to_csv(REVIEW_DIR / f"{bundle.name}_seed42_first_action_by_class.csv", index=False)
    return out


def _old_dqn_cost_by_seed(prior_results: Path, bundle_name: str) -> Dict[int, float]:
    p = prior_results / "v7_all_multiseed_raw.csv"
    if not p.exists():
        p = prior_results / f"{bundle_name}_v7_multiseed_raw.csv"
    df = pd.read_csv(p)
    g = df[(df.dataset == bundle_name) & (df.method.str.contains("ActiGrid-DQN", regex=False))]
    return {int(r.seed): float(r.normalized_cost) for _, r in g.iterrows()}


def _target_validation_cost(prior_results: Path, bundle_name: str) -> float:
    p = prior_results / f"{bundle_name}_v7_selection.json"
    if p.exists():
        with open(p) as f:
            d = json.load(f)
        return float(d["target_validation_cost"])
    p = prior_results / f"{bundle_name}_v7_multiseed_raw.csv"
    if p.exists():
        d = pd.read_csv(p)
        if "target_budget" in d.columns:
            vals = d.target_budget.dropna().unique()
            if len(vals): return float(vals[0])
    # Frozen v7 Stage-A values from the final run, used only as last-resort fallback.
    return 0.5867 if "TU_Delft" in bundle_name else 0.7574


def _aggregate_rows(df: pd.DataFrame, value_cols: List[str], group_cols: List[str]) -> pd.DataFrame:
    out = []
    from scipy.stats import t as student_t
    for keys, g in df.groupby(group_cols, sort=False):
        if not isinstance(keys, tuple): keys = (keys,)
        row = {c: v for c, v in zip(group_cols, keys)}
        n = len(g)
        for c in value_cols:
            x = g[c].astype(float).to_numpy()
            row[c + "_mean"] = float(np.mean(x))
            row[c + "_std"] = float(np.std(x, ddof=1)) if n > 1 else 0.0
            row[c + "_ci95"] = float(student_t.ppf(0.975, n-1) * row[c + "_std"] / math.sqrt(n)) if n > 1 else 0.0
        row["n_seeds"] = n
        out.append(row)
    return pd.DataFrame(out)


def _paired_against_old_dqn(new_rows: pd.DataFrame, prior_results: Path,
                            method_col: str, f1_col: str, cost_col: str) -> pd.DataFrame:
    from scipy.stats import wilcoxon
    old = pd.read_csv(prior_results / "v7_all_multiseed_raw.csv")
    dqn = old[old.method.str.contains("ActiGrid-DQN", regex=False)][["dataset", "seed", "macro_f1", "normalized_cost"]].copy()
    dqn = dqn.rename(columns={"macro_f1":"dqn_f1", "normalized_cost":"dqn_cost"})
    x = new_rows.merge(dqn, on=["dataset", "seed"], how="inner")
    rows = []
    for (dataset, method), g in x.groupby(["dataset", method_col], sort=False):
        if len(g) < 2: continue
        df1 = g[f1_col].to_numpy(float) - g.dqn_f1.to_numpy(float)
        dc = g[cost_col].to_numpy(float) - g.dqn_cost.to_numpy(float)
        def wp(a):
            if np.allclose(a, 0): return 1.0
            try: return float(wilcoxon(a, zero_method="wilcox", alternative="two-sided", mode="auto").pvalue)
            except Exception: return float("nan")
        rows.append({"dataset":dataset, "comparator":method,
                     "comparator_minus_dqn_f1":float(df1.mean()), "p_f1":wp(df1),
                     "comparator_minus_dqn_cost":float(dc.mean()), "p_cost":wp(dc), "n":len(g)})
    return pd.DataFrame(rows)


def run_reviewer_checks_for_bundle(bundle: DatasetBundle, prior_results: Path) -> Dict[str, pd.DataFrame]:
    P.LOG.info("\n%s\nREVIEWER CLEARANCE: %s\n%s", "="*100, bundle.name, "="*100)
    split = group_split(bundle, int(P.CFG.final_split_seed))
    target_cost = _target_validation_cost(prior_results, bundle.name)
    old_dqn_cost = _old_dqn_cost_by_seed(prior_results, bundle.name)

    exhaustive_all, bestk_all, order_best_all, order_curve_all, order_candidates_all, network_best_all, posthoc_all = [], [], [], [], [], [], []

    for si, seed in enumerate(REVIEW_SEEDS, start=1):
        P.LOG.info("Reviewer seed %d/%d = %d | %s", si, len(REVIEW_SEEDS), seed, bundle.name)
        P.CFG.seed = int(seed); seed_everything(P.CFG.seed)
        fusion = train_fusion_classifier(bundle, split, model_tag=f"review_{bundle.name}_{seed}")

        val_cache = _build_all_mask_cache(fusion, bundle, split, split.val_idx)
        test_cache = _build_all_mask_cache(fusion, bundle, split, split.test_idx)

        ex = _exhaustive_subset_rows(bundle, seed, val_cache, test_cache)
        exhaustive_all.append(ex)
        bk = _select_best_subset_per_k(ex)
        bestk_all.append(bk)

        cand, best = _search_fixed_order_entropy(bundle, val_cache, test_cache, target_cost, seed)
        cand["dataset"] = bundle.name
        cand["seed"] = int(seed)
        order_candidates_all.append(cand)
        order_curve_all.append(_validation_selected_order_curve(bundle, cand, test_cache, seed))
        best["method"] = "BestOrder-EntropyStop"
        order_best_all.append(pd.DataFrame([best]))

        nb = _select_network_first_entropy(bundle, cand, val_cache, test_cache, target_cost)
        if nb is not None:
            network_best_all.append(pd.DataFrame([nb]))

        if seed in old_dqn_cost:
            mix = _posthoc_static_mix_at_cost(bundle, seed, old_dqn_cost[seed], bk, test_cache)
            posthoc_all.append(pd.DataFrame([mix]))

        # Save seed-42 full test predictions for exact TU resource sensitivity.
        if seed == 42:
            full_key = "1" * len(bundle.X)
            fr = test_cache[full_key]
            pd.DataFrame({"sample_idx": split.test_idx, "y_true": fr["y_true"], "y_pred": fr["y_pred"]}).to_csv(
                REVIEW_DIR / f"{bundle.name}_seed42_full_samples_recomputed.csv", index=False
            )

        del fusion, val_cache, test_cache
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    exhaustive = pd.concat(exhaustive_all, ignore_index=True)
    bestk = pd.concat(bestk_all, ignore_index=True)
    order_best = pd.concat(order_best_all, ignore_index=True)
    order_curve = pd.concat(order_curve_all, ignore_index=True) if order_curve_all else pd.DataFrame()
    order_candidates = pd.concat(order_candidates_all, ignore_index=True) if order_candidates_all else pd.DataFrame()
    network_best = pd.concat(network_best_all, ignore_index=True) if network_best_all else pd.DataFrame()
    posthoc = pd.concat(posthoc_all, ignore_index=True) if posthoc_all else pd.DataFrame()

    exhaustive.to_csv(REVIEW_DIR / f"{bundle.name}_exhaustive_all_subsets.csv", index=False)
    bestk.to_csv(REVIEW_DIR / f"{bundle.name}_validation_best_subset_per_k.csv", index=False)
    order_best.to_csv(REVIEW_DIR / f"{bundle.name}_best_fixed_order_entropy.csv", index=False)
    if len(order_curve): order_curve.to_csv(REVIEW_DIR / f"{bundle.name}_fixed_order_entropy_budget_curve.csv", index=False)
    if len(order_candidates): order_candidates.to_csv(REVIEW_DIR / f"{bundle.name}_all_order_entropy_validation_candidates.csv", index=False)
    if len(network_best): network_best.to_csv(REVIEW_DIR / f"{bundle.name}_network_first_entropy.csv", index=False)
    if len(posthoc): posthoc.to_csv(REVIEW_DIR / f"{bundle.name}_posthoc_costmatched_static_mix.csv", index=False)

    bestk_summary = _aggregate_rows(bestk, ["test_macro_f1", "test_balanced_accuracy", "normalized_cost"], ["dataset", "k"])
    bestk_summary.to_csv(REVIEW_DIR / f"{bundle.name}_best_subset_per_k_summary.csv", index=False)
    order_summary = _aggregate_rows(order_best, ["test_macro_f1", "test_balanced_accuracy", "test_cost", "test_steps"], ["dataset", "method"])
    order_summary.to_csv(REVIEW_DIR / f"{bundle.name}_best_order_entropy_summary.csv", index=False)
    if len(order_curve):
        _aggregate_rows(order_curve, ["test_macro_f1", "test_balanced_accuracy", "test_cost", "test_steps"], ["dataset", "target_budget"]).to_csv(
            REVIEW_DIR / f"{bundle.name}_fixed_order_entropy_budget_curve_summary.csv", index=False)
    if len(network_best):
        _aggregate_rows(network_best, ["test_macro_f1", "test_balanced_accuracy", "test_cost", "test_steps"], ["dataset", "method"]).to_csv(
            REVIEW_DIR / f"{bundle.name}_network_first_entropy_summary.csv", index=False)
    if len(posthoc):
        _aggregate_rows(posthoc, ["macro_f1", "balanced_accuracy", "normalized_cost"], ["dataset", "method"]).to_csv(
            REVIEW_DIR / f"{bundle.name}_posthoc_costmatched_static_mix_summary.csv", index=False)

    # Paired comparisons versus the frozen old DQN, conditional on seed.
    comp_frames = []
    b = bestk.copy(); b["method_name"] = b.k.map(lambda k: f"BestFixedSubset-k{k}")
    comp_frames.append(_paired_against_old_dqn(b, prior_results, "method_name", "test_macro_f1", "normalized_cost"))
    comp_frames.append(_paired_against_old_dqn(order_best, prior_results, "method", "test_macro_f1", "test_cost"))
    if len(network_best): comp_frames.append(_paired_against_old_dqn(network_best, prior_results, "method", "test_macro_f1", "test_cost"))
    if len(posthoc): comp_frames.append(_paired_against_old_dqn(posthoc, prior_results, "method", "macro_f1", "normalized_cost"))
    paired = pd.concat([x for x in comp_frames if len(x)], ignore_index=True) if any(len(x) for x in comp_frames) else pd.DataFrame()
    if len(paired): paired.to_csv(REVIEW_DIR / f"{bundle.name}_reviewer_new_baselines_vs_dqn.csv", index=False)

    action = _seed42_action_order(prior_results, bundle)
    sensitivity = pd.DataFrame()
    if "TU_Delft" in bundle.name:
        sensitivity = _resource_exhaustion_sensitivity_from_samples(prior_results, bundle)
        # Add recomputed seed42 full fusion sensitivity.
        p = REVIEW_DIR / f"{bundle.name}_seed42_full_samples_recomputed.csv"
        if p.exists():
            df = pd.read_csv(p)
            names = [x.lower().replace(" ", "_") for x in bundle.label_names]
            rid = next((i for i, x in enumerate(names) if "resource" in x and "exhaust" in x), None)
            if rid is not None:
                keep_labels = [i for i in range(len(bundle.label_names)) if i != rid]
                s = df[df.y_true.astype(int) != rid]
                f1 = f1_score(s.y_true.astype(int), s.y_pred.astype(int), labels=keep_labels, average="macro", zero_division=0)
                ba = np.mean([recall_score_safe(s.y_true.astype(int), s.y_pred.astype(int), lab) for lab in keep_labels])
                sensitivity = pd.concat([sensitivity, pd.DataFrame([{
                    "dataset":bundle.name, "method":"Fusion-Full-seed42", "excluded_class":bundle.label_names[rid],
                    "n_remaining":len(s), "macro_f1_5class":float(f1), "balanced_accuracy_5class":float(ba)
                }])], ignore_index=True)
        if len(sensitivity): sensitivity.to_csv(REVIEW_DIR / f"{bundle.name}_resource_exhaustion_exclusion_seed42.csv", index=False)

    return {"exhaustive":exhaustive, "bestk":bestk, "order":order_best, "order_curve":order_curve, "network":network_best,
            "posthoc":posthoc, "paired":paired, "action":action, "sensitivity":sensitivity}


def write_reviewer_readme(results: Dict[str, Dict[str, pd.DataFrame]], holm: pd.DataFrame, prior_results: Path):
    lines = []
    lines.append("# ActiGrid reviewer-clearance outputs\n")
    lines.append("These analyses address conference-critical reviewer questions without retraining the DQN.\n")
    lines.append("## Interpretation rules")
    lines.append("- Exhaustive fixed subsets and fixed-order entropy policies are selected on validation only.")
    lines.append("- `POSTHOC-CostMatchedStaticMix` matches the already-observed DQN test cost and is descriptive only; do not present it as preregistered confirmatory evidence.")
    lines.append("- The ten seeds reuse the frozen group-disjoint split, so uncertainty is over stochastic retraining, not new substations/splits.")
    lines.append("- If a new simple baseline matches or beats DQN on TU Delft, revise the paper claim rather than hiding the result.\n")
    for ds, res in results.items():
        lines.append(f"## {ds}")
        if len(res["bestk"]):
            agg = _aggregate_rows(res["bestk"], ["test_macro_f1", "normalized_cost"], ["dataset","k"])
            lines.append("### Validation-selected best fixed subset by size")
            lines.append("```\n" + agg.to_string(index=False) + "\n```")
        if len(res["order"]):
            lines.append("### Best fixed-order + entropy stopping")
            lines.append("```\n" + res["order"][["seed","order","threshold","val_cost","test_cost","test_macro_f1"]].to_string(index=False) + "\n```")
        if len(res.get("order_curve", pd.DataFrame())):
            lines.append("### Validation-selected fixed-order performance-cost curve")
            curve_agg = _aggregate_rows(res["order_curve"], ["test_macro_f1","test_cost"], ["dataset","target_budget"])
            lines.append("```\n" + curve_agg.to_string(index=False) + "\n```")
        if len(res["network"]):
            lines.append("### Network-first entropy stopping")
            lines.append("```\n" + res["network"][["seed","order","threshold","val_cost","test_cost","test_macro_f1"]].to_string(index=False) + "\n```")
        if len(res["posthoc"]):
            lines.append("### POST-HOC static mixture at observed DQN cost")
            lines.append("```\n" + res["posthoc"][["seed","target_dqn_cost","normalized_cost","macro_f1","low_subset","high_subset"]].to_string(index=False) + "\n```")
        if len(res["sensitivity"]):
            lines.append("### TU resource-exhaustion exclusion (seed 42 sensitivity)")
            lines.append("```\n" + res["sensitivity"].to_string(index=False) + "\n```")
    lines.append("## Corrected multiplicity reporting")
    cols = [c for c in ["dataset","comparator","wilcoxon_p_macro_f1","holm_p_macro_f1","wilcoxon_p_cost","holm_p_cost"] if c in holm.columns]
    lines.append("```\n" + holm[cols].to_string(index=False) + "\n```")
    (REVIEW_DIR / "README_REVIEWER_CLEARANCE.md").write_text("\n".join(lines))


def reviewer_clearance_main(prior_results: Optional[str] = None, fast: bool = False):
    """Run the reviewer-clearance analyses. Call pipeline.setup() first."""
    global PRIOR_RESULTS, REVIEW_DIR, REVIEW_SEEDS
    REVIEW_DIR = P.WORK / "reviewer_clearance"
    REVIEW_DIR.mkdir(parents=True, exist_ok=True)
    REVIEW_SEEDS = (42, 123) if fast else tuple(P.CFG.final_seeds)
    PRIOR_RESULTS = _discover_prior_results_dir(prior_results)
    P.LOG.info("Prior frozen v7 results: %s", PRIOR_RESULTS)
    P.LOG.info("Reviewer outputs: %s", REVIEW_DIR)
    holm = make_corrected_pvalue_table(PRIOR_RESULTS)
    outputs = {}

    if P.CFG.run_msu:
        root = discover_dataset_root("msu", P.CFG.msu_root_override)
        if root:
            msu = load_msu_bundle(maybe_extract_archives(root, "msu"))
            outputs[msu.name] = run_reviewer_checks_for_bundle(msu, PRIOR_RESULTS)
        else:
            P.LOG.warning("MSU dataset not found; skipping reviewer checks")

    if P.CFG.run_tudelft:
        root = discover_dataset_root("tudelft", P.CFG.tudelft_root_override)
        if root:
            tu = load_tudelft_bundle(maybe_extract_archives(root, "tudelft"))
            outputs[tu.name] = run_reviewer_checks_for_bundle(tu, PRIOR_RESULTS)
        else:
            P.LOG.warning("TU Delft dataset not found; skipping reviewer checks")

    write_reviewer_readme(outputs, holm, PRIOR_RESULTS)
    shutil.make_archive(str(REVIEW_DIR), "zip", root_dir=str(REVIEW_DIR))
    P.LOG.info("Reviewer-clearance complete: %s", REVIEW_DIR)
    P.LOG.info("Zip: %s.zip", REVIEW_DIR)
