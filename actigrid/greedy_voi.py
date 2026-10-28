"""Greedy-VOI add-on: non-RL sequential controls on frozen backbones.

Reuses the three fusion checkpoints that the hardening stage trained on the
primary split (backbone seeds 42/123/456) and, without retraining any fusion,
DQN or tree model, adds:
  * Greedy-VOI: one small HGB regressor per modality predicts the one-step
    entropy reduction per unit cost from the current posterior, entropy and
    mask; the policy acquires the best predicted source until the entropy falls
    below a threshold (full 17-threshold frontier),
  * temperature-calibrated fixed-order entropy stopping (one temperature per
    mask, fitted on half of the validation set; orders and thresholds marked
    Pareto optimal on the other half),
  * a same-backbone Pareto comparison with the hardening DQN lambda frontier,
    observed-range AUCs, and a mask-wise calibration summary.

Outputs go to <work_dir>/next_addon/. The released copy is in results/next_addon/
and the run log is results/logs/greedy_voi.log. It needs the hardening outputs
(DQN lambda frontiers, mask calibration and cache/*_hardening_fusion_seed*.pt);
by default they are read from <work_dir>/reviewer_hardening/ when that holds the
checkpoints, otherwise from results/reviewer_hardening/.

Code is the standalone Kaggle add-on script. For the release it reads the
pipeline from actigrid.pipeline instead of an embedded copy of the same source,
takes the hardening folder as an argument, and fixes summarize_calibration(),
which averaged validation instead of test raw ECE (see README). The policy,
calibration and frontier logic is unchanged.
"""
import itertools
import shutil
from typing import Dict, List, Optional, Tuple

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, log_loss
from sklearn.model_selection import StratifiedShuffleSplit

from actigrid import pipeline as P
from actigrid.hardening import load_bundle
from actigrid.pipeline import Path, np, pd, torch

# Three stochastic seeds already available in the conference hardening run.
SEEDS = [42, 123, 456]
# Subsample size for fitting Greedy-VOI regressors. No neural training occurs.
VOI_TRAIN_MAX = {"tu": 3246, "msu": 7000}
BATCH = 1024

# Set by greedy_voi_main() after pipeline.setup().
HARDEN_DIR: Path = None
OUT: Path = None
NAMES = {"tu": "TU_Delft_2026_CPPS", "msu": "MSU_ORNL_Power"}


def _discover_hardening_dir(hardening_dir: Optional[str] = None) -> Path:
    """Folder with the hardening outputs and cache/: argument, work dir, or repo results/."""
    candidates = [hardening_dir, P.WORK / "reviewer_hardening",
                  Path(__file__).resolve().parents[1] / "results" / "reviewer_hardening"]
    for c in candidates:
        if c and len(list((Path(c) / "cache").glob("*_hardening_fusion_seed*.pt"))):
            return Path(c)
    raise FileNotFoundError("No reviewer_hardening folder with cache/*_hardening_fusion_seed*.pt found; "
                            "run --stage hardening first or pass --hardening-dir.")


def checkpoint_path(dataset_name: str, seed: int) -> Path:
    # For DQN lambda-frontier comparison, use hardening_fusion_seedX checkpoints:
    # these share the original fixed group split and differ only by stochastic seed.
    p = HARDEN_DIR / "cache" / f"{dataset_name}_hardening_fusion_seed{seed}.pt"
    if not p.exists():
        raise FileNotFoundError(p)
    return p


def load_frozen_model(bundle, ckpt: Path):
    modalities = list(bundle.X.keys())
    input_dims = {m: bundle.X[m].shape[1] for m in modalities}
    model = P.ActiveFusionClassifier(
        input_dims=input_dims,
        modalities=modalities,
        num_classes=len(bundle.label_names),
        hidden=128,
        latent=64,
        dropout=0.20,
    ).to(P.CFG.device)
    obj = torch.load(ckpt, map_location=P.CFG.device, weights_only=False)
    state = obj["state"] if isinstance(obj, dict) and "state" in obj else obj
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


# -----------------------------------------------------------------------------
# Efficient frozen-backbone inference for every acquisition mask
# -----------------------------------------------------------------------------
def all_masks(M: int, include_empty=True):
    start = 0 if include_empty else 1
    return list(range(start, 1 << M))


def int_to_mask(mask_int: int, M: int) -> np.ndarray:
    return np.array([(mask_int >> j) & 1 for j in range(M)], dtype=np.float32)


def mask_cost(mask_int: int, modalities: List[str], costs: Dict[str, float]) -> float:
    return float(sum(costs[m] for j, m in enumerate(modalities) if (mask_int >> j) & 1))


@torch.no_grad()
def infer_mask(model, split, modalities: List[str], idx: np.ndarray, mask_int: int, device: str):
    M = len(modalities)
    mask_np = int_to_mask(mask_int, M)
    probs_chunks, logits_chunks = [], []
    idx = np.asarray(idx, dtype=int)
    for s in range(0, len(idx), BATCH):
        ii = idx[s:s+BATCH]
        xs = [torch.from_numpy(split.X_scaled[m][ii]).float().to(device) for m in modalities]
        mask = torch.from_numpy(np.repeat(mask_np[None, :], len(ii), axis=0)).float().to(device)
        logits, _ = model(xs, mask)
        probs = torch.softmax(logits, dim=1)
        logits_chunks.append(logits.cpu().numpy())
        probs_chunks.append(probs.cpu().numpy())
    logits = np.concatenate(logits_chunks, axis=0)
    probs = np.concatenate(probs_chunks, axis=0)
    ent = -np.sum(probs * np.log(np.clip(probs, 1e-12, 1.0)), axis=1)
    pred = probs.argmax(axis=1)
    return {"logits": logits, "probs": probs, "entropy": ent, "pred": pred}


def precompute_masks(model, split, modalities, idx, device):
    out = {}
    M = len(modalities)
    for k, mi in enumerate(all_masks(M, include_empty=True), 1):
        out[mi] = infer_mask(model, split, modalities, idx, mi, device)
        if k % 8 == 0 or k == (1 << M):
            print(f"    precomputed {k}/{1<<M} masks")
    return out


# -----------------------------------------------------------------------------
# Greedy-VOI: predict one-step entropy reduction from current frozen state
# -----------------------------------------------------------------------------
def stratified_subsample(y: np.ndarray, idx: np.ndarray, max_n: int, seed: int) -> np.ndarray:
    idx = np.asarray(idx, dtype=int)
    if len(idx) <= max_n:
        return idx
    rng = np.random.default_rng(seed)
    parts = []
    classes = np.unique(y[idx])
    per = max(1, max_n // len(classes))
    for c in classes:
        pool = idx[y[idx] == c]
        take = min(len(pool), per)
        parts.append(rng.choice(pool, size=take, replace=False))
    chosen = np.concatenate(parts)
    if len(chosen) < max_n:
        remaining = np.setdiff1d(idx, chosen, assume_unique=False)
        extra = min(max_n - len(chosen), len(remaining))
        if extra > 0:
            chosen = np.concatenate([chosen, rng.choice(remaining, size=extra, replace=False)])
    rng.shuffle(chosen)
    return chosen[:max_n]


def fit_voi_regressors(pre_train, modalities: List[str], costs: Dict[str, float]):
    M = len(modalities)
    regressors = {}
    for a, mod in enumerate(modalities):
        Xrows, yrows = [], []
        for mask_int in all_masks(M, include_empty=True):
            if (mask_int >> a) & 1:
                continue
            next_mask = mask_int | (1 << a)
            cur = pre_train[mask_int]
            nxt = pre_train[next_mask]
            # State features available BEFORE acquisition: posterior, entropy, mask.
            mask_vec = np.repeat(int_to_mask(mask_int, M)[None, :], len(cur["entropy"]), axis=0)
            feat = np.concatenate([
                cur["probs"],
                cur["entropy"][:, None],
                mask_vec,
            ], axis=1)
            gain = cur["entropy"] - nxt["entropy"]
            # Predict entropy reduction per cost so heterogeneous cost can be supported.
            target = gain / max(float(costs[mod]), 1e-8)
            Xrows.append(feat.astype(np.float32))
            yrows.append(target.astype(np.float32))
        X = np.concatenate(Xrows, axis=0)
        y = np.concatenate(yrows, axis=0)
        reg = HistGradientBoostingRegressor(
            max_iter=120,
            learning_rate=0.06,
            max_leaf_nodes=15,
            l2_regularization=1e-3,
            random_state=1000 + a,
        )
        reg.fit(X, y)
        regressors[a] = reg
        print(f"    fitted VOI regressor for {mod}: n={len(y)}")
    return regressors


def threshold_grid(pre_val, M: int, n=17):
    vals = []
    for mi in range(1, 1 << M):
        vals.append(pre_val[mi]["entropy"])
    vals = np.concatenate(vals)
    qs = np.linspace(0.05, 0.95, n)
    grid = list(np.unique(np.quantile(vals, qs)))
    # Extremes: acquire all / stop as early as allowed.
    grid = [-1.0] + grid + [float(np.max(vals) + 1.0)]
    return grid


def run_greedy_policy(pre, y_true: np.ndarray, idx: np.ndarray, regressors, modalities, costs, threshold: float):
    M = len(modalities)
    full_mask = (1 << M) - 1
    rows = []
    for r, sample_idx in enumerate(idx):
        mask_int = 0
        total_cost = 0.0
        steps = 0
        path = []
        while True:
            cur = pre[mask_int]
            ent = float(cur["entropy"][r])
            pred = int(cur["pred"][r])
            # STOP forbidden before acquiring at least one modality.
            if mask_int != 0 and (ent <= threshold or mask_int == full_mask):
                break
            if mask_int == full_mask:
                break
            mask_vec = int_to_mask(mask_int, M)
            feat = np.concatenate([
                cur["probs"][r],
                np.array([ent], dtype=np.float32),
                mask_vec,
            ])[None, :]
            scores = []
            for a, mod in enumerate(modalities):
                if (mask_int >> a) & 1:
                    continue
                score = float(regressors[a].predict(feat)[0])
                scores.append((score, a, mod))
            # deterministic tie-breaking by modality index
            scores.sort(key=lambda x: (-x[0], x[1]))
            _, a, mod = scores[0]
            mask_int |= (1 << a)
            total_cost += float(costs[mod])
            steps += 1
            path.append(mod)
        rows.append({
            "sample_idx": int(sample_idx),
            "y_true": int(y_true[r]),
            "y_pred": pred,
            "normalized_cost": total_cost,
            "steps": steps,
            "path": " -> ".join(path) + " -> STOP",
            "final_entropy": ent,
        })
    return pd.DataFrame(rows)


def metrics_from_trace(df: pd.DataFrame):
    y, p = df.y_true.to_numpy(), df.y_pred.to_numpy()
    return {
        "macro_f1": float(f1_score(y, p, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y, p)),
        "accuracy": float(accuracy_score(y, p)),
        "normalized_cost": float(df.normalized_cost.mean()),
        "mean_steps": float(df.steps.mean()),
    }


def pareto_mask(cost, score):
    cost = np.asarray(cost, float); score = np.asarray(score, float)
    keep = np.ones(len(cost), dtype=bool)
    for i in range(len(cost)):
        # j dominates i if no more costly and no worse, with one strict inequality.
        dom = (cost <= cost[i] + 1e-12) & (score >= score[i] - 1e-12) & (
            (cost < cost[i] - 1e-12) | (score > score[i] + 1e-12)
        )
        dom[i] = False
        if np.any(dom):
            keep[i] = False
    return keep


def observed_auc(df: pd.DataFrame, x="normalized_cost", y="macro_f1"):
    d = df[[x, y]].dropna().sort_values(x).drop_duplicates(x, keep="last")
    if len(d) < 2:
        return np.nan, np.nan, np.nan, np.nan
    xx = d[x].to_numpy(float); yy = d[y].to_numpy(float)
    auc = float(np.trapz(yy, xx)) if hasattr(np, "trapz") else float(np.trapezoid(yy, xx))
    span = float(xx[-1] - xx[0])
    norm = auc / span if span > 0 else np.nan
    return auc, float(xx[0]), float(xx[-1]), norm



# -----------------------------------------------------------------------------
# Temperature-calibrated fixed-order entropy stopping (no model retraining)
# -----------------------------------------------------------------------------
def softmax_np(logits: np.ndarray, T: float = 1.0) -> np.ndarray:
    z = logits / max(float(T), 1e-6)
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / np.clip(e.sum(axis=1, keepdims=True), 1e-12, None)


def split_val_positions(y_val: np.ndarray, seed: int):
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=seed)
    a, b = next(sss.split(np.zeros(len(y_val)), y_val))
    return np.asarray(a), np.asarray(b)


def fit_mask_temperatures(pre_val, y_val: np.ndarray, calib_pos: np.ndarray, M: int):
    grid = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 4.0]
    temps = {}
    for mi in range(1, 1 << M):
        logits = pre_val[mi]["logits"][calib_pos]
        yy = y_val[calib_pos]
        best_t, best_nll = 1.0, float("inf")
        for T in grid:
            p = softmax_np(logits, T)
            nll = log_loss(yy, p, labels=list(range(p.shape[1])))
            if nll < best_nll:
                best_nll, best_t = nll, T
        temps[mi] = float(best_t)
    return temps


def calibrated_mask_view(pre, temps, M: int):
    out = {}
    for mi in range(1, 1 << M):
        p = softmax_np(pre[mi]["logits"], temps[mi])
        out[mi] = {
            "probs": p,
            "entropy": -np.sum(p * np.log(np.clip(p, 1e-12, 1.0)), axis=1),
            "pred": p.argmax(axis=1),
        }
    return out


def eval_order_threshold(cal_view, y_true: np.ndarray, order: Tuple[int, ...],
                         modalities: List[str], costs: Dict[str, float], threshold: float,
                         positions: np.ndarray | None = None):
    if positions is None:
        positions = np.arange(len(y_true))
    positions = np.asarray(positions, dtype=int)
    M = len(modalities)
    masks, step_costs = [], []
    mi, c = 0, 0.0
    for a in order:
        mi |= (1 << a)
        c += float(costs[modalities[a]])
        masks.append(mi); step_costs.append(c)
    ent = np.stack([cal_view[mi]["entropy"][positions] for mi in masks], axis=1)
    pred = np.stack([cal_view[mi]["pred"][positions] for mi in masks], axis=1)
    stop = ent <= threshold
    any_stop = stop.any(axis=1)
    first = stop.argmax(axis=1)
    first = np.where(any_stop, first, M - 1)
    yp = pred[np.arange(len(positions)), first]
    yy = y_true[positions]
    cc = np.asarray(step_costs)[first]
    return {
        "macro_f1": float(f1_score(yy, yp, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(yy, yp)),
        "accuracy": float(accuracy_score(yy, yp)),
        "normalized_cost": float(cc.mean()),
        "mean_steps": float((first + 1).mean()),
    }


def calibrated_entropy_frontier(pre_val, pre_test, y_val, y_test, modalities, costs, seed: int):
    M = len(modalities)
    calib_pos, select_pos = split_val_positions(y_val, seed)
    temps = fit_mask_temperatures(pre_val, y_val, calib_pos, M)
    cv = calibrated_mask_view(pre_val, temps, M)
    ct = calibrated_mask_view(pre_test, temps, M)

    # Thresholds are derived only from the selection half of validation.
    ecat = np.concatenate([cv[mi]["entropy"][select_pos] for mi in range(1, 1 << M)])
    thresholds = [-1.0] + list(np.unique(np.quantile(ecat, np.linspace(0.05, 0.95, 15)))) + [float(ecat.max() + 1.0)]

    rows = []
    for order in itertools.permutations(range(M)):
        order_name = " -> ".join(modalities[a] for a in order)
        for t in thresholds:
            vm = eval_order_threshold(cv, y_val, order, modalities, costs, float(t), select_pos)
            tm = eval_order_threshold(ct, y_test, order, modalities, costs, float(t), None)
            rows.append({
                "seed": seed,
                "order": order_name,
                "threshold": float(t),
                "val_macro_f1": vm["macro_f1"],
                "val_cost": vm["normalized_cost"],
                "macro_f1": tm["macro_f1"],
                "balanced_accuracy": tm["balanced_accuracy"],
                "accuracy": tm["accuracy"],
                "normalized_cost": tm["normalized_cost"],
                "mean_steps": tm["mean_steps"],
            })
    df = pd.DataFrame(rows)
    # Mark validation Pareto points; these are the defensible candidates to inspect on test.
    keep = pareto_mask(df.val_cost.values, df.val_macro_f1.values)
    df["val_pareto"] = keep
    return df, temps

# -----------------------------------------------------------------------------
# Existing-result postprocessing
# -----------------------------------------------------------------------------
def summarize_calibration(dataset_name: str):
    p = HARDEN_DIR / f"{dataset_name}_mask_calibration_seed42.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p)
    # Test-set ECE before and after scaling. The Kaggle run detected the raw column
    # by name and picked val_ece_raw (the first match), so its released
    # mean/max_raw_ece were validation values; corrected here and in results/.
    raw, cal = "test_ece_raw", "test_ece_temp_scaled"
    out = {
        "dataset": dataset_name,
        "n_masks": len(df),
    }
    if raw:
        out["mean_raw_ece"] = float(df[raw].mean())
        out["max_raw_ece"] = float(df[raw].max())
    if cal:
        out["mean_calibrated_ece"] = float(df[cal].mean())
        out["max_calibrated_ece"] = float(df[cal].max())
    return out


def build_same_backbone_comparison(dataset_name: str, greedy_df: pd.DataFrame, cal_entropy_df: pd.DataFrame):
    dqn_path = HARDEN_DIR / f"{dataset_name}_dqn_lambda_frontier.csv"
    dqn = pd.read_csv(dqn_path)
    dqn = dqn[["seed", "lambda", "macro_f1", "normalized_cost"]].copy()
    dqn["method_family"] = "DQN"
    dqn["setting"] = dqn["lambda"].map(lambda x: f"lambda={x}")

    g = greedy_df[["seed", "threshold", "macro_f1", "normalized_cost"]].copy()
    g["method_family"] = "Greedy-VOI"
    g["setting"] = g["threshold"].map(lambda x: f"H<={x:.4f}")

    ce = cal_entropy_df[cal_entropy_df["val_pareto"]].copy()
    ce["method_family"] = "Calibrated-OrderEntropy"
    ce["setting"] = ce.apply(lambda r: f"{r.order};H<={r.threshold:.4f}", axis=1)

    both = pd.concat([
        dqn[["seed","macro_f1","normalized_cost","method_family","setting"]],
        g[["seed","macro_f1","normalized_cost","method_family","setting"]],
        ce[["seed","macro_f1","normalized_cost","method_family","setting"]],
    ], ignore_index=True)
    both["dataset"] = dataset_name

    # Pareto status within each stochastic seed on the same fixed split/backbone seed.
    both["pareto"] = False
    for seed, ix in both.groupby("seed").groups.items():
        sub = both.loc[ix]
        keep = pareto_mask(sub.normalized_cost.values, sub.macro_f1.values)
        both.loc[np.array(list(ix))[keep], "pareto"] = True

    auc_rows = []
    for (seed, fam), sub in both.groupby(["seed", "method_family"]):
        auc, cmin, cmax, avg = observed_auc(sub)
        auc_rows.append({
            "dataset": dataset_name,
            "seed": int(seed),
            "method_family": fam,
            "observed_auc": auc,
            "cost_min": cmin,
            "cost_max": cmax,
            "mean_f1_over_observed_cost_range": avg,
            "n_points": len(sub),
        })
    return both, pd.DataFrame(auc_rows)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def run_dataset(key: str, seeds: List[int]):
    """Greedy-VOI and calibrated entropy frontiers for one dataset (writes the per-seed files)."""
    bundle = load_bundle(key)
    name = bundle.name
    modalities = list(bundle.X.keys())
    costs = bundle.modality_costs
    print("\n" + "="*88)
    print(f"DATASET: {name} | modalities={modalities}")
    print("="*88)

    # The DQN lambda frontier used the original fixed group split, with stochastic
    # fusion/DQN seeds 42/123/456. Reproduce exactly that split here.
    split = P.group_split(bundle, seed=42)
    cal_entropy_seed_dfs, greedy_seed_dfs = [], []

    for seed in seeds:
        print(f"\n[{name}] frozen backbone seed {seed}")
        ck = checkpoint_path(name, seed)
        model = load_frozen_model(bundle, ck)

        tr_sub = stratified_subsample(bundle.y, split.train_idx, VOI_TRAIN_MAX[key], seed)
        print(f"  VOI policy-fit sample: {len(tr_sub)} / {len(split.train_idx)} training rows")

        print("  precomputing train-mask states...")
        pre_tr = precompute_masks(model, split, modalities, tr_sub, P.CFG.device)
        regressors = fit_voi_regressors(pre_tr, modalities, costs)
        del pre_tr

        print("  precomputing validation masks...")
        pre_va = precompute_masks(model, split, modalities, split.val_idx, P.CFG.device)
        print("  precomputing test masks...")
        pre_te = precompute_masks(model, split, modalities, split.test_idx, P.CFG.device)

        print("  evaluating temperature-calibrated fixed-order entropy frontier...")
        ce_df, temps = calibrated_entropy_frontier(
            pre_va, pre_te, bundle.y[split.val_idx], bundle.y[split.test_idx],
            modalities, costs, seed
        )
        ce_df["dataset"] = name
        cal_entropy_seed_dfs.append(ce_df)
        ce_df.to_csv(OUT / f"{name}_calibrated_order_entropy_frontier_seed{seed}.csv", index=False)
        pd.DataFrame([{
            "dataset": name, "seed": seed, "mask_int": mi, "temperature": T
        } for mi, T in temps.items()]).to_csv(
            OUT / f"{name}_mask_temperatures_seed{seed}.csv", index=False
        )

        grid = threshold_grid(pre_va, len(modalities), n=15)
        seed_rows = []
        for t in grid:
            va_trace = run_greedy_policy(
                pre_va, bundle.y[split.val_idx], split.val_idx,
                regressors, modalities, costs, float(t)
            )
            te_trace = run_greedy_policy(
                pre_te, bundle.y[split.test_idx], split.test_idx,
                regressors, modalities, costs, float(t)
            )
            vm = metrics_from_trace(va_trace); tm = metrics_from_trace(te_trace)
            row = {
                "dataset": name,
                "seed": seed,
                "threshold": float(t),
                "val_macro_f1": vm["macro_f1"],
                "val_cost": vm["normalized_cost"],
                "macro_f1": tm["macro_f1"],
                "balanced_accuracy": tm["balanced_accuracy"],
                "accuracy": tm["accuracy"],
                "normalized_cost": tm["normalized_cost"],
                "mean_steps": tm["mean_steps"],
            }
            seed_rows.append(row)

        seed_df = pd.DataFrame(seed_rows)
        keep = pareto_mask(seed_df.normalized_cost, seed_df.macro_f1)
        seed_df["pareto_within_greedy"] = keep
        greedy_seed_dfs.append(seed_df)
        seed_df.to_csv(OUT / f"{name}_greedy_voi_frontier_seed{seed}.csv", index=False)

        # Save one representative path trace: validation-selected highest-F1 point
        # among the Greedy-VOI Pareto points. Selection uses validation only.
        candidates = seed_df[seed_df.pareto_within_greedy].copy()
        best_row = candidates.sort_values(["val_macro_f1", "val_cost"], ascending=[False, True]).iloc[0]
        best_t = float(best_row.threshold)
        path_trace = run_greedy_policy(
            pre_te, bundle.y[split.test_idx], split.test_idx,
            regressors, modalities, costs, best_t
        )
        path_trace["seed"] = seed
        path_trace["threshold"] = best_t
        path_trace.to_csv(OUT / f"{name}_greedy_voi_paths_seed{seed}.csv", index=False)

        del pre_va, pre_te, regressors, model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    pd.concat(greedy_seed_dfs, ignore_index=True).to_csv(OUT / f"{name}_greedy_voi_frontier.csv", index=False)
    pd.concat(cal_entropy_seed_dfs, ignore_index=True).to_csv(
        OUT / f"{name}_calibrated_order_entropy_frontier.csv", index=False)


def write_summaries(names: List[str]):
    """Comparison, AUC, calibration and markdown summaries from the frontier CSVs in OUT."""
    calibration_rows, comparison_rows, auc_rows = [], [], []
    for name in names:
        cal = summarize_calibration(name)
        if cal:
            calibration_rows.append(cal)
        greedy = pd.read_csv(OUT / f"{name}_greedy_voi_frontier.csv")
        cal_entropy = pd.read_csv(OUT / f"{name}_calibrated_order_entropy_frontier.csv")
        comp, auc = build_same_backbone_comparison(name, greedy, cal_entropy)
        comp.to_csv(OUT / f"{name}_DQN_vs_GreedyVOI_same_backbone.csv", index=False)
        auc.to_csv(OUT / f"{name}_observed_range_auc.csv", index=False)
        comparison_rows.append(comp); auc_rows.append(auc)

    if calibration_rows:
        pd.DataFrame(calibration_rows).to_csv(OUT / "calibration_summary.csv", index=False)
    if comparison_rows:
        pd.concat(comparison_rows, ignore_index=True).to_csv(OUT / "all_same_backbone_pareto_points.csv", index=False)
    if auc_rows:
        pd.concat(auc_rows, ignore_index=True).to_csv(OUT / "all_observed_range_auc.csv", index=False)

    # Compact markdown report.
    lines = [
        "# ActiGrid NEXT add-on summary",
        "",
        "No fusion/DQN/tree models were retrained in this add-on.",
        "Greedy-VOI uses frozen fusion checkpoints and lightweight regressors trained to predict one-step entropy reduction.",
        "It is a non-RL baseline, not a reproduction of JAFA/GSM/ACO.",
        "",
    ]
    for name in ["TU_Delft_2026_CPPS", "MSU_ORNL_Power"]:
        p = OUT / f"{name}_DQN_vs_GreedyVOI_same_backbone.csv"
        if p.exists():
            d = pd.read_csv(p)
            lines += [f"## {name}", "### Pareto points (same fixed split/backbone seeds)", "```"]
            lines += [d[d.pareto].sort_values(["seed","normalized_cost"]).to_string(index=False), "```", ""]
        a = OUT / f"{name}_observed_range_auc.csv"
        if a.exists():
            lines += ["### Observed-range AUC summary", "```", pd.read_csv(a).to_string(index=False), "```", ""]
    if (OUT / "calibration_summary.csv").exists():
        lines += ["## Calibration summary", "```", pd.read_csv(OUT / "calibration_summary.csv").to_string(index=False), "```", ""]
    (OUT / "NEXT_PAPER_READY_SUMMARY.md").write_text("\n".join(lines))


def greedy_voi_main(hardening_dir: Optional[str] = None, fast: bool = False, summaries_only: bool = False):
    """Run the Greedy-VOI add-on. Call pipeline.setup() first.

    hardening_dir   folder with the hardening outputs and cache/ (see module docstring).
    fast            smoke test: backbone seed 42 only. Not for results.
    summaries_only  skip inference; rebuild the comparison/AUC/calibration summaries
                    from the frontier CSVs already in <work_dir>/next_addon/.
    """
    global HARDEN_DIR, OUT
    HARDEN_DIR = _discover_hardening_dir(hardening_dir)
    OUT = P.WORK / "next_addon"
    OUT.mkdir(parents=True, exist_ok=True)
    keys = [k for k, on in (("tu", P.CFG.run_tudelft), ("msu", P.CFG.run_msu)) if on]
    print(f"[ok] using hardening results: {HARDEN_DIR}")
    print(f"Device used for add-on inference: {P.CFG.device}")
    print("NOTE: no fusion/DQN/tree model training will be executed.\n")

    if not summaries_only:
        for key in keys:
            run_dataset(key, [42] if fast else SEEDS)
    write_summaries([NAMES[k] for k in keys])

    # Strong-classifier and repeated-split tables are copied into the add-on output
    # so the final paper package has everything in one place without retraining them.
    for pattern in [
        "*_strong_subset_validation_selected.csv",
        "*_strong_subset_classifier_frontier.csv",
        "*_repeated_group_split_summary.csv",
        "*_repeated_group_split_policy_audit.csv",
        "*_mask_calibration_seed42.csv",
        "*_cluster_bootstrap_seed42.csv",
        "*_premature_stop_corruption_stress.csv",
        "*_dqn_reward_architecture_ablations.csv",
        "*_dqn_lambda_frontier.csv",
    ]:
        for p in HARDEN_DIR.glob(pattern):
            if p.resolve() != (OUT / p.name).resolve():
                shutil.copy2(p, OUT / p.name)

    zip_path = shutil.make_archive(str(OUT), "zip", root_dir=str(OUT))
    print("\nNEXT ADD-ON COMPLETE")
    print("Output:", OUT)
    print("ZIP:", zip_path)
