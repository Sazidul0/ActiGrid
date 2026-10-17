"""Final same-backbone controlled comparison (paper Tables I-II, Figs. 2-3).

For each dataset and each of the ten seeds, ONE mask-aware fusion backbone is
trained, frozen, and shared by every policy:
  * full fusion,
  * every nonempty fixed subset (best subset per size k selected on validation),
  * all M! fixed acquisition orders with entropy stopping (selected on
    validation within +/-0.02 of the DQN validation cost), and
  * ActiGrid Double-DQN (lambda = 0.5, 20,000 episodes) trained on that same
    frozen backbone.
It also exports per-class metrics, DQN confusion counts, acquisition paths,
post-first-acquisition branching, and split/class-support manifests.

Outputs go to <work_dir>/controlled_same_backbone/. The paper's run log is
results/logs/controlled_same_backbone.log.

Code is section 18 of the original Kaggle script. For the release it was moved
into its own module: CFG/WORK/LOG are read from actigrid.pipeline at call time,
the frozen v7 selections are read from results/*_v7_selection.json (with a
full-precision built-in fallback), and output paths are created when the
stage runs. Experimental logic is unchanged.
"""
from actigrid import pipeline as P
from actigrid.pipeline import (
    Any, Dict, List, Optional, Path,
    ActiveEvidenceEnv, DatasetBundle,
    _aggregate_multiseed, _attach_groups_to_env, confusion_matrix,
    discover_dataset_root, evaluate_dqn_detailed, evaluate_full_detailed, gc,
    group_split, json, load_msu_bundle, load_tudelft_bundle, math,
    maybe_extract_archives, np, os, pd, precision_recall_fscore_support,
    seed_everything, shutil, torch, train_dqn_policy, train_fusion_classifier,
)
from actigrid.reviewer_clearance import (
    _build_all_mask_cache, _exhaustive_subset_rows, _mask_from_subset, _mask_key,
    _search_fixed_order_entropy, _select_best_subset_per_k, _simulate_order,
    _validation_selected_order_curve,
)

# Set by controlled_comparison_main() after pipeline.setup().
CONTROL_DIR: Path = None
CONTROL_SEEDS = ()
CONTROL_DQN_EPISODES = 20000


# Frozen validation selections from the v7 sweep (results/*_v7_selection.json),
# kept at full precision so the +/-0.02 cost band is reproduced exactly.
FROZEN_SELECTION = {
    "MSU_ORNL_Power": {"selected_lambda": 0.5, "target_validation_cost": 0.7574122690128805},
    "TU_Delft_2026_CPPS": {"selected_lambda": 0.5, "target_validation_cost": 0.5867293800773292},
}


def _discover_selection_dir(prior_results: Optional[str] = None) -> Optional[Path]:
    """Folder holding *_v7_selection.json: argument, $ACTIGRID_RESULTS, or repo results/."""
    candidates = [prior_results, os.environ.get("ACTIGRID_RESULTS"),
                  Path(__file__).resolve().parents[1] / "results"]
    for c in candidates:
        if c and Path(c).exists() and any(Path(c).glob("*_v7_selection.json")):
            return Path(c)
    return None


def _load_control_selection(prior_results: Optional[Path], bundle_name: str) -> Dict[str, float]:
    frozen = dict(FROZEN_SELECTION[bundle_name])
    p = prior_results / f"{bundle_name}_v7_selection.json" if prior_results else None
    if p is not None and p.exists():
        try:
            d = json.loads(p.read_text())
            return {
                "selected_lambda": float(d.get("selected_lambda", frozen["selected_lambda"])),
                "target_validation_cost": float(d.get("target_validation_cost", frozen["target_validation_cost"])),
            }
        except Exception as e:
            P.LOG.warning("Could not read %s: %s", p, e)
    return frozen


def _summary_row_to_record(summary: pd.DataFrame, dataset: str, seed: int, method: str, **extra) -> Dict[str, Any]:
    r = summary.iloc[0].to_dict()
    r.update({"dataset": dataset, "seed": int(seed), "method": method})
    r.update(extra)
    return r


def _fixed_subset_summary_from_row(row: pd.Series, dataset: str, seed: int) -> Dict[str, Any]:
    return {
        "dataset": dataset,
        "seed": int(seed),
        "method": f"BestFixedSubset-k{int(row['k'])}",
        "accuracy": float(row.get("test_accuracy", np.nan)),
        "balanced_accuracy": float(row["test_balanced_accuracy"]),
        "precision_macro": float(row.get("test_precision_macro", np.nan)),
        "recall_macro": float(row.get("test_recall_macro", row["test_balanced_accuracy"])),
        "macro_f1": float(row["test_macro_f1"]),
        "normalized_cost": float(row["normalized_cost"]),
        "acquisition_ratio": float(row["normalized_cost"]),
        "mean_steps": float(row["k"]),
        "subset": str(row["subset"]),
    }


def _paired_effect_table(raw: pd.DataFrame, dataset: str) -> pd.DataFrame:
    from scipy.stats import wilcoxon, t as student_t
    focal = "ActiGrid-DQN"
    out = []
    f = raw[raw.dataset == dataset].copy()
    methods = [m for m in f.method.unique() if m != focal]
    for comp in methods:
        a = f[f.method == focal][["seed","macro_f1","normalized_cost"]].rename(
            columns={"macro_f1":"f1_a","normalized_cost":"cost_a"})
        b = f[f.method == comp][["seed","macro_f1","normalized_cost"]].rename(
            columns={"macro_f1":"f1_b","normalized_cost":"cost_b"})
        g = a.merge(b, on="seed", how="inner")
        if len(g) < 2:
            continue
        df1 = (g.f1_a - g.f1_b).to_numpy(float)
        dc = (g.cost_a - g.cost_b).to_numpy(float)
        def _wp(d):
            if len(d) < 2 or np.allclose(d, 0): return 1.0
            try: return float(wilcoxon(d, zero_method="wilcox", alternative="two-sided", mode="auto").pvalue)
            except Exception: return float("nan")
        def _ci(d):
            n=len(d); mean=float(np.mean(d)); sd=float(np.std(d,ddof=1)) if n>1 else 0.0
            hw=float(student_t.ppf(0.975,n-1)*sd/math.sqrt(n)) if n>1 else 0.0
            return mean, mean-hw, mean+hw
        mf,lff,uff=_ci(df1); mc,lcc,ucc=_ci(dc)
        out.append({
            "dataset":dataset,"focal":focal,"comparator":comp,"n_seed_pairs":len(g),
            "dqn_minus_comparator_f1":mf,"f1_diff_ci95_low":lff,"f1_diff_ci95_high":uff,
            "wilcoxon_p_f1":_wp(df1),
            "dqn_minus_comparator_cost":mc,"cost_diff_ci95_low":lcc,"cost_diff_ci95_high":ucc,
            "wilcoxon_p_cost":_wp(dc),
        })
    return pd.DataFrame(out)


def _per_class_rows(y_true: np.ndarray, y_pred: np.ndarray, label_names: List[str], dataset: str,
                    seed: int, method: str) -> pd.DataFrame:
    p,r,f,s = precision_recall_fscore_support(
        np.asarray(y_true,int), np.asarray(y_pred,int), labels=np.arange(len(label_names)), zero_division=0)
    rows=[]
    for i,name in enumerate(label_names):
        rows.append({"dataset":dataset,"seed":int(seed),"method":method,"class_id":i,"class_name":name,
                     "precision":float(p[i]),"recall":float(r[i]),"f1":float(f[i]),"support":int(s[i])})
    return pd.DataFrame(rows)


def _branching_rows(dqn_rows: pd.DataFrame, bundle: DatasetBundle, seed: int) -> pd.DataFrame:
    rec=[]
    for _,r in dqn_rows.iterrows():
        seq=[x.strip() for x in str(r.get("acquired","")).split("->") if x.strip()]
        first=seq[0] if len(seq)>=1 else "STOP"
        second=seq[1] if len(seq)>=2 else "STOP"
        third=seq[2] if len(seq)>=3 else "STOP"
        rec.append({
            "dataset":bundle.name,"seed":int(seed),"sample_idx":int(r.sample_idx),
            "class_name":bundle.label_names[int(r.y_true)],"correct":int(r.y_true)==int(r.y_pred),
            "first_action":first,"second_action_or_stop":second,"third_action_or_stop":third,
            "path": " -> ".join(seq+["STOP"]) if seq else "STOP",
            "steps":int(r.steps),"cost":float(r.cost),
        })
    return pd.DataFrame(rec)


def _branching_summary(branch: pd.DataFrame) -> pd.DataFrame:
    # Each seed evaluates the same held-out examples. Compute per-seed proportions first,
    # then summarize those proportions across seeds rather than pretending repeated seed
    # predictions are independent samples.
    rows=[]
    for (dataset,seed,cls),g in branch.groupby(["dataset","seed","class_name"]):
        n=len(g)
        actions=sorted(set(g.second_action_or_stop.astype(str)))
        for a in actions:
            rows.append({"dataset":dataset,"seed":seed,"class_name":cls,"decision_after_first":a,
                         "proportion":float((g.second_action_or_stop.astype(str)==a).mean()),"n_test":n})
    perseed=pd.DataFrame(rows)
    if not len(perseed): return perseed
    agg=(perseed.groupby(["dataset","class_name","decision_after_first"],as_index=False)
         .agg(proportion_mean=("proportion","mean"),proportion_std=("proportion","std"),n_seeds=("seed","nunique"),n_test=("n_test","first")))
    return agg


def _choose_best_subset_test_predictions(bestk: pd.DataFrame, test_cache: Dict[str, Dict[str, Any]], modalities: List[str], k: int):
    row=bestk[bestk.k.astype(int)==int(k)].iloc[0]
    subset=tuple(x.strip() for x in str(row.subset).split("+") if x.strip())
    mask=_mask_from_subset(modalities, subset)
    r=test_cache[_mask_key(mask)]
    return row, r["y_true"], r["y_pred"]


def run_controlled_bundle(bundle: DatasetBundle, prior_results: Optional[Path]) -> Dict[str,pd.DataFrame]:
    P.LOG.info("\n%s\nSAME-BACKBONE FINAL EXPERIMENT: %s\n%s", "="*100, bundle.name, "="*100)
    split=group_split(bundle, int(P.CFG.final_split_seed))
    sel=_load_control_selection(prior_results,bundle.name)
    selected_lambda=float(sel["selected_lambda"])
    target_cost=float(sel["target_validation_cost"])
    P.LOG.info("Frozen validation-selected DQN lambda=%.4f | fixed-order target val cost=%.4f",selected_lambda,target_cost)

    raw_records=[]; exhaustive_all=[]; bestk_all=[]; order_all=[]; curve_all=[]
    branch_all=[]; class_all=[]; confusion_rows=[]

    for si,seed in enumerate(CONTROL_SEEDS,1):
        P.LOG.info("Controlled seed %d/%d = %d | %s",si,len(CONTROL_SEEDS),seed,bundle.name)
        P.CFG.seed=int(seed); seed_everything(P.CFG.seed)
        fusion=train_fusion_classifier(bundle, split, model_tag=f"controlled_{bundle.name}_{seed}")

        # Cache every mask from THIS exact backbone for fixed-subset/order policies.
        val_cache=_build_all_mask_cache(fusion,bundle,split,split.val_idx)
        test_cache=_build_all_mask_cache(fusion,bundle,split,split.test_idx)

        full_sum,full_rows=evaluate_full_detailed(fusion,bundle,split,split.test_idx,"FullFusion")
        raw_records.append(_summary_row_to_record(full_sum,bundle.name,seed,"FullFusion"))
        class_all.append(_per_class_rows(full_rows.y_true,full_rows.y_pred,bundle.label_names,bundle.name,seed,"FullFusion"))

        ex=_exhaustive_subset_rows(bundle,seed,val_cache,test_cache); exhaustive_all.append(ex)
        bk=_select_best_subset_per_k(ex); bestk_all.append(bk)
        for _,br in bk.iterrows():
            raw_records.append(_fixed_subset_summary_from_row(br,bundle.name,seed))

        cand,best=_search_fixed_order_entropy(bundle,val_cache,test_cache,target_cost,seed)
        best_order=tuple(x.strip() for x in str(best.order).split("->"))
        best_test=_simulate_order(test_cache,list(bundle.X.keys()),bundle.modality_costs,best_order,float(best.threshold))
        order_all.append(pd.DataFrame([dict(best)]))
        raw_records.append({
            "dataset":bundle.name,"seed":int(seed),"method":"BestOrder-EntropyStop",
            "accuracy":float(best_test["accuracy"]),"balanced_accuracy":float(best_test["balanced_accuracy"]),
            "precision_macro":float("nan"),"recall_macro":float(best_test["balanced_accuracy"]),
            "macro_f1":float(best_test["macro_f1"]),"normalized_cost":float(best_test["normalized_cost"]),
            "acquisition_ratio":float(best_test["normalized_cost"]),"mean_steps":float(best_test["mean_steps"]),
            "order":str(best.order),"threshold":float(best.threshold),"val_cost":float(best.val_cost),
        })
        class_all.append(_per_class_rows(best_test["y_true"],best_test["y_pred"],bundle.label_names,bundle.name,seed,"BestOrder-EntropyStop"))
        curve_all.append(_validation_selected_order_curve(bundle,cand,test_cache,seed))

        # DQN is trained AFTER the fusion model is frozen, on this SAME fusion instance.
        old_lambda,old_step=P.CFG.acquisition_lambda,P.CFG.step_penalty
        P.CFG.acquisition_lambda=selected_lambda; P.CFG.step_penalty=0.0
        seed_everything(int(seed)+9000)
        env=_attach_groups_to_env(ActiveEvidenceEnv(
            fusion,split.X_scaled,bundle.y,list(bundle.X.keys()),bundle.modality_costs,bundle.label_names,P.CFG.device),bundle)
        policy=train_dqn_policy(env,split.train_idx,CONTROL_DQN_EPISODES,tag=f"controlled_{bundle.name}_{seed}")
        dqn_val_sum,_=evaluate_dqn_detailed(env,policy,split.val_idx,"ActiGrid-DQN")
        dqn_sum,dqn_rows=evaluate_dqn_detailed(env,policy,split.test_idx,"ActiGrid-DQN")
        rec=_summary_row_to_record(dqn_sum,bundle.name,seed,"ActiGrid-DQN",
                                   dqn_val_macro_f1=float(dqn_val_sum.iloc[0].macro_f1),
                                   dqn_val_cost=float(dqn_val_sum.iloc[0].normalized_cost),
                                   selected_lambda=selected_lambda)
        raw_records.append(rec)
        class_all.append(_per_class_rows(dqn_rows.y_true,dqn_rows.y_pred,bundle.label_names,bundle.name,seed,"ActiGrid-DQN"))
        branch_all.append(_branching_rows(dqn_rows,bundle,seed))

        # Operationally important confusion counts for DQN.
        cm=confusion_matrix(dqn_rows.y_true.astype(int),dqn_rows.y_pred.astype(int),labels=np.arange(len(bundle.label_names)))
        for i,tn in enumerate(bundle.label_names):
            for j,pn in enumerate(bundle.label_names):
                confusion_rows.append({"dataset":bundle.name,"seed":int(seed),"true_class":tn,"pred_class":pn,"count":int(cm[i,j])})

        # Save representative same-backbone checkpoints/predictions for audit.
        if int(seed)==42:
            dqn_rows.to_csv(CONTROL_DIR/f"{bundle.name}_seed42_dqn_samples.csv",index=False)
            full_rows.to_csv(CONTROL_DIR/f"{bundle.name}_seed42_full_samples.csv",index=False)
            torch.save({"state":fusion.state_dict(),"seed":seed,"selection":sel},CONTROL_DIR/f"{bundle.name}_seed42_fusion.pt")
            torch.save({"state":policy.state_dict(),"seed":seed,"selection":sel,"state_dim":env.state_dim,"action_dim":env.action_dim},CONTROL_DIR/f"{bundle.name}_seed42_dqn.pt")

        P.CFG.acquisition_lambda,P.CFG.step_penalty=old_lambda,old_step
        del policy,env,fusion,val_cache,test_cache
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    raw=pd.DataFrame(raw_records)
    exhaustive=pd.concat(exhaustive_all,ignore_index=True)
    bestk=pd.concat(bestk_all,ignore_index=True)
    order=pd.concat(order_all,ignore_index=True)
    curve=pd.concat(curve_all,ignore_index=True)
    branch=pd.concat(branch_all,ignore_index=True)
    classdf=pd.concat(class_all,ignore_index=True)
    conf=pd.DataFrame(confusion_rows)

    raw.to_csv(CONTROL_DIR/f"{bundle.name}_controlled_raw.csv",index=False)
    exhaustive.to_csv(CONTROL_DIR/f"{bundle.name}_controlled_exhaustive_subsets.csv",index=False)
    bestk.to_csv(CONTROL_DIR/f"{bundle.name}_controlled_best_subset_per_k.csv",index=False)
    order.to_csv(CONTROL_DIR/f"{bundle.name}_controlled_best_order_entropy.csv",index=False)
    curve.to_csv(CONTROL_DIR/f"{bundle.name}_controlled_order_budget_curve.csv",index=False)
    branch.to_csv(CONTROL_DIR/f"{bundle.name}_controlled_dqn_paths.csv",index=False)
    classdf.to_csv(CONTROL_DIR/f"{bundle.name}_controlled_per_class.csv",index=False)
    conf.to_csv(CONTROL_DIR/f"{bundle.name}_controlled_confusions.csv",index=False)

    summary=_aggregate_multiseed(raw,bundle.name)
    summary.to_csv(CONTROL_DIR/f"{bundle.name}_controlled_summary.csv",index=False)
    paired=_paired_effect_table(raw,bundle.name)
    paired.to_csv(CONTROL_DIR/f"{bundle.name}_controlled_paired_effects.csv",index=False)
    branch_summary=_branching_summary(branch)
    branch_summary.to_csv(CONTROL_DIR/f"{bundle.name}_controlled_branching_summary.csv",index=False)
    class_summary=(classdf.groupby(["dataset","method","class_name"],as_index=False)
                   .agg(precision_mean=("precision","mean"),precision_std=("precision","std"),
                        recall_mean=("recall","mean"),recall_std=("recall","std"),
                        f1_mean=("f1","mean"),f1_std=("f1","std"),support=("support","first")))
    class_summary.to_csv(CONTROL_DIR/f"{bundle.name}_controlled_per_class_summary.csv",index=False)

    # Group/class support manifest used directly in the paper/supplement.
    split_rows=[]
    for split_name,idx in [("train",split.train_idx),("validation",split.val_idx),("test",split.test_idx)]:
        groups=np.asarray(bundle.groups)[idx].astype(str)
        y=np.asarray(bundle.y)[idx].astype(int)
        for cls,name in enumerate(bundle.label_names):
            split_rows.append({"dataset":bundle.name,"split":split_name,"class_id":cls,"class_name":name,
                               "n_samples":int(np.sum(y==cls)),"n_groups_with_class":int(len(np.unique(groups[y==cls]))),
                               "n_groups_total":int(len(np.unique(groups)))})
    split_manifest=pd.DataFrame(split_rows)
    split_manifest.to_csv(CONTROL_DIR/f"{bundle.name}_controlled_split_class_support.csv",index=False)

    P.LOG.info("\nCONTROLLED SUMMARY %s:\n%s",bundle.name,summary.to_string(index=False))
    P.LOG.info("\nCONTROLLED PAIRED EFFECTS %s:\n%s",bundle.name,paired.to_string(index=False))
    return {"raw":raw,"summary":summary,"paired":paired,"bestk":bestk,"order":order,"curve":curve,
            "branch":branch,"branch_summary":branch_summary,"class":classdf,"class_summary":class_summary,
            "confusion":conf,"split_manifest":split_manifest}


def write_controlled_readme(outputs: Dict[str,Dict[str,pd.DataFrame]], prior_results: Optional[Path]):
    lines=[
        "# ActiGrid final same-backbone controlled comparison",
        "",
        "Purpose: remove the classifier-instance ambiguity by evaluating DQN and every strong simple policy on the exact same per-seed fusion backbone.",
        "",
        "Interpretation rules:",
        "- One group-disjoint split per dataset is held fixed across the ten seeds.",
        "- Each seed trains one fusion backbone; full fusion, fixed subsets, fixed-order entropy, and DQN all use that exact instance.",
        "- DQN lambda is frozen from the earlier validation sweep; no test-label tuning is introduced.",
        "- Fixed subsets and fixed-order entropy policies are selected with validation labels only.",
        "- Paired confidence intervals quantify DQN-minus-comparator differences across stochastic seeds on the fixed split.",
        "- Branching summaries focus on the action after the first acquisition, since the initial all-missing state is identical across examples under a deterministic policy.",
        "",
    ]
    for ds,res in outputs.items():
        lines += [f"## {ds}","### Aggregate policy results","```",res["summary"].to_string(index=False),"```",
                  "### Paired DQN-minus-comparator effects","```",res["paired"].to_string(index=False),"```",
                  "### Post-first-acquisition branching","```",res["branch_summary"].to_string(index=False),"```",""]
    (CONTROL_DIR/"README_CONTROLLED_COMPARISON.md").write_text("\n".join(lines))


def controlled_comparison_main(prior_results: Optional[str] = None, fast: bool = False):
    """Run the final same-backbone experiment. Call pipeline.setup() first."""
    global CONTROL_DIR, CONTROL_SEEDS, CONTROL_DQN_EPISODES
    CONTROL_DIR = P.WORK / "controlled_same_backbone"
    CONTROL_DIR.mkdir(parents=True, exist_ok=True)
    CONTROL_SEEDS = (42, 123) if fast else tuple(P.CFG.final_seeds)
    CONTROL_DQN_EPISODES = int(P.CFG.dqn_episodes) if fast else int(P.CFG.final_multiseed_dqn_episodes)
    selection_dir = _discover_selection_dir(prior_results)
    P.LOG.info("Frozen selection source: %s", selection_dir or "built-in FROZEN_SELECTION")
    P.LOG.info("Controlled output directory: %s", CONTROL_DIR)
    outputs = {}
    if P.CFG.run_msu:
        root = discover_dataset_root("msu", P.CFG.msu_root_override)
        if root:
            b = load_msu_bundle(maybe_extract_archives(root, "msu"))
            outputs[b.name] = run_controlled_bundle(b, selection_dir)
        else:
            P.LOG.warning("MSU dataset not found; skipping")
    if P.CFG.run_tudelft:
        root = discover_dataset_root("tudelft", P.CFG.tudelft_root_override)
        if root:
            b = load_tudelft_bundle(maybe_extract_archives(root, "tudelft"))
            outputs[b.name] = run_controlled_bundle(b, selection_dir)
        else:
            P.LOG.warning("TU Delft dataset not found; skipping")
    write_controlled_readme(outputs, selection_dir)
    shutil.make_archive(str(CONTROL_DIR), "zip", root_dir=str(CONTROL_DIR))
    P.LOG.info("Same-backbone final experiment complete: %s", CONTROL_DIR)
    P.LOG.info("Controlled result archive: %s.zip", CONTROL_DIR)
