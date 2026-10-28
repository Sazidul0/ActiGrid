"""Reviewer-hardening suite (paper Tables II and IV, inputs to Figs. 2-4).

Builds on the same pipeline as the controlled experiment and adds:
  * subset-specific RF / HGB / XGBoost classifiers for every nonempty modality
    subset on three group-disjoint partitions (partition seeds 42/123/456;
    validation-selected per size k),
  * three fusion backbones on the primary split (initialization seeds
    42/123/456), each with a DQN lambda frontier (0.25/0.5/0.75) and two reward
    ablations,
  * for initialization seed 42: per-mask ECE with temperature scaling, a run-level
    cluster bootstrap of DQN minus order+entropy, acquisition paths and
    branching, a label-aware oracle, a post-hoc matched-cost static mixture,
    and a first-acquired-modality corruption stress test,
  * a repeated group-split audit (DQN, order+entropy, best fixed subset), whose
    fusion model and DQN use one fixed model seed (4242) on every partition.

Outputs go to <work_dir>/reviewer_hardening/. The released copy is in
results/reviewer_hardening/ and the run log is results/logs/reviewer_hardening.log.
The fusion checkpoints in results/reviewer_hardening/cache/ are the ones the
Greedy-VOI stage (actigrid.greedy_voi) reuses.

Code is section 19 of the original Kaggle script. For the release it was moved
into its own module: pipeline objects are read from actigrid.pipeline at call
time (they are set by pipeline.setup()), the reviewer-clearance and controlled
helpers are imported from their modules, and the run options that the notebook
took from ACTIGRID_* environment variables are arguments of hardening_main().
The paper run used profile="conference" and did not rerun the controlled stage.
Experimental logic is unchanged.
"""
import copy
import random
from itertools import combinations
from types import SimpleNamespace

from sklearn.utils.class_weight import compute_sample_weight

try:
    from scipy import stats
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False

from actigrid import pipeline as P
from actigrid.pipeline import (
    F, HistGradientBoostingClassifier, MLPClassifier, Path, RandomForestClassifier,
    accuracy_score, balanced_accuracy_score, f1_score, gc, json, math, nn, np, pd,
    shutil, torch,
)
from actigrid.reviewer_clearance import (
    _build_all_mask_cache, _exhaustive_subset_rows, _mask_from_subset, _mask_key,
    _search_fixed_order_entropy, _select_best_subset_per_k, _simulate_order,
)
from actigrid.controlled import _branching_rows, _branching_summary

XGBClassifier = getattr(P, "XGBClassifier", None)  # present only if xgboost is installed

CONF = {
    "equivalence_margin_macro_f1": 0.02,
    "conference_split_seeds": [42, 123, 456],
    "full_split_seeds": [42, 123, 456, 789, 2027],
    "conference_dqn_seeds": [42, 123, 456],
    "full_dqn_seeds": [42, 123, 456, 789, 2027],
    "conference_lambda_grid": [0.25, 0.50, 0.75],
    "full_lambda_grid": [0.05, 0.10, 0.20, 0.35, 0.50, 0.75, 1.00],
    "conference_bootstrap_reps": 3000,
    "full_bootstrap_reps": 10000,
    "subset_rf_estimators": 300,
    "subset_xgb_estimators": 400,
    "subset_hgb_iter": 250,
    "run_mlp_in_full": True,
    "corruption_noise_std": 2.0,
}
FROZEN_LAMBDA = 0.50

# Set by hardening_main() after pipeline.setup().
ARGS = SimpleNamespace(profile="conference", force=False, skip_dqn=False)
PROFILE = "conference"
OUT: Path = None
CACHE: Path = None
CONTROL_DIR: Path = None
SPLIT_SEEDS = DQN_SEEDS = LAMBDA_GRID = ()
BOOT_REPS = 0


def done(path: Path) -> bool:
    return path.exists() and not ARGS.force


def save(df: pd.DataFrame, name: str) -> Path:
    p = OUT / name
    df.to_csv(p, index=False)
    print(f"[saved] {p}")
    return p


def set_seed(seed: int):
    P.CFG.seed = int(seed)
    P.seed_everything(int(seed))


def load_bundle(which: str):
    if which == "msu":
        root = P.discover_dataset_root("msu", P.CFG.msu_root_override)
        if root is None:
            raise RuntimeError("MSU/ORNL dataset root not found")
        return P.load_msu_bundle(P.maybe_extract_archives(root, "msu"))
    root = P.discover_dataset_root("tudelft", P.CFG.tudelft_root_override)
    if root is None:
        raise RuntimeError("TU Delft dataset root not found")
    return P.load_tudelft_bundle(P.maybe_extract_archives(root, "tudelft"))


def dataset_keys():
    # The notebook's ACTIGRID_DATASETS=both ran TU Delft first; --datasets maps onto these flags.
    return [k for k, on in (("tu", P.CFG.run_tudelft), ("msu", P.CFG.run_msu)) if on]


# -----------------------------------------------------------------------------
# Metrics / statistics utilities
# -----------------------------------------------------------------------------
def metrics(y, pred):
    y = np.asarray(y, int); pred = np.asarray(pred, int)
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
    }


def holm_adjust(pvals):
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    out = np.empty_like(p)
    running = 0.0
    m = len(p)
    for rank, idx in enumerate(order):
        adj = (m-rank)*p[idx]
        running = max(running, adj)
        out[idx] = min(1.0, running)
    return out


def tost_paired(diffs, margin=0.02):
    """Paired TOST on paired differences. Positive diff means focal-comparator."""
    d = np.asarray(diffs, float)
    d = d[np.isfinite(d)]
    if len(d) < 3 or not HAVE_SCIPY:
        return {"n":len(d), "mean_diff":float(np.mean(d)) if len(d) else np.nan,
                "margin":margin, "p_lower":np.nan, "p_upper":np.nan, "p_tost":np.nan}
    mu, sd, n = float(d.mean()), float(d.std(ddof=1)), len(d)
    se = sd / math.sqrt(n) if sd > 0 else 1e-12
    # H01: mu <= -margin, H11: mu > -margin
    t_low = (mu + margin)/se
    p_low = 1 - stats.t.cdf(t_low, n-1)
    # H02: mu >= +margin, H12: mu < +margin
    t_up = (mu - margin)/se
    p_up = stats.t.cdf(t_up, n-1)
    return {"n":n,"mean_diff":mu,"sd_diff":sd,"margin":margin,
            "p_lower":float(p_low),"p_upper":float(p_up),"p_tost":float(max(p_low,p_up))}


def group_bootstrap_diff(rows_a: pd.DataFrame, rows_b: pd.DataFrame, reps: int, seed=2027):
    """Cluster bootstrap paired policy F1 differences at group/run level."""
    a = rows_a[["sample_idx","y_true","y_pred","group"]].copy()
    b = rows_b[["sample_idx","y_true","y_pred","group"]].copy()
    m = a.merge(b, on=["sample_idx","y_true","group"], suffixes=("_a","_b"))
    groups = np.asarray(sorted(m.group.astype(str).unique()))
    rng = np.random.default_rng(seed)
    diffs=[]
    for _ in range(int(reps)):
        draw = rng.choice(groups, size=len(groups), replace=True)
        chunks=[]
        for j,g in enumerate(draw):
            z=m[m.group.astype(str)==str(g)].copy()
            z["boot_group"] = f"{j}:{g}"
            chunks.append(z)
        q=pd.concat(chunks,ignore_index=True)
        fa=f1_score(q.y_true,q.y_pred_a,average="macro",zero_division=0)
        fb=f1_score(q.y_true,q.y_pred_b,average="macro",zero_division=0)
        diffs.append(float(fa-fb))
    d=np.asarray(diffs)
    return {"bootstrap_reps":len(d),"mean_diff":float(d.mean()),
            "ci95_low":float(np.quantile(d,.025)),"ci95_high":float(np.quantile(d,.975)),
            "prob_diff_gt_0":float(np.mean(d>0))}


# -----------------------------------------------------------------------------
# Strong subset-specific classifier frontier
# -----------------------------------------------------------------------------
def subset_arrays(bundle, split, subset, idx, scaled=False):
    src = split.X_scaled if scaled else bundle.X
    return np.concatenate([src[m][idx] for m in subset], axis=1)


def train_predict_model(kind, Xtr, ytr, Xte, seed, n_classes):
    if kind == "RF":
        model=RandomForestClassifier(n_estimators=CONF["subset_rf_estimators"], class_weight="balanced_subsample",
                                     n_jobs=-1, random_state=seed, max_features="sqrt")
        model.fit(Xtr,ytr); return model.predict(Xte)
    if kind == "HGB":
        sw=compute_sample_weight(class_weight="balanced",y=ytr)
        model=HistGradientBoostingClassifier(max_iter=CONF["subset_hgb_iter"], learning_rate=.08,
                                             max_leaf_nodes=31, l2_regularization=1e-4, random_state=seed)
        model.fit(Xtr,ytr,sample_weight=sw); return model.predict(Xte)
    if kind == "XGB":
        if not P.HAVE_XGB: return None
        model=XGBClassifier(n_estimators=CONF["subset_xgb_estimators"], max_depth=6, learning_rate=.08,
                            subsample=.9, colsample_bytree=.9, objective="multi:softprob",
                            num_class=n_classes, eval_metric="mlogloss", tree_method="hist",
                            n_jobs=-1, random_state=seed)
        model.fit(Xtr,ytr); return model.predict(Xte)
    if kind == "MLP":
        model=MLPClassifier(hidden_layer_sizes=(256,128), max_iter=80, early_stopping=True,
                            validation_fraction=.15, random_state=seed, batch_size=256,
                            learning_rate_init=1e-3)
        model.fit(Xtr,ytr); return model.predict(Xte)
    raise ValueError(kind)


def strong_subset_frontier(bundle):
    """Subset-specific classical frontier with row-level checkpointing.

    Each (split, model, subset) fit is written immediately.  A cancelled Kaggle
    run therefore resumes from the first missing row instead of restarting the
    whole frontier.  Models are fitted once and used for both validation and
    test prediction (the previous implementation fitted twice).
    """
    out_file=OUT/f"{bundle.name}_strong_subset_classifier_frontier.csv"
    partial=OUT/f"{bundle.name}_strong_subset_classifier_frontier.partial.csv"
    if done(out_file):
        print(f"[resume] completed strong frontier: {out_file}")
        return pd.read_csv(out_file)

    modalities=list(bundle.X)
    model_kinds=["RF","HGB","XGB"] + (["MLP"] if PROFILE=="full" and CONF["run_mlp_in_full"] else [])
    if partial.exists() and not ARGS.force:
        df=pd.read_csv(partial)
        print(f"[resume] {bundle.name} strong frontier: {len(df)} rows already complete")
    else:
        df=pd.DataFrame()

    completed=set()
    if len(df):
        completed={(int(r.split_seed),str(r.model),str(r.subset)) for _,r in df.iterrows()}

    def fit_once(kind,Xtr,ytr,seed,n_classes):
        if kind=="RF":
            m=RandomForestClassifier(n_estimators=CONF["subset_rf_estimators"],class_weight="balanced_subsample",
                                     n_jobs=-1,random_state=seed,max_features="sqrt")
            m.fit(Xtr,ytr); return m
        if kind=="HGB":
            sw=compute_sample_weight(class_weight="balanced",y=ytr)
            m=HistGradientBoostingClassifier(max_iter=CONF["subset_hgb_iter"],learning_rate=.08,
                                             max_leaf_nodes=31,l2_regularization=1e-4,random_state=seed)
            m.fit(Xtr,ytr,sample_weight=sw); return m
        if kind=="XGB":
            if not P.HAVE_XGB: return None
            m=XGBClassifier(n_estimators=CONF["subset_xgb_estimators"],max_depth=6,learning_rate=.08,
                            subsample=.9,colsample_bytree=.9,objective="multi:softprob",num_class=n_classes,
                            eval_metric="mlogloss",tree_method="hist",n_jobs=-1,random_state=seed)
            m.fit(Xtr,ytr); return m
        if kind=="MLP":
            m=MLPClassifier(hidden_layer_sizes=(256,128),max_iter=80,early_stopping=True,
                            validation_fraction=.15,random_state=seed,batch_size=256,learning_rate_init=1e-3)
            m.fit(Xtr,ytr); return m
        raise ValueError(kind)

    rows=[] if not len(df) else df.to_dict('records')
    for split_seed in SPLIT_SEEDS:
        set_seed(10000+split_seed)
        split=P.group_split(bundle,seed=split_seed)
        for k in range(1,len(modalities)+1):
            for subset in combinations(modalities,k):
                subset_name="+".join(subset)
                cost=sum(bundle.modality_costs[m] for m in subset)
                for kind in model_kinds:
                    key=(int(split_seed),kind,subset_name)
                    if key in completed:
                        continue
                    scaled=(kind=="MLP")
                    Xtr=subset_arrays(bundle,split,subset,split.train_idx,scaled)
                    Xva=subset_arrays(bundle,split,subset,split.val_idx,scaled)
                    Xte=subset_arrays(bundle,split,subset,split.test_idx,scaled)
                    model=fit_once(kind,Xtr,bundle.y[split.train_idx],10000+split_seed,len(bundle.label_names))
                    if model is None:
                        completed.add(key); continue
                    pred_va=model.predict(Xva); pred_te=model.predict(Xte)
                    mv=metrics(bundle.y[split.val_idx],pred_va); mt=metrics(bundle.y[split.test_idx],pred_te)
                    rec={"dataset":bundle.name,"split_seed":split_seed,"model":kind,"subset":subset_name,"k":k,
                         "normalized_cost":cost,"val_macro_f1":mv["macro_f1"],"test_macro_f1":mt["macro_f1"],
                         "test_balanced_accuracy":mt["balanced_accuracy"]}
                    rows.append(rec); completed.add(key)
                    pd.DataFrame(rows).to_csv(partial,index=False)
                    print("[checkpoint subset]",bundle.name,split_seed,kind,subset_name,mt["macro_f1"])
                    del model,Xtr,Xva,Xte
                    gc.collect()

    df=pd.DataFrame(rows)
    df.to_csv(out_file,index=False)
    if len(df):
        pick=(df.sort_values(["val_macro_f1","subset"],ascending=[False,True])
              .groupby(["dataset","split_seed","model","k"],as_index=False).first())
        save(pick,f"{bundle.name}_strong_subset_validation_selected.csv")
    print(f"[complete] strong frontier: {out_file}")
    return df


# -----------------------------------------------------------------------------
# Temperature scaling / ECE per mask
# -----------------------------------------------------------------------------
def mask_logits(model,bundle,split,idx,mask_np,batch=2048):
    logits=[]; ys=[]
    model.eval()
    with torch.no_grad():
        for s in range(0,len(idx),batch):
            ids=np.asarray(idx[s:s+batch],int)
            xs=[torch.from_numpy(split.X_scaled[m][ids]).float().to(P.CFG.device) for m in bundle.X]
            mask=torch.from_numpy(mask_np).float().to(P.CFG.device)[None,:].repeat(len(ids),1)
            z,_=model(xs,mask)
            logits.append(z.cpu().numpy()); ys.append(bundle.y[ids])
    return np.concatenate(logits), np.concatenate(ys)


def ece_from_probs(probs,y,n_bins=15):
    conf=probs.max(1); pred=probs.argmax(1); y=np.asarray(y,int)
    edges=np.linspace(0,1,n_bins+1); out=0.0
    for lo,hi in zip(edges[:-1],edges[1:]):
        take=(conf>lo)&(conf<=hi) if lo>0 else (conf>=lo)&(conf<=hi)
        if take.any(): out += take.mean()*abs((pred[take]==y[take]).mean()-conf[take].mean())
    return float(out)


def fit_temperature(logits,y):
    t=torch.tensor([1.0],dtype=torch.float32,requires_grad=True)
    x=torch.tensor(logits,dtype=torch.float32); yy=torch.tensor(y,dtype=torch.long)
    opt=torch.optim.LBFGS([t],lr=.05,max_iter=80,line_search_fn="strong_wolfe")
    def closure():
        opt.zero_grad(); loss=F.cross_entropy(x/torch.clamp(t,min=.05,max=20.0),yy); loss.backward(); return loss
    try: opt.step(closure)
    except Exception: pass
    return float(torch.clamp(t.detach(),.05,20.0).item())


def calibration_audit(bundle, split, fusion, seed):
    modalities=list(bundle.X); rows=[]
    for k in range(1,len(modalities)+1):
        for subset in combinations(modalities,k):
            mask=_mask_from_subset(modalities,subset)
            lv,yv=mask_logits(fusion,bundle,split,split.val_idx,mask)
            lt,yt=mask_logits(fusion,bundle,split,split.test_idx,mask)
            temp=fit_temperature(lv,yv)
            pv=torch.softmax(torch.tensor(lv),1).numpy(); pt=torch.softmax(torch.tensor(lt),1).numpy()
            ptc=torch.softmax(torch.tensor(lt)/temp,1).numpy()
            rows.append({"dataset":bundle.name,"seed":seed,"subset":"+".join(subset),"k":k,
                         "temperature":temp,"val_ece_raw":ece_from_probs(pv,yv),
                         "test_ece_raw":ece_from_probs(pt,yt),"test_ece_temp_scaled":ece_from_probs(ptc,yt),
                         "test_macro_f1":f1_score(yt,pt.argmax(1),average="macro",zero_division=0)})
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Oracle lower-bound and randomized matched-cost static mixture
# -----------------------------------------------------------------------------
def oracle_min_cost(bundle,test_cache):
    modalities=list(bundle.X); anyr=next(iter(test_cache.values()))
    yt=anyr["y_true"]; idx=anyr["idx"]; n=len(yt)
    best_cost=np.full(n,np.inf); best_pred=np.zeros(n,int)
    for r in test_cache.values():
        c=float(r["normalized_cost"]); correct=(r["y_pred"]==yt)
        improve=correct & (c<best_cost)
        best_cost[improve]=c; best_pred[improve]=r["y_pred"][improve]
    # If no subset is correct, use full-mask prediction/cost.
    fullmask=np.ones(len(modalities),np.float32); fr=test_cache[_mask_key(fullmask)]
    bad=~np.isfinite(best_cost); best_cost[bad]=fr["normalized_cost"]; best_pred[bad]=fr["y_pred"][bad]
    met=metrics(yt,best_pred)
    return pd.DataFrame([{"dataset":bundle.name,"oracle_label_aware":True,
                          "macro_f1":met["macro_f1"],"mean_min_cost":float(best_cost.mean()),
                          "median_min_cost":float(np.median(best_cost)),"fraction_no_correct_subset":float(bad.mean())}])


def matched_static_mixture(bundle,bestk,test_cache,target_cost,reps=1000,seed=909):
    """Post-hoc descriptive mixture of validation-selected fixed-k subsets at DQN test cost."""
    ks=sorted(bestk.k.astype(int).unique())
    cost_by_k={k:float(bestk[bestk.k==k].iloc[0].normalized_cost) for k in ks}
    lower=max([k for k in ks if cost_by_k[k] <= target_cost], default=min(ks))
    upper=min([k for k in ks if cost_by_k[k] >= target_cost], default=max(ks))
    lo=bestk[bestk.k==lower].iloc[0]; hi=bestk[bestk.k==upper].iloc[0]
    def pred_for(row):
        subset=tuple(str(row.subset).split("+")); mask=_mask_from_subset(list(bundle.X),subset)
        return test_cache[_mask_key(mask)]
    rl,rh=pred_for(lo),pred_for(hi); yt=rl["y_true"]
    if abs(cost_by_k[upper]-cost_by_k[lower])<1e-12:
        q=0.0
    else: q=float(np.clip((target_cost-cost_by_k[lower])/(cost_by_k[upper]-cost_by_k[lower]),0,1))
    rng=np.random.default_rng(seed); vals=[]
    for r in range(reps):
        choose_hi=rng.random(len(yt))<q
        pred=np.where(choose_hi,rh["y_pred"],rl["y_pred"])
        vals.append(f1_score(yt,pred,average="macro",zero_division=0))
    a=np.asarray(vals)
    return pd.DataFrame([{"dataset":bundle.name,"posthoc":True,"target_cost":target_cost,
                          "lower_k":lower,"upper_k":upper,"mix_upper_probability":q,
                          "expected_cost":cost_by_k[lower]*(1-q)+cost_by_k[upper]*q,
                          "macro_f1_mean":a.mean(),"macro_f1_sd":a.std(ddof=1),
                          "macro_f1_ci95_low":np.quantile(a,.025),"macro_f1_ci95_high":np.quantile(a,.975)}])


# -----------------------------------------------------------------------------
# Dueling DDQN and generic trainer
# -----------------------------------------------------------------------------
class DuelingDQN(nn.Module):
    def __init__(self,state_dim,action_dim,hidden):
        super().__init__()
        self.trunk=nn.Sequential(nn.Linear(state_dim,hidden),nn.LayerNorm(hidden),nn.ReLU(),nn.Linear(hidden,hidden),nn.ReLU())
        self.v=nn.Linear(hidden,1); self.a=nn.Linear(hidden,action_dim)
    def forward(self,x):
        h=self.trunk(x); v=self.v(h); a=self.a(h); return v+a-a.mean(1,keepdim=True)


def train_generic_dqn(env,train_idx,episodes,seed,dueling=False,tag="audit"):
    set_seed(seed)
    Q=DuelingDQN if dueling else P.DQN
    online=Q(env.state_dim,env.action_dim,P.CFG.dqn_hidden).to(P.CFG.device)
    target=copy.deepcopy(online).to(P.CFG.device); target.eval()
    opt=torch.optim.AdamW(online.parameters(),lr=P.CFG.dqn_lr,weight_decay=1e-5)
    replay=P.ReplayBuffer(P.CFG.dqn_replay_size)
    rng=np.random.default_rng(seed); train_idx=np.asarray(train_idx,int)
    c2i={int(c):train_idx[env.y[train_idx]==c] for c in np.unique(env.y[train_idx])}
    classes=np.array(sorted(c2i),int); step_count=0
    for ep in range(1,episodes+1):
        cls=int(rng.choice(classes)); i=int(rng.choice(c2i[cls])); state=env.reset(i)
        while True:
            valid=env.valid_actions(); eps=P.epsilon_for_episode(ep,episodes)
            if random.random()<eps: action=int(random.choice(np.where(valid)[0].tolist()))
            else:
                with torch.no_grad(): q=online(torch.from_numpy(state).float()[None,:].to(P.CFG.device))[0].cpu().numpy()
                action=P.masked_argmax(q,valid)
            ns,r,done,info=env.step(action); nv=env.valid_actions().copy() if not done else np.zeros(env.action_dim,bool)
            replay.push(state,action,r,ns,done,nv); state=ns; step_count+=1
            if len(replay)>=max(P.CFG.dqn_warmup,P.CFG.dqn_batch_size):
                bt=replay.sample(P.CFG.dqn_batch_size)
                st=torch.tensor(np.stack(bt.state),dtype=torch.float32,device=P.CFG.device)
                ac=torch.tensor(bt.action,dtype=torch.long,device=P.CFG.device)
                rw=torch.tensor(bt.reward,dtype=torch.float32,device=P.CFG.device)
                nx=torch.tensor(np.stack(bt.next_state),dtype=torch.float32,device=P.CFG.device)
                dn=torch.tensor(bt.done,dtype=torch.float32,device=P.CFG.device)
                nv_t=torch.tensor(np.stack(bt.next_valid),dtype=torch.bool,device=P.CFG.device)
                qsa=online(st).gather(1,ac[:,None]).squeeze(1)
                with torch.no_grad():
                    qo=online(nx).masked_fill(~nv_t,-1e9); na=qo.argmax(1)
                    qt=target(nx).gather(1,na[:,None]).squeeze(1); qt=torch.where(dn>0,torch.zeros_like(qt),qt)
                    tar=rw+P.CFG.dqn_gamma*qt
                loss=F.smooth_l1_loss(qsa,tar); opt.zero_grad(set_to_none=True); loss.backward()
                nn.utils.clip_grad_norm_(online.parameters(),5.0); opt.step()
                if step_count % P.CFG.dqn_target_update == 0: target.load_state_dict(online.state_dict())
            if done: break
        if ep%5000==0: print(f"[dqn] {tag} {ep}/{episodes}")
    return online


# -----------------------------------------------------------------------------
# Simple first-acquisition corruption stress test
# -----------------------------------------------------------------------------
def corrupt_first_acquired(bundle,split,env,policy,idx,mode="zero",noise_std=2.0,seed=77):
    clean_sum,clean=P.evaluate_dqn_detailed(env,policy,idx,method_name="clean")
    Xadv={m:x.copy() for m,x in split.X_scaled.items()}; rng=np.random.default_rng(seed)
    first=[]
    for _,r in clean.iterrows():
        seq=[x.strip() for x in str(r.acquired).split("->") if x.strip()]
        first.append(seq[0] if seq else None)
    for row_i,(sample_idx,m) in enumerate(zip(clean.sample_idx.astype(int),first)):
        if m is None: continue
        if mode=="zero": Xadv[m][sample_idx]=0.0
        elif mode=="noise": Xadv[m][sample_idx]+=rng.normal(0,noise_std,size=Xadv[m][sample_idx].shape).astype(np.float32)
    env_adv=P.ActiveEvidenceEnv(env.model,Xadv,bundle.y,list(bundle.X),bundle.modality_costs,bundle.label_names,P.CFG.device)
    P._attach_groups_to_env(env_adv,bundle)
    adv_sum,adv=P.evaluate_dqn_detailed(env_adv,policy,idx,method_name=f"first_{mode}")
    c=clean_sum.iloc[0]; a=adv_sum.iloc[0]
    return pd.DataFrame([{"dataset":bundle.name,"attack":mode,"clean_macro_f1":c.macro_f1,"adv_macro_f1":a.macro_f1,
                          "delta_macro_f1":a.macro_f1-c.macro_f1,"clean_cost":c.normalized_cost,
                          "adv_cost":a.normalized_cost,"delta_cost":a.normalized_cost-c.normalized_cost}])


# -----------------------------------------------------------------------------
# Core neural audit on original/fixed split
# -----------------------------------------------------------------------------
def neural_audit_fixed_split(bundle):
    """Fixed-split neural hardening with per-run checkpoints.

    Checkpoints are written after every lambda/ablation/Dueling run.  Fusion
    backbones are cached per seed.  On resume, only the missing DQN runs are
    trained.  At worst, cancellation loses the currently executing 20k-episode
    DQN run rather than the entire sweep.
    """
    split=P.group_split(bundle,seed=42)
    modalities=list(bundle.X)
    lambda_file=OUT/f"{bundle.name}_dqn_lambda_frontier.csv"
    ablation_file=OUT/f"{bundle.name}_dqn_reward_architecture_ablations.csv"
    bootstrap_file=OUT/f"{bundle.name}_cluster_bootstrap_seed42.csv"
    robustness_file=OUT/f"{bundle.name}_premature_stop_corruption_stress.csv"
    paths_file=OUT/f"{bundle.name}_dqn_paths_seed42.csv"
    cal_file=OUT/f"{bundle.name}_mask_calibration_seed42.csv"
    oracle_file=OUT/f"{bundle.name}_label_aware_oracle_seed42.csv"
    mix_file=OUT/f"{bundle.name}_posthoc_matched_static_mix_seed42.csv"

    lambda_df=pd.read_csv(lambda_file) if lambda_file.exists() and not ARGS.force else pd.DataFrame()
    ablation_df=pd.read_csv(ablation_file) if ablation_file.exists() and not ARGS.force else pd.DataFrame()
    lambda_rows=lambda_df.to_dict('records') if len(lambda_df) else []
    ablation_rows=ablation_df.to_dict('records') if len(ablation_df) else []

    seeds=DQN_SEEDS if not ARGS.skip_dqn else [42]
    for seed in seeds:
        done_lams=set()
        if len(lambda_df):
            q=lambda_df[lambda_df.seed.astype(int)==int(seed)]
            done_lams={round(float(x),8) for x in q['lambda'].tolist()}
        done_abs=set()
        if len(ablation_df) and 'ablation' in ablation_df:
            q=ablation_df[ablation_df.seed.astype(int)==int(seed)]
            done_abs=set(q.ablation.astype(str))
        deep_needed=(seed==42 and (not cal_file.exists() or not oracle_file.exists() or not mix_file.exists()
                     or not bootstrap_file.exists() or not robustness_file.exists() or not paths_file.exists()))
        requested_lams=[] if ARGS.skip_dqn else [lam for lam in LAMBDA_GRID if round(float(lam),8) not in done_lams]
        requested_abs=[] if ARGS.skip_dqn else [x for x in ['dense_off_critical_on','dense_on_critical_off'] if x not in done_abs]
        duel_needed=(PROFILE=='full' and not ARGS.skip_dqn and 'dueling_ddqn' not in done_abs)
        if not requested_lams and not requested_abs and not duel_needed and not deep_needed:
            print(f"[resume] {bundle.name} seed {seed}: all neural hardening complete")
            continue

        set_seed(seed)
        fusion_ckpt=CACHE/f"{bundle.name}_hardening_fusion_seed{seed}.pt"
        input_dims={m:split.X_scaled[m].shape[1] for m in modalities}
        if fusion_ckpt.exists() and not ARGS.force:
            fusion=P.ActiveFusionClassifier(input_dims,modalities,len(bundle.label_names),hidden=P.CFG.hidden_dim,
                                            latent=P.CFG.latent_dim,dropout=P.CFG.dropout).to(P.CFG.device)
            fusion.load_state_dict(torch.load(fusion_ckpt,map_location=P.CFG.device)['state'])
            print(f"[resume] loaded fusion {fusion_ckpt.name}")
        else:
            fusion=P.train_fusion_classifier(bundle,split,model_tag=f"hardening_{bundle.name}_{seed}")
            torch.save({'state':fusion.state_dict(),'seed':seed},fusion_ckpt)
            print(f"[checkpoint] saved fusion {fusion_ckpt.name}")

        val_cache=_build_all_mask_cache(fusion,bundle,split,split.val_idx)
        test_cache=_build_all_mask_cache(fusion,bundle,split,split.test_idx)
        exhaustive=_exhaustive_subset_rows(bundle,seed,val_cache,test_cache)
        bestk=_select_best_subset_per_k(exhaustive)

        if seed==42 and (ARGS.force or not cal_file.exists()):
            save(calibration_audit(bundle,split,fusion,seed),cal_file.name)
        if seed==42 and (ARGS.force or not oracle_file.exists()):
            save(oracle_min_cost(bundle,test_cache),oracle_file.name)

        if ARGS.skip_dqn:
            del fusion,val_cache,test_cache; gc.collect()
            if torch.cuda.is_available(): torch.cuda.empty_cache()
            continue

        # Lambda frontier.  Force lambda 0.5 if deep audit outputs are missing,
        # even if its summary row was completed in a previous run.
        lams=list(requested_lams)
        if deep_needed and round(FROZEN_LAMBDA,8) not in {round(float(x),8) for x in lams}:
            lams.append(FROZEN_LAMBDA)
        for lam in lams:
            summary_already=(round(float(lam),8) in done_lams)
            old=(P.CFG.acquisition_lambda,P.CFG.dqn_dense_information_reward,P.CFG.critical_confusion_penalty)
            P.CFG.acquisition_lambda=float(lam); P.CFG.dqn_dense_information_reward=True; P.CFG.critical_confusion_penalty=1.0
            env=P.ActiveEvidenceEnv(fusion,split.X_scaled,bundle.y,modalities,bundle.modality_costs,bundle.label_names,P.CFG.device)
            P._attach_groups_to_env(env,bundle)
            pol=train_generic_dqn(env,split.train_idx,P.CFG.final_multiseed_dqn_episodes,seed,False,f"{bundle.name}_s{seed}_lam{lam}")
            sm,rows=P.evaluate_dqn_detailed(env,pol,split.test_idx,method_name="DQN")
            if not summary_already:
                r=sm.iloc[0].to_dict(); r.update({"dataset":bundle.name,"seed":seed,"lambda":lam,"dense_reward":True,"critical_penalty":1.0})
                lambda_rows.append(r); lambda_df=pd.DataFrame(lambda_rows); lambda_df.to_csv(lambda_file,index=False)
                done_lams.add(round(float(lam),8))
                print(f"[checkpoint DQN] {bundle.name} seed={seed} lambda={lam}")

            if abs(float(lam)-FROZEN_LAMBDA)<1e-12 and seed==42 and deep_needed:
                smv,_=P.evaluate_dqn_detailed(env,pol,split.val_idx,method_name="DQN-val")
                target=float(smv.iloc[0].normalized_cost)
                cand,best=_search_fixed_order_entropy(bundle,val_cache,test_cache,target,seed)
                order=tuple(x.strip() for x in str(best.order).split("->"))
                ord_test=_simulate_order(test_cache,modalities,bundle.modality_costs,order,float(best.threshold))
                ord_rows=pd.DataFrame({"sample_idx":split.test_idx,"y_true":ord_test["y_true"],"y_pred":ord_test["y_pred"],
                                       "group":np.asarray(bundle.groups)[split.test_idx].astype(str)})
                if ARGS.force or not bootstrap_file.exists():
                    save(pd.DataFrame([{"dataset":bundle.name,"comparison":"DQN-minus-order_entropy",
                                      **group_bootstrap_diff(rows,ord_rows,BOOT_REPS)}]),bootstrap_file.name)
                if ARGS.force or not mix_file.exists():
                    save(matched_static_mixture(bundle,bestk,test_cache,float(sm.iloc[0].normalized_cost),
                                                reps=3000 if PROFILE=="conference" else 10000),mix_file.name)
                if ARGS.force or not paths_file.exists():
                    paths=_branching_rows(rows,bundle,seed); save(paths,paths_file.name)
                    save(_branching_summary(paths),f"{bundle.name}_dqn_branching_seed42.csv")
                    pcs=(paths.groupby("class_name",as_index=False).agg(mean_cost=("cost","mean"),sd_cost=("cost","std"),
                                                                       mean_steps=("steps","mean"),n=("sample_idx","count")))
                    save(pcs,f"{bundle.name}_per_class_acquisition_cost_seed42.csv")
                if ARGS.force or not robustness_file.exists():
                    rr=pd.concat([corrupt_first_acquired(bundle,split,env,pol,split.test_idx,"zero"),
                                  corrupt_first_acquired(bundle,split,env,pol,split.test_idx,"noise",CONF["corruption_noise_std"])],ignore_index=True)
                    save(rr,robustness_file.name)
                deep_needed=False
            P.CFG.acquisition_lambda,P.CFG.dqn_dense_information_reward,P.CFG.critical_confusion_penalty=old
            del pol,env; gc.collect()
            if torch.cuda.is_available(): torch.cuda.empty_cache()

        for name,dense,crit in [("dense_off_critical_on",False,1.0),("dense_on_critical_off",True,0.0)]:
            if name in done_abs: continue
            old=(P.CFG.acquisition_lambda,P.CFG.dqn_dense_information_reward,P.CFG.critical_confusion_penalty)
            P.CFG.acquisition_lambda=FROZEN_LAMBDA; P.CFG.dqn_dense_information_reward=dense; P.CFG.critical_confusion_penalty=crit
            env=P.ActiveEvidenceEnv(fusion,split.X_scaled,bundle.y,modalities,bundle.modality_costs,bundle.label_names,P.CFG.device); P._attach_groups_to_env(env,bundle)
            pol=train_generic_dqn(env,split.train_idx,P.CFG.final_multiseed_dqn_episodes,seed,False,f"{bundle.name}_{name}_{seed}")
            sm,_=P.evaluate_dqn_detailed(env,pol,split.test_idx,method_name=name)
            r=sm.iloc[0].to_dict(); r.update({"dataset":bundle.name,"seed":seed,"ablation":name,"lambda":FROZEN_LAMBDA})
            ablation_rows.append(r); ablation_df=pd.DataFrame(ablation_rows); ablation_df.to_csv(ablation_file,index=False); done_abs.add(name)
            print(f"[checkpoint ablation] {bundle.name} seed={seed} {name}")
            P.CFG.acquisition_lambda,P.CFG.dqn_dense_information_reward,P.CFG.critical_confusion_penalty=old
            del pol,env; gc.collect()
            if torch.cuda.is_available(): torch.cuda.empty_cache()

        if duel_needed:
            old=(P.CFG.acquisition_lambda,P.CFG.dqn_dense_information_reward,P.CFG.critical_confusion_penalty)
            P.CFG.acquisition_lambda=FROZEN_LAMBDA; P.CFG.dqn_dense_information_reward=True; P.CFG.critical_confusion_penalty=1.0
            env=P.ActiveEvidenceEnv(fusion,split.X_scaled,bundle.y,modalities,bundle.modality_costs,bundle.label_names,P.CFG.device); P._attach_groups_to_env(env,bundle)
            pol=train_generic_dqn(env,split.train_idx,P.CFG.final_multiseed_dqn_episodes,seed,True,f"{bundle.name}_dueling_{seed}")
            sm,_=P.evaluate_dqn_detailed(env,pol,split.test_idx,method_name="Dueling-DDQN")
            r=sm.iloc[0].to_dict(); r.update({"dataset":bundle.name,"seed":seed,"ablation":"dueling_ddqn","lambda":FROZEN_LAMBDA})
            ablation_rows.append(r); ablation_df=pd.DataFrame(ablation_rows); ablation_df.to_csv(ablation_file,index=False)
            print(f"[checkpoint dueling] {bundle.name} seed={seed}")
            P.CFG.acquisition_lambda,P.CFG.dqn_dense_information_reward,P.CFG.critical_confusion_penalty=old
            del pol,env

        del fusion,val_cache,test_cache; gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    return {"lambda":pd.DataFrame(lambda_rows),"ablations":pd.DataFrame(ablation_rows)}


# -----------------------------------------------------------------------------
# Repeated group-split robustness: selected DQN vs simple policies
# -----------------------------------------------------------------------------
def repeated_split_audit(bundle):
    """Repeated group-split policy audit with one checkpoint per split."""
    out_file=OUT/f"{bundle.name}_repeated_group_split_policy_audit.csv"
    partial=OUT/f"{bundle.name}_repeated_group_split_policy_audit.partial.csv"
    if done(out_file) or ARGS.skip_dqn:
        return pd.read_csv(out_file) if out_file.exists() else pd.DataFrame()
    df=pd.read_csv(partial) if partial.exists() and not ARGS.force else pd.DataFrame()
    rows=df.to_dict('records') if len(df) else []
    completed=set(df.split_seed.astype(int).tolist()) if len(df) else set()
    model_seed=4242
    for split_seed in SPLIT_SEEDS:
        if int(split_seed) in completed:
            print(f"[resume] {bundle.name} repeated split {split_seed} complete")
            continue
        set_seed(model_seed)
        split=P.group_split(bundle,seed=split_seed)
        fusion_ckpt=CACHE/f"{bundle.name}_splitrobust_fusion_split{split_seed}.pt"
        modalities=list(bundle.X)
        input_dims={m:split.X_scaled[m].shape[1] for m in modalities}
        if fusion_ckpt.exists() and not ARGS.force:
            fusion=P.ActiveFusionClassifier(input_dims,modalities,len(bundle.label_names),hidden=P.CFG.hidden_dim,
                                            latent=P.CFG.latent_dim,dropout=P.CFG.dropout).to(P.CFG.device)
            fusion.load_state_dict(torch.load(fusion_ckpt,map_location=P.CFG.device)['state'])
        else:
            fusion=P.train_fusion_classifier(bundle,split,model_tag=f"splitrobust_{bundle.name}_{split_seed}")
            torch.save({'state':fusion.state_dict(),'split_seed':split_seed},fusion_ckpt)
        val_cache=_build_all_mask_cache(fusion,bundle,split,split.val_idx); test_cache=_build_all_mask_cache(fusion,bundle,split,split.test_idx)
        ex=_exhaustive_subset_rows(bundle,model_seed,val_cache,test_cache); bk=_select_best_subset_per_k(ex)
        best_all=bk.sort_values(["val_macro_f1","normalized_cost"],ascending=[False,True]).iloc[0]
        split_rows=[{"dataset":bundle.name,"split_seed":split_seed,"method":"BestFixedSubset",
                     "macro_f1":best_all.test_macro_f1,"cost":best_all.normalized_cost,"detail":best_all.subset}]
        old=P.CFG.acquisition_lambda; P.CFG.acquisition_lambda=FROZEN_LAMBDA
        env=P.ActiveEvidenceEnv(fusion,split.X_scaled,bundle.y,list(bundle.X),bundle.modality_costs,bundle.label_names,P.CFG.device); P._attach_groups_to_env(env,bundle)
        pol=train_generic_dqn(env,split.train_idx,P.CFG.final_multiseed_dqn_episodes,model_seed,False,f"split{split_seed}")
        sv,_=P.evaluate_dqn_detailed(env,pol,split.val_idx,"DQN-val"); st,_=P.evaluate_dqn_detailed(env,pol,split.test_idx,"DQN")
        split_rows.append({"dataset":bundle.name,"split_seed":split_seed,"method":"DQN","macro_f1":st.iloc[0].macro_f1,
                           "cost":st.iloc[0].normalized_cost,"detail":"lambda=0.5"})
        cand,best=_search_fixed_order_entropy(bundle,val_cache,test_cache,float(sv.iloc[0].normalized_cost),model_seed)
        split_rows.append({"dataset":bundle.name,"split_seed":split_seed,"method":"OrderEntropy","macro_f1":best.test_macro_f1,
                           "cost":best.test_cost,"detail":best.order})
        P.CFG.acquisition_lambda=old
        rows.extend(split_rows); pd.DataFrame(rows).to_csv(partial,index=False); completed.add(int(split_seed))
        print(f"[checkpoint split] {bundle.name} split={split_seed}")
        del fusion,pol,env,val_cache,test_cache; gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    df=pd.DataFrame(rows); df.to_csv(out_file,index=False)
    agg=(df.groupby(["dataset","method"],as_index=False).agg(macro_f1_mean=("macro_f1","mean"),macro_f1_sd_split=("macro_f1","std"),
                                                               cost_mean=("cost","mean"),cost_sd_split=("cost","std"),n_splits=("split_seed","nunique")))
    save(agg,f"{bundle.name}_repeated_group_split_summary.csv")
    return df




# -----------------------------------------------------------------------------
# Multiplicity / practical-equivalence summaries from controlled outputs
# -----------------------------------------------------------------------------
def controlled_holm_and_tost():
    rows=[]
    paper_rows=[]
    tost_rows=[]
    for name in ["TU_Delft_2026_CPPS", "MSU_ORNL_Power"]:
        p = CONTROL_DIR / f"{name}_controlled_paired_effects.csv"
        r = CONTROL_DIR / f"{name}_controlled_raw.csv"
        if not p.exists():
            continue
        df=pd.read_csv(p).copy()
        df["holm_p_f1_all_within_dataset"] = holm_adjust(df["wilcoxon_p_f1"].astype(float).values)
        df["holm_p_cost_all_within_dataset"] = holm_adjust(df["wilcoxon_p_cost"].astype(float).values)
        rows.append(df)
        # Six paper-facing F1 tests: TU Full/k2/Order; MSU Full/k3/Order.
        comps = ["FullFusion", "BestFixedSubset-k2", "BestOrder-EntropyStop"] if name.startswith("TU_") else ["FullFusion", "BestFixedSubset-k3", "BestOrder-EntropyStop"]
        paper_rows.append(df[df["comparator"].isin(comps)].copy())
        if r.exists():
            raw=pd.read_csv(r)
            a=raw[raw.method=="ActiGrid-DQN"][["seed","macro_f1"]].rename(columns={"macro_f1":"a"})
            b=raw[raw.method=="BestOrder-EntropyStop"][["seed","macro_f1"]].rename(columns={"macro_f1":"b"})
            g=a.merge(b,on="seed")
            if len(g):
                t=tost_paired((g.a-g.b).values, CONF["equivalence_margin_macro_f1"])
                t.update({"dataset":name,"comparison":"DQN-minus-order_entropy"})
                tost_rows.append(t)
    if rows:
        save(pd.concat(rows,ignore_index=True),"controlled_holm_all_comparisons.csv")
    if paper_rows:
        pp=pd.concat(paper_rows,ignore_index=True)
        pp["holm_p_f1_across_six_paper_tests"] = holm_adjust(pp["wilcoxon_p_f1"].astype(float).values)
        save(pp,"controlled_holm_paper_six.csv")
    if tost_rows:
        save(pd.DataFrame(tost_rows),"controlled_tost_margin_0p02.csv")


# -----------------------------------------------------------------------------
# Heterogeneous normalized-cost sensitivity (proxy only; not physical cost)
# -----------------------------------------------------------------------------
def proxy_cost_sensitivity(bundle):
    """Re-price fixed subsets under several normalized cost scenarios.
    This is a proxy sensitivity analysis only; it does not claim measured bytes,
    latency, energy, or operator effort.
    """
    p=CONTROL_DIR/f"{bundle.name}_controlled_exhaustive_subsets.csv"
    if not p.exists():
        return pd.DataFrame()
    ex=pd.read_csv(p)
    mods=list(bundle.X)
    scenarios={"equal":{m:1/len(mods) for m in mods}}
    if bundle.name.startswith("TU_"):
        scenarios.update({
            "network_expensive":{"physical":0.20,"resource":0.20,"network":0.60},
            "resource_expensive":{"physical":0.20,"resource":0.60,"network":0.20},
            "physical_expensive":{"physical":0.60,"resource":0.20,"network":0.20},
        })
    else:
        scenarios.update({
            "cyber_logs_expensive":{**{m:0.15 for m in mods if m!="cyber_logs"},"cyber_logs":0.40},
            "cyber_logs_cheap":{**{m:0.2375 for m in mods if m!="cyber_logs"},"cyber_logs":0.05},
        })
    out=[]
    for _,r in ex.iterrows():
        subset=[x.strip() for x in str(r["subset"]).split("+") if x.strip()]
        for sc,costs in scenarios.items():
            cost=sum(costs.get(m,0) for m in subset)
            out.append({"dataset":bundle.name,"seed":int(r.seed),"scenario":sc,"subset":r.subset,"k":int(r.k),
                        "proxy_cost":float(cost),"test_macro_f1":float(r.test_macro_f1),
                        "note":"normalized proxy scenario; not measured physical cost"})
    df=pd.DataFrame(out)
    save(df,f"{bundle.name}_proxy_cost_sensitivity_fixed_subsets.csv")
    return df


# -----------------------------------------------------------------------------
# Paper-ready compact summaries
# -----------------------------------------------------------------------------
def write_paper_ready_summary():
    lines=[
        "# ActiGrid reviewer-hardening summary",
        "",
        "Use only outputs that completed successfully. Do not reinterpret normalized cost as bytes/latency/energy.",
        "Ten seeds on one split quantify training variability; repeated-split outputs quantify split sensitivity separately.",
        "",
    ]
    for name in ["TU_Delft_2026_CPPS","MSU_ORNL_Power"]:
        cp=CONTROL_DIR/f"{name}_controlled_summary.csv"
        sp=OUT/f"{name}_repeated_group_split_summary.csv"
        tp=OUT/f"{name}_strong_subset_validation_selected.csv"
        if cp.exists():
            lines += [f"## {name}","### Same-backbone policy comparison","```",pd.read_csv(cp).to_string(index=False),"```",""]
        if sp.exists():
            lines += ["### Repeated group-split sensitivity","```",pd.read_csv(sp).to_string(index=False),"```",""]
        if tp.exists():
            best=(pd.read_csv(tp).sort_values("test_macro_f1",ascending=False).head(12))
            lines += ["### Strong subset-specific classifier examples","```",best.to_string(index=False),"```",""]
    (OUT/"PAPER_READY_SUMMARY.md").write_text("\n".join(lines))


def final_stats():
    # DQN selected-point summary.
    rows=[]
    for key in dataset_keys():
        name="TU_Delft_2026_CPPS" if key=="tu" else "MSU_ORNL_Power"
        p=OUT/f"{name}_dqn_lambda_frontier.csv"
        if p.exists():
            df=pd.read_csv(p); d=df[np.isclose(df["lambda"].astype(float),FROZEN_LAMBDA)].copy()
            rows.append({"dataset":name,"analysis":"DQN_lambda_0.5","n":len(d),
                         "macro_f1_mean":d.macro_f1.mean(),"macro_f1_sd":d.macro_f1.std(ddof=1),
                         "cost_mean":d.normalized_cost.mean(),"cost_sd":d.normalized_cost.std(ddof=1)})
    if rows:
        save(pd.DataFrame(rows),"hardening_dqn_selected_summary.csv")
    controlled_holm_and_tost()


def hardening_main(profile: str = "conference", force: bool = False, skip_dqn: bool = False,
                   fast: bool = False):
    """Run the reviewer-hardening suite. Call pipeline.setup() first.

    profile   "conference" (the paper: 3 partitions, 3 initialization seeds, lambda 0.25/0.5/0.75,
              3,000 bootstrap replicates) or "full" (5 splits/seeds, 7 lambdas, MLP
              subsets, Dueling DDQN, 10,000 replicates).
    force     recompute outputs that already exist; otherwise every stage resumes
              from its checkpoint files.
    skip_dqn  train no DQN (subset frontier and seed-42 calibration/oracle only).
    fast      smoke test: one split/seed, lambda 0.5, few episodes. Not for results.
    """
    global ARGS, PROFILE, OUT, CACHE, CONTROL_DIR, SPLIT_SEEDS, DQN_SEEDS, LAMBDA_GRID, BOOT_REPS
    if profile not in {"conference", "full"}:
        raise ValueError(f"profile must be 'conference' or 'full', got: {profile}")
    PROFILE = profile
    ARGS = SimpleNamespace(profile=profile, force=force, skip_dqn=skip_dqn)
    OUT = P.WORK / "reviewer_hardening"
    OUT.mkdir(parents=True, exist_ok=True)
    CACHE = OUT / "cache"
    CACHE.mkdir(exist_ok=True)
    CONTROL_DIR = P.WORK / "controlled_same_backbone"
    (OUT / "hardening_protocol.json").write_text(json.dumps(CONF, indent=2))

    SPLIT_SEEDS = CONF["conference_split_seeds"] if PROFILE == "conference" else CONF["full_split_seeds"]
    DQN_SEEDS = CONF["conference_dqn_seeds"] if PROFILE == "conference" else CONF["full_dqn_seeds"]
    LAMBDA_GRID = CONF["conference_lambda_grid"] if PROFILE == "conference" else CONF["full_lambda_grid"]
    BOOT_REPS = CONF["conference_bootstrap_reps"] if PROFILE == "conference" else CONF["full_bootstrap_reps"]

    # Keep exact paper training budget.
    P.CFG.final_multiseed_dqn_episodes = 20000
    P.CFG.dqn_episodes = 20000
    P.CFG.step_penalty = 0.0
    if fast:
        SPLIT_SEEDS, DQN_SEEDS, LAMBDA_GRID, BOOT_REPS = [42], [42], [FROZEN_LAMBDA], 50
        P.CFG.final_multiseed_dqn_episodes = P.CFG.dqn_episodes = 300

    manifest={
        "profile":PROFILE,"datasets":dataset_keys(),"torch":torch.__version__,"device":P.CFG.device,
        "xgboost":P.HAVE_XGB,"scipy":HAVE_SCIPY,"msu_root":P.CFG.msu_root_override,
        "tu_root":P.CFG.tudelft_root_override,"work_dir":P.CFG.work_dir,
    }
    (OUT/"run_manifest.json").write_text(json.dumps(manifest,indent=2))
    print("="*100)
    print("ActiGrid reviewer-hardening suite")
    print("Profile:",PROFILE,"| datasets:",dataset_keys(),"| device:",P.CFG.device)
    print("MSU:",P.CFG.msu_root_override); print("TU :",P.CFG.tudelft_root_override); print("OUT:",P.CFG.work_dir)
    print("="*100)
    for key in dataset_keys():
        bundle=load_bundle(key); bundle.validate()
        strong_subset_frontier(bundle)
        neural_audit_fixed_split(bundle)
        repeated_split_audit(bundle)
        proxy_cost_sensitivity(bundle)
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    final_stats()
    write_paper_ready_summary()
    readme=f"""# ActiGrid final reviewer-hardening outputs

Profile: **{PROFILE}**

## Covered experimentally
- same-backbone DQN/full/fixed-subset/fixed-order comparison
- subset-specific RF/HGB/XGBoost frontiers (MLP in full profile)
- repeated group-disjoint split sensitivity
- DQN lambda frontier and reward ablations
- optional Dueling Double-DQN in full profile
- group/run-level bootstrap uncertainty
- per-mask ECE and validation temperature scaling
- acquisition paths, branching, and per-class acquisition cost
- label-aware minimum-cost oracle (analysis-only upper bound)
- post-hoc matched-cost randomized fixed-subset mixture
- simple first-acquired-modality corruption stress test
- Holm multiplicity summaries and 0.02-margin TOST
- heterogeneous *normalized proxy* cost sensitivity for fixed subsets

## Not cleared by offline code
- measured bandwidth, latency, energy, storage-I/O, or operator-effort cost
- true temporal/online deployment behavior
- generalization to additional unseen datasets
- faithful reproduction of every published AFA algorithm

Do not claim those items were established.
"""
    (OUT/"README.md").write_text(readme)
    shutil.make_archive(str(OUT),"zip",root_dir=str(OUT))
    print("\nREVIEW HARDENING COMPLETE")
    print("Output:",OUT)
    print("ZIP:",str(OUT)+".zip")
