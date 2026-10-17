# ============================================================
# ActiGrid: Resource-Aware Sequential Evidence Acquisition
# for Cyber-Physical Power-System Diagnosis
#
# End-to-end research pipeline (v7 final protocol). This module is the code that
# produced the results in results/ (originally run as a Kaggle notebook).
# Changes for the public release are limited to: configurable
# paths/CLI entry point (setup()), a missing `Counter` import, and moving
# import-time side effects into setup(). Model, reward, split and evaluation
# logic are unchanged.
# ============================================================
# Designed for:
#   A) UAH / MSU-ORNL Power System Attack Dataset
#   B) TU Delft 2026 Cyber-Physical Power System Dataset
# Optional:
#   C) UAH EMS logs as a normal-operation stress test
#
# What this script does:
#   1. Reproducibility + debug logging
#   2. Automatic dataset discovery and data audit
#   3. MSU-ORNL loader + scenario labels + PMU/log modality groups
#   4. TU Delft raw CSV/PCAPNG loader with 1-second aggregation
#   5. Group-aware train/val/test splitting
#   6. Per-modality scaling
#   7. Missing-modality-aware multimodal classifier
#   8. Modality-dropout training
#   9. Classical/full-telemetry baselines
#  10. Random / fixed / utility active-acquisition baselines
#  11. Double-DQN active acquisition policy
#  12. Acquisition cost, F1, FAR, steps, and Pareto evaluation
#  13. Prototype/Mahalanobis open-set evaluation (single LOAO run)
#  14. Ablation hooks and artifact saving
#
# IMPORTANT RESEARCH NOTES
# ------------------------
# * Do not report the target/illustrative values from the paper plan as results.
# * The script prints real measured results only.
# * Verify auto-detected TU Delft file types in the manifest before publication.
# * For the UAH dataset, this script uses source files as groups to reduce
#   leakage from row-wise random splitting.
# * For TU Delft, each physical recording is treated as a run and windows from
#   one run stay in one split.
# ============================================================

from __future__ import annotations

import os
import re
import gc
import sys
import json
import math
import time
import copy
import random
import shutil
import logging
import warnings
import platform
import subprocess
from dataclasses import dataclass, asdict, field
from pathlib import Path
from collections import Counter, defaultdict, deque, namedtuple
from typing import Dict, List, Tuple, Optional, Any, Iterable

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_recall_fscore_support,
    f1_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# xgboost is normally available on Kaggle, but keep it optional.
try:
    from xgboost import XGBClassifier
    HAVE_XGB = True
except Exception:
    HAVE_XGB = False

# Scapy is optional. Classic .pcap files have a native fallback parser; Scapy is only required for .pcapng.
try:
    from scapy.utils import PcapNgReader
    from scapy.layers.l2 import Ether
    HAVE_SCAPY = True
except Exception:
    HAVE_SCAPY = False


# ============================================================
# 0. CONFIGURATION
# ============================================================
@dataclass
class Config:
    # Paths (override from the CLI or a JSON config; see README).
    data_root: str = "data"
    work_dir: str = "outputs"

    # If auto-discovery picks the wrong folder, set these to exact dataset paths.
    msu_root_override: Optional[str] = None
    tudelft_root_override: Optional[str] = None
    ems_root_override: Optional[str] = None

    run_msu: bool = True
    run_tudelft: bool = True
    run_ems_stress: bool = False
    run_ablations: bool = False  # v5.2 ablations are already valid; v6 focuses on final Pareto/statistics
    ablation_dqn_episodes: int = 10000
    auto_extract_archives: bool = True

    # Global reproducibility
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Debug mode: use a smaller dataset / fewer epochs to validate the pipeline.
    fast_debug: bool = False
    debug_max_rows: int = 25000

    # Split
    train_frac: float = 0.60
    val_frac: float = 0.20
    test_frac: float = 0.20
    # Publication protocol
    strict_class_coverage: bool = True
    split_search_trials: int = 500
    run_leakage_audit: bool = True
    primary_cost_mode: str = "unit"  # each acquired evidence source costs equally
    strict_tu_pairing: bool = True  # never silently mix duplicate physical/resource runs

    # TU Delft window aggregation
    tudelft_window_seconds: float = 1.0
    tudelft_max_packets_per_file: Optional[int] = None  # frame-level GT requires the full capture; native PCAP parser is used for speed
    tu_use_frame_ground_truth: bool = True
    tu_drop_resource_identifiers: bool = True
    tu_absolute_time_alignment: bool = True
    tu_attack_label_context_windows: int = 0  # keep 0 for primary results; use >0 only as a sensitivity analysis

    # Classifier
    latent_dim: int = 64
    hidden_dim: int = 128
    dropout: float = 0.20
    clf_batch_size: int = 512
    clf_epochs: int = 35
    clf_patience: int = 7
    clf_lr: float = 1e-3
    weight_decay: float = 1e-4
    modality_dropout_p: float = 0.25
    probability_all_missing: float = 0.02

    # DQN
    dqn_hidden: int = 256
    dqn_lr: float = 2e-4
    dqn_gamma: float = 0.97
    dqn_batch_size: int = 256
    dqn_replay_size: int = 80000
    dqn_warmup: int = 1500
    dqn_episodes: int = 30000
    dqn_target_update: int = 500
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_fraction: float = 0.70

    # Reward
    correct_reward: float = 2.0
    wrong_penalty: float = 2.0
    critical_confusion_penalty: float = 1.0
    acquisition_lambda: float = 0.35
    step_penalty: float = 0.0  # v6 primary reward: classification - lambda * acquisition cost

    # Active baselines
    entropy_stop_threshold: float = 0.35
    min_acquisitions_before_stop: int = 1

    # Classical baselines
    baseline_max_train_rows: int = 80000
    rf_estimators: int = 300
    xgb_estimators: int = 400

    # Open-set / LOAO
    run_open_set: bool = False  # excluded from the core PES-GM experiment
    # MSU attack family names are: data_injection, remote_trip, relay_setting
    msu_holdout_attack: str = "data_injection"
    # TU Delft choices normally: reconnaissance, resource_exhaustion,
    # goose_injection, sv_injection
    tudelft_holdout_attack: str = "goose_injection"
    openset_quantile: float = 0.99
    openset_use_mahalanobis: bool = True
    openset_classifier_epochs: int = 25
    openset_dqn_episodes: int = 12000

    # V6 FINAL EXPERIMENT PROTOCOL
    # Stage A: one-seed validation Pareto sweep to select lambda/heuristic threshold.
    # Stage B: five-seed confirmation using the selected operating point.
    final_run_protocol: bool = True
    final_split_seed: int = 42
    final_seeds: Tuple[int, ...] = (42, 123, 456, 789, 2027, 31415, 27182, 16180, 14142, 17320)
    final_lambda_grid: Tuple[float, ...] = (0.05, 0.10, 0.20, 0.35, 0.50, 0.75)
    final_entropy_grid: Tuple[float, ...] = tuple(round(x * 0.05, 2) for x in range(1, 37))
    final_sweep_dqn_episodes: int = 8000
    final_multiseed_dqn_episodes: int = 20000
    final_f1_tolerance: float = 0.015
    final_lambda_fallback_cost_weight: float = 0.08
    final_run_classical_once: bool = True
    final_save_seed42_traces: bool = True
    dqn_balanced_episode_sampling: bool = True
    dqn_dense_information_reward: bool = True
    dqn_dense_reward_scale: float = 1.0
    dqn_dense_reward_clip: float = 2.0

    # Set True only for a quick smoke test of v6.
    final_quick_mode: bool = False

    # Save plots/models
    save_models: bool = True
    save_plots: bool = True


CFG = Config()
WORK = Path(CFG.work_dir)
LOG = logging.getLogger("actigrid")


def _apply_fast_debug(cfg: "Config") -> None:
    cfg.clf_epochs = 5
    cfg.clf_patience = 3
    cfg.dqn_episodes = 1500
    cfg.dqn_warmup = 300
    cfg.dqn_target_update = 200
    cfg.baseline_max_train_rows = 15000
    cfg.rf_estimators = 80
    cfg.xgb_estimators = 100
    cfg.openset_classifier_epochs = 5
    cfg.openset_dqn_episodes = 800
    cfg.ablation_dqn_episodes = 600


# ============================================================
# 1. DEBUG / LOGGING / REPRODUCIBILITY
# ============================================================
def build_logger() -> logging.Logger:
    logger = logging.getLogger("actigrid")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.INFO)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    fh = logging.FileHandler(WORK / "debug.log", mode="w")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def print_environment():
    LOG.info("=" * 80)
    LOG.info("ActiGrid v7 balanced dense-reward + fair-budget final pipeline starting")
    LOG.info("Python: %s", sys.version.replace("\n", " "))
    LOG.info("Platform: %s", platform.platform())
    LOG.info("NumPy: %s | pandas: %s | torch: %s", np.__version__, pd.__version__, torch.__version__)
    LOG.info("Device: %s", CFG.device)
    if torch.cuda.is_available():
        LOG.info("GPU: %s", torch.cuda.get_device_name(0))
        LOG.info("CUDA memory: %.2f GB", torch.cuda.get_device_properties(0).total_memory / 1e9)
    LOG.info("XGBoost available: %s", HAVE_XGB)
    LOG.info("Scapy available: %s", HAVE_SCAPY)
    LOG.info("Work dir: %s", WORK)
    LOG.info("=" * 80)


def setup(overrides: Optional[Dict[str, Any]] = None) -> "Config":
    """Initialise the global config, output directory, logger and seeds.

    Must be called before main(). `overrides` maps Config field names to values.
    """
    global CFG, WORK, LOG
    CFG = Config()
    for key, value in (overrides or {}).items():
        if not hasattr(CFG, key):
            raise KeyError(f"Unknown config key: {key}")
        if isinstance(getattr(CFG, key), tuple) and isinstance(value, list):
            value = tuple(value)
        setattr(CFG, key, value)
    if CFG.device.startswith("cuda") and not torch.cuda.is_available():
        CFG.device = "cpu"
    if CFG.fast_debug:
        _apply_fast_debug(CFG)
    WORK = Path(CFG.work_dir)
    WORK.mkdir(parents=True, exist_ok=True)
    LOG = build_logger()
    seed_everything(CFG.seed)
    print_environment()
    with open(WORK / "config.json", "w") as f:
        json.dump(asdict(CFG), f, indent=2)
    if Path(CFG.data_root).exists():
        print_tree(CFG.data_root)
    return CFG


def print_tree(root: str | Path, max_files: int = 180, max_depth: int = 5):
    root = Path(root)
    if not root.exists():
        LOG.warning("Dataset root does not exist: %s", root)
        return
    LOG.info("Dataset tree preview: %s", root)
    n = 0
    for p in sorted(root.rglob("*")):
        try:
            depth = len(p.relative_to(root).parts)
        except Exception:
            depth = 999
        if depth > max_depth:
            continue
        if p.is_file():
            size_mb = p.stat().st_size / (1024 ** 2)
            LOG.info("  %s (%.2f MB)", p.relative_to(root), size_mb)
            n += 1
            if n >= max_files:
                LOG.info("  ... tree truncated at %d files", max_files)
                break




# ============================================================
# 2. COMMON DATA CONTAINERS AND AUDIT UTILITIES
# ============================================================
@dataclass
class DatasetBundle:
    name: str
    X: Dict[str, np.ndarray]
    y: np.ndarray
    label_names: List[str]
    groups: np.ndarray
    feature_names: Dict[str, List[str]]
    modality_costs: Dict[str, float]
    meta: Optional[pd.DataFrame] = None

    def validate(self):
        n = len(self.y)
        assert n > 0, f"{self.name}: no samples"
        assert len(self.groups) == n
        for m, arr in self.X.items():
            assert arr.ndim == 2, (m, arr.shape)
            assert arr.shape[0] == n, (m, arr.shape, n)
            assert np.isfinite(arr).all(), f"{self.name}/{m}: NaN/Inf remains"
        assert set(self.modality_costs) == set(self.X)
        assert len(self.label_names) > int(np.max(self.y))
        return True


def describe_bundle(bundle: DatasetBundle):
    bundle.validate()
    LOG.info("\n%s", "=" * 80)
    LOG.info("DATASET BUNDLE: %s", bundle.name)
    LOG.info("Samples: %d", len(bundle.y))
    LOG.info("Groups/runs: %d", pd.Series(bundle.groups).nunique())
    LOG.info("Modalities:")
    for m, x in bundle.X.items():
        LOG.info("  %-24s shape=%s cost=%.4f features=%d", m, x.shape, bundle.modality_costs[m], x.shape[1])
    counts = pd.Series(bundle.y).value_counts().sort_index()
    for i, name in enumerate(bundle.label_names):
        LOG.info("  class %d %-28s n=%d", i, name, int(counts.get(i, 0)))
    LOG.info("%s\n", "=" * 80)


def clean_numeric_frame(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    out = df[cols].copy()
    for c in cols:
        if not pd.api.types.is_numeric_dtype(out[c]):
            out[c] = pd.to_numeric(out[c], errors="coerce")
    out = out.replace([np.inf, -np.inf], np.nan)
    # median then zero fallback
    med = out.median(numeric_only=True)
    out = out.fillna(med).fillna(0.0)
    return out.astype(np.float32)


def normalize_costs(raw_costs: Dict[str, float]) -> Dict[str, float]:
    vals = np.array([max(float(v), 1e-9) for v in raw_costs.values()], dtype=np.float64)
    # Costs sum to 1.0: full telemetry has normalized cost 1.0.
    vals = vals / vals.sum()
    return {k: float(v) for k, v in zip(raw_costs.keys(), vals)}


# ============================================================
# 3A. OPTIONAL ARCHIVE EXTRACTION (Kaggle-friendly)
# ============================================================
def maybe_extract_archives(root: Path, tag: str) -> Path:
    """Extract a dataset archive into the work directory when needed.

    TU Delft is distributed on Zenodo as a .rar archive. Kaggle datasets may
    already be extracted; if not, this helper tries 7z/unrar. ZIP/TAR files are
    handled with shutil. The source dataset folder is never modified.
    """
    if not CFG.auto_extract_archives or not root.exists():
        return root
    # If useful files already exist, leave the root untouched.
    useful = list(root.rglob("*.csv")) + list(root.rglob("*.pcapng")) + list(root.rglob("*.pcap"))
    if useful:
        return root
    archives = []
    for ext in ["*.rar", "*.zip", "*.tar", "*.tar.gz", "*.tgz", "*.7z"]:
        archives.extend(root.rglob(ext))
    if not archives:
        return root
    archive = sorted(archives, key=lambda x: x.stat().st_size, reverse=True)[0]
    out = WORK / f"extracted_{tag}"
    out.mkdir(parents=True, exist_ok=True)
    LOG.info("Dataset appears archived. Attempting extraction: %s -> %s", archive, out)
    try:
        if archive.suffix.lower() == ".zip" or archive.name.lower().endswith((".tar", ".tar.gz", ".tgz")):
            shutil.unpack_archive(str(archive), str(out))
        else:
            commands = [
                ["7z", "x", str(archive), f"-o{out}", "-y"],
                ["unrar", "x", "-o+", str(archive), str(out) + os.sep],
            ]
            ok = False
            for cmd in commands:
                if shutil.which(cmd[0]) is None:
                    continue
                LOG.info("Trying extractor: %s", " ".join(cmd))
                cp = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                LOG.debug("Extractor output:\n%s", cp.stdout[-5000:])
                if cp.returncode == 0:
                    ok = True
                    break
            if not ok:
                raise RuntimeError("No working 7z/unrar extractor found in this Kaggle image")
        LOG.info("Archive extraction completed")
        return out
    except Exception as e:
        LOG.exception("Archive extraction failed: %s", e)
        LOG.warning("Recommended fix: extract the archive locally, create a Kaggle Dataset from the extracted folder, then set the *_root_override path.")
        return root


# ============================================================
# 3. DATASET AUTO-DISCOVERY
# ============================================================
def discover_dataset_root(kind: str, override: Optional[str] = None) -> Optional[Path]:
    """Discover dataset roots robustly under Kaggle's sometimes deeply nested mounts.

    Datasets may be mounted as <data_root>/<slug>/... or, in some runtimes,
    <data_root>/datasets/<owner>/<slug>/....  The original v1 scanner only
    inspected the first directory level and therefore missed valid MSU .arff files.
    """
    if override:
        p = Path(override)
        if p.exists():
            LOG.info("Using explicit %s root: %s", kind, p)
            return p
        LOG.warning("Explicit %s root does not exist: %s", kind, p)

    root = Path(CFG.data_root)
    if not root.exists():
        return None

    kind = kind.lower()

    if kind == "msu":
        # Strongest signature for the UAH release seen on Kaggle: 15
        # `dataN Sampled Scenarios.csv.arff` files. Also support converted CSVs.
        arffs = [p for p in root.rglob("*.arff") if "sampled scenario" in p.name.lower()]
        if not arffs:
            arffs = list(root.rglob("*.arff"))
        if arffs:
            counts = Counter(p.parent for p in arffs)
            parent, n = counts.most_common(1)[0]
            LOG.info("Auto-discovery msu: found %d ARFF files; selected %s (%d files)", len(arffs), parent, n)
            return parent

        # Fall back to likely CSV directories.
        csvs = list(root.rglob("*.csv"))
        scored = []
        for p in csvs:
            text = (str(p.parent) + " " + p.name).lower()
            score = sum(k in text for k in ["power", "msu", "ornl", "sampled", "scenario"])
            if score:
                scored.append((score, p.parent))
        if scored:
            agg = defaultdict(int)
            for score, parent in scored:
                agg[parent] += score
            parent = max(agg, key=agg.get)
            LOG.info("Auto-discovery msu CSV root: %s", parent)
            return parent

    # TU Delft and EMS: score directories recursively, not only the first level.
    candidates = defaultdict(int)
    max_probe = 20000
    for j, f in enumerate(root.rglob("*")):
        if j >= max_probe:
            break
        if not f.is_file():
            continue
        text = (str(f.parent) + " " + f.name).lower()
        if kind == "tudelft":
            keys = ["delft", "digital", "substation", "cyberphysical", "cyber-physical", "iec61850", "rtds", "goose", "sv_injection"]
            score = sum(k in text for k in keys)
            if f.suffix.lower() in {".pcap", ".pcapng"}:
                score += 2
        elif kind == "ems":
            keys = ["ems", "energy-management", "energy_management", "scada"]
            score = sum(k in text for k in keys)
        else:
            score = 0
        if score:
            # Accumulate evidence for the file's parent and a few ancestors.
            cur = f.parent
            for _ in range(4):
                try:
                    cur.relative_to(root)
                except Exception:
                    break
                candidates[cur] += score
                if cur == root:
                    break
                cur = cur.parent

    if not candidates:
        LOG.warning("Could not auto-discover %s dataset root under %s", kind, root)
        return None
    ranked = sorted(((v, k) for k, v in candidates.items()), reverse=True, key=lambda x: x[0])
    LOG.info("Auto-discovery %s candidates: %s", kind, [(s, str(p)) for s, p in ranked[:5]])
    return ranked[0][1]


# ============================================================
# 4. MSU-ORNL POWER SYSTEM DATASET
# ============================================================
MSU_ATTACK_SCENARIOS = {
    "normal": {41},
    "physical_fault_or_maintenance": set(range(1, 7)) | {13, 14},
    "data_injection": set(range(7, 13)),
    "remote_trip": set(range(15, 21)),
    "relay_setting": set(range(21, 31)) | set(range(35, 41)),
}


def scenario_to_label_name(scenario: int) -> Optional[str]:
    for name, scenarios in MSU_ATTACK_SCENARIOS.items():
        if scenario in scenarios:
            return name
    return None


def parse_msu_scenario(v: Any) -> Optional[int]:
    """Parse scenario from the original marker.

    UAH README: marker = [scenario][3 digits fault location][3 digits load].
    Therefore scenario is the leading 1-2 digits before the last 6 digits.
    Also supports already-decoded scenario integers.
    """
    if pd.isna(v):
        return None
    s = str(v).strip()
    try:
        # handle floats serialized as 41000000.0
        if re.fullmatch(r"[-+]?\d+(\.0+)?", s):
            s = str(int(float(s)))
    except Exception:
        pass

    digits = re.sub(r"\D", "", s)
    if not digits:
        return None
    try:
        direct = int(digits)
        if direct in set(range(1, 42)) - {31, 32, 33, 34}:
            return direct
    except Exception:
        pass

    if len(digits) > 6:
        prefix = digits[:-6]
        try:
            scenario = int(prefix)
            if scenario in set(range(1, 42)) - {31, 32, 33, 34}:
                return scenario
        except Exception:
            pass

    # fallback: first 1 or 2 digits that make a valid scenario
    for k in [2, 1]:
        if len(digits) >= k:
            try:
                scenario = int(digits[:k])
                if scenario in set(range(1, 42)) - {31, 32, 33, 34}:
                    return scenario
            except Exception:
                pass
    return None


def _decode_object_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Decode scipy ARFF byte strings without touching numeric columns."""
    out = df.copy()
    for c in out.columns:
        if out[c].dtype == object:
            out[c] = out[c].map(lambda v: v.decode("utf-8", errors="replace") if isinstance(v, (bytes, bytearray)) else v)
    return out


def read_msu_table(path: Path, nrows: Optional[int] = None) -> pd.DataFrame:
    """Read an original UAH CSV or Kaggle-hosted ARFF copy."""
    suf = path.suffix.lower()
    if suf == ".arff":
        try:
            from scipy.io import arff
        except Exception as e:
            raise RuntimeError("scipy.io.arff is required to read the MSU .arff files") from e
        data, _meta = arff.loadarff(str(path))
        df = pd.DataFrame(data)
        df = _decode_object_columns(df)
        if nrows is not None:
            df = df.head(nrows)
        return df
    # Handles true .csv plus files ending in .csv generated by users.
    return pd.read_csv(path, nrows=nrows)


def inspect_msu_table(path: Path, nrows: int = 5) -> Tuple[int, List[str]]:
    try:
        x = read_msu_table(path, nrows=nrows)
        return x.shape[1], [str(c).strip() for c in x.columns]
    except Exception as e:
        LOG.debug("Could not inspect MSU candidate %s: %s", path, e)
        return 0, []


def find_msu_files(root: Path) -> List[Path]:
    files = sorted(list(root.rglob("*.csv")) + list(root.rglob("*.arff")))
    candidates = []
    for p in files:
        ncols, cols = inspect_msu_table(p, 3)
        low = [c.lower() for c in cols]
        # Original data has ~128 predictors + Marker. Some ARFF conversions
        # sanitize names but preserve the dimensionality.
        if ncols >= 120 and ("marker" in low or ncols >= 128):
            candidates.append(p)

    if not candidates:
        raise FileNotFoundError(
            f"No MSU-ORNL-like CSV/ARFF files found under {root}. "
            "Expected >=120 columns and preferably a 'Marker' column."
        )

    # Keep one coherent schema so binary/3-class/multiclass derivative copies
    # are never concatenated as independent data.
    signature_groups = defaultdict(list)
    for p in candidates:
        _, cols = inspect_msu_table(p, 2)
        sig = tuple(cols)
        signature_groups[sig].append(p)
    _best_sig, best_files = max(signature_groups.items(), key=lambda kv: len(kv[1]))
    LOG.info("MSU table signatures found: %d; using largest coherent set with %d files", len(signature_groups), len(best_files))
    for p in best_files:
        LOG.info("  MSU file: %s", p)
    return sorted(best_files)


def infer_msu_modalities(feature_cols: List[str]) -> Dict[str, List[str]]:
    groups: Dict[str, List[str]] = {f"pmu_r{i}": [] for i in range(1, 5)}
    remainder = []
    for c in feature_cols:
        cs = str(c).strip()
        matched = False
        for i in range(1, 5):
            if re.match(rf"^R{i}[-_: ]", cs, flags=re.IGNORECASE) or cs.upper().startswith(f"R{i}-"):
                groups[f"pmu_r{i}"].append(c)
                matched = True
                break
        if not matched:
            remainder.append(c)

    # If prefixes were damaged by a Kaggle-converted copy, exploit the original
    # documented layout: 4 PMUs x 29 columns = 116, then 12 cyber/log columns.
    if sum(len(groups[k]) for k in groups) < 100 and len(feature_cols) >= 116:
        LOG.warning("MSU PMU prefixes were not reliably detected. Falling back to documented column order (4 x 29 PMU).")
        groups = {
            "pmu_r1": feature_cols[0:29],
            "pmu_r2": feature_cols[29:58],
            "pmu_r3": feature_cols[58:87],
            "pmu_r4": feature_cols[87:116],
        }
        remainder = feature_cols[116:]

    if remainder:
        groups["cyber_logs"] = remainder
    groups = {k: v for k, v in groups.items() if len(v) > 0}
    return groups


def load_msu_bundle(root: Path) -> DatasetBundle:
    files = find_msu_files(root)
    frames = []
    for p in files:
        LOG.info("Reading MSU: %s", p.name)
        df = read_msu_table(p)
        df.columns = [str(c).strip() for c in df.columns]
        df["__source_file"] = p.name
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)

    if CFG.fast_debug and len(df) > CFG.debug_max_rows:
        df = df.sample(CFG.debug_max_rows, random_state=CFG.seed).reset_index(drop=True)
        LOG.warning("FAST_DEBUG: sampled MSU to %d rows", len(df))

    lower_map = {c.lower(): c for c in df.columns}
    marker_col = lower_map.get("marker")

    # Some derivative files may contain a class/label instead of marker. For
    # this research pipeline we strongly prefer the original Marker because it
    # preserves scenario identity needed for the fault-vs-attack taxonomy.
    if marker_col is None:
        for k in ["scenario", "scenario_id", "event", "class", "label"]:
            if k in lower_map:
                marker_col = lower_map[k]
                LOG.warning("No Marker column found; attempting scenario parsing from '%s'", marker_col)
                break
    if marker_col is None:
        raise ValueError("MSU loader cannot locate Marker/scenario column.")

    scenarios = df[marker_col].map(parse_msu_scenario)
    parse_rate = scenarios.notna().mean()
    LOG.info("MSU scenario parse success: %.2f%%", 100 * parse_rate)
    if parse_rate < 0.90:
        sample_bad = df.loc[scenarios.isna(), marker_col].head(20).tolist()
        raise ValueError(f"Scenario parsing failed for too many rows. Bad marker examples: {sample_bad}")

    df["__scenario"] = scenarios
    df["__label_name"] = df["__scenario"].map(scenario_to_label_name)
    before = len(df)
    df = df[df["__label_name"].notna()].reset_index(drop=True)
    if len(df) < before:
        LOG.warning("Dropped %d rows with unknown scenario labels", before - len(df))

    label_names = [
        "normal",
        "physical_fault_or_maintenance",
        "data_injection",
        "remote_trip",
        "relay_setting",
    ]
    label_to_id = {x: i for i, x in enumerate(label_names)}
    y = df["__label_name"].map(label_to_id).astype(int).values

    exclude = {marker_col, "__source_file", "__scenario", "__label_name"}
    feature_cols = [c for c in df.columns if c not in exclude]

    # Use only columns that are at least mostly numeric.
    numeric_cols = []
    for c in feature_cols:
        if pd.api.types.is_numeric_dtype(df[c]):
            numeric_cols.append(c)
        else:
            converted = pd.to_numeric(df[c], errors="coerce")
            if converted.notna().mean() >= 0.95:
                df[c] = converted
                numeric_cols.append(c)
            else:
                LOG.debug("MSU excluding nonnumeric feature: %s", c)
    feature_cols = numeric_cols

    groups_cols = infer_msu_modalities(feature_cols)
    LOG.info("MSU modality grouping:")
    for k, cols in groups_cols.items():
        LOG.info("  %s: %d columns | first=%s", k, len(cols), cols[:4])

    X = {}
    feature_names = {}
    raw_costs = {}
    for m, cols in groups_cols.items():
        clean = clean_numeric_frame(df, cols)
        X[m] = clean.values.astype(np.float32)
        feature_names[m] = cols
        # Data-volume proxy: number of scalar features observed for a row.
        raw_costs[m] = len(cols)

    if CFG.primary_cost_mode == "unit":
        modality_costs = {k: 1.0 / len(raw_costs) for k in raw_costs}
    else:
        modality_costs = normalize_costs(raw_costs)
    meta = df[["__source_file", "__scenario", "__label_name"]].copy()
    bundle = DatasetBundle(
        name="MSU_ORNL_Power",
        X=X,
        y=y.astype(np.int64),
        label_names=label_names,
        groups=df["__source_file"].astype(str).values,
        feature_names=feature_names,
        modality_costs=modality_costs,
        meta=meta,
    )
    describe_bundle(bundle)
    return bundle


# ============================================================
# 5. TU DELFT 2026 CPPS DATASET
# ============================================================
TU_LABELS = [
    "normal",
    "physical_fault",
    "reconnaissance",
    "resource_exhaustion",
    "goose_injection",
    "sv_injection",
]


def infer_tu_scenario(text: str) -> Optional[str]:
    s = text.lower().replace("-", "_").replace(" ", "_")
    # Order matters.
    if "resource" in s and any(k in s for k in ["exhaust", "attack", "dos"]):
        return "resource_exhaustion"
    if "recon" in s or "nmap" in s or "scan" in s:
        return "reconnaissance"
    if "goose" in s and any(k in s for k in ["inject", "attack", "malicious"]):
        return "goose_injection"
    if ("sv" in s or "sampled_value" in s or "sampledvalues" in s) and any(k in s for k in ["inject", "attack", "malicious"]):
        return "sv_injection"
    if "fault" in s or "slg" in s or "single_phase" in s:
        return "physical_fault"
    if "normal" in s or "baseline" in s or "no_attack" in s:
        return "normal"
    return None


def sniff_csv_type(path: Path) -> str:
    """Classify TU Delft CSVs using the dataset's filename conventions first.

    `*_rscad.csv` contains RTDS/RSCAD physical-system measurements, while
    `resource_monitor_*.csv` contains vIED resource telemetry.  Column-name
    heuristics alone can confuse these because RSCAD exports may mention IEDs.
    """
    n = path.name.lower()
    if "resource_monitor" in n or n.startswith("resource_"):
        return "resource"
    if "_rscad" in n or n.endswith("rscad.csv") or "rtds" in n:
        return "physical"
    if "label" in n or "ground" in n:
        return "packet_label"

    try:
        df = pd.read_csv(path, nrows=8)
    except Exception:
        return "unknown"
    cols = [str(c).lower() for c in df.columns]
    blob = " ".join(cols)
    phys_score = sum(k in blob for k in ["voltage", "current", "breaker", "fault", "rtds", "va", "vb", "vc", "ia", "ib", "ic"])
    res_score = sum(k in blob for k in ["cpu", "memory", "%cpu", "%mem", "rss", "virt", "loadavg"])
    label_score = sum(k in blob for k in ["malicious", "label", "ground_truth", "groundtruth", "attack_label"])
    if label_score >= 1 and df.shape[1] <= 12:
        return "packet_label"
    if phys_score >= 2:
        return "physical"
    if res_score >= 1:
        return "resource"
    return "unknown"


def canonical_tu_run_key(path: Path) -> str:
    """Map streams from one experiment to one key while preserving parent context.

    Filename-only keys caused collisions in Resource Exhaustion (e.g. several
    distinct fault_priority1 experiments). The parent experiment family is now
    included so independent runs cannot be merged accidentally.
    """
    n = path.stem.lower()
    n = re.sub(r"^resource_monitor_", "", n)
    # Remove view/modality suffixes repeatedly.
    for _ in range(3):
        n = re.sub(r"_(rscad|anon|labels?|ground_truth|groundtruth)$", "", n)
        n = re.sub(r"_(attacker|vied)$", "", n)
        n = re.sub(r"_(attacker|vied)_(anon)$", "", n)
    n = n.strip("_-")

    # Immediate parent is usually the run folder; parent.parent identifies the
    # experiment family (e.g. different resource-exhaustion sub-scenarios).
    family = path.parent.parent.name.lower() if len(path.parents) >= 2 else ""
    family = re.sub(r"[^a-z0-9]+", "_", family).strip("_")
    # Avoid redundant key when parent family is just the same run stem.
    if family and family != n and family not in {"scenarios", "scenario"}:
        return f"{family}__{n}"
    return n


def build_tu_manifest(root: Path) -> pd.DataFrame:
    rows = []
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        ext = p.suffix.lower()
        if ext not in {".csv", ".pcapng", ".pcap"}:
            continue
        scenario = infer_tu_scenario(str(p.relative_to(root)))
        if ext == ".csv":
            typ = sniff_csv_type(p)
        elif ext in {".pcapng", ".pcap"}:
            typ = "network"
        else:
            typ = "unknown"
        rows.append({
            "path": str(p),
            "relpath": str(p.relative_to(root)),
            "scenario": scenario,
            "type": typ,
            "run_key": canonical_tu_run_key(p),
            "size_mb": p.stat().st_size / (1024 ** 2),
            "parent": str(p.parent),
        })
    manifest = pd.DataFrame(rows)
    manifest.to_csv(WORK / "tudelft_manifest.csv", index=False)
    LOG.info("TU Delft manifest written to %s", WORK / "tudelft_manifest.csv")
    if len(manifest):
        LOG.info("TU manifest counts:\n%s", manifest.groupby(["scenario", "type"], dropna=False).size().to_string())
        LOG.info("TU run-key preview:\n%s", manifest[["scenario", "type", "run_key", "relpath"]].head(30).to_string(index=False))
    return manifest


def _coerce_decimal_series(x: pd.Series) -> pd.Series:
    """Parse ordinary numeric strings plus European decimal-comma values."""
    if pd.api.types.is_numeric_dtype(x):
        return pd.to_numeric(x, errors="coerce")
    y = x.astype(str).str.strip().str.replace(",", ".", regex=False)
    return pd.to_numeric(y, errors="coerce")


def _timeplus_to_seconds(v: Any) -> float:
    """Parse Linux top-like TIME+ values (MM:SS.xx or HH:MM:SS.xx)."""
    if pd.isna(v):
        return np.nan
    txt = str(v).strip()
    if not txt:
        return np.nan
    try:
        parts = txt.split(":")
        if len(parts) == 2:
            return float(parts[0]) * 60.0 + float(parts[1].replace(",", "."))
        if len(parts) == 3:
            return float(parts[0]) * 3600.0 + float(parts[1]) * 60.0 + float(parts[2].replace(",", "."))
        return float(txt.replace(",", "."))
    except Exception:
        return np.nan


def preprocess_tu_resource(df: pd.DataFrame) -> pd.DataFrame:
    """Remove run/process identifiers and convert cumulative counters to rates.

    PID, USER, COMMAND and process state can identify a particular experiment or
    process instance rather than a transferable cyber-physical condition.  The
    cumulative network/TIME+ counters also encode capture chronology, so they are
    replaced by within-run rates/deltas.
    """
    d = df.copy()
    d.columns = [str(c).strip() for c in d.columns]

    # Preserve Timestamp for absolute alignment.
    ts_col = next((c for c in d.columns if c.lower() == "timestamp"), None)
    if ts_col is not None:
        ts = pd.to_datetime(d[ts_col], errors="coerce", utc=True)
        dt = ts.diff().dt.total_seconds()
        med = float(dt[dt > 0].median()) if (dt > 0).any() else 2.0
        dt = dt.where(dt > 0, med).fillna(med).clip(lower=1e-6)
    else:
        dt = pd.Series(np.ones(len(d)) * 2.0, index=d.index)

    # Numeric fields that arrive with decimal commas.
    for c in ["%CPU", "%MEM"]:
        if c in d.columns:
            d[c] = _coerce_decimal_series(d[c])

    # Replace cumulative network counters with per-second increments.
    for c in ["Network Sent (GiB)", "Network Received (GiB)"]:
        if c in d.columns:
            vals = _coerce_decimal_series(d[c])
            delta = vals.diff().clip(lower=0)
            rate = (delta / dt).replace([np.inf, -np.inf], np.nan).fillna(0.0)
            d[c + " Rate (GiB/s)"] = rate
            d.drop(columns=[c], inplace=True)

    # TIME+ is cumulative CPU time. Use a rate, not the absolute clock value.
    if "TIME+" in d.columns:
        secs = d["TIME+"].map(_timeplus_to_seconds)
        d["Process CPU Time Rate"] = (secs.diff().clip(lower=0) / dt).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        d.drop(columns=["TIME+"], inplace=True)

    # Explicit identifiers / categorical metadata are not model inputs.
    if CFG.tu_drop_resource_identifiers:
        drop = [c for c in ["PID", "USER", "COMMAND", "S"] if c in d.columns]
        if drop:
            LOG.info("TU resource preprocessing: dropping identifier/categorical fields %s", drop)
            d.drop(columns=drop, inplace=True)

    return d


def _select_time_seconds(d: pd.DataFrame, time_col: Optional[str] = None) -> Tuple[pd.Series, bool, Optional[str]]:
    """Return seconds and whether they are absolute Unix time.

    TU Delft streams expose absolute timestamps in the physical/resource files,
    while PCAP timestamps are absolute epoch seconds. Prefer those so modalities
    are aligned in real time rather than independently resetting every stream to 0.
    """
    cols = list(d.columns)
    candidates: List[str] = []
    if time_col is not None and time_col in d.columns:
        candidates.append(time_col)
    for preferred in ["formatted_datetime", "Timestamp", "timestamp", "datetime", "date_time"]:
        if preferred in d.columns and preferred not in candidates:
            candidates.append(preferred)
    for c in cols:
        if c not in candidates and any(k in c.lower() for k in ["timestamp", "datetime", "date_time"]):
            candidates.append(c)
    for c in cols:
        if c not in candidates and c.lower() in {"time", "relative_time", "relative time"}:
            candidates.append(c)

    for c in candidates:
        raw = d[c]
        # Try datetime first for nonnumeric values.
        if not pd.api.types.is_numeric_dtype(raw):
            parsed = pd.to_datetime(raw, errors="coerce", utc=True)
            if parsed.notna().mean() > 0.7:
                sec = pd.Series(np.nan, index=d.index, dtype=float)
                good = parsed.notna()
                sec.loc[good] = parsed.loc[good].astype("int64").astype(float) / 1e9
                return sec, True, c
        vals = pd.to_numeric(raw, errors="coerce")
        if vals.notna().mean() > 0.7:
            vals = vals.astype(float)
            vals = vals - np.nanmin(vals.values)
            return vals, False, c

    LOG.warning("No usable time column detected for TU stream; using row index as pseudo-time.")
    return pd.Series(np.arange(len(d), dtype=float), index=d.index), False, None


def numeric_aggregate(
    df: pd.DataFrame,
    prefix: str,
    time_col: Optional[str] = None,
    window_seconds: float = 1.0,
    exclude_cols: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    """Aggregate a numeric stream into aligned fixed-duration windows."""
    d = df.copy()
    d.columns = [str(c).strip() for c in d.columns]
    t, is_absolute, chosen_time_col = _select_time_seconds(d, time_col=time_col)
    d["__time_s"] = t
    d = d[np.isfinite(d["__time_s"])].copy()
    if is_absolute and CFG.tu_absolute_time_alignment:
        d["__window"] = np.floor(d["__time_s"] / window_seconds).astype(np.int64)
    else:
        rel = d["__time_s"] - float(d["__time_s"].min())
        d["__window"] = np.floor(rel / window_seconds).astype(np.int64)

    ignore = {"__time_s", "__window"}
    if chosen_time_col:
        ignore.add(chosen_time_col)
    ignore.update(set(exclude_cols or []))

    numeric_cols = []
    for c in d.columns:
        if c in ignore:
            continue
        if pd.api.types.is_numeric_dtype(d[c]):
            numeric_cols.append(c)
        else:
            conv = _coerce_decimal_series(d[c])
            if conv.notna().mean() > 0.8:
                d[c] = conv
                numeric_cols.append(c)

    if not numeric_cols:
        raise ValueError("No numeric columns available for aggregation")

    agg = d.groupby("__window")[numeric_cols].agg(["mean", "std", "min", "max"])
    agg.columns = [f"{prefix}__{a}__{b}" for a, b in agg.columns]
    agg = agg.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    agg["__window"] = agg.index.astype(np.int64)
    return agg.reset_index(drop=True)


def aggregate_physical_ground_truth(df: pd.DataFrame, window_seconds: float = 1.0) -> pd.DataFrame:
    """Create physical-fault ground truth from FLT_input, never as a feature."""
    d = df.copy()
    d.columns = [str(c).strip() for c in d.columns]
    if "FLT_input" not in d.columns:
        return pd.DataFrame(columns=["__window", "__fault_active"])
    t, is_absolute, _ = _select_time_seconds(d)
    d["__time_s"] = t
    d = d[np.isfinite(d["__time_s"])].copy()
    if is_absolute and CFG.tu_absolute_time_alignment:
        d["__window"] = np.floor(d["__time_s"] / window_seconds).astype(np.int64)
    else:
        d["__window"] = np.floor((d["__time_s"] - d["__time_s"].min()) / window_seconds).astype(np.int64)
    v = pd.to_numeric(d["FLT_input"], errors="coerce").fillna(0.0)
    d["__fault_signal"] = v
    out = d.groupby("__window", as_index=False)["__fault_signal"].max()
    out["__fault_active"] = (out["__fault_signal"] > 0.5).astype(int)
    return out[["__window", "__fault_active"]]

def _pcap_bucket_factory():
    return {
        "packet_count": 0,
        "bytes": 0,
        "sum_len2": 0.0,
        "min_len": float("inf"),
        "max_len": 0,
        "goose_count": 0,
        "sv_count": 0,
        "ipv4_count": 0,
        "arp_count": 0,
        "tcp_count": 0,
        "udp_count": 0,
        "broadcast_count": 0,
        "multicast_count": 0,
        "src_macs": set(),
        "dst_macs": set(),
        "malicious_count": 0,
    }


def _mac_str(raw: bytes) -> str:
    return ":".join(f"{b:02x}" for b in raw)


def _update_packet_bucket(b: Dict[str, Any], frame: bytes) -> None:
    plen = len(frame)
    b["packet_count"] += 1
    b["bytes"] += plen
    b["sum_len2"] += plen * plen
    b["min_len"] = min(b["min_len"], plen)
    b["max_len"] = max(b["max_len"], plen)
    if plen < 14:
        return

    dst_raw, src_raw = frame[0:6], frame[6:12]
    dst, src = _mac_str(dst_raw), _mac_str(src_raw)
    b["src_macs"].add(src)
    b["dst_macs"].add(dst)
    if dst_raw == b"\xff" * 6:
        b["broadcast_count"] += 1
    if dst_raw[0] & 1:
        b["multicast_count"] += 1

    et_off = 12
    et = int.from_bytes(frame[et_off:et_off+2], "big")
    l3_off = 14
    # Support one or more VLAN tags.
    while et in (0x8100, 0x88A8, 0x9100) and len(frame) >= l3_off + 4:
        et = int.from_bytes(frame[l3_off+2:l3_off+4], "big")
        l3_off += 4

    if et == 0x88B8:
        b["goose_count"] += 1
    elif et == 0x88BA:
        b["sv_count"] += 1
    elif et == 0x0806:
        b["arp_count"] += 1
    elif et == 0x0800:
        b["ipv4_count"] += 1
        if len(frame) >= l3_off + 20:
            version_ihl = frame[l3_off]
            ihl = (version_ihl & 0x0F) * 4
            if (version_ihl >> 4) == 4 and ihl >= 20 and len(frame) >= l3_off + ihl:
                proto = frame[l3_off + 9]
                if proto == 6:
                    b["tcp_count"] += 1
                elif proto == 17:
                    b["udp_count"] += 1



def _detect_capture_format(path: Path) -> str:
    """Detect capture format from file magic, never from the filename suffix.

    The TU Delft release uses files ending in `.pcap` whose contents are
    actually PCAPNG (magic 0a0d0d0a), so suffix-based dispatch is unsafe.
    """
    try:
        with path.open("rb") as f:
            magic = f.read(4)
    except Exception as e:
        raise RuntimeError(f"Could not read capture header from {path}: {e}") from e

    classic_magic = {
        b"\xd4\xc3\xb2\xa1",  # little-endian microsecond PCAP
        b"\xa1\xb2\xc3\xd4",  # big-endian microsecond PCAP
        b"\x4d\x3c\xb2\xa1",  # little-endian nanosecond PCAP
        b"\xa1\xb2\x3c\x4d",  # big-endian nanosecond PCAP
    }
    if magic in classic_magic:
        return "pcap"
    if magic == b"\x0a\x0d\x0d\x0a":
        return "pcapng"
    return f"unknown:{magic.hex()}"


def _iter_classic_pcap(path: Path):
    """Yield (timestamp, raw_frame) from classic PCAP without third-party dependencies."""
    import struct
    with path.open("rb") as f:
        magic = f.read(4)
        if len(magic) != 4:
            return
        formats = {
            b"\xd4\xc3\xb2\xa1": ("<", 1e-6),
            b"\xa1\xb2\xc3\xd4": (">", 1e-6),
            b"\x4d\x3c\xb2\xa1": ("<", 1e-9),
            b"\xa1\xb2\x3c\x4d": (">", 1e-9),
        }
        if magic not in formats:
            raise ValueError(f"Not a classic PCAP file (magic={magic.hex()}): {path}")
        endian, frac_scale = formats[magic]
        rest = f.read(20)
        if len(rest) != 20:
            return
        ph = struct.Struct(endian + "IIII")
        while True:
            hdr = f.read(16)
            if not hdr:
                break
            if len(hdr) != 16:
                break
            ts_sec, ts_frac, incl_len, orig_len = ph.unpack(hdr)
            frame = f.read(incl_len)
            if len(frame) != incl_len:
                break
            yield ts_sec + ts_frac * frac_scale, frame



def _iter_pcapng_native(path: Path):
    """Yield ``(timestamp, raw_frame)`` from PCAPNG without third-party packages.

    Supports the block types used by the TU Delft captures:
      * Section Header Block (SHB)
      * Interface Description Block (IDB), including if_tsresol / if_tsoffset
      * Enhanced Packet Block (EPB)
      * legacy Packet Block (PB)

    Packet order is preserved exactly, which keeps the supplied
    ``frame_number`` ground truth aligned with capture frames.
    """
    import struct

    SHB = 0x0A0D0D0A
    IDB = 0x00000001
    PB  = 0x00000002
    SPB = 0x00000003
    EPB = 0x00000006

    endian = None
    interfaces = []  # [{linktype, ts_scale, ts_offset}, ...]
    skipped_non_eth = 0
    skipped_untimed = 0

    def parse_options(buf: bytes, endian_: str):
        opts = {}
        pos = 0
        while pos + 4 <= len(buf):
            code, length = struct.unpack_from(endian_ + 'HH', buf, pos)
            pos += 4
            if code == 0:
                break
            value = buf[pos:pos + length]
            opts.setdefault(code, []).append(value)
            pos += (length + 3) & ~3
        return opts

    with path.open('rb') as f:
        while True:
            block_start = f.tell()
            type_bytes = f.read(4)
            if not type_bytes:
                break
            if len(type_bytes) != 4:
                break
            len_bytes = f.read(4)
            if len(len_bytes) != 4:
                break

            # SHB must be decoded from its byte-order magic because a new
            # section may change endianness.
            if type_bytes == b'\x0a\x0d\x0d\x0a':
                bom = f.read(4)
                if bom == b'\x4d\x3c\x2b\x1a':
                    endian = '<'
                elif bom == b'\x1a\x2b\x3c\x4d':
                    endian = '>'
                else:
                    raise ValueError(f'Invalid PCAPNG byte-order magic {bom.hex()} in {path}')
                total_len = struct.unpack(endian + 'I', len_bytes)[0]
                if total_len < 28:
                    raise ValueError(f'Invalid PCAPNG SHB length {total_len} in {path}')
                # 12 bytes already consumed (type, length, BOM).
                rest = f.read(total_len - 12)
                if len(rest) != total_len - 12:
                    break
                trailer = struct.unpack_from(endian + 'I', rest, len(rest) - 4)[0]
                if trailer != total_len:
                    raise ValueError(f'PCAPNG SHB length trailer mismatch in {path}')
                interfaces = []
                continue

            if endian is None:
                raise ValueError(f'PCAPNG data before Section Header Block in {path}')

            block_type = struct.unpack(endian + 'I', type_bytes)[0]
            total_len = struct.unpack(endian + 'I', len_bytes)[0]
            if total_len < 12 or total_len % 4:
                raise ValueError(f'Invalid PCAPNG block length {total_len} at offset {block_start} in {path}')
            body_len = total_len - 12
            body = f.read(body_len)
            trailer_bytes = f.read(4)
            if len(body) != body_len or len(trailer_bytes) != 4:
                break
            trailer = struct.unpack(endian + 'I', trailer_bytes)[0]
            if trailer != total_len:
                raise ValueError(f'PCAPNG block length trailer mismatch at offset {block_start} in {path}')

            if block_type == IDB:
                if len(body) < 8:
                    continue
                linktype, _reserved, _snaplen = struct.unpack_from(endian + 'HHI', body, 0)
                opts = parse_options(body[8:], endian)
                # Default pcapng timestamp resolution is microseconds.
                ts_scale = 1e-6
                if 9 in opts and opts[9] and len(opts[9][0]) >= 1:
                    v = opts[9][0][0]
                    ts_scale = (2.0 ** -(v & 0x7F)) if (v & 0x80) else (10.0 ** -v)
                ts_offset = 0.0
                if 14 in opts and opts[14] and len(opts[14][0]) >= 8:
                    ts_offset = float(struct.unpack_from(endian + 'q', opts[14][0], 0)[0])
                interfaces.append({'linktype': int(linktype), 'ts_scale': float(ts_scale), 'ts_offset': ts_offset})
                continue

            if block_type == EPB:
                if len(body) < 20:
                    continue
                interface_id, ts_hi, ts_lo, cap_len, _orig_len = struct.unpack_from(endian + 'IIIII', body, 0)
                if interface_id >= len(interfaces):
                    continue
                iface = interfaces[interface_id]
                data = body[20:20 + cap_len]
                if len(data) < cap_len:
                    continue
                if iface['linktype'] != 1:  # LINKTYPE_ETHERNET
                    skipped_non_eth += 1
                    continue
                raw_ts = (int(ts_hi) << 32) | int(ts_lo)
                ts = raw_ts * iface['ts_scale'] + iface['ts_offset']
                yield float(ts), bytes(data)
                continue

            if block_type == PB:
                # Obsolete Packet Block; supported for completeness.
                if len(body) < 20:
                    continue
                interface_id, _drops = struct.unpack_from(endian + 'HH', body, 0)
                ts_hi, ts_lo, cap_len, _orig_len = struct.unpack_from(endian + 'IIII', body, 4)
                if interface_id >= len(interfaces):
                    continue
                iface = interfaces[interface_id]
                data = body[20:20 + cap_len]
                if len(data) < cap_len or iface['linktype'] != 1:
                    skipped_non_eth += int(iface['linktype'] != 1)
                    continue
                raw_ts = (int(ts_hi) << 32) | int(ts_lo)
                ts = raw_ts * iface['ts_scale'] + iface['ts_offset']
                yield float(ts), bytes(data)
                continue

            if block_type == SPB:
                # SPB has no timestamp, so it cannot be aligned to the CSV streams.
                skipped_untimed += 1
                continue

    if skipped_non_eth:
        LOG.warning('Native PCAPNG parser skipped %d non-Ethernet packets in %s', skipped_non_eth, path.name)
    if skipped_untimed:
        LOG.warning('Native PCAPNG parser skipped %d untimestamped SPB packets in %s', skipped_untimed, path.name)

def _load_frame_labels(path: Optional[Path]) -> Optional[np.ndarray]:
    if path is None or not path.exists():
        return None
    try:
        d = pd.read_csv(path, usecols=["frame_number", "label"])
    except Exception as e:
        LOG.warning("Could not read packet labels %s: %s", path.name, e)
        return None
    if d.empty:
        return None
    frames = pd.to_numeric(d["frame_number"], errors="coerce")
    labels = pd.to_numeric(d["label"], errors="coerce").fillna(0).astype(np.uint8)
    # The released files are contiguous frame_number=1..N. Verify before using
    # O(1) array indexing so silent misalignment cannot occur.
    if frames.notna().all() and int(frames.iloc[0]) == 1 and int(frames.iloc[-1]) == len(d):
        return labels.to_numpy(dtype=np.uint8)
    LOG.warning("Non-contiguous frame_number sequence in %s; using sparse lookup", path.name)
    max_frame = int(frames.max()) if frames.notna().any() else 0
    arr = np.zeros(max_frame, dtype=np.uint8)
    good = frames.notna() & (frames >= 1)
    idx = frames[good].astype(int).to_numpy() - 1
    arr[idx] = labels[good].to_numpy(dtype=np.uint8)
    return arr


def packet_features_from_pcap(
    path: Path,
    window_seconds: float = 1.0,
    max_packets: Optional[int] = None,
    labels_path: Optional[Path] = None,
) -> pd.DataFrame:
    buckets: Dict[int, Dict[str, Any]] = defaultdict(_pcap_bucket_factory)
    frame_labels = _load_frame_labels(labels_path) if CFG.tu_use_frame_ground_truth else None

    # Detect by content signature, NOT extension.  The TU Delft release uses
    # `.pcap` filenames containing PCAPNG bytes (magic 0a0d0d0a).
    capture_format = _detect_capture_format(path)
    if capture_format.startswith("unknown:"):
        raise ValueError(
            f"Unsupported capture format ({capture_format}) for {path}"
        )
    n = 0
    if capture_format == "pcap":
        LOG.info(
            "Native classic-PCAP parser: %s%s",
            path.name,
            f" | labels={labels_path.name}" if labels_path else "",
        )
        iterator = _iter_classic_pcap(path)
        for ts, frame in iterator:
            n += 1
            if max_packets is not None and n > max_packets:
                break
            # Absolute Unix-time bucket: aligns directly with CSV timestamps.
            w = int(math.floor(ts / window_seconds)) if CFG.tu_absolute_time_alignment else n
            _update_packet_bucket(buckets[w], frame)
            if frame_labels is not None and n <= len(frame_labels) and frame_labels[n - 1] > 0:
                buckets[w]["malicious_count"] += 1
    else:
        LOG.info(
            "Native PCAPNG parser (signature-detected): %s%s",
            path.name,
            f" | labels={labels_path.name}" if labels_path else "",
        )
        t0 = None
        for ts, frame in _iter_pcapng_native(path):
            n += 1
            if max_packets is not None and n > max_packets:
                break
            if t0 is None:
                t0 = ts
            w = (
                int(math.floor(ts / window_seconds))
                if CFG.tu_absolute_time_alignment
                else int(math.floor((ts - t0) / window_seconds))
            )
            _update_packet_bucket(buckets[w], frame)

            # Native parser preserves capture order, so packet n maps directly
            # to frame_number=n in the supplied packet-label CSV.
            if frame_labels is not None and n <= len(frame_labels) and frame_labels[n - 1] > 0:
                buckets[w]["malicious_count"] += 1

    if frame_labels is not None and max_packets is None and n != len(frame_labels):
        LOG.warning(
            "Frame-count mismatch for %s: packets_read=%d label_rows=%d. "
            "Frame-level GT may be misaligned if the reader skipped packets.",
            path.name, n, len(frame_labels),
        )

    rows = []
    for w, b in sorted(buckets.items()):
        c = max(b["packet_count"], 1)
        mean_len = b["bytes"] / c
        var_len = max(b["sum_len2"] / c - mean_len ** 2, 0.0)
        rows.append({
            "__window": w,
            # Ground truth only: deliberately NOT prefixed net__, so it can
            # never enter the network feature matrix.
            "__malicious_packets": int(b["malicious_count"]),
            "net__packet_count": b["packet_count"],
            "net__bytes": b["bytes"],
            "net__mean_len": mean_len,
            "net__std_len": math.sqrt(var_len),
            "net__min_len": 0 if b["min_len"] == float("inf") else b["min_len"],
            "net__max_len": b["max_len"],
            "net__goose_count": b["goose_count"],
            "net__sv_count": b["sv_count"],
            "net__ipv4_count": b["ipv4_count"],
            "net__arp_count": b["arp_count"],
            "net__tcp_count": b["tcp_count"],
            "net__udp_count": b["udp_count"],
            "net__broadcast_count": b["broadcast_count"],
            "net__multicast_count": b["multicast_count"],
            "net__unique_src": len(b["src_macs"]),
            "net__unique_dst": len(b["dst_macs"]),
        })
    if frame_labels is not None:
        LOG.info(
            "Packet GT %s: malicious frames=%d/%d | observed malicious windows=%d",
            path.name,
            int(frame_labels.sum()),
            len(frame_labels),
            sum(1 for r in rows if r["__malicious_packets"] > 0),
        )
    return pd.DataFrame(rows)


def _label_for_pcap(pcap: Path, label_paths: List[Path]) -> Optional[Path]:
    wanted = (pcap.stem + "_labels").lower()
    exact = [p for p in label_paths if p.stem.lower() == wanted]
    if len(exact) == 1:
        return exact[0]
    if len(exact) > 1:
        LOG.warning("Multiple packet-label files match %s; using first", pcap.name)
        return sorted(exact)[0]
    return None


def combine_network_pcaps(
    paths: List[Path],
    label_paths: Optional[List[Path]] = None,
    window_seconds: float = 1.0,
    max_packets: Optional[int] = None,
) -> Optional[pd.DataFrame]:
    """Combine synchronized capture views while retaining frame-level GT separately."""
    label_paths = label_paths or []
    frames = []
    for p in paths:
        lp = _label_for_pcap(p, label_paths)
        f = packet_features_from_pcap(
            p,
            window_seconds=window_seconds,
            max_packets=max_packets,
            labels_path=lp,
        )
        if f is not None and len(f):
            frames.append(f)
    if not frames:
        return None
    if len(frames) == 1:
        return frames[0]

    d = pd.concat(frames, ignore_index=True, sort=False)
    count_cols = [
        c for c in d.columns
        if c.startswith("net__") and c not in {"net__mean_len", "net__std_len", "net__min_len", "net__max_len"}
    ]
    agg_map = {c: "sum" for c in count_cols}
    agg_map["__malicious_packets"] = "sum"
    if "net__min_len" in d:
        agg_map["net__min_len"] = "min"
    if "net__max_len" in d:
        agg_map["net__max_len"] = "max"
    out = d.groupby("__window", as_index=False).agg(agg_map)
    if "net__bytes" in out and "net__packet_count" in out:
        out["net__mean_len"] = out["net__bytes"] / out["net__packet_count"].clip(lower=1)
    return out

def _clock_align_integer_windows(
    reference: pd.DataFrame,
    other: pd.DataFrame,
    stream_name: str,
) -> pd.DataFrame:
    """Align an absolute-time stream to a reference timeline without using labels.

    TU Delft CSV timestamps are wall-clock strings while packet captures store epoch
    timestamps. Depending on how the wall-clock strings are interpreted, the two
    streams can differ by an integer clock/time-zone offset (commonly whole hours).

    This routine estimates a shift ONLY from the unlabeled window ranges.  Candidate
    offsets are generated from start/end/median/quantile differences and the offset
    maximizing exact one-second window overlap is selected.  No target labels or
    attack markers participate in this synchronization.
    """
    if reference is None or other is None or len(reference) == 0 or len(other) == 0:
        return other
    if "__window" not in reference.columns or "__window" not in other.columns:
        return other

    ref = np.unique(pd.to_numeric(reference["__window"], errors="coerce").dropna().astype(np.int64))
    oth = np.unique(pd.to_numeric(other["__window"], errors="coerce").dropna().astype(np.int64))
    if len(ref) == 0 or len(oth) == 0:
        return other

    def _score(delta: int) -> int:
        return int(np.intersect1d(ref, oth + int(delta), assume_unique=True).size)

    candidates = {0}
    # Range anchors.
    candidates.add(int(ref[0] - oth[0]))
    candidates.add(int(ref[-1] - oth[-1]))
    candidates.add(int(round(float(np.median(ref)) - float(np.median(oth)))))
    # Robust quantile anchors help if one stream starts slightly earlier/later.
    for q in (0.1, 0.25, 0.5, 0.75, 0.9):
        candidates.add(int(round(float(np.quantile(ref, q)) - float(np.quantile(oth, q)))))

    # Keep clock corrections within one day. Anything larger is likely a date/parsing bug.
    candidates = {d for d in candidates if abs(int(d)) <= 24 * 3600}
    scored = sorted(((d, _score(d)) for d in candidates), key=lambda x: (x[1], -abs(x[0])), reverse=True)
    best_delta, best_overlap = scored[0]
    raw_overlap = _score(0)
    possible = max(1, min(len(ref), len(oth)))

    # Require a meaningful overlap improvement before modifying timestamps.
    min_good = max(3, int(math.ceil(0.10 * possible)))
    apply_shift = (
        best_delta != 0
        and best_overlap > raw_overlap
        and best_overlap >= min_good
    )

    LOG.info(
        "TU clock audit %s: ref=[%d,%d] other=[%d,%d] raw_overlap=%d/%d best_shift=%+ds best_overlap=%d/%d%s",
        stream_name,
        int(ref[0]), int(ref[-1]), int(oth[0]), int(oth[-1]),
        raw_overlap, possible, int(best_delta), best_overlap, possible,
        " APPLY" if apply_shift else " KEEP",
    )

    if not apply_shift:
        return other

    out = other.copy()
    out["__window"] = pd.to_numeric(out["__window"], errors="coerce").astype(np.int64) + int(best_delta)

    # Extra diagnostic: verify that malicious GT windows now overlap the reference.
    if "__malicious_packets" in out.columns:
        mw = out.loc[pd.to_numeric(out["__malicious_packets"], errors="coerce").fillna(0) > 0, "__window"]
        mal_overlap = int(np.intersect1d(ref, np.unique(mw.astype(np.int64))).size) if len(mw) else 0
        LOG.info(
            "TU clock audit %s after shift: malicious_windows_in_reference=%d/%d",
            stream_name, mal_overlap, int(len(np.unique(mw))) if len(mw) else 0,
        )
    return out


def align_tu_streams(
    physical: Optional[pd.DataFrame],
    resource: Optional[pd.DataFrame],
    network: Optional[pd.DataFrame],
    fault_gt: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Align all TU modalities on absolute 1-s windows without future leakage.

    Physical telemetry defines the diagnosis timeline whenever available.
    Resource monitoring is lower-rate (~2 s), so each physical second receives
    the most recent *past* resource sample via backward as-of matching.
    """
    if physical is not None and len(physical):
        base = physical.copy().sort_values("__window")
    elif network is not None and len(network):
        base = network.copy().sort_values("__window")
    elif resource is not None and len(resource):
        base = resource.copy().sort_values("__window")
    else:
        return pd.DataFrame()

    # Packet captures and CSV telemetry can describe the same wall-clock interval
    # with different clock/time-zone origins.  Estimate an integer shift using only
    # their unlabeled time ranges before exact network/physical merging.
    if (
        physical is not None and len(physical)
        and network is not None and len(network)
        and network is not base
    ):
        network = _clock_align_integer_windows(base, network, "network->physical")

    # Causal lower-rate resource alignment: never use a future resource sample.
    if resource is not None and len(resource) and resource is not physical:
        r = resource.copy().sort_values("__window")
        overlap = [c for c in r.columns if c != "__window" and c in base.columns]
        if overlap:
            r = r.drop(columns=overlap)
        base = pd.merge_asof(
            base.sort_values("__window"),
            r.sort_values("__window"),
            on="__window",
            direction="backward",
            tolerance=max(1, int(math.ceil(2.5 / CFG.tudelft_window_seconds))),
        )

    # Network and explicit fault GT are already 1-s buckets: exact alignment.
    for x in [network, fault_gt]:
        if x is not None and len(x) and x is not base:
            xx = x.copy()
            overlap = [c for c in xx.columns if c != "__window" and c in base.columns]
            if overlap:
                xx = xx.drop(columns=overlap)
            base = pd.merge(base, xx, on="__window", how="left")

    base = base.sort_values("__window").reset_index(drop=True)
    net_cols = [c for c in base.columns if c.startswith("net__")]
    if net_cols:
        base[net_cols] = base[net_cols].fillna(0.0)
    if "__malicious_packets" in base:
        base["__malicious_packets"] = base["__malicious_packets"].fillna(0).astype(int)
    if "__fault_active" in base:
        base["__fault_active"] = base["__fault_active"].fillna(0).astype(int)

    feature_cols = [c for c in base.columns if c.startswith(("phys__", "res__", "net__"))]
    if feature_cols:
        base[feature_cols] = base[feature_cols].replace([np.inf, -np.inf], np.nan)
        # Do not interpolate resource data across time after causal as-of match.
        phys_net = [c for c in feature_cols if c.startswith(("phys__", "net__"))]
        if phys_net:
            base[phys_net] = base[phys_net].interpolate(limit_direction="both")
        med = base[feature_cols].median(numeric_only=True)
        base[feature_cols] = base[feature_cols].fillna(med).fillna(0.0)
    return base

def _dilate_binary_mask(mask: np.ndarray, radius: int) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    if radius <= 0 or not mask.any():
        return mask
    out = mask.copy()
    hit = np.flatnonzero(mask)
    for d in range(1, radius + 1):
        out[np.clip(hit - d, 0, len(mask) - 1)] = True
        out[np.clip(hit + d, 0, len(mask) - 1)] = True
    return out


def assign_tu_window_labels(merged: pd.DataFrame, scenario: str) -> np.ndarray:
    """Create per-window target labels with explicit ground truth when available.

    Priority:
      1. Malicious frame present -> network attack class for that scenario.
      2. FLT_input active -> physical fault.
      3. Resource-exhaustion captures lack frame labels; scenario label is used.
      4. Otherwise -> normal.
    """
    y = np.full(len(merged), "normal", dtype=object)
    fault = merged.get("__fault_active", pd.Series(np.zeros(len(merged), dtype=int))).to_numpy() > 0
    malicious = merged.get("__malicious_packets", pd.Series(np.zeros(len(merged), dtype=int))).to_numpy() > 0
    malicious = _dilate_binary_mask(malicious, int(CFG.tu_attack_label_context_windows))

    if fault.any():
        y[fault] = "physical_fault"

    if scenario in {"reconnaissance", "goose_injection", "sv_injection"}:
        y[malicious] = scenario
        if not malicious.any():
            LOG.warning("TU run scenario=%s has no malicious frame-aligned windows after parsing", scenario)
    elif scenario == "resource_exhaustion":
        # No packet-label ground truth is supplied for this process-resource attack.
        # The released scenario recording is therefore treated as attack-active.
        y[:] = "resource_exhaustion"
    elif scenario == "physical_fault":
        # Already set only where FLT_input is active; pre/post-fault windows stay normal.
        pass
    elif scenario == "normal":
        y[:] = "normal"
    return y

def load_tudelft_bundle(root: Path) -> DatasetBundle:
    manifest = build_tu_manifest(root)
    if manifest.empty:
        raise FileNotFoundError(f"No TU Delft CSV/PCAP files found under {root}")

    # Drop files for which a scenario cannot be inferred; save these for user inspection.
    unknown = manifest[manifest["scenario"].isna()]
    if len(unknown):
        LOG.warning("TU Delft: %d files have unknown scenario labels. Inspect tudelft_manifest.csv", len(unknown))

    valid = manifest[manifest["scenario"].notna()].copy()
    if valid.empty:
        raise ValueError("TU Delft scenario names could not be inferred from paths. Edit infer_tu_scenario().")

    # v6.1 parses both classic PCAP and PCAPNG natively. Scapy is optional and
    # is no longer a runtime dependency for TU Delft captures.
    if (valid["type"] == "network").any() and not HAVE_SCAPY:
        LOG.info("Scapy unavailable: using dependency-free native PCAP/PCAPNG readers.")

    all_runs = []
    # Pair streams by semantic run key, not by independent sorted positions.
    # This correctly groups e.g. nmap_aggressive_rscad.csv,
    # resource_monitor_nmap_aggressive.csv, and both attacker/vIED PCAP views.
    for (scenario, run_key), g in valid.groupby(["scenario", "run_key"], dropna=False):
        phys = [Path(x) for x in g.loc[g.type == "physical", "path"]]
        res = [Path(x) for x in g.loc[g.type == "resource", "path"]]
        net = [Path(x) for x in g.loc[g.type == "network", "path"]]
        lbl = [Path(x) for x in g.loc[g.type == "packet_label", "path"]]
        run_id = f"{scenario}__{run_key}"
        LOG.info("TU processing %s | physical=%d resource=%d network_views=%d labels=%d",
                 run_id, len(phys), len(res), len(net), len(lbl))

        if len(phys) > 1 or len(res) > 1:
            msg = f"TU {run_id} ambiguous pairing: physical={len(phys)} resource={len(res)}"
            if CFG.strict_tu_pairing:
                LOG.error("%s; skipping run instead of silently selecting a file", msg)
                continue
            LOG.warning("%s; using first after sort because strict_tu_pairing=False", msg)
        p_phys = sorted(phys)[0] if phys else None
        p_res = sorted(res)[0] if res else None

        physical_agg = None
        resource_agg = None
        network_agg = None
        fault_gt = None
        try:
            if p_phys:
                d = pd.read_csv(p_phys)
                fault_gt = aggregate_physical_ground_truth(d, window_seconds=CFG.tudelft_window_seconds)
                # FLT_input is target ground truth and MUST NOT be an input feature.
                physical_agg = numeric_aggregate(
                    d,
                    "phys",
                    window_seconds=CFG.tudelft_window_seconds,
                    exclude_cols={"FLT_input"},
                )
            if p_res:
                d = preprocess_tu_resource(pd.read_csv(p_res))
                resource_agg = numeric_aggregate(d, "res", window_seconds=CFG.tudelft_window_seconds)
            if net:
                network_agg = combine_network_pcaps(
                    sorted(net),
                    label_paths=sorted(lbl),
                    window_seconds=CFG.tudelft_window_seconds,
                    max_packets=CFG.tudelft_max_packets_per_file,
                )
        except Exception as e:
            LOG.exception("TU failed processing %s: %s", run_id, e)
            continue

        merged = align_tu_streams(physical_agg, resource_agg, network_agg, fault_gt=fault_gt)
        if merged.empty:
            LOG.warning("TU run %s produced no windows", run_id)
            continue
        merged["__run_id"] = run_id
        merged["__source_scenario"] = scenario
        merged["__scenario"] = assign_tu_window_labels(merged, scenario)
        LOG.info(
            "TU window labels %s: %s",
            run_id,
            pd.Series(merged["__scenario"]).value_counts().to_dict(),
        )
        all_runs.append(merged)

    if not all_runs:
        raise RuntimeError("TU Delft loader produced no usable run windows. Inspect debug.log and tudelft_manifest.csv")

    df = pd.concat(all_runs, ignore_index=True, sort=False)
    if CFG.fast_debug and len(df) > CFG.debug_max_rows:
        # Sample full runs rather than individual windows if possible.
        run_ids = df["__run_id"].unique().tolist()
        random.Random(CFG.seed).shuffle(run_ids)
        keep = []
        total = 0
        for r in run_ids:
            n = (df["__run_id"] == r).sum()
            keep.append(r)
            total += n
            if total >= CFG.debug_max_rows:
                break
        df = df[df["__run_id"].isin(keep)].reset_index(drop=True)

    # Modalities are inferred by prefix from the aggregation functions.
    groups_cols = {}
    for prefix, name in [("phys__", "physical"), ("res__", "resource"), ("net__", "network")]:
        cols = [c for c in df.columns if c.startswith(prefix)]
        if cols:
            groups_cols[name] = cols
    if len(groups_cols) < 2:
        raise RuntimeError(
            f"TU Delft produced only {len(groups_cols)} usable modality/modality groups: {list(groups_cols)}. "
            "Active acquisition requires at least two. Inspect tudelft_manifest.csv/debug.log before training."
        )
    # Hard fail if target/identifier fields somehow leaked into model features.
    forbidden = ["pid", "flt_input", "frame_number", "malicious_packets", "ground_truth", "__scenario", "command", "user"]
    bad_features = []
    for m, cols in groups_cols.items():
        for c in cols:
            low = c.lower()
            if any(tok in low for tok in forbidden):
                bad_features.append((m, c))
    if bad_features:
        raise RuntimeError(f"TU target/identifier leakage guard triggered: {bad_features[:20]}")
    LOG.info("TU leakage guard: no target/identifier fields in model features")

    expected = {"physical", "resource", "network"}
    missing_modalities = expected - set(groups_cols)
    if missing_modalities:
        LOG.warning("TU Delft is missing modalities %s. Two-modality training is allowed for debugging, but use all three for final paper experiments.", sorted(missing_modalities))

    # Ensure all columns numeric / finite.
    X, feature_names, raw_costs = {}, {}, {}
    for m, cols in groups_cols.items():
        clean = clean_numeric_frame(df, cols)
        X[m] = clean.values.astype(np.float32)
        feature_names[m] = cols
        # More defensible data-volume proxy than simply assigning arbitrary costs.
        # Here feature count is multiplied by an approximate number of source
        # observations per 1-s window when known.
        if m == "physical":
            source_rate = max(1.0, 1.0 / 0.0016)  # paper reports 1.6 ms exported resolution
        elif m == "resource":
            source_rate = 0.5  # paper reports 2-s monitoring resolution
        else:
            # Network packet count itself will vary. Feature-count proxy keeps
            # the cost fixed across samples; replace with measured bytes for a
            # publication-grade packet-level cost experiment if desired.
            source_rate = 1.0
        raw_costs[m] = len(cols) * source_rate

    if CFG.primary_cost_mode == "unit":
        costs = {k: 1.0 / len(raw_costs) for k in raw_costs}
    else:
        costs = normalize_costs(raw_costs)
    label_to_id = {x: i for i, x in enumerate(TU_LABELS)}
    y = df["__scenario"].map(label_to_id).astype(int).values
    meta = df[["__run_id", "__scenario", "__window"]].copy()
    bundle = DatasetBundle(
        name="TU_Delft_2026_CPPS",
        X=X,
        y=y.astype(np.int64),
        label_names=TU_LABELS,
        groups=df["__run_id"].astype(str).values,
        feature_names=feature_names,
        modality_costs=costs,
        meta=meta,
    )
    describe_bundle(bundle)
    return bundle


# ============================================================
# 6. GROUP-AWARE SPLIT + SCALING
# ============================================================
@dataclass
class SplitBundle:
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray
    scalers: Dict[str, StandardScaler]
    X_scaled: Dict[str, np.ndarray]


def _class_distribution(y: np.ndarray, idx: np.ndarray, names: List[str]) -> str:
    s = pd.Series(y[idx]).value_counts().sort_index()
    return ", ".join(f"{names[i]}={int(s.get(i,0))}" for i in range(len(names)))


def _allocate_group_counts(n: int) -> Tuple[int, int, int]:
    """Allocate pure-class groups to train/val/test while preserving coverage."""
    if n < 3:
        raise ValueError(f"Need >=3 independent groups per class for strict coverage; got {n}")
    n_val = max(1, int(round(n * CFG.val_frac)))
    n_test = max(1, int(round(n * CFG.test_frac)))
    n_train = n - n_val - n_test
    if n_train < 1:
        n_train = 1
        if n_val >= n_test and n_val > 1:
            n_val -= 1
        elif n_test > 1:
            n_test -= 1
    assert n_train >= 1 and n_val >= 1 and n_test >= 1 and n_train + n_val + n_test == n
    return n_train, n_val, n_test


def _split_score(y: np.ndarray, tr: np.ndarray, va: np.ndarray, te: np.ndarray, C: int) -> float:
    global_p = np.bincount(y, minlength=C).astype(float)
    global_p /= global_p.sum()
    score = 0.0
    for ii in (tr, va, te):
        cnt = np.bincount(y[ii], minlength=C).astype(float)
        score += 1000.0 * float(np.sum(cnt == 0))
        if cnt.sum() > 0:
            score += float(np.abs(cnt / cnt.sum() - global_p).sum())
    return score


def _strict_group_indices(bundle: DatasetBundle, seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    y = np.asarray(bundle.y)
    groups = np.asarray(bundle.groups).astype(str)
    idx = np.arange(len(y))
    C = len(bundle.label_names)
    uniq = np.unique(groups)

    group_labels = {}
    pure = True
    for g in uniq:
        vals = np.unique(y[groups == g])
        if len(vals) == 1:
            group_labels[g] = int(vals[0])
        else:
            pure = False
            break

    if pure:
        rng = np.random.default_rng(seed)
        tr_g, va_g, te_g = [], [], []
        for c in range(C):
            gs = np.array([g for g, gc in group_labels.items() if gc == c], dtype=object)
            if len(gs) < 3:
                raise ValueError(
                    f"{bundle.name}: class '{bundle.label_names[c]}' has only {len(gs)} independent runs; "
                    "strict train/val/test coverage is impossible."
                )
            rng.shuffle(gs)
            nt, nv, ne = _allocate_group_counts(len(gs))
            tr_g.extend(gs[:nt].tolist())
            va_g.extend(gs[nt:nt+nv].tolist())
            te_g.extend(gs[nt+nv:nt+nv+ne].tolist())
        return idx[np.isin(groups, tr_g)], idx[np.isin(groups, va_g)], idx[np.isin(groups, te_g)]

    best, best_score = None, float("inf")
    for trial in range(CFG.split_search_trials):
        rs = seed + trial * 17
        gss1 = GroupShuffleSplit(n_splits=1, train_size=CFG.train_frac, random_state=rs)
        tr, rest = next(gss1.split(idx, y, groups))
        rest_groups = groups[rest]
        val_share_rest = CFG.val_frac / (CFG.val_frac + CFG.test_frac)
        gss2 = GroupShuffleSplit(n_splits=1, train_size=val_share_rest, random_state=rs + 1)
        vr, er = next(gss2.split(rest, y[rest], rest_groups))
        va, te = rest[vr], rest[er]
        sc = _split_score(y, tr, va, te, C)
        if sc < best_score:
            best_score, best = sc, (tr.copy(), va.copy(), te.copy())
    if best is None:
        raise RuntimeError(f"{bundle.name}: could not build a group-disjoint split")
    return best


def group_split(bundle: DatasetBundle, seed: int = 42) -> SplitBundle:
    groups = np.asarray(bundle.groups).astype(str)
    unique_groups = np.unique(groups)
    if len(unique_groups) < 5:
        raise ValueError(
            f"{bundle.name}: only {len(unique_groups)} independent groups. "
            "Publication protocol refuses row-level fallback."
        )
    tr, va, te = _strict_group_indices(bundle, seed)
    gt, gv, ge = set(groups[tr]), set(groups[va]), set(groups[te])
    assert gt.isdisjoint(gv) and gt.isdisjoint(ge) and gv.isdisjoint(ge), "Group leakage in split"

    LOG.info("Split %s: train=%d val=%d test=%d", bundle.name, len(tr), len(va), len(te))
    LOG.info("Train classes: %s", _class_distribution(bundle.y, tr, bundle.label_names))
    LOG.info("Val   classes: %s", _class_distribution(bundle.y, va, bundle.label_names))
    LOG.info("Test  classes: %s", _class_distribution(bundle.y, te, bundle.label_names))
    LOG.info("Train groups=%d Val groups=%d Test groups=%d", len(gt), len(gv), len(ge))

    missing_any = False
    for split_name, ii in [("train", tr), ("val", va), ("test", te)]:
        missing = [bundle.label_names[c] for c in range(len(bundle.label_names)) if np.sum(bundle.y[ii] == c) == 0]
        if missing:
            missing_any = True
            LOG.error("%s split missing classes: %s", split_name, missing)
    if CFG.strict_class_coverage and missing_any:
        raise RuntimeError(f"{bundle.name}: strict class coverage failed")

    group_rows = []
    for name, ii in [("train", tr), ("val", va), ("test", te)]:
        for g in sorted(set(groups[ii])):
            yy = bundle.y[groups == g]
            counts = np.bincount(yy, minlength=len(bundle.label_names))
            row = {"split": name, "group": g, "n_rows": int(len(yy))}
            row.update({f"n_{bundle.label_names[c]}": int(counts[c]) for c in range(len(bundle.label_names))})
            group_rows.append(row)
    pd.DataFrame(group_rows).to_csv(WORK / f"{bundle.name}_group_split_audit.csv", index=False)

    scalers, X_scaled = {}, {}
    for m, X in bundle.X.items():
        scaler = StandardScaler()
        scaler.fit(X[tr])
        x = scaler.transform(X).astype(np.float32)
        x = np.nan_to_num(x, nan=0.0, posinf=8.0, neginf=-8.0)
        x = np.clip(x, -12.0, 12.0).astype(np.float32)
        scalers[m], X_scaled[m] = scaler, x
        LOG.info("Scaled %-20s train mean=%.4f std=%.4f", m, float(x[tr].mean()), float(x[tr].std()))
    return SplitBundle(tr, va, te, scalers, X_scaled)


# ============================================================
# 7. TORCH DATASET + MULTIMODAL MODEL
# ============================================================
class MultiModalDataset(Dataset):
    def __init__(self, X: Dict[str, np.ndarray], y: np.ndarray, idx: np.ndarray, modalities: List[str]):
        self.X = X
        self.y = y
        self.idx = np.asarray(idx)
        self.modalities = modalities

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        j = self.idx[i]
        xs = [torch.from_numpy(self.X[m][j]).float() for m in self.modalities]
        return xs, torch.tensor(int(self.y[j]), dtype=torch.long), torch.tensor(int(j), dtype=torch.long)


def collate_multimodal(batch):
    xs_list, ys, js = zip(*batch)
    M = len(xs_list[0])
    stacked = [torch.stack([x[m] for x in xs_list], dim=0) for m in range(M)]
    return stacked, torch.stack(ys), torch.stack(js)


class ModalityEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden: int, latent: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, latent),
            nn.GELU(),
        )

    def forward(self, x):
        return self.net(x)


class ActiveFusionClassifier(nn.Module):
    """Missing-modality-aware fusion that preserves modality identity."""
    def __init__(self, input_dims: Dict[str, int], modalities: List[str], num_classes: int,
                 hidden: int = 128, latent: int = 64, dropout: float = 0.2):
        super().__init__()
        self.modalities = modalities
        self.num_modalities = len(modalities)
        self.per_modality_latent = latent
        self.fused_dim = latent * self.num_modalities
        self.latent_dim = self.fused_dim
        self.encoders = nn.ModuleDict({
            m: ModalityEncoder(input_dims[m], hidden, latent, dropout) for m in modalities
        })
        width = max(hidden, 192)
        self.classifier = nn.Sequential(
            nn.Linear(self.fused_dim + self.num_modalities, width),
            nn.LayerNorm(width),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(width, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )

    def fuse(self, xs: List[torch.Tensor], mask: torch.Tensor) -> torch.Tensor:
        zs = []
        for i, m in enumerate(self.modalities):
            z = self.encoders[m](xs[i]) * mask[:, i:i+1]
            zs.append(z)
        return torch.cat(zs, dim=1)

    def forward(self, xs: List[torch.Tensor], mask: torch.Tensor):
        fused = self.fuse(xs, mask)
        logits = self.classifier(torch.cat([fused, mask], dim=1))
        return logits, fused


def random_modality_mask(batch_size: int, M: int, p_drop: float, p_all_missing: float, device: str) -> torch.Tensor:
    mask = (torch.rand(batch_size, M, device=device) > p_drop).float()
    # Some all-missing examples teach the empty state used by RL.
    all_missing = torch.rand(batch_size, device=device) < p_all_missing
    mask[all_missing] = 0.0
    # Otherwise ensure at least one modality remains.
    no_obs = (mask.sum(dim=1) == 0) & (~all_missing)
    for row in torch.where(no_obs)[0].tolist():
        mask[row, random.randrange(M)] = 1.0
    return mask


def class_weights_from_indices(y: np.ndarray, idx: np.ndarray, num_classes: int) -> torch.Tensor:
    counts = np.bincount(y[idx], minlength=num_classes).astype(float)
    counts[counts == 0] = 1.0
    w = counts.sum() / (num_classes * counts)
    w = w / w.mean()
    return torch.tensor(w, dtype=torch.float32, device=CFG.device)


@torch.no_grad()
def evaluate_classifier(model: nn.Module, X: Dict[str, np.ndarray], y: np.ndarray, idx: np.ndarray,
                        modalities: List[str], mask_mode: str = "full", batch_size: int = 2048):
    model.eval()
    ds = MultiModalDataset(X, y, idx, modalities)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate_multimodal)
    all_y, all_pred, all_prob, all_z = [], [], [], []
    for xs, yy, _ in dl:
        xs = [x.to(CFG.device) for x in xs]
        yy = yy.to(CFG.device)
        B = len(yy)
        if mask_mode == "full":
            mask = torch.ones(B, len(modalities), device=CFG.device)
        elif mask_mode == "none":
            mask = torch.zeros(B, len(modalities), device=CFG.device)
        else:
            raise ValueError(mask_mode)
        logits, z = model(xs, mask)
        probs = torch.softmax(logits, dim=1)
        all_y.append(yy.cpu().numpy())
        all_pred.append(probs.argmax(1).cpu().numpy())
        all_prob.append(probs.cpu().numpy())
        all_z.append(z.cpu().numpy())
    return (
        np.concatenate(all_y),
        np.concatenate(all_pred),
        np.concatenate(all_prob),
        np.concatenate(all_z),
    )


def macro_metrics(y_true, y_pred, num_classes: Optional[int] = None) -> Dict[str, float]:
    labels = np.arange(num_classes) if num_classes is not None else np.unique(np.concatenate([np.asarray(y_true), np.asarray(y_pred)]))
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision_macro": p,
        "recall_macro": r,
        "macro_f1": f,
    }


def train_fusion_classifier(bundle: DatasetBundle, split: SplitBundle, epochs: Optional[int] = None,
                            train_idx_override: Optional[np.ndarray] = None,
                            val_idx_override: Optional[np.ndarray] = None,
                            y_override: Optional[np.ndarray] = None,
                            num_classes_override: Optional[int] = None,
                            model_tag: str = "main") -> ActiveFusionClassifier:
    modalities = list(bundle.X.keys())
    X = split.X_scaled
    y = bundle.y if y_override is None else y_override
    train_idx = split.train_idx if train_idx_override is None else train_idx_override
    val_idx = split.val_idx if val_idx_override is None else val_idx_override
    num_classes = len(bundle.label_names) if num_classes_override is None else num_classes_override
    input_dims = {m: X[m].shape[1] for m in modalities}
    model = ActiveFusionClassifier(input_dims, modalities, num_classes,
                                   hidden=CFG.hidden_dim, latent=CFG.latent_dim, dropout=CFG.dropout).to(CFG.device)
    opt = torch.optim.AdamW(model.parameters(), lr=CFG.clf_lr, weight_decay=CFG.weight_decay)
    weights = class_weights_from_indices(y, train_idx, num_classes)
    criterion = nn.CrossEntropyLoss(weight=weights)

    tr_ds = MultiModalDataset(X, y, train_idx, modalities)
    va_ds = MultiModalDataset(X, y, val_idx, modalities)
    tr_dl = DataLoader(tr_ds, batch_size=CFG.clf_batch_size, shuffle=True,
                       num_workers=2, pin_memory=torch.cuda.is_available(), collate_fn=collate_multimodal)
    va_dl = DataLoader(va_ds, batch_size=2048, shuffle=False, num_workers=2,
                       pin_memory=torch.cuda.is_available(), collate_fn=collate_multimodal)

    epochs = epochs or CFG.clf_epochs
    best = None
    best_f1 = -1
    bad = 0

    LOG.info("Training fusion classifier [%s] | epochs=%d | classes=%d | modalities=%s", model_tag, epochs, num_classes, modalities)
    for ep in range(1, epochs + 1):
        model.train()
        losses = []
        for xs, yy, _ in tr_dl:
            xs = [x.to(CFG.device, non_blocking=True) for x in xs]
            yy = yy.to(CFG.device, non_blocking=True)
            mask = random_modality_mask(len(yy), len(modalities), CFG.modality_dropout_p,
                                        CFG.probability_all_missing, CFG.device)
            logits, _ = model(xs, mask)
            loss = criterion(logits, yy)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            losses.append(float(loss.item()))

        # Validate full telemetry for stable checkpoint selection.
        model.eval()
        yy_all, pp_all = [], []
        with torch.no_grad():
            for xs, yy, _ in va_dl:
                xs = [x.to(CFG.device) for x in xs]
                mask = torch.ones(len(yy), len(modalities), device=CFG.device)
                logits, _ = model(xs, mask)
                pred = logits.argmax(1).cpu().numpy()
                yy_all.append(yy.numpy())
                pp_all.append(pred)
        yy_all = np.concatenate(yy_all)
        pp_all = np.concatenate(pp_all)
        f1 = f1_score(yy_all, pp_all, average="macro", zero_division=0)
        LOG.info("Classifier %-12s epoch %03d | loss %.5f | val Macro-F1 %.5f", model_tag, ep, np.mean(losses), f1)
        if f1 > best_f1 + 1e-5:
            best_f1 = f1
            best = copy.deepcopy(model.state_dict())
            bad = 0
        else:
            bad += 1
            if bad >= CFG.clf_patience:
                LOG.info("Early stopping classifier at epoch %d", ep)
                break

    if best is not None:
        model.load_state_dict(best)
    return model


# ============================================================
# 7B. LEAKAGE / TRIVIAL-SIGNATURE AUDIT
# ============================================================
def run_leakage_audit(bundle: DatasetBundle, split: SplitBundle) -> pd.DataFrame:
    rows = []
    suspicious_tokens = ("label", "target", "scenario", "attack_type", "class", "run_id", "filename", "filepath")
    for m, names in bundle.feature_names.items():
        sus = [n for n in names if any(tok in n.lower() for tok in suspicious_tokens)]
        rows.append({"check":"suspicious_feature_names", "modality":m, "value":len(sus), "detail":";".join(sus[:20])})
        if sus:
            LOG.error("LEAKAGE AUDIT %s/%s suspicious features: %s", bundle.name, m, sus[:20])

    groups = np.asarray(bundle.groups).astype(str)
    for key, a, b in [("train_val", split.train_idx, split.val_idx), ("train_test", split.train_idx, split.test_idx), ("val_test", split.val_idx, split.test_idx)]:
        overlap = len(set(groups[a]) & set(groups[b]))
        rows.append({"check":"group_overlap", "modality":key, "value":overlap, "detail":"must be 0"})
        if overlap:
            raise RuntimeError(f"Group leakage detected: {key}={overlap}")

    from sklearn.tree import DecisionTreeClassifier
    C = len(bundle.label_names)
    for m, X in split.X_scaled.items():
        best_f1, best_j = -1.0, -1
        for j in range(X.shape[1]):
            clf = DecisionTreeClassifier(max_depth=2, random_state=CFG.seed, class_weight="balanced")
            clf.fit(X[split.train_idx, j:j+1], bundle.y[split.train_idx])
            pred = clf.predict(X[split.test_idx, j:j+1])
            f = f1_score(bundle.y[split.test_idx], pred, labels=np.arange(C), average="macro", zero_division=0)
            if f > best_f1:
                best_f1, best_j = float(f), j
        fname = bundle.feature_names[m][best_j] if best_j >= 0 else ""
        rows.append({"check":"best_single_feature_depth2_f1", "modality":m, "value":best_f1, "detail":fname})
        LOG.info("LEAKAGE AUDIT %-20s best single-feature depth-2 Macro-F1=%.4f feature=%s", m, best_f1, fname)
    out = pd.DataFrame(rows)
    out.to_csv(WORK / f"{bundle.name}_leakage_audit.csv", index=False)
    return out


# ============================================================
# 8. CLASSICAL / FULL-TELEMETRY BASELINES
# ============================================================
def concat_features(X: Dict[str, np.ndarray], modalities: List[str]) -> np.ndarray:
    return np.concatenate([X[m] for m in modalities], axis=1)


def maybe_subsample(idx: np.ndarray, max_n: int, seed: int = 42) -> np.ndarray:
    if len(idx) <= max_n:
        return idx
    rng = np.random.default_rng(seed)
    return rng.choice(idx, size=max_n, replace=False)


def run_classical_baselines(bundle: DatasetBundle, split: SplitBundle) -> pd.DataFrame:
    modalities = list(bundle.X.keys())
    XX = concat_features(split.X_scaled, modalities)
    tr = maybe_subsample(split.train_idx, CFG.baseline_max_train_rows, CFG.seed)
    te = split.test_idx
    Xtr, ytr = XX[tr], bundle.y[tr]
    Xte, yte = XX[te], bundle.y[te]

    models = {
        "LogisticRegression": LogisticRegression(max_iter=1200, class_weight="balanced", n_jobs=-1),
        "RandomForest": RandomForestClassifier(
            n_estimators=CFG.rf_estimators, class_weight="balanced_subsample", n_jobs=-1,
            random_state=CFG.seed, max_features="sqrt"
        ),
        "ExtraTrees": ExtraTreesClassifier(
            n_estimators=CFG.rf_estimators, class_weight="balanced", n_jobs=-1,
            random_state=CFG.seed, max_features="sqrt"
        ),
        "HistGradientBoosting": HistGradientBoostingClassifier(
            learning_rate=0.08, max_iter=250 if not CFG.fast_debug else 60,
            max_leaf_nodes=31, random_state=CFG.seed
        ),
        "SklearnMLP": MLPClassifier(
            hidden_layer_sizes=(128, 64), max_iter=80 if not CFG.fast_debug else 15,
            early_stopping=True, random_state=CFG.seed, batch_size=512
        ),
    }
    if HAVE_XGB:
        models["XGBoost"] = XGBClassifier(
            n_estimators=CFG.xgb_estimators,
            max_depth=7,
            learning_rate=0.06,
            subsample=0.85,
            colsample_bytree=0.85,
            objective="multi:softprob",
            num_class=len(bundle.label_names),
            eval_metric="mlogloss",
            n_jobs=-1,
            random_state=CFG.seed,
            tree_method="hist",
        )

    rows = []
    for name, model in models.items():
        LOG.info("Baseline training: %s (%d rows, %d features)", name, len(tr), Xtr.shape[1])
        t0 = time.time()
        try:
            model.fit(Xtr, ytr)
            pred = model.predict(Xte)
            elapsed = time.time() - t0
            met = macro_metrics(yte, pred, len(bundle.label_names))
            rows.append({"method": name, **met, "train_seconds": elapsed,
                         "normalized_cost": 1.0, "acquisition_ratio": 1.0, "mean_steps": len(modalities)})
            LOG.info("Baseline %-24s Macro-F1=%.5f Acc=%.5f time=%.1fs", name, met["macro_f1"], met["accuracy"], elapsed)
        except Exception as e:
            LOG.exception("Baseline %s failed: %s", name, e)
    return pd.DataFrame(rows)


# ============================================================
# 9. DQN ACTIVE ACQUISITION
# ============================================================
Transition = namedtuple("Transition", "state action reward next_state done next_valid")


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buf = deque(maxlen=capacity)

    def push(self, *args):
        self.buf.append(Transition(*args))

    def sample(self, batch_size: int):
        batch = random.sample(self.buf, batch_size)
        return Transition(*zip(*batch))

    def __len__(self):
        return len(self.buf)


class DQN(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_dim),
        )

    def forward(self, x):
        return self.net(x)


class ActiveEvidenceEnv:
    def __init__(self, model: ActiveFusionClassifier, X: Dict[str, np.ndarray], y: np.ndarray,
                 modalities: List[str], modality_costs: Dict[str, float], label_names: List[str],
                 device: str):
        self.model = model
        self.model.eval()
        self.X = X
        self.y = y
        self.modalities = modalities
        self.costs = np.array([modality_costs[m] for m in modalities], dtype=np.float32)
        self.label_names = label_names
        self.device = device
        self.M = len(modalities)
        self.stop_action = self.M
        self.action_dim = self.M + 1
        self.state_dim = model.latent_dim + len(label_names) + 1 + self.M + 1
        self.reset(0)

    def reset(self, sample_idx: int):
        self.i = int(sample_idx)
        self.mask = np.zeros(self.M, dtype=np.float32)
        self.total_cost = 0.0
        self.steps = 0
        self.done = False
        # Training-only diagnostic potential used for dense reward shaping.
        # It is never included in the observable state and is unavailable to the
        # deployed policy; the true label is used only by the training environment.
        _, probs, _, _ = self._diagnosis()
        true_y = int(self.y[self.i])
        self._true_logp = float(np.log(max(float(probs[true_y]), 1e-8)))
        return self._state()

    def _inputs(self):
        xs = [torch.from_numpy(self.X[m][self.i:self.i+1]).float().to(self.device) for m in self.modalities]
        mask = torch.from_numpy(self.mask[None, :]).float().to(self.device)
        return xs, mask

    @torch.no_grad()
    def _diagnosis(self):
        xs, mask = self._inputs()
        logits, z = self.model(xs, mask)
        p = torch.softmax(logits, 1)[0]
        pred = int(torch.argmax(p).item())
        entropy = float((-(p * torch.log(p + 1e-12)).sum()).item())
        return pred, p.cpu().numpy().astype(np.float32), z[0].cpu().numpy().astype(np.float32), entropy

    def _state(self):
        pred, probs, z, entropy = self._diagnosis()
        remaining_budget = max(0.0, 1.0 - self.total_cost)
        state = np.concatenate([
            z,
            probs,
            np.array([entropy], dtype=np.float32),
            self.mask.astype(np.float32),
            np.array([remaining_budget], dtype=np.float32),
        ]).astype(np.float32)
        assert len(state) == self.state_dim
        return state

    def valid_actions(self) -> np.ndarray:
        valid = np.ones(self.action_dim, dtype=bool)
        valid[:self.M] = self.mask < 0.5
        # Prevent immediate stop before minimum evidence is acquired.
        if self.mask.sum() < CFG.min_acquisitions_before_stop:
            valid[self.stop_action] = False
        return valid

    def _critical_confusion(self, pred: int, true: int) -> bool:
        pn = self.label_names[pred].lower()
        tn = self.label_names[true].lower()
        p_fault = "fault" in pn or "maintenance" in pn
        t_fault = "fault" in tn or "maintenance" in tn
        p_attack = pred != true and any(k in pn for k in ["inject", "trip", "relay", "attack", "recon", "resource"])
        t_attack = pred != true and any(k in tn for k in ["inject", "trip", "relay", "attack", "recon", "resource"])
        return (p_fault and t_attack) or (t_fault and p_attack)

    def step(self, action: int):
        assert not self.done
        true_y = int(self.y[self.i])
        reward = 0.0
        if action == self.stop_action:
            pred, probs, z, entropy = self._diagnosis()
            if pred == true_y:
                reward += CFG.correct_reward
            else:
                reward -= CFG.wrong_penalty
                if self._critical_confusion(pred, true_y):
                    reward -= CFG.critical_confusion_penalty
            self.done = True
            next_state = self._state()
            return next_state, reward, True, {"pred": pred, "probs": probs, "entropy": entropy}

        if action < 0 or action >= self.M or self.mask[action] > 0.5:
            # invalid/repeated acquisition
            reward -= 1.0
            next_state = self._state()
            return next_state, reward, False, {"invalid": True}

        self.mask[action] = 1.0
        c = float(self.costs[action])
        self.total_cost += c
        self.steps += 1
        reward -= CFG.acquisition_lambda * c
        reward -= CFG.step_penalty

        # Dense diagnostic-improvement reward: credit an acquisition when it
        # increases the classifier probability of the true training class.
        # This is training-only reward shaping; no true label enters the state
        # or inference-time policy. It substantially reduces the sparse-credit
        # problem in multi-source settings such as MSU/ORNL.
        if CFG.dqn_dense_information_reward:
            _, new_probs, _, _ = self._diagnosis()
            new_logp = float(np.log(max(float(new_probs[true_y]), 1e-8)))
            delta = CFG.dqn_dense_reward_scale * (new_logp - self._true_logp)
            clip = float(CFG.dqn_dense_reward_clip)
            reward += float(np.clip(delta, -clip, clip))
            self._true_logp = new_logp

        # If all modalities are acquired, force a terminal diagnosis.
        if self.mask.sum() >= self.M:
            pred, probs, z, entropy = self._diagnosis()
            if pred == true_y:
                reward += CFG.correct_reward
            else:
                reward -= CFG.wrong_penalty
                if self._critical_confusion(pred, true_y):
                    reward -= CFG.critical_confusion_penalty
            self.done = True
            return self._state(), reward, True, {"pred": pred, "probs": probs, "entropy": entropy, "forced_stop": True}

        return self._state(), reward, False, {}


def epsilon_for_episode(ep: int, total: int) -> float:
    decay_eps = max(1, int(total * CFG.epsilon_decay_fraction))
    frac = min(ep / decay_eps, 1.0)
    return CFG.epsilon_start + frac * (CFG.epsilon_end - CFG.epsilon_start)


def masked_argmax(q: np.ndarray, valid: np.ndarray) -> int:
    qq = q.copy()
    qq[~valid] = -1e30
    return int(np.argmax(qq))


def train_dqn_policy(env: ActiveEvidenceEnv, train_idx: np.ndarray, episodes: Optional[int] = None,
                     tag: str = "main") -> DQN:
    episodes = episodes or CFG.dqn_episodes
    online = DQN(env.state_dim, env.action_dim, CFG.dqn_hidden).to(CFG.device)
    target = copy.deepcopy(online).to(CFG.device)
    target.eval()
    opt = torch.optim.AdamW(online.parameters(), lr=CFG.dqn_lr, weight_decay=1e-5)
    replay = ReplayBuffer(CFG.dqn_replay_size)

    reward_hist = deque(maxlen=500)
    cost_hist = deque(maxlen=500)
    step_count = 0
    rng = np.random.default_rng(CFG.seed)
    train_idx = np.asarray(train_idx, dtype=np.int64)
    class_to_idx = {}
    if CFG.dqn_balanced_episode_sampling:
        for cls in np.unique(env.y[train_idx]):
            arr = train_idx[env.y[train_idx] == cls]
            if len(arr):
                class_to_idx[int(cls)] = arr
        balanced_classes = np.array(sorted(class_to_idx.keys()), dtype=np.int64)
        LOG.info("DQN [%s] balanced episode sampling across classes=%s", tag, balanced_classes.tolist())
    else:
        balanced_classes = None

    LOG.info("Training DQN [%s] episodes=%d state_dim=%d actions=%d", tag, episodes, env.state_dim, env.action_dim)
    for ep in range(1, episodes + 1):
        if balanced_classes is not None and len(balanced_classes):
            cls = int(rng.choice(balanced_classes))
            i = int(rng.choice(class_to_idx[cls]))
        else:
            i = int(rng.choice(train_idx))
        state = env.reset(i)
        ep_reward = 0.0
        while True:
            valid = env.valid_actions()
            eps = epsilon_for_episode(ep, episodes)
            if random.random() < eps:
                choices = np.where(valid)[0]
                action = int(random.choice(choices.tolist()))
            else:
                with torch.no_grad():
                    q = online(torch.from_numpy(state).float().unsqueeze(0).to(CFG.device))[0].cpu().numpy()
                action = masked_argmax(q, valid)

            next_state, reward, done, info = env.step(action)
            next_valid = env.valid_actions().copy() if not done else np.zeros(env.action_dim, dtype=bool)
            replay.push(state, action, reward, next_state, done, next_valid)
            state = next_state
            ep_reward += reward
            step_count += 1

            if len(replay) >= max(CFG.dqn_warmup, CFG.dqn_batch_size):
                batch = replay.sample(CFG.dqn_batch_size)
                states = torch.tensor(np.stack(batch.state), dtype=torch.float32, device=CFG.device)
                actions = torch.tensor(batch.action, dtype=torch.long, device=CFG.device)
                rewards = torch.tensor(batch.reward, dtype=torch.float32, device=CFG.device)
                next_states = torch.tensor(np.stack(batch.next_state), dtype=torch.float32, device=CFG.device)
                dones = torch.tensor(batch.done, dtype=torch.float32, device=CFG.device)
                next_valid = torch.tensor(np.stack(batch.next_valid), dtype=torch.bool, device=CFG.device)

                q_sa = online(states).gather(1, actions[:, None]).squeeze(1)
                with torch.no_grad():
                    # Double DQN: online chooses, target evaluates.
                    q_next_online = online(next_states)
                    q_next_online = q_next_online.masked_fill(~next_valid, -1e9)
                    next_actions = q_next_online.argmax(1)
                    q_next_target = target(next_states).gather(1, next_actions[:, None]).squeeze(1)
                    q_next_target = torch.where(dones > 0, torch.zeros_like(q_next_target), q_next_target)
                    target_q = rewards + CFG.dqn_gamma * q_next_target

                loss = F.smooth_l1_loss(q_sa, target_q)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(online.parameters(), 5.0)
                opt.step()

                if step_count % CFG.dqn_target_update == 0:
                    target.load_state_dict(online.state_dict())

            if done:
                break

        reward_hist.append(ep_reward)
        cost_hist.append(env.total_cost)
        if ep == 1 or ep % max(100, episodes // 30) == 0:
            LOG.info("DQN %-10s ep=%d/%d eps=%.3f avgR500=%.3f avgCost500=%.3f replay=%d",
                     tag, ep, episodes, epsilon_for_episode(ep, episodes),
                     np.mean(reward_hist), np.mean(cost_hist), len(replay))

    return online


# ============================================================
# 10. POLICY EVALUATION / ACTIVE BASELINES
# ============================================================
@torch.no_grad()
def diagnose_with_mask(model, X, y, modalities, sample_idx: int, mask_np: np.ndarray):
    xs = [torch.from_numpy(X[m][sample_idx:sample_idx+1]).float().to(CFG.device) for m in modalities]
    mask = torch.from_numpy(mask_np[None, :]).float().to(CFG.device)
    logits, z = model(xs, mask)
    probs = torch.softmax(logits, 1)[0].cpu().numpy()
    pred = int(np.argmax(probs))
    ent = float(-np.sum(probs * np.log(probs + 1e-12)))
    return pred, probs, z[0].cpu().numpy(), ent


def evaluate_dqn(env: ActiveEvidenceEnv, policy: DQN, idx: np.ndarray, method_name="ActiGrid-DQN") -> Tuple[pd.DataFrame, pd.DataFrame]:
    policy.eval()
    rows = []
    traces = []
    for k, i in enumerate(idx):
        state = env.reset(int(i))
        acquired = []
        while True:
            valid = env.valid_actions()
            with torch.no_grad():
                q = policy(torch.from_numpy(state).float().unsqueeze(0).to(CFG.device))[0].cpu().numpy()
            action = masked_argmax(q, valid)
            if action < env.M:
                acquired.append(env.modalities[action])
            state, reward, done, info = env.step(action)
            if done:
                pred = int(info["pred"])
                probs = info["probs"]
                break
        rows.append({
            "sample_idx": int(i),
            "y_true": int(env.y[i]),
            "y_pred": pred,
            "cost": env.total_cost,
            "acquisition_ratio": float(env.mask.mean()),
            "steps": int(env.steps),
        })
        traces.append({
            "sample_idx": int(i),
            "true": env.label_names[int(env.y[i])],
            "pred": env.label_names[pred],
            "acquired": " -> ".join(acquired),
            "cost": env.total_cost,
        })
        if (k + 1) % 5000 == 0:
            LOG.info("Evaluated DQN %d/%d", k + 1, len(idx))

    df = pd.DataFrame(rows)
    met = macro_metrics(df.y_true.values, df.y_pred.values, len(env.label_names))
    summary = pd.DataFrame([{
        "method": method_name,
        **met,
        "normalized_cost": df.cost.mean(),
        "acquisition_ratio": df.acquisition_ratio.mean(),
        "mean_steps": df.steps.mean(),
    }])
    return summary, pd.DataFrame(traces)


def standalone_modality_utility(model, X, y, val_idx, modalities, costs) -> Dict[str, float]:
    utilities = {}
    for j, m in enumerate(modalities):
        yy, pp = [], []
        mask = np.zeros(len(modalities), dtype=np.float32)
        mask[j] = 1.0
        for i in val_idx:
            pred, _, _, _ = diagnose_with_mask(model, X, y, modalities, int(i), mask)
            yy.append(int(y[i])); pp.append(pred)
        f1 = f1_score(yy, pp, labels=np.arange(model.classifier[-1].out_features), average="macro", zero_division=0)
        utilities[m] = f1 / max(costs[m], 1e-8)
        LOG.info("Standalone modality utility %-20s F1=%.4f cost=%.4f utility=%.4f", m, f1, costs[m], utilities[m])
    return utilities


def evaluate_heuristic_policy(model, bundle: DatasetBundle, split: SplitBundle, idx: np.ndarray,
                              strategy: str, utilities: Optional[Dict[str, float]] = None,
                              method_name: Optional[str] = None) -> pd.DataFrame:
    modalities = list(bundle.X.keys())
    M = len(modalities)
    costs_arr = np.array([bundle.modality_costs[m] for m in modalities])
    rng = np.random.default_rng(CFG.seed + 123)
    rows = []

    if strategy == "fixed_cost":
        base_order = list(np.argsort(costs_arr))
    elif strategy == "utility":
        assert utilities is not None
        base_order = sorted(range(M), key=lambda j: utilities[modalities[j]], reverse=True)
    elif strategy == "random":
        base_order = None
    else:
        raise ValueError(strategy)

    for i in idx:
        mask = np.zeros(M, dtype=np.float32)
        order = list(base_order) if base_order is not None else rng.permutation(M).tolist()
        total_cost = 0.0
        steps = 0
        pred = 0
        for action in order:
            mask[action] = 1.0
            total_cost += float(costs_arr[action])
            steps += 1
            pred, probs, z, ent = diagnose_with_mask(model, split.X_scaled, bundle.y, modalities, int(i), mask)
            if steps >= CFG.min_acquisitions_before_stop and ent <= CFG.entropy_stop_threshold:
                break
        rows.append((int(bundle.y[i]), pred, total_cost, mask.mean(), steps))

    arr = pd.DataFrame(rows, columns=["y_true", "y_pred", "cost", "ratio", "steps"])
    met = macro_metrics(arr.y_true, arr.y_pred, len(bundle.label_names))
    return pd.DataFrame([{
        "method": method_name or strategy,
        **met,
        "normalized_cost": arr.cost.mean(),
        "acquisition_ratio": arr.ratio.mean(),
        "mean_steps": arr.steps.mean(),
    }])


# ============================================================
# 11. FULL TELEMETRY FUSION BASELINE + PER-CLASS REPORT
# ============================================================
def full_fusion_result(model, bundle: DatasetBundle, split: SplitBundle, idx: np.ndarray, method="Fusion-Full"):
    yt, yp, prob, z = evaluate_classifier(model, split.X_scaled, bundle.y, idx, list(bundle.X.keys()), "full")
    met = macro_metrics(yt, yp, len(bundle.label_names))
    row = pd.DataFrame([{
        "method": method,
        **met,
        "normalized_cost": 1.0,
        "acquisition_ratio": 1.0,
        "mean_steps": len(bundle.X),
    }])
    report = classification_report(yt, yp, labels=np.arange(len(bundle.label_names)), target_names=bundle.label_names,
                                   zero_division=0, output_dict=True)
    return row, pd.DataFrame(report).T, confusion_matrix(yt, yp), (yt, yp, prob, z)


# ============================================================
# 12. OPEN-SET / LEAVE-ONE-ATTACK-OUT
# ============================================================
def fit_class_prototypes(z: np.ndarray, y: np.ndarray, num_classes: int, use_mahalanobis: bool = True):
    means = []
    inv_covs = []
    for c in range(num_classes):
        Zc = z[y == c]
        if len(Zc) == 0:
            raise ValueError(f"No prototype samples for class {c}")
        mu = Zc.mean(axis=0)
        means.append(mu)
        if use_mahalanobis:
            if len(Zc) < 3:
                cov = np.eye(z.shape[1], dtype=np.float32)
            else:
                cov = np.cov(Zc.T)
                cov = cov + np.eye(cov.shape[0]) * 1e-3
            inv_covs.append(np.linalg.pinv(cov).astype(np.float32))
        else:
            inv_covs.append(None)
    return np.stack(means), inv_covs


def prototype_distance(z: np.ndarray, means: np.ndarray, inv_covs: List[Optional[np.ndarray]]) -> np.ndarray:
    all_d = []
    for c in range(len(means)):
        d = z - means[c]
        if inv_covs[c] is None:
            dist = np.sqrt(np.sum(d * d, axis=1) + 1e-12)
        else:
            dist = np.sqrt(np.maximum(np.einsum("bi,ij,bj->b", d, inv_covs[c], d), 0.0) + 1e-12)
        all_d.append(dist)
    return np.min(np.stack(all_d, axis=1), axis=1)


def build_loao_mapping(bundle: DatasetBundle, holdout_name: str):
    if holdout_name not in bundle.label_names:
        raise ValueError(f"Holdout '{holdout_name}' not found in labels: {bundle.label_names}")
    hold_id = bundle.label_names.index(holdout_name)
    known_old = [c for c in range(len(bundle.label_names)) if c != hold_id]
    old_to_new = {old: new for new, old in enumerate(known_old)}
    new_names = [bundle.label_names[c] for c in known_old]
    y_new = np.full_like(bundle.y, -1)
    for old, new in old_to_new.items():
        y_new[bundle.y == old] = new
    return hold_id, known_old, old_to_new, new_names, y_new


def run_single_loao(bundle: DatasetBundle, split: SplitBundle, holdout_name: str) -> pd.DataFrame:
    LOG.info("\n%s", "#" * 80)
    LOG.info("OPEN-SET LOAO | dataset=%s holdout=%s", bundle.name, holdout_name)
    hold_id, known_old, old_to_new, known_names, y_new = build_loao_mapping(bundle, holdout_name)

    # Strictly remove unknown class from train AND validation threshold fit.
    tr = split.train_idx[bundle.y[split.train_idx] != hold_id]
    va = split.val_idx[bundle.y[split.val_idx] != hold_id]
    te = split.test_idx.copy()  # includes known + unknown if split has both
    if len(tr) == 0 or len(va) == 0:
        raise ValueError("LOAO train/val has no known samples")
    if np.sum(bundle.y[te] == hold_id) == 0:
        LOG.warning("LOAO test split contains zero holdout samples due to group split. Result will be skipped.")
        return pd.DataFrame()

    # Some known classes could disappear from train due to group split.
    present_new = sorted(np.unique(y_new[tr]).tolist())
    if present_new != list(range(len(known_names))):
        LOG.warning("LOAO training does not contain every known class. present=%s expected=%s", present_new, list(range(len(known_names))))

    # We need a lightweight temporary bundle definition only for class names.
    tmp_bundle = copy.copy(bundle)
    tmp_bundle.label_names = known_names

    model = train_fusion_classifier(
        tmp_bundle, split,
        epochs=CFG.openset_classifier_epochs,
        train_idx_override=tr,
        val_idx_override=va,
        y_override=y_new,
        num_classes_override=len(known_names),
        model_tag=f"loao_{holdout_name}",
    )

    modalities = list(bundle.X.keys())
    # Get full-evidence latent embeddings for known train/val sets.
    _, _, _, ztr = evaluate_classifier(model, split.X_scaled, y_new, tr, modalities, "full")
    _, _, _, zva = evaluate_classifier(model, split.X_scaled, y_new, va, modalities, "full")
    ytr_new = y_new[tr]
    yva_new = y_new[va]
    means, inv_covs = fit_class_prototypes(ztr, ytr_new, len(known_names), CFG.openset_use_mahalanobis)
    val_dist = prototype_distance(zva, means, inv_covs)
    threshold = float(np.quantile(val_dist, CFG.openset_quantile))
    LOG.info("LOAO prototype threshold q=%.3f => %.6f", CFG.openset_quantile, threshold)

    # DQN is trained only on known training samples/classes.
    env = ActiveEvidenceEnv(model, split.X_scaled, y_new, modalities, bundle.modality_costs, known_names, CFG.device)
    policy = train_dqn_policy(env, tr, episodes=CFG.openset_dqn_episodes, tag=f"loao_{holdout_name}")

    # Evaluate each test event with active policy and then use the acquired-mask
    # embedding for open-set distance. Unknown y_new=-1 is never passed to env.step,
    # so we implement inference manually here.
    rows = []
    for i in te:
        state = env.reset(int(i))
        # env.y has -1 on unknown; do NOT call terminal env.step on unknown because
        # reward logic indexes invalid class. Instead emulate acquisition decisions.
        env.mask[:] = 0
        env.total_cost = 0.0
        env.steps = 0
        acquired_steps = 0
        while True:
            state = env._state()
            valid = env.valid_actions()
            with torch.no_grad():
                q = policy(torch.from_numpy(state).float().unsqueeze(0).to(CFG.device))[0].cpu().numpy()
            action = masked_argmax(q, valid)
            if action == env.stop_action:
                break
            env.mask[action] = 1.0
            env.total_cost += float(env.costs[action])
            env.steps += 1
            acquired_steps += 1
            if env.mask.sum() >= env.M:
                break

        pred_known, probs, z, entropy = env._diagnosis()
        dist = float(prototype_distance(z[None, :], means, inv_covs)[0])
        pred_unknown = dist > threshold
        true_unknown = int(bundle.y[i] == hold_id)
        rows.append({
            "sample_idx": int(i),
            "true_unknown": true_unknown,
            "pred_unknown": int(pred_unknown),
            "distance": dist,
            "known_pred": pred_known,
            "cost": env.total_cost,
            "ratio": env.mask.mean(),
            "steps": env.steps,
        })

    out = pd.DataFrame(rows)
    if out.true_unknown.nunique() < 2:
        auroc = np.nan; auprc = np.nan
    else:
        auroc = roc_auc_score(out.true_unknown, out.distance)
        auprc = average_precision_score(out.true_unknown, out.distance)
    unknown_f1 = f1_score(out.true_unknown, out.pred_unknown, zero_division=0)
    row = pd.DataFrame([{
        "dataset": bundle.name,
        "holdout_attack": holdout_name,
        "unknown_auroc": auroc,
        "unknown_auprc": auprc,
        "unknown_f1": unknown_f1,
        "threshold": threshold,
        "normalized_cost": out.cost.mean(),
        "acquisition_ratio": out.ratio.mean(),
        "mean_steps": out.steps.mean(),
        "n_unknown_test": int(out.true_unknown.sum()),
        "n_known_test": int((1 - out.true_unknown).sum()),
    }])
    LOG.info("LOAO result:\n%s", row.to_string(index=False))
    out.to_csv(WORK / f"{bundle.name}_loao_{holdout_name}_samples.csv", index=False)
    return row


# ============================================================
# 12B. ABLATIONS
# ============================================================
def evaluate_static_mask(model, bundle: DatasetBundle, split: SplitBundle, idx: np.ndarray,
                         selected_modalities: List[str], method_name: str) -> pd.DataFrame:
    modalities = list(bundle.X.keys())
    mask = np.array([1.0 if m in selected_modalities else 0.0 for m in modalities], dtype=np.float32)
    if mask.sum() == 0:
        raise ValueError("Static-mask ablation selected no modalities")
    yt, yp = [], []
    for i in idx:
        pred, _, _, _ = diagnose_with_mask(model, split.X_scaled, bundle.y, modalities, int(i), mask)
        yt.append(int(bundle.y[i])); yp.append(pred)
    met = macro_metrics(np.asarray(yt), np.asarray(yp), len(bundle.label_names))
    cost = sum(bundle.modality_costs[m] for m in selected_modalities)
    return pd.DataFrame([{
        "method": method_name,
        **met,
        "normalized_cost": cost,
        "acquisition_ratio": len(selected_modalities) / len(modalities),
        "mean_steps": len(selected_modalities),
    }])


def run_ablations(bundle: DatasetBundle, split: SplitBundle, main_fusion: ActiveFusionClassifier) -> pd.DataFrame:
    """Publication-oriented ablations.

    To keep the default notebook practical, CFG.run_ablations is False. For the
    final experiment set, turn it on. This executes actual re-training for the
    two reward/model ablations instead of only masking outputs.
    """
    modalities = list(bundle.X.keys())
    rows = []

    # A) Physical-like vs cyber/log-like evidence only.
    physical_like = [m for m in modalities if any(k in m.lower() for k in ["pmu", "physical", "rtds"])]
    cyber_like = [m for m in modalities if any(k in m.lower() for k in ["cyber", "network", "resource", "log"])]
    if physical_like:
        rows.append(evaluate_static_mask(main_fusion, bundle, split, split.test_idx, physical_like, "Ablation-PhysicalOnly"))
    if cyber_like:
        rows.append(evaluate_static_mask(main_fusion, bundle, split, split.test_idx, cyber_like, "Ablation-CyberOnly"))

    # B) No modality dropout: re-train classifier with all modalities present in
    # training except rare empty examples disabled as well.
    old_drop, old_empty = CFG.modality_dropout_p, CFG.probability_all_missing
    try:
        CFG.modality_dropout_p = 0.0
        CFG.probability_all_missing = 0.0
        no_drop_model = train_fusion_classifier(bundle, split, epochs=max(10, CFG.clf_epochs // 2), model_tag="ablation_no_modality_dropout")
        full, _, _, _ = full_fusion_result(no_drop_model, bundle, split, split.test_idx, method="Ablation-NoModalityDropout-Full")
        rows.append(full)
    finally:
        CFG.modality_dropout_p, CFG.probability_all_missing = old_drop, old_empty

    # C) DQN without acquisition-cost penalty.
    old_lambda = CFG.acquisition_lambda
    try:
        CFG.acquisition_lambda = 0.0
        env = ActiveEvidenceEnv(main_fusion, split.X_scaled, bundle.y, modalities, bundle.modality_costs, bundle.label_names, CFG.device)
        pol = train_dqn_policy(env, split.train_idx, episodes=CFG.ablation_dqn_episodes, tag="ablation_no_cost")
        row, _ = evaluate_dqn(env, pol, split.test_idx, "Ablation-DQN-NoCostPenalty")
        rows.append(row)
    finally:
        CFG.acquisition_lambda = old_lambda

    # D) DQN without step/delay penalty.
    old_step = CFG.step_penalty
    try:
        CFG.step_penalty = 0.0
        env = ActiveEvidenceEnv(main_fusion, split.X_scaled, bundle.y, modalities, bundle.modality_costs, bundle.label_names, CFG.device)
        pol = train_dqn_policy(env, split.train_idx, episodes=CFG.ablation_dqn_episodes, tag="ablation_no_step")
        row, _ = evaluate_dqn(env, pol, split.test_idx, "Ablation-DQN-NoStepPenalty")
        rows.append(row)
    finally:
        CFG.step_penalty = old_step

    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    out.to_csv(WORK / f"{bundle.name}_ablations.csv", index=False)
    LOG.info("ABLATION RESULTS %s:\n%s", bundle.name, out.to_string(index=False))
    return out


# ============================================================
# 13. PLOTS
# ============================================================
def plot_pareto(results: pd.DataFrame, dataset_name: str):
    if results.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 6))
    for _, r in results.iterrows():
        ax.scatter(r["normalized_cost"], r["macro_f1"], s=70)
        ax.annotate(r["method"], (r["normalized_cost"], r["macro_f1"]), xytext=(5, 4), textcoords="offset points", fontsize=8)
    ax.set_xlabel("Normalized telemetry acquisition cost (lower is better)")
    ax.set_ylabel("Macro-F1 (higher is better)")
    ax.set_title(f"{dataset_name}: performance-cost trade-off")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    p = WORK / f"{dataset_name}_pareto.png"
    fig.savefig(p, dpi=180)
    plt.close(fig)
    LOG.info("Saved Pareto plot: %s", p)


def plot_confusion(cm: np.ndarray, names: List[str], dataset_name: str):
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm)
    fig.colorbar(im, ax=ax)
    ax.set_xticks(np.arange(len(names))); ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(names))); ax.set_yticklabels(names)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(f"{dataset_name}: full-telemetry fusion confusion matrix")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=8)
    fig.tight_layout()
    p = WORK / f"{dataset_name}_confusion.png"
    fig.savefig(p, dpi=180)
    plt.close(fig)


# ============================================================
# 14. CORE END-TO-END EXPERIMENT FOR ONE DATASET
# ============================================================
def run_dataset_experiment(bundle: DatasetBundle, holdout_attack: Optional[str] = None):
    LOG.info("\n%s", "=" * 100)
    LOG.info("START EXPERIMENT: %s", bundle.name)
    LOG.info("%s", "=" * 100)

    split = group_split(bundle, CFG.seed)
    if CFG.run_leakage_audit:
        run_leakage_audit(bundle, split)
    modalities = list(bundle.X.keys())

    # Save split metadata for exact reproducibility.
    split_df = pd.DataFrame({
        "idx": np.arange(len(bundle.y)),
        "group": bundle.groups,
        "label_id": bundle.y,
        "label": [bundle.label_names[x] for x in bundle.y],
        "split": "",
    })
    split_df.loc[split.train_idx, "split"] = "train"
    split_df.loc[split.val_idx, "split"] = "val"
    split_df.loc[split.test_idx, "split"] = "test"
    split_df.to_csv(WORK / f"{bundle.name}_split.csv", index=False)

    # 1) Classical baselines.
    classical = run_classical_baselines(bundle, split)

    # 2) Main missing-modality-aware fusion classifier.
    fusion = train_fusion_classifier(bundle, split, model_tag=bundle.name)
    full_row, report, cm, full_outputs = full_fusion_result(fusion, bundle, split, split.test_idx)
    report.to_csv(WORK / f"{bundle.name}_full_classification_report.csv")
    np.savetxt(WORK / f"{bundle.name}_full_confusion.csv", cm, delimiter=",", fmt="%d")
    if CFG.save_plots:
        plot_confusion(cm, bundle.label_names, bundle.name)

    # 3) Active heuristic baselines.
    utilities = standalone_modality_utility(
        fusion, split.X_scaled, bundle.y, split.val_idx, modalities, bundle.modality_costs
    )
    random_row = evaluate_heuristic_policy(fusion, bundle, split, split.test_idx, "random", utilities, "Active-Random")
    fixed_row = evaluate_heuristic_policy(fusion, bundle, split, split.test_idx, "fixed_cost", utilities, "Active-CostOrder")
    utility_row = evaluate_heuristic_policy(fusion, bundle, split, split.test_idx, "utility", utilities, "Active-UtilityEntropy")

    # 4) DQN policy.
    env = ActiveEvidenceEnv(fusion, split.X_scaled, bundle.y, modalities, bundle.modality_costs, bundle.label_names, CFG.device)
    policy = train_dqn_policy(env, split.train_idx, CFG.dqn_episodes, tag=bundle.name)
    dqn_row, traces = evaluate_dqn(env, policy, split.test_idx, "ActiGrid-DQN")
    traces.to_csv(WORK / f"{bundle.name}_dqn_traces.csv", index=False)

    # 5) Merge results.
    results = pd.concat([classical, full_row, random_row, fixed_row, utility_row, dqn_row], ignore_index=True)
    results = results.sort_values(["macro_f1", "normalized_cost"], ascending=[False, True])
    results.to_csv(WORK / f"{bundle.name}_results.csv", index=False)
    LOG.info("\nFINAL MAIN RESULTS: %s\n%s", bundle.name, results.to_string(index=False))

    if CFG.save_plots:
        plot_pareto(results, bundle.name)

    # Optional publication-grade ablations (computationally expensive).
    if CFG.run_ablations:
        try:
            run_ablations(bundle, split, fusion)
        except Exception as e:
            LOG.exception("Ablation suite failed for %s: %s", bundle.name, e)

    # Save models / scalers.
    if CFG.save_models:
        torch.save({
            "model_state": fusion.state_dict(),
            "modalities": modalities,
            "label_names": bundle.label_names,
            "input_dims": {m: split.X_scaled[m].shape[1] for m in modalities},
            "costs": bundle.modality_costs,
            "config": asdict(CFG),
        }, WORK / f"{bundle.name}_fusion.pt")
        torch.save({
            "policy_state": policy.state_dict(),
            "state_dim": env.state_dim,
            "action_dim": env.action_dim,
            "modalities": modalities,
        }, WORK / f"{bundle.name}_dqn.pt")
        import pickle
        with open(WORK / f"{bundle.name}_scalers.pkl", "wb") as f:
            pickle.dump(split.scalers, f)

    # 6) Open set.
    loao = pd.DataFrame()
    if CFG.run_open_set and holdout_attack:
        try:
            loao = run_single_loao(bundle, split, holdout_attack)
            loao.to_csv(WORK / f"{bundle.name}_loao_summary.csv", index=False)
        except Exception as e:
            LOG.exception("Open-set evaluation failed for %s: %s", bundle.name, e)

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return results, loao


# ============================================================
# 15. OPTIONAL UAH EMS NORMAL-OPERATION STRESS TEST
# ============================================================
def load_ems_numeric_windows(root: Path, window_rows: int = 50) -> Tuple[np.ndarray, List[str]]:
    """Generic EMS log numeric/categorical aggregation.

    This is intentionally conservative because UAH EMS files can be distributed
    in different text/CSV forms. It creates fixed-row windows after one-hot-like
    frequency features for selected categorical columns and numeric statistics.

    Publication note: inspect the actual EMS schema and replace this generic
    function with schema-aware parsing before final results.
    """
    files = sorted(list(root.rglob("*.csv")) + list(root.rglob("*.txt")) + list(root.rglob("*.log")))
    if not files:
        raise FileNotFoundError(f"No EMS text/CSV files under {root}")
    p = files[0]
    LOG.info("EMS stress loader using: %s", p)
    try:
        df = pd.read_csv(p)
    except Exception:
        df = pd.read_csv(p, sep=None, engine="python")
    df.columns = [str(c).strip() for c in df.columns]
    LOG.info("EMS columns: %s", df.columns.tolist())

    # Numeric statistics + stable categorical hash frequencies.
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in df.columns if c not in num_cols][:8]
    rows = []
    names = []
    for start in range(0, len(df), window_rows):
        w = df.iloc[start:start + window_rows]
        if len(w) < max(5, window_rows // 4):
            continue
        feats = []
        feat_names = []
        for c in num_cols:
            x = pd.to_numeric(w[c], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0)
            for stat, val in [("mean", x.mean()), ("std", x.std()), ("min", x.min()), ("max", x.max())]:
                feats.append(float(0 if pd.isna(val) else val)); feat_names.append(f"{c}__{stat}")
        for c in cat_cols:
            vals = w[c].astype(str)
            feats.extend([vals.nunique(), vals.value_counts(normalize=True).iloc[0] if len(vals) else 0])
            feat_names.extend([f"{c}__nunique", f"{c}__topfreq"])
        rows.append(feats)
        names = feat_names
    return np.asarray(rows, dtype=np.float32), names



# ============================================================
# 15B. V6 FINAL PARETO + MULTI-SEED PROTOCOL
# ============================================================
def _summary_from_policy_rows(rows: pd.DataFrame, label_names: List[str], method: str) -> pd.DataFrame:
    met = macro_metrics(rows["y_true"].to_numpy(), rows["y_pred"].to_numpy(), len(label_names))
    return pd.DataFrame([{
        "method": method,
        **met,
        "normalized_cost": float(rows["cost"].mean()),
        "acquisition_ratio": float(rows["ratio"].mean()),
        "mean_steps": float(rows["steps"].mean()),
    }])


def evaluate_full_detailed(model, bundle: DatasetBundle, split: SplitBundle, idx: np.ndarray,
                           method: str = "Fusion-Full") -> Tuple[pd.DataFrame, pd.DataFrame]:
    modalities = list(bundle.X.keys())
    mask = np.ones(len(modalities), dtype=np.float32)
    rows = []
    for i in idx:
        pred, probs, z, ent = diagnose_with_mask(model, split.X_scaled, bundle.y, modalities, int(i), mask)
        rows.append({
            "sample_idx": int(i), "y_true": int(bundle.y[i]), "y_pred": int(pred),
            "cost": 1.0, "ratio": 1.0, "steps": len(modalities), "entropy": float(ent),
            "group": str(bundle.groups[i]),
        })
    rows = pd.DataFrame(rows)
    return _summary_from_policy_rows(rows, bundle.label_names, method), rows


def evaluate_heuristic_detailed(model, bundle: DatasetBundle, split: SplitBundle, idx: np.ndarray,
                                strategy: str, utilities: Optional[Dict[str, float]] = None,
                                entropy_threshold: Optional[float] = None,
                                max_budget: Optional[float] = None,
                                rng_seed: Optional[int] = None,
                                method_name: Optional[str] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    modalities = list(bundle.X.keys())
    M = len(modalities)
    costs = np.array([bundle.modality_costs[m] for m in modalities], dtype=np.float64)
    rng = np.random.default_rng(CFG.seed + 123 if rng_seed is None else rng_seed)

    if strategy == "fixed_cost":
        base_order = list(np.argsort(costs))
    elif strategy == "utility":
        if utilities is None:
            raise ValueError("utilities required for utility strategy")
        base_order = sorted(range(M), key=lambda j: utilities[modalities[j]], reverse=True)
    elif strategy == "random":
        base_order = None
    else:
        raise ValueError(strategy)

    out = []
    for i in idx:
        mask = np.zeros(M, dtype=np.float32)
        order = list(base_order) if base_order is not None else rng.permutation(M).tolist()
        total_cost = 0.0
        steps = 0
        pred, ent = 0, float("inf")
        acquired = []

        for action in order:
            c = float(costs[action])
            # Hard maximum budget. Always permit at least one evidence source.
            if max_budget is not None and steps > 0 and total_cost + c > max_budget + 1e-9:
                break
            mask[action] = 1.0
            total_cost += c
            steps += 1
            acquired.append(modalities[action])
            pred, probs, z, ent = diagnose_with_mask(
                model, split.X_scaled, bundle.y, modalities, int(i), mask
            )
            if entropy_threshold is not None and steps >= CFG.min_acquisitions_before_stop and ent <= entropy_threshold:
                break
            if max_budget is not None and total_cost >= max_budget - 1e-9:
                break

        out.append({
            "sample_idx": int(i), "y_true": int(bundle.y[i]), "y_pred": int(pred),
            "cost": float(total_cost), "ratio": float(mask.mean()), "steps": int(steps),
            "entropy": float(ent), "group": str(bundle.groups[i]),
            "acquired": " -> ".join(acquired),
        })

    rows = pd.DataFrame(out)
    return _summary_from_policy_rows(rows, bundle.label_names, method_name or strategy), rows


def evaluate_dqn_detailed(env: ActiveEvidenceEnv, policy: DQN, idx: np.ndarray,
                          method_name: str = "ActiGrid-DQN") -> Tuple[pd.DataFrame, pd.DataFrame]:
    policy.eval()
    out = []
    for k, i in enumerate(idx):
        state = env.reset(int(i))
        acquired = []
        terminal_info = None
        while True:
            valid = env.valid_actions()
            with torch.no_grad():
                q = policy(torch.from_numpy(state).float().unsqueeze(0).to(CFG.device))[0].cpu().numpy()
            action = masked_argmax(q, valid)
            if action < env.M:
                acquired.append(env.modalities[action])
            state, reward, done, info = env.step(action)
            if done:
                terminal_info = info
                break
        pred = int(terminal_info["pred"])
        out.append({
            "sample_idx": int(i), "y_true": int(env.y[i]), "y_pred": pred,
            "cost": float(env.total_cost), "ratio": float(env.mask.mean()), "steps": int(env.steps),
            "group": str(getattr(env, "groups", [""] * len(env.y))[i]) if hasattr(env, "groups") else "",
            "acquired": " -> ".join(acquired),
        })
        if (k + 1) % 5000 == 0:
            LOG.info("Evaluated final DQN %d/%d", k + 1, len(idx))
    rows = pd.DataFrame(out)
    return _summary_from_policy_rows(rows, env.label_names, method_name), rows


def _attach_groups_to_env(env: ActiveEvidenceEnv, bundle: DatasetBundle):
    env.groups = np.asarray(bundle.groups).astype(str)
    return env


def _budget_grid(bundle: DatasetBundle) -> List[float]:
    costs = sorted(float(v) for v in bundle.modality_costs.values())
    grid = []
    s = 0.0
    for c in costs:
        s += c
        grid.append(min(1.0, float(s)))
    return sorted(set(round(x, 6) for x in grid))


def _nearest_feasible_budget(bundle: DatasetBundle, target_cost: float) -> float:
    """Nearest discrete cumulative acquisition budget to an adaptive policy cost.

    Under the unit-cost protocol this maps TU ~0.61/0.67 -> 2/3 and
    MSU ~0.77/0.79 -> 4/5, avoiding the old floor-to-one/three-modality bias.
    """
    grid = _budget_grid(bundle)
    return float(min(grid, key=lambda b: (abs(float(b) - float(target_cost)), -float(b))))


def _select_dqn_operating_point(sweep: pd.DataFrame, full_val_f1: float) -> Tuple[float, str]:
    d = sweep[sweep["split"] == "val"].copy()
    target = full_val_f1 - CFG.final_f1_tolerance
    eligible = d[d["macro_f1"] >= target]
    if len(eligible):
        chosen = eligible.sort_values(["normalized_cost", "macro_f1"], ascending=[True, False]).iloc[0]
        reason = f"min validation cost with Macro-F1 >= full_val_f1-{CFG.final_f1_tolerance:.3f}"
    else:
        d["selection_utility"] = d["macro_f1"] - CFG.final_lambda_fallback_cost_weight * d["normalized_cost"]
        chosen = d.sort_values(["selection_utility", "macro_f1"], ascending=[False, False]).iloc[0]
        reason = "fallback max(validation Macro-F1 - cost_weight*cost)"
    return float(chosen["lambda"]), reason


def _choose_entropy_threshold(entropy_val: pd.DataFrame, target_cost: float) -> float:
    d = entropy_val.copy()
    d["cost_gap"] = (d["normalized_cost"] - target_cost).abs()
    # Prefer close budget match; break ties by higher F1.
    chosen = d.sort_values(["cost_gap", "macro_f1"], ascending=[True, False]).iloc[0]
    return float(chosen["threshold"])


def _save_class_report(rows: pd.DataFrame, label_names: List[str], path: Path):
    rep = classification_report(
        rows["y_true"].to_numpy(), rows["y_pred"].to_numpy(),
        labels=np.arange(len(label_names)), target_names=label_names,
        zero_division=0, output_dict=True,
    )
    pd.DataFrame(rep).T.to_csv(path)


def run_v7_sweep(bundle: DatasetBundle, split: SplitBundle) -> Dict[str, Any]:
    """Single-seed validation-controlled Pareto sweep.

    Selection is based ONLY on validation performance/cost. Test values are
    recorded for the final Pareto figure but never used to choose lambda or
    entropy threshold.
    """
    base_seed = int(CFG.final_split_seed)
    old_seed, old_lambda, old_step = CFG.seed, CFG.acquisition_lambda, CFG.step_penalty
    CFG.seed = base_seed
    CFG.step_penalty = 0.0
    seed_everything(base_seed)

    LOG.info("\n%s", "=" * 100)
    LOG.info("V6 STAGE A SWEEP: %s | seed=%d", bundle.name, base_seed)
    LOG.info("%s", "=" * 100)

    fusion = train_fusion_classifier(bundle, split, model_tag=f"{bundle.name}_v7_sweep")
    full_val, full_val_rows = evaluate_full_detailed(fusion, bundle, split, split.val_idx, "Fusion-Full")
    full_test, full_test_rows = evaluate_full_detailed(fusion, bundle, split, split.test_idx, "Fusion-Full")
    full_val_f1 = float(full_val.iloc[0]["macro_f1"])

    utilities = standalone_modality_utility(
        fusion, split.X_scaled, bundle.y, split.val_idx,
        list(bundle.X.keys()), bundle.modality_costs,
    )

    sweep_rows = []
    # Full point.
    for split_name, row in [("val", full_val), ("test", full_test)]:
        rr = row.iloc[0].to_dict(); rr.update({"family":"full", "split":split_name, "lambda":np.nan, "threshold":np.nan, "budget":1.0})
        sweep_rows.append(rr)

    # Entropy-threshold sweep using utility ordering.
    entropy_val_records = []
    for thr in CFG.final_entropy_grid:
        val_sum, _ = evaluate_heuristic_detailed(
            fusion, bundle, split, split.val_idx, "utility", utilities,
            entropy_threshold=float(thr), method_name=f"UtilityEntropy@{thr:.2f}", rng_seed=base_seed + 7,
        )
        test_sum, _ = evaluate_heuristic_detailed(
            fusion, bundle, split, split.test_idx, "utility", utilities,
            entropy_threshold=float(thr), method_name=f"UtilityEntropy@{thr:.2f}", rng_seed=base_seed + 7,
        )
        for split_name, row in [("val", val_sum), ("test", test_sum)]:
            rr = row.iloc[0].to_dict(); rr.update({"family":"entropy", "split":split_name, "lambda":np.nan, "threshold":float(thr), "budget":np.nan})
            sweep_rows.append(rr)
            if split_name == "val": entropy_val_records.append(rr)

    # Fixed hard-budget baselines create an explicit cost curve.
    for budget in _budget_grid(bundle):
        for strategy, name in [("random", "RandomBudget"), ("fixed_cost", "CostOrderBudget"), ("utility", "UtilityBudget")]:
            val_sum, _ = evaluate_heuristic_detailed(
                fusion, bundle, split, split.val_idx, strategy, utilities,
                entropy_threshold=None, max_budget=float(budget), rng_seed=base_seed + 11,
                method_name=f"{name}@{budget:.3f}",
            )
            test_sum, _ = evaluate_heuristic_detailed(
                fusion, bundle, split, split.test_idx, strategy, utilities,
                entropy_threshold=None, max_budget=float(budget), rng_seed=base_seed + 11,
                method_name=f"{name}@{budget:.3f}",
            )
            for split_name, row in [("val", val_sum), ("test", test_sum)]:
                rr = row.iloc[0].to_dict(); rr.update({"family":name, "split":split_name, "lambda":np.nan, "threshold":np.nan, "budget":float(budget)})
                sweep_rows.append(rr)

    # DQN lambda sweep. No explicit step penalty in v6.
    sweep_episodes = 250 if CFG.final_quick_mode else int(CFG.final_sweep_dqn_episodes)
    dqn_sweep_records = []
    for j, lam in enumerate(CFG.final_lambda_grid):
        CFG.acquisition_lambda = float(lam)
        train_seed = base_seed + 1000 + j * 37
        CFG.seed = train_seed
        seed_everything(train_seed)
        env = _attach_groups_to_env(ActiveEvidenceEnv(
            fusion, split.X_scaled, bundle.y, list(bundle.X.keys()), bundle.modality_costs,
            bundle.label_names, CFG.device,
        ), bundle)
        policy = train_dqn_policy(env, split.train_idx, sweep_episodes, tag=f"{bundle.name}_lambda_{lam:.2f}")
        val_sum, _ = evaluate_dqn_detailed(env, policy, split.val_idx, f"DQN-lambda-{lam:.2f}")
        test_sum, _ = evaluate_dqn_detailed(env, policy, split.test_idx, f"DQN-lambda-{lam:.2f}")
        for split_name, row in [("val", val_sum), ("test", test_sum)]:
            rr = row.iloc[0].to_dict(); rr.update({"family":"dqn", "split":split_name, "lambda":float(lam), "threshold":np.nan, "budget":np.nan})
            sweep_rows.append(rr)
            if split_name == "val": dqn_sweep_records.append(rr)
        del policy, env
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    sweep = pd.DataFrame(sweep_rows)
    sweep.to_csv(WORK / f"{bundle.name}_v7_pareto_sweep.csv", index=False)

    dqn_val = pd.DataFrame(dqn_sweep_records)
    selected_lambda, reason = _select_dqn_operating_point(dqn_val, full_val_f1)
    selected_dqn_val = dqn_val[np.isclose(dqn_val["lambda"], selected_lambda)].sort_values("macro_f1", ascending=False).iloc[0]
    target_cost = float(selected_dqn_val["normalized_cost"])
    entropy_val = pd.DataFrame(entropy_val_records)
    selected_threshold = _choose_entropy_threshold(entropy_val, target_cost)

    selection = {
        "dataset": bundle.name,
        "split_seed": base_seed,
        "selected_lambda": selected_lambda,
        "selected_entropy_threshold": selected_threshold,
        "target_validation_cost": target_cost,
        "full_validation_macro_f1": full_val_f1,
        "selection_reason": reason,
        "step_penalty": 0.0,
        "sweep_episodes": sweep_episodes,
    }
    with open(WORK / f"{bundle.name}_v7_selection.json", "w") as f:
        json.dump(selection, f, indent=2)
    LOG.info("V6 selection %s: lambda=%.3f entropy_thr=%.3f target_val_cost=%.4f | %s",
             bundle.name, selected_lambda, selected_threshold, target_cost, reason)

    # One classical baseline pass is enough; these are not the stochastic policy claim.
    if CFG.final_run_classical_once:
        CFG.seed = base_seed; seed_everything(base_seed)
        classical = run_classical_baselines(bundle, split)
        classical.to_csv(WORK / f"{bundle.name}_v7_classical_once.csv", index=False)

    # Sweep Pareto plot (test only).
    if CFG.save_plots:
        test_sweep = sweep[sweep["split"] == "test"].copy()
        fig, ax = plt.subplots(figsize=(9, 6))
        for fam, g in test_sweep.groupby("family"):
            ax.plot(g["normalized_cost"], g["macro_f1"], marker="o", linestyle="-", label=fam)
        ax.set_xlabel("Normalized telemetry cost")
        ax.set_ylabel("Macro-F1")
        ax.set_title(f"{bundle.name}: v6 validation-selected Pareto sweep")
        ax.grid(True, alpha=0.25); ax.legend(fontsize=8)
        fig.tight_layout(); fig.savefig(WORK / f"{bundle.name}_v7_pareto_sweep.png", dpi=180); plt.close(fig)

    CFG.seed, CFG.acquisition_lambda, CFG.step_penalty = old_seed, old_lambda, old_step
    seed_everything(CFG.seed)
    return selection


def _aggregate_multiseed(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    from scipy.stats import t as student_t
    metrics = ["accuracy", "balanced_accuracy", "precision_macro", "recall_macro", "macro_f1",
               "normalized_cost", "acquisition_ratio", "mean_steps"]
    rows = []
    for method, g in df.groupby("method"):
        row = {"dataset": dataset_name, "method": method, "n_seeds": len(g)}
        for m in metrics:
            x = g[m].astype(float).to_numpy()
            n = len(x); mean = float(np.mean(x)); std = float(np.std(x, ddof=1)) if n > 1 else 0.0
            if n > 1:
                crit = float(student_t.ppf(0.975, df=n-1)); ci = crit * std / math.sqrt(n)
            else:
                ci = 0.0
            row[f"{m}_mean"] = mean; row[f"{m}_std"] = std; row[f"{m}_ci95"] = ci
        row["telemetry_saving_mean"] = 1.0 - row["normalized_cost_mean"]
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["macro_f1_mean", "normalized_cost_mean"], ascending=[False, True])


def _paired_seed_tests(seed_df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    from scipy.stats import wilcoxon
    pivot_f1 = seed_df.pivot(index="seed", columns="method", values="macro_f1")
    pivot_cost = seed_df.pivot(index="seed", columns="method", values="normalized_cost")
    focal = "ActiGrid-DQN-Final"
    comparators = ["UtilityEntropy-Matched", "UtilityNearestBudget", "RandomNearestBudget", "CostOrderNearestBudget", "Fusion-Full"]
    rows = []
    for comp in comparators:
        if focal not in pivot_f1.columns or comp not in pivot_f1.columns:
            continue
        common = pivot_f1[[focal, comp]].dropna()
        common_cost = pivot_cost[[focal, comp]].dropna()
        def safe_wilcoxon(a, b):
            d = np.asarray(a) - np.asarray(b)
            if len(d) < 2 or np.allclose(d, 0):
                return 1.0
            try:
                return float(wilcoxon(a, b, zero_method="wilcox", alternative="two-sided", mode="auto").pvalue)
            except Exception:
                return np.nan
        rows.append({
            "dataset": dataset_name, "focal": focal, "comparator": comp,
            "n_seed_pairs": len(common),
            "mean_delta_macro_f1": float((common[focal] - common[comp]).mean()),
            "wilcoxon_p_macro_f1": safe_wilcoxon(common[focal], common[comp]),
            "mean_delta_cost": float((common_cost[focal] - common_cost[comp]).mean()),
            "wilcoxon_p_cost": safe_wilcoxon(common_cost[focal], common_cost[comp]),
        })
    return pd.DataFrame(rows)


def run_v7_multiseed(bundle: DatasetBundle, split: SplitBundle, selection: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    old_seed, old_lambda, old_step = CFG.seed, CFG.acquisition_lambda, CFG.step_penalty
    CFG.step_penalty = 0.0
    selected_lambda = float(selection["selected_lambda"])
    selected_threshold = float(selection["selected_entropy_threshold"])
    target_budget = float(selection["target_validation_cost"])
    fair_budget = _nearest_feasible_budget(bundle, target_budget)
    seeds = (42, 123) if CFG.final_quick_mode else tuple(CFG.final_seeds)
    final_episodes = 500 if CFG.final_quick_mode else int(CFG.final_multiseed_dqn_episodes)

    LOG.info("\n%s", "=" * 100)
    LOG.info("V7 STAGE B MULTI-SEED: %s | seeds=%s | lambda=%.3f | entropy_thr=%.3f | dqn_target=%.4f | fair_discrete_budget=%.4f",
             bundle.name, seeds, selected_lambda, selected_threshold, target_budget, fair_budget)
    LOG.info("%s", "=" * 100)

    all_rows = []
    for seed in seeds:
        CFG.seed = int(seed); seed_everything(CFG.seed)
        LOG.info("V6 seed %d/%s: training fusion", seed, bundle.name)
        fusion = train_fusion_classifier(bundle, split, model_tag=f"{bundle.name}_seed_{seed}")
        utilities = standalone_modality_utility(
            fusion, split.X_scaled, bundle.y, split.val_idx,
            list(bundle.X.keys()), bundle.modality_costs,
        )

        full_sum, full_rows = evaluate_full_detailed(fusion, bundle, split, split.test_idx, "Fusion-Full")
        entropy_sum, entropy_rows = evaluate_heuristic_detailed(
            fusion, bundle, split, split.test_idx, "utility", utilities,
            entropy_threshold=selected_threshold, rng_seed=seed + 700,
            method_name="UtilityEntropy-Matched",
        )
        utility_budget_sum, utility_budget_rows = evaluate_heuristic_detailed(
            fusion, bundle, split, split.test_idx, "utility", utilities,
            max_budget=fair_budget, rng_seed=seed + 701,
            method_name="UtilityNearestBudget",
        )
        random_budget_sum, random_budget_rows = evaluate_heuristic_detailed(
            fusion, bundle, split, split.test_idx, "random", utilities,
            max_budget=fair_budget, rng_seed=seed + 702,
            method_name="RandomNearestBudget",
        )
        cost_budget_sum, cost_budget_rows = evaluate_heuristic_detailed(
            fusion, bundle, split, split.test_idx, "fixed_cost", utilities,
            max_budget=fair_budget, rng_seed=seed + 703,
            method_name="CostOrderNearestBudget",
        )

        CFG.acquisition_lambda = selected_lambda
        seed_everything(seed + 9000)
        env = _attach_groups_to_env(ActiveEvidenceEnv(
            fusion, split.X_scaled, bundle.y, list(bundle.X.keys()), bundle.modality_costs,
            bundle.label_names, CFG.device,
        ), bundle)
        policy = train_dqn_policy(env, split.train_idx, final_episodes, tag=f"{bundle.name}_final_seed_{seed}")
        dqn_sum, dqn_rows = evaluate_dqn_detailed(env, policy, split.test_idx, "ActiGrid-DQN-Final")

        for sm in [full_sum, entropy_sum, utility_budget_sum, random_budget_sum, cost_budget_sum, dqn_sum]:
            rr = sm.iloc[0].to_dict(); rr.update({"dataset":bundle.name, "seed":int(seed), "selected_lambda":selected_lambda,
                                                  "selected_entropy_threshold":selected_threshold, "target_budget":target_budget, "fair_discrete_budget":fair_budget})
            all_rows.append(rr)

        # Per-class reports for the representative seed used in the paper tables.
        if seed == seeds[0]:
            _save_class_report(full_rows, bundle.label_names, WORK / f"{bundle.name}_v7_seed{seed}_full_report.csv")
            _save_class_report(entropy_rows, bundle.label_names, WORK / f"{bundle.name}_v7_seed{seed}_entropy_report.csv")
            _save_class_report(dqn_rows, bundle.label_names, WORK / f"{bundle.name}_v7_seed{seed}_dqn_report.csv")
            if CFG.final_save_seed42_traces:
                dqn_rows.to_csv(WORK / f"{bundle.name}_v7_seed{seed}_dqn_samples.csv", index=False)
                entropy_rows.to_csv(WORK / f"{bundle.name}_v7_seed{seed}_entropy_samples.csv", index=False)

        if CFG.save_models and seed == seeds[0]:
            torch.save({"state": fusion.state_dict(), "seed":seed, "selection":selection}, WORK / f"{bundle.name}_v7_seed{seed}_fusion.pt")
            torch.save({"state": policy.state_dict(), "seed":seed, "selection":selection,
                        "state_dim":env.state_dim, "action_dim":env.action_dim}, WORK / f"{bundle.name}_v7_seed{seed}_dqn.pt")

        del policy, env, fusion
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    seed_df = pd.DataFrame(all_rows)
    seed_df.to_csv(WORK / f"{bundle.name}_v7_multiseed_raw.csv", index=False)
    agg = _aggregate_multiseed(seed_df, bundle.name)
    agg.to_csv(WORK / f"{bundle.name}_v7_multiseed_summary.csv", index=False)
    stats = _paired_seed_tests(seed_df, bundle.name)
    stats.to_csv(WORK / f"{bundle.name}_v7_paired_tests.csv", index=False)

    # Seed-paired deltas against full telemetry.
    full = seed_df[seed_df.method == "Fusion-Full"][["seed", "macro_f1", "normalized_cost"]].rename(
        columns={"macro_f1":"full_macro_f1", "normalized_cost":"full_cost"}
    )
    delta = seed_df.merge(full, on="seed", how="left")
    delta["macro_f1_loss_vs_full"] = delta["full_macro_f1"] - delta["macro_f1"]
    delta["telemetry_saving_vs_full"] = delta["full_cost"] - delta["normalized_cost"]
    delta.to_csv(WORK / f"{bundle.name}_v7_seed_deltas.csv", index=False)

    LOG.info("\nV7 MULTI-SEED SUMMARY %s:\n%s", bundle.name, agg.to_string(index=False))
    LOG.info("\nV7 PAIRED TESTS %s:\n%s", bundle.name, stats.to_string(index=False) if len(stats) else "<none>")

    if CFG.save_plots:
        fig, ax = plt.subplots(figsize=(9, 6))
        for _, r in agg.iterrows():
            ax.errorbar(r["normalized_cost_mean"], r["macro_f1_mean"],
                        xerr=r["normalized_cost_ci95"], yerr=r["macro_f1_ci95"],
                        marker="o", capsize=4)
            ax.annotate(r["method"], (r["normalized_cost_mean"], r["macro_f1_mean"]), xytext=(5,4), textcoords="offset points", fontsize=8)
        ax.set_xlabel("Mean normalized telemetry cost (95% CI)")
        ax.set_ylabel("Mean Macro-F1 (95% CI)")
        ax.set_title(f"{bundle.name}: five-seed final performance-cost trade-off")
        ax.grid(True, alpha=0.25); fig.tight_layout()
        fig.savefig(WORK / f"{bundle.name}_v7_multiseed_pareto.png", dpi=180); plt.close(fig)

    CFG.seed, CFG.acquisition_lambda, CFG.step_penalty = old_seed, old_lambda, old_step
    seed_everything(CFG.seed)
    return seed_df, agg, stats


def run_v7_final_protocol(bundle: DatasetBundle) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # Freeze one data split for all seeds so seed variation measures optimization/policy randomness,
    # not a changing test set.
    split = group_split(bundle, int(CFG.final_split_seed))
    if CFG.run_leakage_audit:
        run_leakage_audit(bundle, split)
    selection = run_v7_sweep(bundle, split)
    return run_v7_multiseed(bundle, split, selection)


# ============================================================
# 16. V6 FINAL MAIN
# ============================================================
def main():
    all_raw, all_summary, all_stats = [], [], []

    if not CFG.final_run_protocol:
        raise RuntimeError("v6 is intended for final_run_protocol=True")

    if CFG.run_msu:
        msu_root = discover_dataset_root("msu", CFG.msu_root_override)
        if msu_root:
            try:
                msu_root = maybe_extract_archives(msu_root, "msu")
                msu = load_msu_bundle(msu_root)
                raw, summary, stats = run_v7_final_protocol(msu)
                all_raw.append(raw); all_summary.append(summary)
                if len(stats): all_stats.append(stats)
            except Exception as e:
                LOG.exception("MSU v6 final protocol failed: %s", e)
        else:
            LOG.warning("MSU experiment skipped: dataset not found")

    if CFG.run_tudelft:
        tu_root = discover_dataset_root("tudelft", CFG.tudelft_root_override)
        if tu_root:
            try:
                tu_root = maybe_extract_archives(tu_root, "tudelft")
                tu = load_tudelft_bundle(tu_root)
                raw, summary, stats = run_v7_final_protocol(tu)
                all_raw.append(raw); all_summary.append(summary)
                if len(stats): all_stats.append(stats)
            except Exception as e:
                LOG.exception("TU Delft v6 final protocol failed: %s", e)
        else:
            LOG.warning("TU Delft experiment skipped: dataset not found")

    # Recovery-friendly merge: if the failed v6 run already produced MSU
    # combined CSVs in this Kaggle session, preserve those rows and replace
    # only TU Delft rows. This avoids losing completed MSU five-seed results.
    raw_path = WORK / "v7_all_multiseed_raw.csv"
    summary_path = WORK / "v7_all_multiseed_summary.csv"
    stats_path = WORK / "v7_all_paired_tests.csv"

    if all_raw:
        new_raw = pd.concat(all_raw, ignore_index=True)
        if raw_path.exists():
            try:
                old_raw = pd.read_csv(raw_path)
                if "dataset" in old_raw.columns:
                    old_raw = old_raw[~old_raw["dataset"].isin(new_raw["dataset"].unique())]
                    new_raw = pd.concat([old_raw, new_raw], ignore_index=True)
                    LOG.info("Recovery merge: preserved %d prior raw rows", len(old_raw))
            except Exception as e:
                LOG.warning("Could not merge prior raw results: %s", e)
        new_raw.to_csv(raw_path, index=False)

    if all_summary:
        final_summary = pd.concat(all_summary, ignore_index=True)
        if summary_path.exists():
            try:
                old_summary = pd.read_csv(summary_path)
                if "dataset" in old_summary.columns:
                    old_summary = old_summary[~old_summary["dataset"].isin(final_summary["dataset"].unique())]
                    final_summary = pd.concat([old_summary, final_summary], ignore_index=True)
                    LOG.info("Recovery merge: preserved %d prior summary rows", len(old_summary))
            except Exception as e:
                LOG.warning("Could not merge prior summary: %s", e)
        final_summary.to_csv(summary_path, index=False)
        LOG.info("\n%s\nV7 FINAL COMBINED MULTI-SEED SUMMARY\n%s\n%s", "="*110, final_summary.to_string(index=False), "="*110)

    if all_stats:
        final_stats = pd.concat(all_stats, ignore_index=True)
        if stats_path.exists():
            try:
                old_stats = pd.read_csv(stats_path)
                if "dataset" in old_stats.columns:
                    old_stats = old_stats[~old_stats["dataset"].isin(final_stats["dataset"].unique())]
                    final_stats = pd.concat([old_stats, final_stats], ignore_index=True)
                    LOG.info("Recovery merge: preserved %d prior paired-test rows", len(old_stats))
            except Exception as e:
                LOG.warning("Could not merge prior paired tests: %s", e)
        final_stats.to_csv(stats_path, index=False)
        LOG.info("\nV7 FINAL PAIRED TESTS\n%s", final_stats.to_string(index=False))

    shutil.make_archive(str(WORK), "zip", root_dir=str(WORK))
    LOG.info("Finished v7. Results directory: %s", WORK)
    LOG.info("Zipped results: %s.zip", WORK)
    LOG.info("DEBUG LOG: %s", WORK / "debug.log")


if __name__ == "__main__":
    setup()
    main()
