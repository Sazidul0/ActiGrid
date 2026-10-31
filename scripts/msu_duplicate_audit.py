"""Count exact duplicate MSU/ORNL rows, within files as well as across them.

The final reviewer-clearance audit (actigrid/final_reviewer_clearance.py) only
counted duplicate hashes that span more than one source file. This script
applies the same hash (all 128 predictors, before imputation) to every row, so
duplicates inside a single file are counted too.

Input is the raw, unimputed MSU cache written by the final audit
(<work_dir>/final_reviewer_clearance/cache/msu_raw_no_impute.joblib). It holds
dataset features, so it is not redistributed here. joblib unpickles the file,
so load only a cache that your own run produced.

Usage (from the repository root):
    python scripts/msu_duplicate_audit.py path/to/msu_raw_no_impute.joblib
Needs:  numpy, pandas, joblib
Writes: results/final_reviewer_clearance/MSU_ALL_DUPLICATE_GROUPS.csv
"""
import csv
import sys
from pathlib import Path

import joblib
import joblib.numpy_pickle as npk
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SPLITS = ROOT / "results" / "MSU_ORNL_Power_group_split_audit.csv"
OUT = ROOT / "results" / "final_reviewer_clearance" / "MSU_ALL_DUPLICATE_GROUPS.csv"


class _Record:
    """Stand-in for the audit's RawMSU dataclass, so the cache loads without
    importing the Kaggle script (which creates its output folders on import)."""

    def __setstate__(self, state):
        self.__dict__.update(state)


_find_class = npk.NumpyUnpickler.find_class


def _lenient_find_class(self, module, name):
    try:
        return _find_class(self, module, name)
    except (AttributeError, ImportError):
        return type(name, (_Record,), {})


npk.NumpyUnpickler.find_class = _lenient_find_class


def main(path):
    raw = joblib.load(path)
    full = np.concatenate([raw.X[m] for m in raw.X], axis=1)
    # Same hash as the final audit: NaNs hash consistently.
    h = pd.util.hash_pandas_object(pd.DataFrame(full), index=False).astype("uint64").to_numpy()
    with open(SPLITS, newline="") as f:
        split_of = {r["group"]: r["split"] for r in csv.DictReader(f)}
    d = pd.DataFrame({"hash": h, "file": raw.groups, "label": raw.y})
    d["split"] = d["file"].map(split_of)

    g = d.groupby("hash").agg(n=("hash", "size"), n_files=("file", "nunique"),
                              n_splits=("split", "nunique"), n_labels=("label", "nunique"))
    dup = g[g.n > 1]
    rows = []
    for hsh, s in dup.iterrows():
        members = d[d.hash == hsh]
        rows.append(dict(
            hash=hsh, rows=int(s.n),
            files=" | ".join(sorted(members.file.unique())),
            splits=" | ".join(sorted(members.split.unique())),
            label=" | ".join(raw.labels[i] for i in sorted(members.label.unique())),
            within_one_file=bool(s.n_files == 1),
            conflicting_labels=bool(s.n_labels > 1),
        ))
    pd.DataFrame(rows).to_csv(OUT, index=False)

    print(f"rows {len(d)}, unique {len(g)}, surplus {len(d) - len(g)}")
    for r in rows:
        print(f"  {r['rows']} rows | {r['files']} | {r['splits']} | {r['label']}")
    print("involves test:", any("test" in r["splits"] for r in rows))
    print("wrote", OUT)


if __name__ == "__main__":
    main(sys.argv[1])
