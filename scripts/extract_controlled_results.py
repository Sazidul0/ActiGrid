"""Extract the controlled same-backbone tables from the run log into CSV files.

Usage (from the repository root):
    python scripts/extract_controlled_results.py

Reads  results/logs/controlled_same_backbone.log
Writes results/controlled_same_backbone/controlled_summary.csv
       results/controlled_same_backbone/controlled_paired_effects.csv

The log prints each table with pandas `to_string`; this script parses the
"CONTROLLED SUMMARY <dataset>:" and "CONTROLLED PAIRED EFFECTS <dataset>:"
blocks verbatim (values are not recomputed or rounded).
"""
import csv
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / "results" / "logs" / "controlled_same_backbone.log"
OUT = ROOT / "results" / "controlled_same_backbone"
TEXT_COLS = {"dataset", "method", "focal", "comparator"}


def parse_block(lines, start):
    header = lines[start + 1].split()
    rows = []
    for ln in lines[start + 2:]:
        if not ln.strip() or re.match(r"^\d{4}-\d{2}-\d{2} ", ln):
            break
        vals = ln.split()
        if len(vals) != len(header):
            raise ValueError(f"Unexpected row in log: {ln!r}")
        rows.append(dict(zip(header, vals)))
    return header, rows


def main():
    lines = LOG.read_text().splitlines()
    tables = {"SUMMARY": [], "PAIRED EFFECTS": []}
    headers = {}
    for i, ln in enumerate(lines):
        m = re.match(r"^CONTROLLED (SUMMARY|PAIRED EFFECTS) (\S+):$", ln.strip())
        if m:
            header, rows = parse_block(lines, i)
            headers[m.group(1)] = header
            tables[m.group(1)].extend(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    for key, name in (("SUMMARY", "controlled_summary.csv"),
                      ("PAIRED EFFECTS", "controlled_paired_effects.csv")):
        if not tables[key]:
            raise RuntimeError(f"No CONTROLLED {key} blocks found in {LOG}")
        with open(OUT / name, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=headers[key])
            w.writeheader()
            w.writerows(tables[key])
        print(f"wrote {OUT / name} ({len(tables[key])} rows)")


if __name__ == "__main__":
    main()
