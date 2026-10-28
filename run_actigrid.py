"""Command-line entry point for the ActiGrid experiments.

Stages
------
controlled  same-backbone ten-seed experiment (actigrid.controlled; paper
            Table I); default
v7        lambda sweep plus the ten-seed v7 evaluation (actigrid.pipeline.main)
reviewer  exhaustive fixed subsets and fixed-order entropy stopping on the
          per-seed backbones (actigrid.reviewer_clearance); reads the frozen
          v7 results from --prior-results (default: results/)
hardening   subset-specific tree classifiers on three group splits, DQN lambda
            frontier on three backbones, calibration, bootstrap, branching and
            corruption tests (actigrid.hardening; paper Tables II and IV,
            Figs. 3-4)
greedy-voi  Greedy-VOI and calibrated entropy stopping on the frozen hardening
            backbones, no retraining (actigrid.greedy_voi; paper Fig. 2); reads
            the hardening outputs from --hardening-dir
hgb-sequential  entropy stopping over per-subset HGB models versus static HGB
            subsets on the three group splits (actigrid.hgb_sequential; paper
            Table III, Fig. 2)

Examples
--------
Same-backbone experiment (paper Table I):

    python run_actigrid.py --stage controlled --config configs/paper_v7.json \
        --msu-root data/msu_ornl --tudelft-root data/tudelft_cpps --work-dir outputs

Full v7 protocol on both datasets:

    python run_actigrid.py --stage v7 --config configs/paper_v7.json \
        --msu-root data/msu_ornl --tudelft-root data/tudelft_cpps --work-dir outputs

Reviewer-clearance baselines:

    python run_actigrid.py --stage reviewer --config configs/paper_v7.json \
        --msu-root data/msu_ornl --tudelft-root data/tudelft_cpps --work-dir outputs

Reviewer-hardening suite, as run for the paper (conference profile):

    python run_actigrid.py --stage hardening --config configs/paper_v7.json \
        --msu-root data/msu_ornl --tudelft-root data/tudelft_cpps --work-dir outputs

Greedy-VOI on the released hardening checkpoints (results/reviewer_hardening/cache):

    python run_actigrid.py --stage greedy-voi --config configs/paper_v7.json \
        --msu-root data/msu_ornl --tudelft-root data/tudelft_cpps --work-dir outputs \
        --hardening-dir results/reviewer_hardening

Sequential stopping over per-subset HGB (paper Table III):

    python run_actigrid.py --stage hgb-sequential --config configs/paper_v7.json \
        --msu-root data/msu_ornl --tudelft-root data/tudelft_cpps --work-dir outputs

Quick smoke test of the controlled stage (few epochs/episodes, two seeds):

    python run_actigrid.py --config configs/paper_v7.json --quick \
        --tudelft-root data/tudelft_cpps --datasets tudelft
"""
import argparse
import json
from pathlib import Path

from actigrid import pipeline


def parse_args():
    p = argparse.ArgumentParser(description="Run an ActiGrid experiment stage.")
    p.add_argument("--stage", choices=["controlled", "v7", "reviewer", "hardening", "greedy-voi",
                            "hgb-sequential"],
                   default="controlled",
                   help="controlled: same-backbone experiment (Table I); hardening: stronger "
                        "baselines and stress tests (Tables II, IV); greedy-voi: non-RL sequential "
                        "controls (Fig. 2); hgb-sequential: stopping over per-subset HGB "
                        "(Table III); v7/reviewer: earlier stages (see README).")
    p.add_argument("--prior-results", default=None,
                   help="Folder with the frozen v7 selection/result files (default: results/).")
    p.add_argument("--profile", choices=["conference", "full"], default="conference",
                   help="hardening: conference = the paper's run; full = more splits, seeds and lambdas.")
    p.add_argument("--force", action="store_true",
                   help="hardening, hgb-sequential: recompute outputs that already exist instead of resuming.")
    p.add_argument("--skip-dqn", action="store_true",
                   help="hardening: train no DQN (subset frontier and calibration only).")
    p.add_argument("--hardening-dir", default=None,
                   help="greedy-voi, hgb-sequential: folder with the hardening outputs and cache/ "
                        "(default: <work-dir>/reviewer_hardening, else results/reviewer_hardening).")
    p.add_argument("--summaries-only", action="store_true",
                   help="greedy-voi: rebuild the summary files from existing frontier CSVs.")
    p.add_argument("--config", type=Path, default=None,
                   help="JSON file of Config overrides (e.g. configs/paper_v7.json).")
    p.add_argument("--data-root", default=None,
                   help="Folder searched for datasets when explicit roots are not given.")
    p.add_argument("--msu-root", default=None, help="Folder with the MSU/ORNL ARFF/CSV files.")
    p.add_argument("--tudelft-root", default=None, help="Extracted TU Delft CPPS dataset folder.")
    p.add_argument("--work-dir", default=None, help="Output folder (default: outputs).")
    p.add_argument("--datasets", nargs="+", choices=["msu", "tudelft"], default=None,
                   help="Subset of datasets to run (default: both).")
    p.add_argument("--seeds", nargs="+", type=int, default=None,
                   help="Override the evaluation seeds (paper: 10 seeds).")
    p.add_argument("--device", default=None, help="cuda or cpu (default: auto).")
    p.add_argument("--quick", action="store_true",
                   help="Smoke-test mode: fast_debug + final_quick_mode. Not for reported results.")
    return p.parse_args()


def main():
    args = parse_args()
    overrides = {}
    if args.config:
        overrides.update(json.loads(args.config.read_text()))
    if args.data_root:
        overrides["data_root"] = args.data_root
    if args.msu_root:
        overrides["msu_root_override"] = args.msu_root
    if args.tudelft_root:
        overrides["tudelft_root_override"] = args.tudelft_root
    if args.work_dir:
        overrides["work_dir"] = args.work_dir
    if args.datasets:
        overrides["run_msu"] = "msu" in args.datasets
        overrides["run_tudelft"] = "tudelft" in args.datasets
    if args.seeds:
        overrides["final_seeds"] = args.seeds
    if args.device:
        overrides["device"] = args.device
    if args.quick:
        overrides["fast_debug"] = True
        overrides["final_quick_mode"] = True

    pipeline.setup(overrides)
    if args.stage == "controlled":
        from actigrid import controlled
        controlled.controlled_comparison_main(args.prior_results, fast=args.quick)
    elif args.stage == "v7":
        pipeline.main()
    elif args.stage == "hardening":
        from actigrid import hardening
        hardening.hardening_main(args.profile, force=args.force, skip_dqn=args.skip_dqn, fast=args.quick)
    elif args.stage == "greedy-voi":
        from actigrid import greedy_voi
        greedy_voi.greedy_voi_main(args.hardening_dir, fast=args.quick, summaries_only=args.summaries_only)
    elif args.stage == "hgb-sequential":
        from actigrid import hgb_sequential
        hgb_sequential.hgb_sequential_main(args.hardening_dir, force=args.force, fast=args.quick)
    else:
        from actigrid import reviewer_clearance
        reviewer_clearance.reviewer_clearance_main(args.prior_results, fast=args.quick)


if __name__ == "__main__":
    main()
