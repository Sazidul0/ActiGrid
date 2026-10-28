# ActiGrid

**Comparing fixed and sequential evidence selection for cyber–physical power-system diagnosis.**

ActiGrid treats power-system fault-versus-attack diagnosis as a sequential decision. A mask-aware multimodal fusion classifier can classify from any subset of evidence sources. A Double-DQN policy decides which source to acquire next, or stops once the evidence is sufficient. The agent never controls grid equipment: it only chooses which diagnostic evidence to look at.

The paper asks when such a learned policy is worth its added complexity. It compares the DQN with every fixed subset and fixed acquisition order on one shared classifier, then retests on repeated group-disjoint splits against tree ensembles trained per subset, calibrated entropy stopping and a greedy value-of-information (Greedy-VOI) rule. The DQN does adapt to what it observes, but simple stopping rules match or beat it, and compact subset classifiers outperform it at lower cost on both benchmarks.

This repository contains the code, the full result files, the run logs and the trained models behind the paper:

> *ActiGrid: Comparing Fixed and Sequential Evidence Selection for Power System Diagnosis*, submitted to IEEE PES General Meeting 2027.

## Main results

**Shared backbone, ten training seeds on the primary group-disjoint split (paper Table I).** Every policy of a seed uses the same frozen fusion classifier.

| Dataset | Method | Macro-F1 | Normalized cost |
|---|---|---|---|
| TU Delft | Full fusion | 0.8158 ± 0.0323 | 1.000 |
| TU Delft | DQN (λ = 0.5) | 0.7917 ± 0.0333 | 0.703 |
| TU Delft | Order + entropy stopping | 0.7868 ± 0.0347 | 0.690 |
| TU Delft | Fixed subset, k = 2 | 0.6933 ± 0.0583 | 0.667 |
| MSU/ORNL | Full fusion | 0.3924 ± 0.0077 | 1.000 |
| MSU/ORNL | DQN (λ = 0.5) | 0.3769 ± 0.0118 | 0.754 |
| MSU/ORNL | Order + entropy stopping | 0.3859 ± 0.0131 | 0.787 |
| MSU/ORNL | Fixed subset, k = 2 | 0.3931 ± 0.0093 | 0.400 |
| MSU/ORNL | Fixed subset, k = 3 | **0.4011 ± 0.0101** | 0.600 |

**Three group-disjoint splits with subset-specific classifiers (paper Table II).** HGB rows are the validation-selected histogram gradient boosting model for each subset size k.

| Dataset | Method | Macro-F1 | Normalized cost |
|---|---|---|---|
| TU Delft | DQN | 0.8045 ± 0.0165 | 0.706 |
| TU Delft | Order + entropy stopping | 0.8095 ± 0.0113 | 0.732 |
| TU Delft | HGB, k = 1 (network) | 0.8673 ± 0.0272 | 0.333 |
| TU Delft | HGB, k = 3 | **0.9077 ± 0.0466** | 1.000 |
| MSU/ORNL | DQN | 0.3632 ± 0.0268 | 0.728 |
| MSU/ORNL | Order + entropy stopping | 0.3800 ± 0.0264 | 0.751 |
| MSU/ORNL | HGB, k = 3 (R1, R3, R4) | 0.4389 ± 0.0069 | 0.600 |
| MSU/ORNL | HGB, k = 5 | **0.4402 ± 0.0061** | 1.000 |

On the shared backbone the DQN keeps 97.0% of full-fusion Macro-F1 on TU Delft with 29.7% fewer modality units, but its paired difference from order + entropy stopping is only +0.005 (95% CI [−0.004, 0.014], Wilcoxon p = 0.43), and on MSU/ORNL a fixed three-source subset beats it. With classifiers trained per subset, a single network-modality HGB on TU Delft and a three-relay HGB on MSU/ORNL score higher than every sequential policy at lower cost. Cost is the fraction of available modality units read (equal cost per modality), not bytes, latency or energy.

<p align="center"><img src="figures/fig_frontier.png" width="640" alt="Macro-F1 against normalized cost for calibrated entropy stopping, Greedy-VOI, the DQN and per-subset HGB on TU Delft and MSU/ORNL"></p>

## Repository layout

```
actigrid/pipeline.py               Data loading, splits, fusion model, DQN, v7 protocol, evaluation
actigrid/controlled.py             Shared-backbone ten-seed experiment (paper Table I)
actigrid/hardening.py              Repeated splits, per-subset trees, lambda frontier, calibration,
                                   bootstrap, branching, corruption (paper Tables II-III, Figs. 2-3)
actigrid/greedy_voi.py             Greedy-VOI and calibrated entropy stopping on frozen backbones (Fig. 1)
actigrid/reviewer_clearance.py     Fixed-subset and fixed-order entropy helpers and analyses
run_actigrid.py                    Command-line entry point (--stage controlled | hardening | greedy-voi | v7 | reviewer)
configs/paper_v7.json              Configuration used for the paper
scripts/verify_results.py          Recomputes every hardening/Greedy-VOI summary and the paper's numbers
scripts/make_paper_tables.py       Rebuilds paper Tables I-III
scripts/make_paper_figures.py      Rebuilds paper Figs. 1-3
scripts/extract_controlled_results.py  Parses the controlled-run log into CSV
results/controlled_same_backbone/  Ten-seed summary and paired effects (Table I)
results/reviewer_hardening/        Hardening outputs (Tables II-III); cache/ holds its 12 fusion checkpoints
results/next_addon/                Greedy-VOI and calibrated entropy frontiers, Pareto points, AUCs
results/reviewer_clearance/        Per-seed exhaustive subsets, order+entropy search, budget curves
results/logs/                      Run logs (controlled, hardening, Greedy-VOI, v7) and the TU Delft manifest
results/models/                    Seed-42 fusion classifier and DQN weights from the v7 run (PyTorch)
results/*_v7_selection.json        Frozen validation selections (lambda, DQN validation cost)
results/*.csv                      Earlier v7 protocol outputs (classical baselines, audits)
figures/                           Generated figures
```

## Installation

Python 3.10 or newer is required. The paper results were produced with Python 3.13, PyTorch 2.11, NumPy 2.1, pandas 2.3 and scikit-learn on Kaggle. The controlled and hardening runs used CPU; the v7 run and the Greedy-VOI inference used a Tesla T4 GPU.

```bash
git clone https://github.com/sazidul0/ActiGrid.git
cd ActiGrid
pip install -r requirements.txt
```

The code runs on CPU; a GPU only speeds up the neural parts.

## Data

Neither dataset is redistributed here. Download both from their original sources.

**MSU/ORNL power-system attack dataset.** This is the multiclass release with 15 `dataN Sampled Scenarios.csv.arff` files. It is available from the [UAH ICS data sets page](https://www.sites.google.com/a/uah.edu/tommy-morris-uah/ics-data-sets). Please cite Pan *et al.*, IEEE TII 2015.

**TU Delft cyber–physical digital-substation dataset.** This dataset is described in N. Cibin *et al.*, "Cyber–physical power system dataset for cyber security of digital substations," *IEEE Access*, vol. 14, 2026, [doi:10.1109/ACCESS.2026.3699153](https://doi.org/10.1109/ACCESS.2026.3699153). Follow the data-availability statement in that paper. Extract the archive before running; the pipeline can call `7z`/`unrar` if it finds only an archive.

A suggested layout:

```
data/
  msu_ornl/        data1 Sampled Scenarios.csv.arff ... data15 Sampled Scenarios.csv.arff
  tudelft_cpps/    2026_03_17_CPPS_Dataset_CyberSec_DigitalSubstation_TUD/...
```

## Reproducing the paper

All commands run from the repository root. The data options are the same for every stage:

```bash
DATA="--config configs/paper_v7.json --msu-root data/msu_ornl --tudelft-root data/tudelft_cpps --work-dir outputs"
```

**1. Rebuild the tables and figures from the released results.** This needs no data and no GPU (only `numpy` and `matplotlib`):

```bash
python scripts/verify_results.py      # recomputes every derived hardening/Greedy-VOI file and the paper's numbers
python scripts/make_paper_tables.py   # Tables I-III as LaTeX rows
python scripts/make_paper_figures.py  # figures/fig_frontier, fig_calibration, fig_branching
```

**2. Shared-backbone experiment (Table I).** For each dataset and each of ten seeds, this trains one fusion backbone and evaluates full fusion, all fixed subsets, all fixed orders with entropy stopping and the DQN on it. The frozen selections (λ = 0.5 and the DQN validation costs) are read from `results/*_v7_selection.json`. The paper run took about 7 hours on CPU; its log is [`results/logs/controlled_same_backbone.log`](results/logs/controlled_same_backbone.log).

```bash
python run_actigrid.py --stage controlled $DATA
```

**3. Hardening suite (Tables II-III, Figs. 2-3).** For each dataset this fits RF, HGB and XGBoost on every modality subset of three group splits (seeds 42, 123, 456), trains three fusion backbones on the primary split with a DQN for λ = 0.25, 0.5 and 0.75 plus two reward ablations each, runs the seed-42 calibration, bootstrap, branching and corruption analyses, and repeats the DQN versus order + entropy comparison on the three splits. Every step writes a checkpoint, so an interrupted run resumes where it stopped; `--force` recomputes. The paper run (`--profile conference`, the default) spans about 12 hours of log time on CPU; its log is [`results/logs/reviewer_hardening.log`](results/logs/reviewer_hardening.log). `--profile full` adds splits, seeds, λ values, MLP subsets and a Dueling DDQN; the paper does not use it.

```bash
python run_actigrid.py --stage hardening $DATA
```

Outputs go to `outputs/reviewer_hardening/`; the released copy is [`results/reviewer_hardening/`](results/reviewer_hardening/).

**4. Greedy-VOI and calibrated entropy stopping (Fig. 1).** This reuses the hardening backbones (seeds 42, 123, 456) without retraining any fusion, DQN or tree model. It fits one small HGB regressor per modality for Greedy-VOI, temperature-scales every mask, and evaluates all fixed orders and thresholds. To reproduce the paper's frontier exactly, point it at the released checkpoints:

```bash
python run_actigrid.py --stage greedy-voi $DATA --hardening-dir results/reviewer_hardening
```

Without `--hardening-dir` it uses `outputs/reviewer_hardening/` if step 3 has run there, otherwise the released folder. `--summaries-only` rebuilds the Pareto, AUC and calibration summaries from frontier CSVs already in `outputs/next_addon/`. The log of the paper run is [`results/logs/greedy_voi.log`](results/logs/greedy_voi.log).

**5. Optional earlier stages.** `--stage v7` re-runs the λ sweep and the earlier ten-seed v7 evaluation that produced the frozen selections. `--stage reviewer` re-runs the fixed-subset and order + entropy analyses on their own (results in `results/reviewer_clearance/`). Neither is needed for the paper's tables.

**6. Quick smoke test.** `--quick` uses few epochs and episodes, one split and seed for the hardening stage, and seed 42 only for Greedy-VOI. The numbers are not meaningful; it only checks that a stage runs:

```bash
python run_actigrid.py --stage hardening --quick --datasets tudelft --config configs/paper_v7.json \
    --tudelft-root data/tudelft_cpps --work-dir outputs_smoke
python run_actigrid.py --stage greedy-voi --quick --datasets tudelft --config configs/paper_v7.json \
    --tudelft-root data/tudelft_cpps --work-dir outputs_smoke
```

Other useful options are `--datasets msu tudelft`, `--seeds 42 123` (controlled stage), `--device cpu` and `--data-root data` (which auto-discovers the dataset folders).

## Protocol summary

- **Splits.** MSU/ORNL is split by source file (9/3/3 files; 47,471/15,272/15,634 samples) and TU Delft by recording run (47/16/16 runs; 3,246/1,276/1,316 one-second windows), so no file or run appears in two splits. All classes occur in every split. This primary split is `group_split(seed=42)`. The hardening stage rebuilds it with seeds 42, 123 and 456; seed 42 reproduces the primary split exactly.
- **Modalities and cost.** MSU/ORNL has four PMU/relay groups plus cyber logs (cost 0.2 each). TU Delft has physical RSCAD measurements, resource-monitoring features and PCAP network features (cost 1/3 each).
- **Shared backbone.** For each dataset and seed, one fusion classifier is trained with modality dropout, then frozen and shared by every policy evaluated on that seed.
- **Policy.** A Double DQN with class-balanced episodes (20,000 per run). Its reward is a clipped log-probability gain for the true class minus λ × cost, with ±2 at termination and an extra −1 for confusing a fault with an attack. The true label is used only for the training reward; it is never part of the policy state. λ = 0.5 was selected on validation data in the v7 sweep and frozen.
- **Baselines.** Full fusion; every nonempty fixed subset, with the best subset of each size selected on validation Macro-F1; all M! fixed orders with entropy stopping, selected on validation within ±0.02 of the DQN's validation cost; RF, HGB and XGBoost trained per subset and selected on validation; calibrated entropy stopping (one temperature per mask fitted on half of the validation set, settings chosen on the other half); and Greedy-VOI (acquire the source with the largest predicted entropy reduction per unit cost, fitted on 3,246 TU Delft and 7,000 MSU/ORNL training rows).
- **Three meanings of "seed 42".** The controlled run's seed 42 is one of its ten training seeds. The hardening stage's *backbone* seed 42 is a separate model on the same primary split; Table III, Figs. 1-3 and the bootstrap use it. *Split* seed 42 is the primary split itself.
- **Statistics.** Paired DQN-minus-comparator differences over the ten seeds with 95% t intervals and exploratory Wilcoxon signed-rank p-values; a run-level (TU Delft) or file-level (MSU/ORNL) cluster bootstrap with 3,000 replicates; mask-wise ECE with 15 bins.

## Notes on this release

`actigrid/pipeline.py` is the code that produced the v7 results (originally run as a Kaggle notebook). For the release, only the following changed:

- Paths and the configuration can now be set from the command line (`setup()` / `run_actigrid.py`).
- Code that ran at import time was moved into `setup()`.
- A missing `Counter` import in dataset auto-discovery was added. The paper run used explicit dataset paths and never reached that code path.

`actigrid/reviewer_clearance.py`, `actigrid/controlled.py` and `actigrid/hardening.py` are sections 17, 18 and 19 of the same script, moved into their own modules. Sections 0-18 of the script that ran the hardening stage are identical to the released modules apart from the changes above. The modules read `CFG`/`WORK`/`LOG` from `actigrid.pipeline` at call time, take result folders as arguments instead of searching `/kaggle`, and take the hardening options (profile, force, skip DQN) as arguments instead of `ACTIGRID_*` environment variables. Their experimental logic is unchanged.

`actigrid/greedy_voi.py` is the standalone Greedy-VOI script. The Kaggle version carried a compressed copy of the base pipeline; that copy is byte-identical to the hardening script, so the release imports `actigrid.pipeline` instead. One bug was fixed: its calibration summary detected the raw-ECE column by name and picked `val_ece_raw`, so the original `mean_raw_ece`/`max_raw_ece` were validation values (0.185/0.166). The released `results/next_addon/calibration_summary.csv` and `NEXT_PAPER_READY_SUMMARY.md` carry the corrected test values (0.162/0.163), which are the ones the paper reports. Nothing else in the add-on depends on that file.

Other points:

- The Kaggle add-on also copied 18 hardening tables (nine per dataset) into its output folder for packaging. `results/next_addon/` omits those copies; the originals are in `results/reviewer_hardening/`. The hardening stage's `*.partial.csv` resume files are also omitted, because they equal the final files.
- `scripts/verify_results.py` recomputes every derived file in `results/reviewer_hardening/` and `results/next_addon/` from its per-row inputs and matches them with the run logs. Rebuilding the summaries with a newer pandas changes only the last printed digit of some floats.
- Neural training is not bit-identical across hardware and library versions, so a rerun of the controlled or hardening stage gives close but not identical numbers. The Greedy-VOI stage trains no neural model; with the released checkpoints, its differences are limited to GPU/CPU floating-point effects.
- Open-set evaluation, ablations of the v7 run and the EMS stress test remain in the code but are switched off and are not part of the paper's claims.

## Citation

If you use this code, please cite the paper (citation details will be updated after publication):

```bibtex
@inproceedings{actigrid2027,
  title     = {ActiGrid: Comparing Fixed and Sequential Evidence Selection for Power System Diagnosis},
  author    = {<Authors>},
  booktitle = {Proc. IEEE PES General Meeting},
  year      = {2027}
}
```

## License

The code is released under the MIT License (see [LICENSE](LICENSE)). The datasets are subject to their own licenses and terms of use.
