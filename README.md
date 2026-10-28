# ActiGrid

**Comparing fixed and sequential evidence selection for cyber–physical power-system diagnosis.**

ActiGrid treats power-system fault-versus-attack diagnosis as a sequential decision. A mask-aware multimodal fusion classifier can classify from any subset of evidence sources. A Double-DQN policy decides which source to acquire next, or stops once the evidence is sufficient. The agent never controls grid equipment: it only chooses which diagnostic evidence to look at.

The paper asks when such a learned policy is worth its added complexity. It compares the DQN with every fixed subset and fixed acquisition order on one shared classifier, then retests on repeated group-disjoint splits against tree ensembles trained per subset, calibrated entropy stopping and a greedy value-of-information (Greedy-VOI) rule. The DQN does adapt to what it observes, but simple stopping rules match or beat it, and compact subset classifiers outperform it at lower cost on both benchmarks. Entropy stopping run over those subset classifiers improves on them for TU Delft and matches them for MSU/ORNL. Selective acquisition can pay off, but on these benchmarks it does not need a learned policy.

This repository contains the code, the full result files, the run logs and the trained models behind the paper:

> *ActiGrid: Comparing Fixed and Sequential Evidence Selection for Power System Diagnosis*, submitted to IEEE PES General Meeting 2027.

## Main results

**Shared backbone, ten training seeds on the primary group-disjoint split (paper Table I).** Every policy of a seed uses the same frozen fusion classifier.

| Dataset | Method | Macro-F1 | Normalized cost |
|---|---|---|---|
| TU Delft | Full fusion | **0.8158 ± 0.0323** | 1.000 |
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

**Static versus sequential HGB on the same three splits (paper Table III).** Sequential HGB runs fixed-order entropy stopping over the per-subset HGB models. Temperatures are fitted on one half of the validation set; the order, the threshold and the static comparator are chosen on the other half.

| Dataset | Method | Macro-F1 | Normalized cost |
|---|---|---|---|
| TU Delft | Static HGB | 0.8972 ± 0.0580 | 0.778 |
| TU Delft | Sequential HGB, calibrated | **0.9282 ± 0.0378** | 0.571 |
| MSU/ORNL | Static HGB | 0.4376 ± 0.0077 | 0.933 |
| MSU/ORNL | Sequential HGB, calibrated | 0.4400 ± 0.0066 | 0.861 |

On the shared backbone the DQN keeps 97.0% of full-fusion Macro-F1 on TU Delft with 29.7% fewer modality units, but its paired difference from order + entropy stopping is only +0.005 (95% CI [−0.004, 0.014], Wilcoxon p = 0.43), and on MSU/ORNL a fixed three-source subset beats it. With classifiers trained per subset, a single network-modality HGB on TU Delft and a three-relay HGB on MSU/ORNL score higher than every policy on the shared backbone at lower cost. Entropy stopping over those same classifiers reads the network modality first on TU Delft and beats the best static subset at a validation budget of 2/3 on all three splits (by 0.039 to 0.100); on MSU/ORNL the two tie. Cost is the fraction of available modality units read (equal cost per modality), not bytes, latency or energy.

<p align="center"><img src="figures/fig_frontier.png" width="640" alt="Macro-F1 against normalized cost for calibrated entropy stopping, Greedy-VOI, the DQN, static HGB and sequential HGB on TU Delft and MSU/ORNL"></p>

## Repository layout

```
actigrid/pipeline.py               Data loading, splits, fusion model, DQN, v7 protocol, evaluation
actigrid/controlled.py             Shared-backbone ten-seed experiment (paper Table I)
actigrid/hardening.py              Repeated splits, per-subset trees, lambda frontier, calibration,
                                   bootstrap, branching, corruption (paper Tables II and IV, Figs. 3-4)
actigrid/greedy_voi.py             Greedy-VOI and calibrated entropy stopping on frozen backbones (Fig. 2)
actigrid/hgb_sequential.py         Entropy stopping over per-subset HGB versus static HGB (Table III, Fig. 2)
actigrid/reviewer_clearance.py     Fixed-subset and fixed-order entropy helpers and analyses
run_actigrid.py                    Command-line entry point (--stage controlled | hardening | greedy-voi |
                                   hgb-sequential | v7 | reviewer)
configs/paper_v7.json              Configuration used for the paper
scripts/verify_results.py          Recomputes every hardening/Greedy-VOI/sequential-HGB summary and the paper's numbers
scripts/make_paper_tables.py       Rebuilds paper Tables I-IV
scripts/make_paper_figures.py      Rebuilds paper Figs. 2-4
scripts/make_method_diagram.py     Draws paper Fig. 1 (the acquisition loop)
scripts/extract_controlled_results.py  Parses the controlled-run log into CSV
results/controlled_same_backbone/  Ten-seed summary and paired effects (Table I)
results/reviewer_hardening/        Hardening outputs (Tables II and IV); cache/ holds its 12 fusion checkpoints
results/next_addon/                Greedy-VOI and calibrated entropy frontiers, Pareto points, AUCs
results/hgb_sequential/            Static and sequential HGB per partition, matched budgets, predictions (Table III)
results/reviewer_clearance/        Per-seed exhaustive subsets, order+entropy search, budget curves
results/logs/                      Run logs (controlled, hardening, Greedy-VOI, sequential HGB, v7), TU Delft manifest
results/models/                    Seed-42 fusion classifier and DQN weights from the v7 run (PyTorch)
results/*_v7_selection.json        Frozen validation selections (lambda, DQN validation cost)
results/*.csv                      Earlier v7 protocol outputs (classical baselines, audits)
figures/                           Generated figures
```

## Installation

Python 3.10 or newer is required. The paper results were produced with Python 3.13, PyTorch 2.11, NumPy 2.1, pandas 2.3 and scikit-learn on Kaggle. The controlled and hardening runs used CPU; the v7 run and the Greedy-VOI inference used a Tesla T4 GPU. The sequential-HGB stage trains only tree models, which run on CPU.

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
python scripts/verify_results.py      # recomputes every derived hardening/Greedy-VOI/sequential-HGB file and the paper's numbers
python scripts/make_paper_tables.py   # Tables I-IV as LaTeX rows
python scripts/make_paper_figures.py  # figures/fig_frontier, fig_calibration, fig_branching (Figs. 2-4)
python scripts/make_method_diagram.py # figures/fig_method (Fig. 1)
```

**2. Shared-backbone experiment (Table I).** For each dataset and each of ten seeds, this trains one fusion backbone and evaluates full fusion, all fixed subsets, all fixed orders with entropy stopping and the DQN on it. The frozen selections (λ = 0.5 and the DQN validation costs) are read from `results/*_v7_selection.json`. The paper run took about 7 hours on CPU; its log is [`results/logs/controlled_same_backbone.log`](results/logs/controlled_same_backbone.log).

```bash
python run_actigrid.py --stage controlled $DATA
```

**3. Hardening suite (Tables II and IV, Figs. 3-4).** For each dataset this fits RF, HGB and XGBoost on every modality subset of three group-disjoint partitions (partition seeds 42, 123, 456), trains three fusion backbones on the primary split (initialization seeds 42, 123, 456) with a DQN for λ = 0.25, 0.5 and 0.75 plus two reward ablations each, runs the seed-42 calibration, bootstrap, branching and corruption analyses, and repeats the DQN versus order + entropy comparison on the three partitions. Every step writes a checkpoint, so an interrupted run resumes where it stopped; `--force` recomputes. The paper run (`--profile conference`, the default) spans about 12 hours of log time on CPU; its log is [`results/logs/reviewer_hardening.log`](results/logs/reviewer_hardening.log). `--profile full` adds splits, seeds, λ values, MLP subsets and a Dueling DDQN; the paper does not use it.

```bash
python run_actigrid.py --stage hardening $DATA
```

Outputs go to `outputs/reviewer_hardening/`; the released copy is [`results/reviewer_hardening/`](results/reviewer_hardening/).

**4. Greedy-VOI and calibrated entropy stopping (Fig. 2).** This reuses the hardening backbones (initialization seeds 42, 123, 456) without retraining any fusion, DQN or tree model. It fits one small HGB regressor per modality for Greedy-VOI, temperature-scales every mask, and evaluates all fixed orders and thresholds. To reproduce the paper's frontier exactly, point it at the released checkpoints:

```bash
python run_actigrid.py --stage greedy-voi $DATA --hardening-dir results/reviewer_hardening
```

Without `--hardening-dir` it uses `outputs/reviewer_hardening/` if step 3 has run there, otherwise the released folder. `--summaries-only` rebuilds the Pareto, AUC and calibration summaries from frontier CSVs already in `outputs/next_addon/`. The log of the paper run is [`results/logs/greedy_voi.log`](results/logs/greedy_voi.log).

**5. Sequential versus static HGB (Table III, Fig. 2).** For each dataset and partition seed this trains one HGB model per nonempty modality subset (the hardening stage's settings), fits one temperature per subset on half of the validation set, and evaluates every fixed order with entropy stopping on raw and calibrated posteriors. The order, the threshold and the static comparator are chosen on the other validation half; the test set is only scored. It trains no neural model and checks its static scores against the hardening frontier. The log of the paper run is [`results/logs/hgb_sequential.log`](results/logs/hgb_sequential.log).

```bash
python run_actigrid.py --stage hgb-sequential $DATA
```

**6. Optional earlier stages.** `--stage v7` re-runs the λ sweep and the earlier ten-seed v7 evaluation that produced the frozen selections. `--stage reviewer` re-runs the fixed-subset and order + entropy analyses on their own (results in `results/reviewer_clearance/`). Neither is needed for the paper's tables.

**7. Quick smoke test.** `--quick` uses few epochs and episodes, one split and seed for the hardening stage, and seed 42 only for Greedy-VOI and sequential HGB. The numbers are not meaningful; it only checks that a stage runs:

```bash
python run_actigrid.py --stage hardening --quick --datasets tudelft --config configs/paper_v7.json \
    --tudelft-root data/tudelft_cpps --work-dir outputs_smoke
python run_actigrid.py --stage greedy-voi --quick --datasets tudelft --config configs/paper_v7.json \
    --tudelft-root data/tudelft_cpps --work-dir outputs_smoke
python run_actigrid.py --stage hgb-sequential --quick --datasets tudelft --config configs/paper_v7.json \
    --tudelft-root data/tudelft_cpps --work-dir outputs_smoke
```

Other useful options are `--datasets msu tudelft`, `--seeds 42 123` (controlled stage), `--device cpu` and `--data-root data` (which auto-discovers the dataset folders).

## Protocol summary

- **Splits.** MSU/ORNL is split by source file (9/3/3 files; 47,471/15,272/15,634 samples) and TU Delft by recording run (47/16/16 runs; 3,246/1,276/1,316 one-second windows), so no file or run appears in two splits. All classes occur in every split. This primary split is `group_split(seed=42)`. The hardening and sequential-HGB stages build three partitions with partition seeds 42, 123 and 456; partition 42 reproduces the primary split exactly. The test windows of MSU/ORNL therefore come from files never seen in training, and no model trained under this protocol exceeds 0.45 Macro-F1.
- **Modalities and cost.** MSU/ORNL has four PMU/relay groups plus cyber logs (cost 0.2 each). TU Delft has physical RSCAD measurements, resource-monitoring features and PCAP network features (cost 1/3 each).
- **Shared backbone.** For each dataset and seed, one fusion classifier is trained with modality dropout, then frozen and shared by every policy evaluated on that seed.
- **Policy.** A Double DQN with class-balanced episodes (20,000 per run). Its reward is a clipped log-probability gain for the true class minus λ × cost, with ±2 at termination and an extra −1 for confusing a fault with an attack. The true label is used only for the training reward; it is never part of the policy state. λ = 0.5 was selected on validation data in the v7 sweep and frozen.
- **Baselines.** Full fusion; every nonempty fixed subset, with the best subset of each size selected on validation Macro-F1; all M! fixed orders with entropy stopping, selected on validation within ±0.02 of the DQN's validation cost; RF, HGB and XGBoost trained per subset and selected on validation; calibrated entropy stopping (one temperature per mask fitted on half of the validation set, settings chosen on the other half); Greedy-VOI (acquire the source with the largest predicted entropy reduction per unit cost, fitted on 3,246 TU Delft and 7,000 MSU/ORNL training rows); and sequential HGB (entropy stopping over the per-subset HGB models, compared with static HGB at matched validation budgets).
- **Three meanings of "seed 42".** The controlled run's seed 42 is one of its ten training seeds. *Initialization seed* 42 is one of three hardening backbones on the primary split; the λ frontier, Greedy-VOI, Table IV, Figs. 2-4 and the bootstrap use it. *Partition seed* 42 is the primary split itself; the repeated-split runs train every partition's fusion model and DQN with one fixed model seed (4242), so the partition-42 model in Table II shares data but not weights with initialization seed 42.
- **Statistics.** Paired DQN-minus-comparator differences over the ten seeds with 95% t intervals and exploratory Wilcoxon signed-rank p-values; a run-level (TU Delft) or file-level (MSU/ORNL) cluster bootstrap with 3,000 replicates; mask-wise ECE with 15 bins.

## Notes on this release

`actigrid/pipeline.py` is the code that produced the v7 results (originally run as a Kaggle notebook). For the release, only the following changed:

- Paths and the configuration can now be set from the command line (`setup()` / `run_actigrid.py`).
- Code that ran at import time was moved into `setup()`.
- A missing `Counter` import in dataset auto-discovery was added. The paper run used explicit dataset paths and never reached that code path.

`actigrid/reviewer_clearance.py`, `actigrid/controlled.py` and `actigrid/hardening.py` are sections 17, 18 and 19 of the same script, moved into their own modules. Sections 0-18 of the script that ran the hardening stage are identical to the released modules apart from the changes above. The modules read `CFG`/`WORK`/`LOG` from `actigrid.pipeline` at call time, take result folders as arguments instead of searching `/kaggle`, and take the hardening options (profile, force, skip DQN) as arguments instead of `ACTIGRID_*` environment variables. Their experimental logic is unchanged.

`actigrid/greedy_voi.py` is the standalone Greedy-VOI script. The Kaggle version carried a compressed copy of the base pipeline; that copy is byte-identical to the hardening script, so the release imports `actigrid.pipeline` instead. One bug was fixed: its calibration summary detected the raw-ECE column by name and picked `val_ece_raw`, so the original `mean_raw_ece`/`max_raw_ece` were validation values (0.185/0.166). The released `results/next_addon/calibration_summary.csv` and `NEXT_PAPER_READY_SUMMARY.md` carry the corrected test values (0.162/0.163), which are the ones the paper reports. Nothing else in the add-on depends on that file.

`actigrid/hgb_sequential.py` is the standalone sequential-over-HGB script. Its compressed copy of the base pipeline is also byte-identical to the hardening script, so the release imports `actigrid.pipeline` and loads datasets through `actigrid.hardening`. Its analysis functions are unchanged. Its `HGB_CALIBRATION_SUMMARY.csv` reports ECE on the same validation half that fits the temperatures, so that file is in-sample; the paper does not use it.

Other points:

- The Kaggle add-on also copied 18 hardening tables (nine per dataset) into its output folder for packaging. `results/next_addon/` omits those copies; the originals are in `results/reviewer_hardening/`. The `*.partial.csv` resume files of the hardening and sequential-HGB stages are also omitted, because they equal the final files. `results/hgb_sequential/` omits its 319 MB `cache/` (processed datasets, 114 fitted HGB models and their probabilities); a rerun regenerates it, since every model has a fixed `random_state`.
- `scripts/verify_results.py` recomputes every derived file in `results/reviewer_hardening/`, `results/next_addon/` and `results/hgb_sequential/` from its per-row inputs and matches them with the run logs. Rebuilding the summaries with a newer pandas changes only the last printed digit of some floats.
- Neural training is not bit-identical across hardware and library versions, so a rerun of the controlled or hardening stage gives close but not identical numbers. The Greedy-VOI stage trains no neural model; with the released checkpoints, its differences are limited to GPU/CPU floating-point effects. The sequential-HGB stage trains only seeded tree models.
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
