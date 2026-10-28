# ActiGrid final reviewer-hardening outputs

Profile: **conference**

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
