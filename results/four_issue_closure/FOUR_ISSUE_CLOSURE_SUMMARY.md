# ActiGrid Four-Issue Closure

This report is generated only from the new closure run. It does not claim equivalence.

## Protocol changes

- MSU/ORNL: median imputation is fit on the training partition separately for every split.
- HGB: early_stopping=False; no internal random validation fraction is used.
- Static and sequential HGB: 10 unique group-disjoint partitions.
- DQN repeated-split control: 5 unique group-disjoint partitions, fixed model seed 4242 and lambda=0.5.
- HGB corruption: clean, train-mean replacement, Gaussian noise at 2 train SD, and normal-class mimicry.
- Greedy-VOI: VOI regressors are fit only on rows excluded from fusion-backbone fitting.

## TU 10-split corrected HGB matched-budget
```
           dataset   budget  static_f1_mean  static_f1_sd  seq_f1_mean  seq_f1_sd  seq_cost_mean  delta_f1_mean  delta_f1_sd  n_splits
TU_Delft_2026_CPPS 0.333333        0.871419      0.031468     0.871533   0.031602       0.333765       0.000114     0.000361        10
TU_Delft_2026_CPPS 0.666667        0.848221      0.069157     0.918299   0.029353       0.522834       0.070077     0.065014        10
TU_Delft_2026_CPPS 1.000000        0.875051      0.043173     0.918299   0.029353       0.522834       0.043247     0.033730        10
```

## MSU 10-split corrected HGB matched-budget
```
       dataset  budget  static_f1_mean  static_f1_sd  seq_f1_mean  seq_f1_sd  seq_cost_mean  delta_f1_mean  delta_f1_sd  n_splits
MSU_ORNL_Power     0.2        0.370287      0.006680     0.370294   0.006684       0.200003       0.000006     0.000020        10
MSU_ORNL_Power     0.4        0.418537      0.012303     0.401550   0.010814       0.379399      -0.016987     0.008774        10
MSU_ORNL_Power     0.6        0.428528      0.016373     0.417116   0.011420       0.592943      -0.011412     0.011361        10
MSU_ORNL_Power     0.8        0.435159      0.014740     0.430420   0.012935       0.764328      -0.004739     0.005602        10
MSU_ORNL_Power     1.0        0.435123      0.014901     0.433191   0.013137       0.849233      -0.001932     0.005726        10
```

## TU 5-split DQN expansion
```
           dataset       method  macro_f1_mean  macro_f1_sd  cost_mean  cost_sd  n_splits
TU_Delft_2026_CPPS          DQN       0.783063     0.076369   0.691974 0.052266         5
TU_Delft_2026_CPPS   FullFusion       0.831342     0.061535   1.000000 0.000000         5
TU_Delft_2026_CPPS OrderEntropy       0.794354     0.065228   0.700688 0.077696         5
```

## MSU 5-split corrected DQN expansion
```
       dataset       method  macro_f1_mean  macro_f1_sd  cost_mean  cost_sd  n_splits
MSU_ORNL_Power          DQN       0.367430     0.009787   0.744405 0.021189         5
MSU_ORNL_Power   FullFusion       0.388058     0.007901   1.000000 0.000000         5
MSU_ORNL_Power OrderEntropy       0.375439     0.014099   0.758839 0.025180         5
```

## TU Greedy-VOI holdout
```
           dataset  test_macro_f1_mean  test_macro_f1_sd  test_cost_mean  test_cost_sd  n
TU_Delft_2026_CPPS            0.830585          0.027968         0.80049      0.074596  3
```

## MSU Greedy-VOI holdout
```
       dataset  test_macro_f1_mean  test_macro_f1_sd  test_cost_mean  test_cost_sd  n
MSU_ORNL_Power              0.3484          0.019339             1.0           0.0  3
```
