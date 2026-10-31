# ActiGrid Final Reviewer-Clearance Audit

This report separates measured findings from hypotheses. No claim of equivalence is made.

## TU_Delft_2026_CPPS: train-vs-validation posterior audit
```
 seed  mean_train_minus_val_confidence  mean_train_minus_val_entropy
   42                        -0.016868                      0.026131
  123                        -0.021142                      0.031634
  456                        -0.010161                      0.012217
```

## MSU_ORNL_Power: train-vs-validation posterior audit
```
 seed  mean_train_minus_val_confidence  mean_train_minus_val_entropy
   42                         0.009493                     -0.020176
  123                         0.005400                     -0.011750
  456                         0.007696                     -0.017104
```

## MSU preprocessing hygiene
```
  modality split  rows  rows_with_nan_or_inf  cells_nan_or_inf
    pmu_r1 train 47471                  1781              1781
    pmu_r1   val 15272                   551               551
    pmu_r1  test 15634                   545               545
    pmu_r2 train 47471                  1606              1606
    pmu_r2   val 15272                   481               481
    pmu_r2  test 15634                   520               520
    pmu_r3 train 47471                  1639              1639
    pmu_r3   val 15272                   467               467
    pmu_r3  test 15634                   527               527
    pmu_r4 train 47471                  1757              1757
    pmu_r4   val 15272                   529               529
    pmu_r4  test 15634                   503               503
cyber_logs train 47471                     0                 0
cyber_logs   val 15272                     0                 0
cyber_logs  test 15634                     0                 0
```
```
 n_rows  exact_duplicate_hashes_across_files  rows_in_cross_file_duplicate_groups  duplicate_hashes_crossing_train_val_test  rows_in_cross_split_duplicate_groups  duplicate_hashes_with_conflicting_labels
  78377                                    1                                    2                                         1                                     2                                         0
```

### MSU HGB split/preprocessing sanity
```
                   protocol         preprocessing  accuracy  balanced_accuracy  macro_f1
              file_disjoint current_pooled_median  0.461750           0.458981  0.440033
              file_disjoint     train_only_median  0.460151           0.453981  0.437049
stratified_random_row_80_20     train_only_median  0.775581           0.812628  0.769722
```

- Train-only imputation changed file-disjoint HGB Macro-F1 by -0.0030.
- Random-row sanity HGB Macro-F1: 0.7697 versus corrected file-disjoint 0.4370.
- Interpret this as protocol sensitivity, not proof of leakage in prior papers.

## Single-modality neural vs HGB
### TU_Delft_2026_CPPS
```
           dataset modality  neural_masked_macro_f1  HGB_subset_macro_f1  HGB_minus_neural
TU_Delft_2026_CPPS physical                0.345982             0.373315          0.027332
TU_Delft_2026_CPPS resource                0.539400             0.534811         -0.004589
TU_Delft_2026_CPPS  network                0.443306             0.849588          0.406282
```

### MSU_ORNL_Power
```
       dataset   modality  neural_masked_macro_f1  HGB_subset_macro_f1  HGB_minus_neural
MSU_ORNL_Power     pmu_r1                0.266783             0.350136          0.083353
MSU_ORNL_Power     pmu_r2                0.234449             0.385364          0.150915
MSU_ORNL_Power     pmu_r3                0.160744             0.353408          0.192664
MSU_ORNL_Power     pmu_r4                0.331113             0.377708          0.046594
MSU_ORNL_Power cyber_logs                0.086959             0.088529          0.001571
```

## Backbone/policy holdout DQN control
### TU_Delft_2026_CPPS
```
           dataset  seed                     method holdout_unit  group_overlap  backbone_groups  policy_groups  backbone_train_rows  policy_train_rows  val_macro_f1  val_cost  test_macro_f1  test_cost                     order  threshold
TU_Delft_2026_CPPS    42          PolicyHoldout-DQN          row             47               47             47                 2272                974      0.666817  0.737983       0.770720   0.729737                       NaN        NaN
TU_Delft_2026_CPPS    42 PolicyHoldout-OrderEntropy          row             47               47             47                 2272                974      0.767613  0.614420       0.792996   0.741135 resource>physical>network   0.381160
TU_Delft_2026_CPPS    42   PolicyHoldout-FullFusion          row             47               47             47                 2272                974           NaN  1.000000       0.799204   1.000000                       NaN        NaN
TU_Delft_2026_CPPS   123          PolicyHoldout-DQN          row             47               47             47                 2272                974      0.767318  0.617816       0.814333   0.706940                       NaN        NaN
TU_Delft_2026_CPPS   123 PolicyHoldout-OrderEntropy          row             47               47             47                 2272                974      0.803641  0.626437       0.825004   0.717072 resource>physical>network   0.535480
TU_Delft_2026_CPPS   123   PolicyHoldout-FullFusion          row             47               47             47                 2272                974           NaN  1.000000       0.856064   1.000000                       NaN        NaN
TU_Delft_2026_CPPS   456          PolicyHoldout-DQN          row             47               47             47                 2272                974      0.782715  0.613898       0.799739   0.665907                       NaN        NaN
TU_Delft_2026_CPPS   456 PolicyHoldout-OrderEntropy          row             47               47             47                 2272                974      0.835962  0.613375       0.852776   0.705167 resource>physical>network   0.254772
TU_Delft_2026_CPPS   456   PolicyHoldout-FullFusion          row             47               47             47                 2272                974           NaN  1.000000       0.850106   1.000000                       NaN        NaN
```

Original hardening DQN lambda=0.5 mean Macro-F1=0.8007, cost=0.6795.

### MSU_ORNL_Power
```
       dataset  seed                     method holdout_unit  group_overlap  backbone_groups  policy_groups  backbone_train_rows  policy_train_rows  val_macro_f1  val_cost  test_macro_f1  test_cost                                  order  threshold
MSU_ORNL_Power    42          PolicyHoldout-DQN        group              0                6              3                31311              16160      0.361211  0.626938       0.355672   0.626481                                    NaN        NaN
MSU_ORNL_Power    42 PolicyHoldout-OrderEntropy        group              0                6              3                31311              16160      0.346529  0.643766       0.350530   0.649827 cyber_logs>pmu_r4>pmu_r1>pmu_r2>pmu_r3   1.270078
MSU_ORNL_Power    42   PolicyHoldout-FullFusion        group              0                6              3                31311              16160           NaN  1.000000       0.374662   1.000000                                    NaN        NaN
MSU_ORNL_Power   123          PolicyHoldout-DQN        group              0                6              3                32050              15421      0.304469  0.689039       0.328290   0.688794                                    NaN        NaN
MSU_ORNL_Power   123 PolicyHoldout-OrderEntropy        group              0                6              3                32050              15421      0.344218  0.694591       0.364296   0.692926 pmu_r4>pmu_r1>pmu_r2>cyber_logs>pmu_r3   1.087204
MSU_ORNL_Power   123   PolicyHoldout-FullFusion        group              0                6              3                32050              15421           NaN  1.000000       0.371275   1.000000                                    NaN        NaN
MSU_ORNL_Power   456          PolicyHoldout-DQN        group              0                6              3                31563              15908      0.309146  0.749699       0.316212   0.747704                                    NaN        NaN
MSU_ORNL_Power   456 PolicyHoldout-OrderEntropy        group              0                6              3                31563              15908      0.312338  0.768714       0.317858   0.773059 pmu_r3>pmu_r4>pmu_r1>cyber_logs>pmu_r2   1.266104
MSU_ORNL_Power   456   PolicyHoldout-FullFusion        group              0                6              3                31563              15908           NaN  1.000000       0.343687   1.000000                                    NaN        NaN
```

Original hardening DQN lambda=0.5 mean Macro-F1=0.3700, cost=0.7584.

## Automatic interpretation flags
- TU_Delft_2026_CPPS posterior coupling flag: **NO** (mean train-val confidence=-0.0161, entropy=+0.0233).
- MSU_ORNL_Power posterior coupling flag: **NO** (mean train-val confidence=+0.0075, entropy=-0.0163).
- MSU exact cross-split duplicate-hash groups: **1**.
- MSU train-only-imputation materiality flag (|Delta HGB F1| >= 0.005): **NO**.

## Paper-use guidance
- If the backbone/policy holdout DQN conclusion is similar to the original DQN conclusion, describe M1 as a checked sensitivity rather than a fatal protocol flaw. State whether the holdout unit was group or row.
- If corrected MSU HGB differs materially, use corrected MSU preprocessing/results and do not report the pooled-imputation numbers as final.
- Report MSU random-row versus file-disjoint results only as a sanity comparison; do not call random-row evaluation leakage unless overlap is directly demonstrated.
- The old MSU three-file cluster-bootstrap interval is too coarse for a strong inferential claim; omit it from headline conclusions.
- For the Pareto statement, write 'none of the tested DQN operating points' rather than a universal claim.
- Use the already-generated sequential-HGB matched-budget table for the static-vs-sequential tree comparison.
