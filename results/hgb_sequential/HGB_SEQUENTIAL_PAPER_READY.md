# ActiGrid — sequential-over-HGB reviewer hardening

## Protocol

- Same subset-specific HGB classifier family is used for both static and sequential policies.
- HGB hyperparameters match the reviewer-hardening frontier: 250 boosting iterations, learning rate 0.08, 31 max leaves, L2=1e-4, class-balanced sample weights.
- Original validation groups are split into a stratified calibration half and policy-selection half.
- Per-subset temperature scaling is fitted only on the calibration half.
- Acquisition order and entropy threshold are chosen only from the policy-selection half.
- Test data are used only for final evaluation.
- Group splits 42, 123, and 456 are evaluated independently.

## Split-level representative results
```
           dataset                          method  mean_test_macro_f1  sd_test_macro_f1  mean_test_cost  sd_test_cost
    MSU_ORNL_Power       Sequential HGB calibrated            0.439978          0.006596        0.860784      0.053155
    MSU_ORNL_Power              Sequential HGB raw            0.439539          0.005667        0.878576      0.053481
    MSU_ORNL_Power Static HGB (best validation F1)            0.437603          0.007699        0.933333      0.115470
TU_Delft_2026_CPPS       Sequential HGB calibrated            0.928200          0.037793        0.571185      0.135419
TU_Delft_2026_CPPS              Sequential HGB raw            0.925354          0.044345        0.519870      0.118104
TU_Delft_2026_CPPS Static HGB (best validation F1)            0.897175          0.058050        0.777778      0.192450
```

## Matched-budget summary
```
           dataset   budget  sequential_method  static_test_f1_mean  static_test_f1_sd  sequential_test_f1_mean  sequential_test_f1_sd  sequential_test_cost_mean  delta_f1_mean  delta_f1_sd
    MSU_ORNL_Power 0.200000 HGB-seq-calibrated             0.383408           0.009948                 0.383408               0.009948                   0.200000       0.000000     0.000000
    MSU_ORNL_Power 0.200000        HGB-seq-raw             0.383408           0.009948                 0.383408               0.009948                   0.200000       0.000000     0.000000
    MSU_ORNL_Power 0.400000 HGB-seq-calibrated             0.430124           0.008603                 0.406961               0.006350                   0.389469      -0.023164     0.006149
    MSU_ORNL_Power 0.400000        HGB-seq-raw             0.430124           0.008603                 0.412608               0.004469                   0.394395      -0.017516     0.008217
    MSU_ORNL_Power 0.600000 HGB-seq-calibrated             0.436750           0.004950                 0.431785               0.005884                   0.584901      -0.004965     0.007426
    MSU_ORNL_Power 0.600000        HGB-seq-raw             0.436750           0.004950                 0.432763               0.006849                   0.585231      -0.003987     0.007266
    MSU_ORNL_Power 0.800000 HGB-seq-calibrated             0.438968           0.006121                 0.438813               0.006553                   0.790952      -0.000154     0.007746
    MSU_ORNL_Power 0.800000        HGB-seq-raw             0.438968           0.006121                 0.438837               0.007321                   0.795952      -0.000131     0.007459
    MSU_ORNL_Power 1.000000 HGB-seq-calibrated             0.437603           0.007699                 0.439978               0.006596                   0.860784       0.002375     0.004497
    MSU_ORNL_Power 1.000000        HGB-seq-raw             0.437603           0.007699                 0.439539               0.005667                   0.878576       0.001936     0.004550
TU_Delft_2026_CPPS 0.333333 HGB-seq-calibrated             0.867267           0.027178                 0.867267               0.027178                   0.333333       0.000000     0.000000
TU_Delft_2026_CPPS 0.333333        HGB-seq-raw             0.867267           0.027178                 0.867267               0.027178                   0.333425       0.000000     0.000000
TU_Delft_2026_CPPS 0.666667 HGB-seq-calibrated             0.861752           0.021590                 0.928200               0.037793                   0.571185       0.066448     0.030563
TU_Delft_2026_CPPS 0.666667        HGB-seq-raw             0.861752           0.021590                 0.925354               0.044345                   0.519870       0.063602     0.031414
TU_Delft_2026_CPPS 1.000000 HGB-seq-calibrated             0.897175           0.058050                 0.928200               0.037793                   0.571185       0.031025     0.034327
TU_Delft_2026_CPPS 1.000000        HGB-seq-raw             0.897175           0.058050                 0.925354               0.044345                   0.519870       0.028179     0.043025
```

## HGB calibration summary
```
           dataset  mean_temperature  mean_calibration_ece_raw  mean_calibration_ece_scaled
    MSU_ORNL_Power          1.816099                  0.167855                     0.072473
TU_Delft_2026_CPPS          3.769406                  0.187107                     0.082111
```

Static-HGB reproduction check maximum absolute Macro-F1 difference versus rev-hard: 1.11022302463e-16.

## Interpretation template

Use the matched-budget table to decide the Reviewer-#2 claim. If calibrated sequential HGB improves Macro-F1 at the same or lower cost across splits, sequential acquisition remains useful once classifier-family confounding is removed. If the static HGB frontier still dominates, the appropriate conclusion is that the small modality spaces do not justify sequential acquisition on these benchmarks, even with a strong classifier family.

Do not compare only the single highest-F1 sequential point with a cheaper static subset; use matched budgets or the Pareto frontier.

## MSU/ORNL reporting note

The lower MSU/ORNL scores should be described as performance under strict whole-file group-disjoint generalization, not as evidence that the dataset is intrinsically difficult or that earlier work leaked data. A literature citation is required before contrasting this protocol with prior random/row-level splits.