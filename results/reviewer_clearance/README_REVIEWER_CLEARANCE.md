# ActiGrid reviewer-clearance outputs

These analyses address conference-critical reviewer questions without retraining the DQN.

## Interpretation rules
- Exhaustive fixed subsets and fixed-order entropy policies are selected on validation only.
- `POSTHOC-CostMatchedStaticMix` matches the already-observed DQN test cost and is descriptive only; do not present it as preregistered confirmatory evidence.
- The ten seeds reuse the frozen group-disjoint split, so uncertainty is over stochastic retraining, not new substations/splits.
- If a new simple baseline matches or beats DQN on TU Delft, revise the paper claim rather than hiding the result.

## MSU_ORNL_Power
### Validation-selected best fixed subset by size
```
       dataset  k  test_macro_f1_mean  test_macro_f1_std  test_macro_f1_ci95  normalized_cost_mean  normalized_cost_std  normalized_cost_ci95  n_seeds
MSU_ORNL_Power  1            0.322009           0.013091            0.009365                   0.2         0.000000e+00          0.000000e+00       10
MSU_ORNL_Power  2            0.393097           0.009346            0.006685                   0.4         0.000000e+00          0.000000e+00       10
MSU_ORNL_Power  3            0.401085           0.010102            0.007227                   0.6         1.170278e-16          8.371663e-17       10
MSU_ORNL_Power  4            0.398658           0.009451            0.006761                   0.8         0.000000e+00          0.000000e+00       10
MSU_ORNL_Power  5            0.392381           0.007717            0.005520                   1.0         0.000000e+00          0.000000e+00       10
```
### Best fixed-order + entropy stopping
```
 seed                                              order  threshold  val_cost  test_cost  test_macro_f1
   42 pmu_r4 -> pmu_r1 -> cyber_logs -> pmu_r2 -> pmu_r3   1.071367  0.767417   0.782397       0.391966
  123 pmu_r4 -> pmu_r1 -> pmu_r2 -> cyber_logs -> pmu_r3   1.078350  0.775419   0.793143       0.354994
  456 pmu_r4 -> pmu_r1 -> pmu_r3 -> pmu_r2 -> cyber_logs   1.026700  0.754780   0.774300       0.385411
  789 pmu_r4 -> pmu_r1 -> pmu_r2 -> pmu_r3 -> cyber_logs   0.943074  0.770836   0.788308       0.399313
 2027 pmu_r4 -> pmu_r1 -> pmu_r2 -> cyber_logs -> pmu_r3   1.085179  0.769120   0.788896       0.383448
31415 pmu_r4 -> pmu_r1 -> pmu_r2 -> pmu_r3 -> cyber_logs   0.971365  0.771765   0.787655       0.384989
27182 pmu_r4 -> pmu_r1 -> pmu_r2 -> pmu_r3 -> cyber_logs   0.980733  0.776244   0.790994       0.393047
16180 pmu_r4 -> pmu_r1 -> pmu_r2 -> pmu_r3 -> cyber_logs   1.005875  0.768753   0.784585       0.387035
14142 pmu_r4 -> pmu_r1 -> cyber_logs -> pmu_r2 -> pmu_r3   1.011487  0.768478   0.781579       0.401432
17320 pmu_r4 -> pmu_r1 -> pmu_r2 -> pmu_r3 -> cyber_logs   1.053639  0.776716   0.796137       0.377034
```
### Validation-selected fixed-order performance-cost curve
```
       dataset  target_budget  test_macro_f1_mean  test_macro_f1_std  test_macro_f1_ci95  test_cost_mean  test_cost_std  test_cost_ci95  n_seeds
MSU_ORNL_Power            0.2            0.323012           0.012755            0.009125        0.204005       0.004578        0.003275       10
MSU_ORNL_Power            0.3            0.344044           0.012133            0.008679        0.312864       0.007294        0.005218       10
MSU_ORNL_Power            0.4            0.357697           0.010627            0.007602        0.416556       0.005376        0.003846       10
MSU_ORNL_Power            0.5            0.366291           0.008727            0.006243        0.513150       0.010327        0.007388       10
MSU_ORNL_Power            0.6            0.374709           0.010364            0.007414        0.612575       0.019066        0.013639       10
MSU_ORNL_Power            0.7            0.381274           0.012472            0.008922        0.719962       0.005974        0.004273       10
MSU_ORNL_Power            0.8            0.387754           0.012646            0.009046        0.820889       0.004550        0.003255       10
MSU_ORNL_Power            0.9            0.393501           0.009400            0.006724        0.903001       0.005224        0.003737       10
MSU_ORNL_Power            1.0            0.394053           0.008652            0.006189        0.913103       0.008686        0.006214       10
```
### POST-HOC static mixture at observed DQN cost
```
 seed  target_dqn_cost  normalized_cost  macro_f1               low_subset                     high_subset
   42         0.767507         0.767651  0.404313 pmu_r1+pmu_r4+cyber_logs pmu_r1+pmu_r3+pmu_r4+cyber_logs
  123         0.748100         0.747997  0.388125     pmu_r1+pmu_r2+pmu_r4 pmu_r1+pmu_r2+pmu_r4+cyber_logs
  456         0.736574         0.736628  0.391184 pmu_r1+pmu_r4+cyber_logs pmu_r1+pmu_r2+pmu_r4+cyber_logs
  789         0.761456         0.761512  0.403653 pmu_r1+pmu_r4+cyber_logs pmu_r1+pmu_r2+pmu_r4+cyber_logs
 2027         0.761737         0.761540  0.395739     pmu_r1+pmu_r2+pmu_r4     pmu_r1+pmu_r2+pmu_r3+pmu_r4
31415         0.757567         0.757492  0.407000 pmu_r1+pmu_r4+cyber_logs pmu_r1+pmu_r2+pmu_r4+cyber_logs
27182         0.754689         0.754469  0.409610 pmu_r1+pmu_r4+cyber_logs pmu_r1+pmu_r2+pmu_r4+cyber_logs
16180         0.761264         0.761354  0.400436 pmu_r1+pmu_r4+cyber_logs pmu_r1+pmu_r2+pmu_r4+cyber_logs
14142         0.716899         0.716945  0.409478 pmu_r1+pmu_r4+cyber_logs pmu_r1+pmu_r2+pmu_r4+cyber_logs
17320         0.713522         0.713760  0.384185     pmu_r1+pmu_r3+pmu_r4     pmu_r1+pmu_r2+pmu_r3+pmu_r4
```
## TU_Delft_2026_CPPS
### Validation-selected best fixed subset by size
```
           dataset  k  test_macro_f1_mean  test_macro_f1_std  test_macro_f1_ci95  normalized_cost_mean  normalized_cost_std  normalized_cost_ci95  n_seeds
TU_Delft_2026_CPPS  1            0.562856           0.030513            0.021828              0.333333         5.851389e-17          4.185832e-17       10
TU_Delft_2026_CPPS  2            0.693306           0.058345            0.041738              0.666667         1.170278e-16          8.371663e-17       10
TU_Delft_2026_CPPS  3            0.815825           0.032259            0.023077              1.000000         0.000000e+00          0.000000e+00       10
```
### Best fixed-order + entropy stopping
```
 seed                           order  threshold  val_cost  test_cost  test_macro_f1
   42 resource -> physical -> network   0.489710  0.574974   0.691996       0.778750
  123 resource -> physical -> network   0.295195  0.606061   0.694529       0.800962
  456 resource -> physical -> network   0.471774  0.603710   0.675025       0.766417
  789 resource -> physical -> network   0.596809  0.595873   0.691996       0.834976
 2027 resource -> network -> physical   0.726385  0.577586   0.630193       0.807452
31415 resource -> physical -> network   0.740480  0.603971   0.696049       0.780163
27182 resource -> physical -> network   0.253211  0.598746   0.722898       0.782472
16180 resource -> physical -> network   0.408517  0.605538   0.690983       0.824530
14142 resource -> physical -> network   0.615891  0.581243   0.697062       0.782231
17320 resource -> physical -> network   0.496790  0.599530   0.704914       0.710136
```
### Validation-selected fixed-order performance-cost curve
```
           dataset  target_budget  test_macro_f1_mean  test_macro_f1_std  test_macro_f1_ci95  test_cost_mean  test_cost_std  test_cost_ci95  n_seeds
TU_Delft_2026_CPPS       0.333333            0.565762           0.031164            0.022293        0.335410       0.004419        0.003161       10
TU_Delft_2026_CPPS       0.416667            0.659505           0.054462            0.038960        0.434043       0.025563        0.018287       10
TU_Delft_2026_CPPS       0.500000            0.734339           0.050702            0.036270        0.551039       0.036365        0.026014       10
TU_Delft_2026_CPPS       0.583333            0.771753           0.041129            0.029422        0.645111       0.046290        0.033114       10
TU_Delft_2026_CPPS       0.666667            0.812046           0.033285            0.023811        0.757168       0.018924        0.013538       10
TU_Delft_2026_CPPS       0.750000            0.808189           0.040205            0.028761        0.774139       0.031221        0.022334       10
TU_Delft_2026_CPPS       0.833333            0.812172           0.033706            0.024112        0.782219       0.038362        0.027442       10
TU_Delft_2026_CPPS       0.916667            0.812172           0.033706            0.024112        0.782219       0.038362        0.027442       10
TU_Delft_2026_CPPS       1.000000            0.812172           0.033706            0.024112        0.782219       0.038362        0.027442       10
```
### Network-first entropy stopping
```
 seed                           order  threshold  val_cost  test_cost  test_macro_f1
   42 network -> physical -> resource   0.647214  0.584117   0.584093       0.603113
  123 network -> physical -> resource   0.520064  0.600313   0.601824       0.614495
  456 network -> physical -> resource   0.715006  0.601097   0.566616       0.533795
  789 network -> physical -> resource   0.658166  0.600836   0.588399       0.612275
 2027 network -> physical -> resource   0.686082  0.605799   0.624873       0.670293
31415 network -> physical -> resource   0.740480  0.605016   0.611449       0.615977
27182 network -> physical -> resource   0.754580  0.582027   0.576494       0.700159
16180 network -> physical -> resource   0.659002  0.594305   0.607396       0.620779
14142 network -> physical -> resource   0.782068  0.599530   0.599797       0.672733
17320 network -> physical -> resource   0.754747  0.587774   0.572948       0.543169
```
### POST-HOC static mixture at observed DQN cost
```
 seed  target_dqn_cost  normalized_cost  macro_f1       low_subset               high_subset
   42         0.775329         0.774430  0.679066 resource+network physical+resource+network
  123         0.691489         0.691426  0.687619 resource+network physical+resource+network
  456         0.741641         0.742427  0.766399 resource+network physical+resource+network
  789         0.686930         0.688019  0.776626 resource+network physical+resource+network
 2027         0.694529         0.694364  0.784432 resource+network physical+resource+network
31415         0.825481         0.825519  0.740556 resource+network physical+resource+network
27182         0.640578         0.639995  0.665136          network          resource+network
16180         0.780395         0.780332  0.732157 resource+network physical+resource+network
14142         0.690983         0.691489  0.670227 resource+network physical+resource+network
17320         0.671479         0.671479  0.599126 resource+network physical+resource+network
```
### TU resource-exhaustion exclusion (seed 42 sensitivity)
```
           dataset              method      excluded_class  n_remaining  macro_f1_5class  balanced_accuracy_5class
TU_Delft_2026_CPPS ActiGrid-DQN-seed42 resource_exhaustion         1011         0.763142                  0.841300
TU_Delft_2026_CPPS      Entropy-seed42 resource_exhaustion         1011         0.668818                  0.790969
TU_Delft_2026_CPPS  Fusion-Full-seed42 resource_exhaustion         1011         0.756733                  0.839824
```
## Corrected multiplicity reporting
```
           dataset             comparator  wilcoxon_p_macro_f1  holm_p_macro_f1  wilcoxon_p_cost  holm_p_cost
    MSU_ORNL_Power UtilityEntropy-Matched             0.083984         0.167969         0.001953     0.009766
    MSU_ORNL_Power   UtilityNearestBudget             0.003906         0.019531         0.001953     0.009766
    MSU_ORNL_Power    RandomNearestBudget             0.695312         0.695312         0.001953     0.009766
    MSU_ORNL_Power CostOrderNearestBudget             0.003906         0.019531         0.001953     0.009766
    MSU_ORNL_Power            Fusion-Full             0.003906         0.019531         0.001953     0.009766
TU_Delft_2026_CPPS UtilityEntropy-Matched             0.001953         0.009766         0.005859     0.023438
TU_Delft_2026_CPPS   UtilityNearestBudget             0.001953         0.009766         0.019531     0.058594
TU_Delft_2026_CPPS    RandomNearestBudget             0.001953         0.009766         0.019531     0.058594
TU_Delft_2026_CPPS CostOrderNearestBudget             0.001953         0.009766         0.019531     0.058594
TU_Delft_2026_CPPS            Fusion-Full             0.048828         0.048828         0.001953     0.009766
```