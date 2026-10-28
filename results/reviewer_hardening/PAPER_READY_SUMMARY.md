# ActiGrid reviewer-hardening summary

Use only outputs that completed successfully. Do not reinterpret normalized cost as bytes/latency/energy.
Ten seeds on one split quantify training variability; repeated-split outputs quantify split sensitivity separately.

### Repeated group-split sensitivity
```
           dataset          method  macro_f1_mean  macro_f1_sd_split  cost_mean  cost_sd_split  n_splits
TU_Delft_2026_CPPS BestFixedSubset       0.833447           0.050623   1.000000       0.000000         3
TU_Delft_2026_CPPS             DQN       0.804519           0.016498   0.706416       0.075621         3
TU_Delft_2026_CPPS    OrderEntropy       0.809518           0.011293   0.731610       0.072950         3
```

### Strong subset-specific classifier examples
```
           dataset  split_seed model  k                    subset  normalized_cost  val_macro_f1  test_macro_f1  test_balanced_accuracy
TU_Delft_2026_CPPS         123   HGB  3 physical+resource+network         1.000000      0.926370       0.959922                0.958280
TU_Delft_2026_CPPS         123    RF  3 physical+resource+network         1.000000      0.920793       0.958089                0.947283
TU_Delft_2026_CPPS         123   XGB  3 physical+resource+network         1.000000      0.902346       0.957309                0.955352
TU_Delft_2026_CPPS         456    RF  3 physical+resource+network         1.000000      0.883273       0.907901                0.936759
TU_Delft_2026_CPPS         456   XGB  3 physical+resource+network         1.000000      0.894944       0.904519                0.931451
TU_Delft_2026_CPPS         123   HGB  2          resource+network         0.666667      0.881902       0.901963                0.914188
TU_Delft_2026_CPPS         456    RF  1                   network         0.333333      0.827368       0.899770                0.892250
TU_Delft_2026_CPPS         456   HGB  1                   network         0.333333      0.854424       0.898562                0.892737
TU_Delft_2026_CPPS         456   HGB  3 physical+resource+network         1.000000      0.897722       0.892592                0.933330
TU_Delft_2026_CPPS         123   XGB  2          resource+network         0.666667      0.876879       0.892181                0.894040
TU_Delft_2026_CPPS         456   XGB  1                   network         0.333333      0.842602       0.891933                0.883288
TU_Delft_2026_CPPS         123    RF  2          resource+network         0.666667      0.870339       0.886909                0.884378
```

### Repeated group-split sensitivity
```
       dataset          method  macro_f1_mean  macro_f1_sd_split  cost_mean  cost_sd_split  n_splits
MSU_ORNL_Power BestFixedSubset       0.397057           0.030804   0.800000       0.000000         3
MSU_ORNL_Power             DQN       0.363175           0.026817   0.728120       0.026997         3
MSU_ORNL_Power    OrderEntropy       0.380041           0.026435   0.751311       0.038785         3
```

### Strong subset-specific classifier examples
```
       dataset  split_seed model  k                                 subset  normalized_cost  val_macro_f1  test_macro_f1  test_balanced_accuracy
MSU_ORNL_Power         123   HGB  4            pmu_r1+pmu_r2+pmu_r3+pmu_r4              0.8      0.416947       0.449416                0.457927
MSU_ORNL_Power         123   HGB  5 pmu_r1+pmu_r2+pmu_r3+pmu_r4+cyber_logs              1.0      0.418419       0.446413                0.455044
MSU_ORNL_Power         123   HGB  3                   pmu_r1+pmu_r3+pmu_r4              0.6      0.417121       0.444362                0.452164
MSU_ORNL_Power         456   HGB  3                   pmu_r1+pmu_r3+pmu_r4              0.6      0.424338       0.441243                0.471521
MSU_ORNL_Power         123   HGB  2                          pmu_r1+pmu_r4              0.4      0.411636       0.440045                0.445773
MSU_ORNL_Power          42   HGB  5 pmu_r1+pmu_r2+pmu_r3+pmu_r4+cyber_logs              1.0      0.425235       0.440033                0.458981
MSU_ORNL_Power         456   HGB  5 pmu_r1+pmu_r2+pmu_r3+pmu_r4+cyber_logs              1.0      0.428666       0.434227                0.459513
MSU_ORNL_Power         456   XGB  4            pmu_r1+pmu_r2+pmu_r3+pmu_r4              0.8      0.406414       0.433012                0.435954
MSU_ORNL_Power         456   HGB  4        pmu_r1+pmu_r3+pmu_r4+cyber_logs              0.8      0.426792       0.432216                0.459338
MSU_ORNL_Power          42   HGB  4            pmu_r1+pmu_r2+pmu_r3+pmu_r4              0.8      0.427322       0.432170                0.449890
MSU_ORNL_Power          42   XGB  5 pmu_r1+pmu_r2+pmu_r3+pmu_r4+cyber_logs              1.0      0.415001       0.432071                0.429687
MSU_ORNL_Power         456   XGB  3                   pmu_r1+pmu_r2+pmu_r4              0.6      0.408273       0.431890                0.435877
```
