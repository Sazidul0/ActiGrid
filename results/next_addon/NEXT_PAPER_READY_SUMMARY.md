# ActiGrid NEXT add-on summary

No fusion/DQN/tree models were retrained in this add-on.
Greedy-VOI uses frozen fusion checkpoints and lightweight regressors trained to predict one-step entropy reduction.
It is a non-RL baseline, not a reproduction of JAFA/GSM/ACO.

## TU_Delft_2026_CPPS
### Pareto points (same fixed split/backbone seeds)
```
 seed  macro_f1  normalized_cost           method_family                                   setting            dataset  pareto
   42  0.539400         0.333333              Greedy-VOI                                 H<=2.7582 TU_Delft_2026_CPPS    True
   42  0.539400         0.333333 Calibrated-OrderEntropy resource -> physical -> network;H<=2.7833 TU_Delft_2026_CPPS    True
   42  0.539400         0.333333 Calibrated-OrderEntropy resource -> network -> physical;H<=2.7833 TU_Delft_2026_CPPS    True
   42  0.621918         0.402989              Greedy-VOI                                 H<=1.1485 TU_Delft_2026_CPPS    True
   42  0.702261         0.450608 Calibrated-OrderEntropy resource -> physical -> network;H<=1.4728 TU_Delft_2026_CPPS    True
   42  0.738001         0.513171              Greedy-VOI                                 H<=0.9795 TU_Delft_2026_CPPS    True
   42  0.755002         0.548886              Greedy-VOI                                 H<=0.9096 TU_Delft_2026_CPPS    True
   42  0.767612         0.573961 Calibrated-OrderEntropy resource -> physical -> network;H<=1.3273 TU_Delft_2026_CPPS    True
   42  0.772386         0.611955 Calibrated-OrderEntropy resource -> physical -> network;H<=1.2707 TU_Delft_2026_CPPS    True
   42  0.773590         0.637538              Greedy-VOI                                 H<=0.6430 TU_Delft_2026_CPPS    True
   42  0.782269         0.681611              Greedy-VOI                                 H<=0.5356 TU_Delft_2026_CPPS    True
   42  0.788728         0.730750              Greedy-VOI                                 H<=0.4432 TU_Delft_2026_CPPS    True
   42  0.794652         0.766211              Greedy-VOI                                 H<=0.2938 TU_Delft_2026_CPPS    True
   42  0.795289         0.789514              Greedy-VOI                                 H<=0.1549 TU_Delft_2026_CPPS    True
   42  0.797277         0.791540 Calibrated-OrderEntropy resource -> physical -> network;H<=0.8012 TU_Delft_2026_CPPS    True
  123  0.571361         0.333333 Calibrated-OrderEntropy resource -> physical -> network;H<=2.7053 TU_Delft_2026_CPPS    True
  123  0.571361         0.333333 Calibrated-OrderEntropy resource -> network -> physical;H<=2.7053 TU_Delft_2026_CPPS    True
  123  0.580661         0.463526              Greedy-VOI                                 H<=0.7548 TU_Delft_2026_CPPS    True
  123  0.588263         0.477710              Greedy-VOI                                 H<=0.7214 TU_Delft_2026_CPPS    True
  123  0.647445         0.479230 Calibrated-OrderEntropy resource -> physical -> network;H<=1.5489 TU_Delft_2026_CPPS    True
  123  0.715546         0.533688 Calibrated-OrderEntropy resource -> physical -> network;H<=1.4111 TU_Delft_2026_CPPS    True
  123  0.724708         0.589412 Calibrated-OrderEntropy resource -> physical -> network;H<=1.3260 TU_Delft_2026_CPPS    True
  123  0.780977         0.624113 Calibrated-OrderEntropy resource -> physical -> network;H<=1.2261 TU_Delft_2026_CPPS    True
  123  0.781336         0.649189 Calibrated-OrderEntropy resource -> physical -> network;H<=1.1435 TU_Delft_2026_CPPS    True
  123  0.808519         0.654002 Calibrated-OrderEntropy resource -> network -> physical;H<=1.2261 TU_Delft_2026_CPPS    True
  123  0.814945         0.673252 Calibrated-OrderEntropy resource -> network -> physical;H<=1.1435 TU_Delft_2026_CPPS    True
  123  0.827203         0.696302 Calibrated-OrderEntropy resource -> network -> physical;H<=1.0849 TU_Delft_2026_CPPS    True
  123  0.830590         0.730496 Calibrated-OrderEntropy resource -> network -> physical;H<=1.0300 TU_Delft_2026_CPPS    True
  123  0.834559         0.793313 Calibrated-OrderEntropy resource -> network -> physical;H<=0.9125 TU_Delft_2026_CPPS    True
  123  0.840850         0.918946              Greedy-VOI                                 H<=0.0166 TU_Delft_2026_CPPS    True
  456  0.582385         0.333333 Calibrated-OrderEntropy resource -> physical -> network;H<=2.7143 TU_Delft_2026_CPPS    True
  456  0.582385         0.333333 Calibrated-OrderEntropy resource -> network -> physical;H<=2.7143 TU_Delft_2026_CPPS    True
  456  0.590481         0.352584 Calibrated-OrderEntropy resource -> physical -> network;H<=1.4730 TU_Delft_2026_CPPS    True
  456  0.631499         0.500253 Calibrated-OrderEntropy resource -> physical -> network;H<=1.3883 TU_Delft_2026_CPPS    True
  456  0.667235         0.517730 Calibrated-OrderEntropy resource -> network -> physical;H<=1.3883 TU_Delft_2026_CPPS    True
  456  0.746010         0.591945 Calibrated-OrderEntropy resource -> network -> physical;H<=1.2592 TU_Delft_2026_CPPS    True
  456  0.769719         0.638804 Calibrated-OrderEntropy resource -> network -> physical;H<=1.1691 TU_Delft_2026_CPPS    True
  456  0.791475         0.669200 Calibrated-OrderEntropy resource -> network -> physical;H<=1.1091 TU_Delft_2026_CPPS    True
  456  0.812822         0.690730                     DQN                               lambda=0.25 TU_Delft_2026_CPPS    True
  456  0.828644         0.715046                     DQN                                lambda=0.5 TU_Delft_2026_CPPS    True
  456  0.828824         0.719098                     DQN                               lambda=0.75 TU_Delft_2026_CPPS    True
  456  0.829770         0.894377              Greedy-VOI                                 H<=0.0237 TU_Delft_2026_CPPS    True
```

### Observed-range AUC summary
```
           dataset  seed           method_family  observed_auc  cost_min  cost_max  mean_f1_over_observed_cost_range  n_points
TU_Delft_2026_CPPS    42 Calibrated-OrderEntropy      0.383181  0.333333  0.861449                          0.725563        11
TU_Delft_2026_CPPS    42                     DQN      0.077149  0.636272  0.737589                          0.761457         3
TU_Delft_2026_CPPS    42              Greedy-VOI      0.497029  0.333333  1.000000                          0.745544        17
TU_Delft_2026_CPPS   123 Calibrated-OrderEntropy      0.363148  0.333333  0.842199                          0.713643        17
TU_Delft_2026_CPPS   123                     DQN      0.074527  0.687183  0.778875                          0.812802         3
TU_Delft_2026_CPPS   123              Greedy-VOI      0.455361  0.333333  1.000000                          0.683042        17
TU_Delft_2026_CPPS   456 Calibrated-OrderEntropy      0.294393  0.333333  0.756079                          0.696382        11
TU_Delft_2026_CPPS   456                     DQN      0.023316  0.690730  0.719098                          0.821876         3
TU_Delft_2026_CPPS   456              Greedy-VOI      0.461575  0.333333  1.000000                          0.692362        17
```

## MSU_ORNL_Power
### Pareto points (same fixed split/backbone seeds)
```
 seed  macro_f1  normalized_cost           method_family                                                      setting        dataset  pareto
   42  0.331113         0.200000 Calibrated-OrderEntropy pmu_r4 -> pmu_r1 -> pmu_r2 -> pmu_r3 -> cyber_logs;H<=2.6094 MSU_ORNL_Power    True
   42  0.331113         0.200000 Calibrated-OrderEntropy pmu_r4 -> pmu_r1 -> pmu_r2 -> cyber_logs -> pmu_r3;H<=2.6094 MSU_ORNL_Power    True
   42  0.331113         0.200000 Calibrated-OrderEntropy pmu_r4 -> pmu_r1 -> pmu_r3 -> pmu_r2 -> cyber_logs;H<=2.6094 MSU_ORNL_Power    True
   42  0.331113         0.200000 Calibrated-OrderEntropy pmu_r4 -> pmu_r1 -> pmu_r3 -> cyber_logs -> pmu_r2;H<=2.6094 MSU_ORNL_Power    True
   42  0.331113         0.200000 Calibrated-OrderEntropy pmu_r4 -> pmu_r1 -> cyber_logs -> pmu_r2 -> pmu_r3;H<=2.6094 MSU_ORNL_Power    True
   42  0.331113         0.200000 Calibrated-OrderEntropy pmu_r4 -> pmu_r1 -> cyber_logs -> pmu_r3 -> pmu_r2;H<=2.6094 MSU_ORNL_Power    True
   42  0.331113         0.200000 Calibrated-OrderEntropy pmu_r4 -> pmu_r2 -> pmu_r1 -> pmu_r3 -> cyber_logs;H<=2.6094 MSU_ORNL_Power    True
   42  0.331113         0.200000 Calibrated-OrderEntropy pmu_r4 -> pmu_r2 -> pmu_r1 -> cyber_logs -> pmu_r3;H<=2.6094 MSU_ORNL_Power    True
   42  0.331113         0.200000 Calibrated-OrderEntropy pmu_r4 -> pmu_r2 -> pmu_r3 -> pmu_r1 -> cyber_logs;H<=2.6094 MSU_ORNL_Power    True
   42  0.331113         0.200000 Calibrated-OrderEntropy pmu_r4 -> pmu_r2 -> pmu_r3 -> cyber_logs -> pmu_r1;H<=2.6094 MSU_ORNL_Power    True
   42  0.331113         0.200000 Calibrated-OrderEntropy pmu_r4 -> pmu_r2 -> cyber_logs -> pmu_r1 -> pmu_r3;H<=2.6094 MSU_ORNL_Power    True
   42  0.331113         0.200000 Calibrated-OrderEntropy pmu_r4 -> pmu_r2 -> cyber_logs -> pmu_r3 -> pmu_r1;H<=2.6094 MSU_ORNL_Power    True
   42  0.331113         0.200000 Calibrated-OrderEntropy pmu_r4 -> pmu_r3 -> pmu_r1 -> pmu_r2 -> cyber_logs;H<=2.6094 MSU_ORNL_Power    True
   42  0.331113         0.200000 Calibrated-OrderEntropy pmu_r4 -> pmu_r3 -> pmu_r1 -> cyber_logs -> pmu_r2;H<=2.6094 MSU_ORNL_Power    True
   42  0.331113         0.200000 Calibrated-OrderEntropy pmu_r4 -> pmu_r3 -> pmu_r2 -> pmu_r1 -> cyber_logs;H<=2.6094 MSU_ORNL_Power    True
   42  0.331113         0.200000 Calibrated-OrderEntropy pmu_r4 -> pmu_r3 -> pmu_r2 -> cyber_logs -> pmu_r1;H<=2.6094 MSU_ORNL_Power    True
   42  0.331113         0.200000 Calibrated-OrderEntropy pmu_r4 -> pmu_r3 -> cyber_logs -> pmu_r1 -> pmu_r2;H<=2.6094 MSU_ORNL_Power    True
   42  0.331113         0.200000 Calibrated-OrderEntropy pmu_r4 -> pmu_r3 -> cyber_logs -> pmu_r2 -> pmu_r1;H<=2.6094 MSU_ORNL_Power    True
   42  0.331113         0.200000 Calibrated-OrderEntropy pmu_r4 -> cyber_logs -> pmu_r1 -> pmu_r2 -> pmu_r3;H<=2.6094 MSU_ORNL_Power    True
   42  0.331113         0.200000 Calibrated-OrderEntropy pmu_r4 -> cyber_logs -> pmu_r1 -> pmu_r3 -> pmu_r2;H<=2.6094 MSU_ORNL_Power    True
   42  0.331113         0.200000 Calibrated-OrderEntropy pmu_r4 -> cyber_logs -> pmu_r2 -> pmu_r1 -> pmu_r3;H<=2.6094 MSU_ORNL_Power    True
   42  0.331113         0.200000 Calibrated-OrderEntropy pmu_r4 -> cyber_logs -> pmu_r2 -> pmu_r3 -> pmu_r1;H<=2.6094 MSU_ORNL_Power    True
   42  0.331113         0.200000 Calibrated-OrderEntropy pmu_r4 -> cyber_logs -> pmu_r3 -> pmu_r1 -> pmu_r2;H<=2.6094 MSU_ORNL_Power    True
   42  0.331113         0.200000 Calibrated-OrderEntropy pmu_r4 -> cyber_logs -> pmu_r3 -> pmu_r2 -> pmu_r1;H<=2.6094 MSU_ORNL_Power    True
   42  0.331275         0.200537 Calibrated-OrderEntropy pmu_r4 -> pmu_r1 -> pmu_r2 -> pmu_r3 -> cyber_logs;H<=1.6048 MSU_ORNL_Power    True
   42  0.331275         0.200537 Calibrated-OrderEntropy pmu_r4 -> pmu_r1 -> pmu_r2 -> cyber_logs -> pmu_r3;H<=1.6048 MSU_ORNL_Power    True
   42  0.331345         0.200921 Calibrated-OrderEntropy pmu_r4 -> cyber_logs -> pmu_r1 -> pmu_r2 -> pmu_r3;H<=1.6048 MSU_ORNL_Power    True
   42  0.333121         0.207407 Calibrated-OrderEntropy pmu_r4 -> pmu_r1 -> pmu_r2 -> pmu_r3 -> cyber_logs;H<=1.5900 MSU_ORNL_Power    True
   42  0.340179         0.241909 Calibrated-OrderEntropy pmu_r4 -> cyber_logs -> pmu_r1 -> pmu_r2 -> pmu_r3;H<=1.5637 MSU_ORNL_Power    True
   42  0.346156         0.265856 Calibrated-OrderEntropy pmu_r4 -> pmu_r1 -> cyber_logs -> pmu_r2 -> pmu_r3;H<=1.5298 MSU_ORNL_Power    True
   42  0.348142         0.287489 Calibrated-OrderEntropy pmu_r4 -> pmu_r1 -> pmu_r3 -> cyber_logs -> pmu_r2;H<=1.4913 MSU_ORNL_Power    True
   42  0.348429         0.314341 Calibrated-OrderEntropy pmu_r4 -> pmu_r1 -> pmu_r3 -> cyber_logs -> pmu_r2;H<=1.4567 MSU_ORNL_Power    True
   42  0.350355         0.316758 Calibrated-OrderEntropy pmu_r4 -> pmu_r1 -> pmu_r2 -> cyber_logs -> pmu_r3;H<=1.4567 MSU_ORNL_Power    True
   42  0.350380         0.316810 Calibrated-OrderEntropy pmu_r4 -> pmu_r1 -> pmu_r2 -> pmu_r3 -> cyber_logs;H<=1.4567 MSU_ORNL_Power    True
   42  0.350912         0.338467 Calibrated-OrderEntropy pmu_r4 -> pmu_r1 -> pmu_r3 -> cyber_logs -> pmu_r2;H<=1.4285 MSU_ORNL_Power    True
   42  0.353177         0.341717 Calibrated-OrderEntropy pmu_r4 -> pmu_r1 -> pmu_r2 -> pmu_r3 -> cyber_logs;H<=1.4285 MSU_ORNL_Power    True
   42  0.355892         0.363899 Calibrated-OrderEntropy pmu_r4 -> pmu_r1 -> pmu_r3 -> cyber_logs -> pmu_r2;H<=1.4063 MSU_ORNL_Power    True
   42  0.358376         0.367852 Calibrated-OrderEntropy pmu_r4 -> pmu_r1 -> pmu_r2 -> cyber_logs -> pmu_r3;H<=1.4063 MSU_ORNL_Power    True
   42  0.365510         0.395510 Calibrated-OrderEntropy pmu_r4 -> pmu_r1 -> pmu_r3 -> cyber_logs -> pmu_r2;H<=1.3906 MSU_ORNL_Power    True
   42  0.366048         0.406742 Calibrated-OrderEntropy pmu_r4 -> pmu_r1 -> cyber_logs -> pmu_r3 -> pmu_r2;H<=1.3906 MSU_ORNL_Power    True
   42  0.369500         0.455021 Calibrated-OrderEntropy pmu_r4 -> pmu_r1 -> pmu_r3 -> cyber_logs -> pmu_r2;H<=1.3792 MSU_ORNL_Power    True
   42  0.376810         0.546578 Calibrated-OrderEntropy pmu_r4 -> pmu_r1 -> pmu_r3 -> pmu_r2 -> cyber_logs;H<=1.3634 MSU_ORNL_Power    True
   42  0.380106         0.641921 Calibrated-OrderEntropy pmu_r4 -> pmu_r1 -> pmu_r3 -> cyber_logs -> pmu_r2;H<=1.3369 MSU_ORNL_Power    True
   42  0.382059         0.648458 Calibrated-OrderEntropy pmu_r4 -> pmu_r1 -> pmu_r2 -> pmu_r3 -> cyber_logs;H<=1.3369 MSU_ORNL_Power    True
   42  0.382340         0.648996 Calibrated-OrderEntropy pmu_r4 -> pmu_r1 -> pmu_r2 -> cyber_logs -> pmu_r3;H<=1.3369 MSU_ORNL_Power    True
   42  0.391889         0.748049 Calibrated-OrderEntropy pmu_r4 -> pmu_r1 -> pmu_r3 -> cyber_logs -> pmu_r2;H<=1.2879 MSU_ORNL_Power    True
   42  0.394533         0.749507 Calibrated-OrderEntropy pmu_r4 -> pmu_r1 -> pmu_r2 -> cyber_logs -> pmu_r3;H<=1.2879 MSU_ORNL_Power    True
  123  0.331435         0.200000              Greedy-VOI                                                    H<=2.6094 MSU_ORNL_Power    True
  123  0.331435         0.200000 Calibrated-OrderEntropy pmu_r4 -> pmu_r1 -> pmu_r2 -> pmu_r3 -> cyber_logs;H<=2.6094 MSU_ORNL_Power    True
  123  0.331435         0.200000 Calibrated-OrderEntropy pmu_r4 -> pmu_r1 -> pmu_r2 -> cyber_logs -> pmu_r3;H<=2.6094 MSU_ORNL_Power    True
  123  0.331435         0.200000 Calibrated-OrderEntropy pmu_r4 -> pmu_r1 -> pmu_r3 -> pmu_r2 -> cyber_logs;H<=2.6094 MSU_ORNL_Power    True
  123  0.331435         0.200000 Calibrated-OrderEntropy pmu_r4 -> pmu_r1 -> pmu_r3 -> cyber_logs -> pmu_r2;H<=2.6094 MSU_ORNL_Power    True
  123  0.331435         0.200000 Calibrated-OrderEntropy pmu_r4 -> pmu_r1 -> cyber_logs -> pmu_r2 -> pmu_r3;H<=2.6094 MSU_ORNL_Power    True
  123  0.331435         0.200000 Calibrated-OrderEntropy pmu_r4 -> pmu_r1 -> cyber_logs -> pmu_r3 -> pmu_r2;H<=2.6094 MSU_ORNL_Power    True
  123  0.331435         0.200000 Calibrated-OrderEntropy pmu_r4 -> pmu_r2 -> pmu_r1 -> pmu_r3 -> cyber_logs;H<=2.6094 MSU_ORNL_Power    True
  123  0.331435         0.200000 Calibrated-OrderEntropy pmu_r4 -> pmu_r2 -> pmu_r1 -> cyber_logs -> pmu_r3;H<=2.6094 MSU_ORNL_Power    True
  123  0.331435         0.200000 Calibrated-OrderEntropy pmu_r4 -> pmu_r2 -> pmu_r3 -> pmu_r1 -> cyber_logs;H<=2.6094 MSU_ORNL_Power    True
  123  0.331435         0.200000 Calibrated-OrderEntropy pmu_r4 -> pmu_r2 -> pmu_r3 -> cyber_logs -> pmu_r1;H<=2.6094 MSU_ORNL_Power    True
  123  0.331435         0.200000 Calibrated-OrderEntropy pmu_r4 -> pmu_r2 -> cyber_logs -> pmu_r1 -> pmu_r3;H<=2.6094 MSU_ORNL_Power    True
  123  0.331435         0.200000 Calibrated-OrderEntropy pmu_r4 -> pmu_r2 -> cyber_logs -> pmu_r3 -> pmu_r1;H<=2.6094 MSU_ORNL_Power    True
  123  0.331435         0.200000 Calibrated-OrderEntropy pmu_r4 -> pmu_r3 -> pmu_r1 -> pmu_r2 -> cyber_logs;H<=2.6094 MSU_ORNL_Power    True
  123  0.331435         0.200000 Calibrated-OrderEntropy pmu_r4 -> pmu_r3 -> pmu_r1 -> cyber_logs -> pmu_r2;H<=2.6094 MSU_ORNL_Power    True
  123  0.331435         0.200000 Calibrated-OrderEntropy pmu_r4 -> pmu_r3 -> pmu_r2 -> pmu_r1 -> cyber_logs;H<=2.6094 MSU_ORNL_Power    True
  123  0.331435         0.200000 Calibrated-OrderEntropy pmu_r4 -> pmu_r3 -> pmu_r2 -> cyber_logs -> pmu_r1;H<=2.6094 MSU_ORNL_Power    True
  123  0.331435         0.200000 Calibrated-OrderEntropy pmu_r4 -> pmu_r3 -> cyber_logs -> pmu_r1 -> pmu_r2;H<=2.6094 MSU_ORNL_Power    True
  123  0.331435         0.200000 Calibrated-OrderEntropy pmu_r4 -> pmu_r3 -> cyber_logs -> pmu_r2 -> pmu_r1;H<=2.6094 MSU_ORNL_Power    True
  123  0.331435         0.200000 Calibrated-OrderEntropy pmu_r4 -> cyber_logs -> pmu_r1 -> pmu_r2 -> pmu_r3;H<=2.6094 MSU_ORNL_Power    True
  123  0.331435         0.200000 Calibrated-OrderEntropy pmu_r4 -> cyber_logs -> pmu_r1 -> pmu_r3 -> pmu_r2;H<=2.6094 MSU_ORNL_Power    True
  123  0.331435         0.200000 Calibrated-OrderEntropy pmu_r4 -> cyber_logs -> pmu_r2 -> pmu_r1 -> pmu_r3;H<=2.6094 MSU_ORNL_Power    True
  123  0.331435         0.200000 Calibrated-OrderEntropy pmu_r4 -> cyber_logs -> pmu_r2 -> pmu_r3 -> pmu_r1;H<=2.6094 MSU_ORNL_Power    True
  123  0.331435         0.200000 Calibrated-OrderEntropy pmu_r4 -> cyber_logs -> pmu_r3 -> pmu_r1 -> pmu_r2;H<=2.6094 MSU_ORNL_Power    True
  123  0.331435         0.200000 Calibrated-OrderEntropy pmu_r4 -> cyber_logs -> pmu_r3 -> pmu_r2 -> pmu_r1;H<=2.6094 MSU_ORNL_Power    True
  123  0.331448         0.200614 Calibrated-OrderEntropy pmu_r4 -> pmu_r1 -> pmu_r2 -> pmu_r3 -> cyber_logs;H<=1.6060 MSU_ORNL_Power    True
  123  0.331448         0.200614 Calibrated-OrderEntropy pmu_r4 -> pmu_r1 -> pmu_r2 -> cyber_logs -> pmu_r3;H<=1.6060 MSU_ORNL_Power    True
  123  0.331448         0.200614 Calibrated-OrderEntropy pmu_r4 -> pmu_r1 -> pmu_r3 -> pmu_r2 -> cyber_logs;H<=1.6060 MSU_ORNL_Power    True
  123  0.331448         0.200614 Calibrated-OrderEntropy pmu_r4 -> pmu_r1 -> pmu_r3 -> cyber_logs -> pmu_r2;H<=1.6060 MSU_ORNL_Power    True
  123  0.331575         0.201190              Greedy-VOI                                                    H<=1.5893 MSU_ORNL_Power    True
  123  0.335844         0.246552 Calibrated-OrderEntropy pmu_r4 -> pmu_r2 -> pmu_r1 -> pmu_r3 -> cyber_logs;H<=1.5716 MSU_ORNL_Power    True
  123  0.335853         0.254010              Greedy-VOI                                                    H<=1.4677 MSU_ORNL_Power    True
  123  0.340740         0.279122 Calibrated-OrderEntropy pmu_r4 -> pmu_r2 -> pmu_r1 -> pmu_r3 -> cyber_logs;H<=1.5447 MSU_ORNL_Power    True
  123  0.341126         0.284137              Greedy-VOI                                                    H<=1.4129 MSU_ORNL_Power    True
  123  0.341621         0.285416 Calibrated-OrderEntropy pmu_r4 -> pmu_r1 -> pmu_r2 -> pmu_r3 -> cyber_logs;H<=1.5447 MSU_ORNL_Power    True
  123  0.346906         0.318370 Calibrated-OrderEntropy pmu_r4 -> pmu_r1 -> pmu_r2 -> pmu_r3 -> cyber_logs;H<=1.5117 MSU_ORNL_Power    True
  123  0.349188         0.326519              Greedy-VOI                                                    H<=1.3844 MSU_ORNL_Power    True
  123  0.352512         0.353934 Calibrated-OrderEntropy pmu_r4 -> pmu_r1 -> pmu_r2 -> pmu_r3 -> cyber_logs;H<=1.4772 MSU_ORNL_Power    True
  123  0.352935         0.366125 Calibrated-OrderEntropy pmu_r4 -> pmu_r1 -> cyber_logs -> pmu_r2 -> pmu_r3;H<=1.4772 MSU_ORNL_Power    True
  123  0.361394         0.393540 Calibrated-OrderEntropy pmu_r4 -> pmu_r1 -> pmu_r2 -> pmu_r3 -> cyber_logs;H<=1.4484 MSU_ORNL_Power    True
  123  0.361677         0.406051 Calibrated-OrderEntropy pmu_r4 -> pmu_r1 -> cyber_logs -> pmu_r2 -> pmu_r3;H<=1.4484 MSU_ORNL_Power    True
  123  0.366321         0.435935 Calibrated-OrderEntropy pmu_r4 -> pmu_r1 -> pmu_r2 -> pmu_r3 -> cyber_logs;H<=1.4272 MSU_ORNL_Power    True
  123  0.366550         0.440617 Calibrated-OrderEntropy pmu_r4 -> pmu_r1 -> pmu_r2 -> cyber_logs -> pmu_r3;H<=1.4272 MSU_ORNL_Power    True
  123  0.366727         0.542165 Calibrated-OrderEntropy pmu_r4 -> pmu_r1 -> pmu_r2 -> cyber_logs -> pmu_r3;H<=1.3982 MSU_ORNL_Power    True
  123  0.369525         0.545682 Calibrated-OrderEntropy cyber_logs -> pmu_r4 -> pmu_r1 -> pmu_r2 -> pmu_r3;H<=1.4272 MSU_ORNL_Power    True
  123  0.375396         0.579429 Calibrated-OrderEntropy cyber_logs -> pmu_r4 -> pmu_r1 -> pmu_r2 -> pmu_r3;H<=1.4109 MSU_ORNL_Power    True
  123  0.376628         0.855047 Calibrated-OrderEntropy pmu_r4 -> pmu_r1 -> cyber_logs -> pmu_r2 -> pmu_r3;H<=1.2081 MSU_ORNL_Power    True
  123  0.381246         0.905411              Greedy-VOI                                                    H<=0.2640 MSU_ORNL_Power    True
  123  0.381547         0.911526 Calibrated-OrderEntropy pmu_r4 -> pmu_r1 -> cyber_logs -> pmu_r2 -> pmu_r3;H<=0.9124 MSU_ORNL_Power    True
  123  0.381737         0.919649 Calibrated-OrderEntropy cyber_logs -> pmu_r4 -> pmu_r1 -> pmu_r2 -> pmu_r3;H<=0.9124 MSU_ORNL_Power    True
  123  0.382649         1.000000              Greedy-VOI                                                   H<=-1.0000 MSU_ORNL_Power    True
  456  0.301546         0.200000 Calibrated-OrderEntropy pmu_r4 -> pmu_r1 -> pmu_r2 -> pmu_r3 -> cyber_logs;H<=2.6094 MSU_ORNL_Power    True
  456  0.301546         0.200000 Calibrated-OrderEntropy pmu_r4 -> pmu_r1 -> pmu_r2 -> cyber_logs -> pmu_r3;H<=2.6094 MSU_ORNL_Power    True
  456  0.301546         0.200000 Calibrated-OrderEntropy pmu_r4 -> pmu_r1 -> pmu_r3 -> pmu_r2 -> cyber_logs;H<=2.6094 MSU_ORNL_Power    True
  456  0.301546         0.200000 Calibrated-OrderEntropy pmu_r4 -> pmu_r1 -> pmu_r3 -> cyber_logs -> pmu_r2;H<=2.6094 MSU_ORNL_Power    True
  456  0.301546         0.200000 Calibrated-OrderEntropy pmu_r4 -> pmu_r1 -> cyber_logs -> pmu_r2 -> pmu_r3;H<=2.6094 MSU_ORNL_Power    True
  456  0.301546         0.200000 Calibrated-OrderEntropy pmu_r4 -> pmu_r1 -> cyber_logs -> pmu_r3 -> pmu_r2;H<=2.6094 MSU_ORNL_Power    True
  456  0.301546         0.200000 Calibrated-OrderEntropy pmu_r4 -> pmu_r2 -> pmu_r1 -> pmu_r3 -> cyber_logs;H<=2.6094 MSU_ORNL_Power    True
  456  0.301546         0.200000 Calibrated-OrderEntropy pmu_r4 -> pmu_r2 -> pmu_r1 -> cyber_logs -> pmu_r3;H<=2.6094 MSU_ORNL_Power    True
  456  0.301546         0.200000 Calibrated-OrderEntropy pmu_r4 -> pmu_r2 -> pmu_r3 -> pmu_r1 -> cyber_logs;H<=2.6094 MSU_ORNL_Power    True
  456  0.301546         0.200000 Calibrated-OrderEntropy pmu_r4 -> pmu_r2 -> pmu_r3 -> cyber_logs -> pmu_r1;H<=2.6094 MSU_ORNL_Power    True
  456  0.301546         0.200000 Calibrated-OrderEntropy pmu_r4 -> pmu_r2 -> cyber_logs -> pmu_r1 -> pmu_r3;H<=2.6094 MSU_ORNL_Power    True
  456  0.301546         0.200000 Calibrated-OrderEntropy pmu_r4 -> pmu_r2 -> cyber_logs -> pmu_r3 -> pmu_r1;H<=2.6094 MSU_ORNL_Power    True
  456  0.301546         0.200000 Calibrated-OrderEntropy pmu_r4 -> pmu_r3 -> pmu_r1 -> pmu_r2 -> cyber_logs;H<=2.6094 MSU_ORNL_Power    True
  456  0.301546         0.200000 Calibrated-OrderEntropy pmu_r4 -> pmu_r3 -> pmu_r1 -> cyber_logs -> pmu_r2;H<=2.6094 MSU_ORNL_Power    True
  456  0.301546         0.200000 Calibrated-OrderEntropy pmu_r4 -> pmu_r3 -> pmu_r2 -> pmu_r1 -> cyber_logs;H<=2.6094 MSU_ORNL_Power    True
  456  0.301546         0.200000 Calibrated-OrderEntropy pmu_r4 -> pmu_r3 -> pmu_r2 -> cyber_logs -> pmu_r1;H<=2.6094 MSU_ORNL_Power    True
  456  0.301546         0.200000 Calibrated-OrderEntropy pmu_r4 -> pmu_r3 -> cyber_logs -> pmu_r1 -> pmu_r2;H<=2.6094 MSU_ORNL_Power    True
  456  0.301546         0.200000 Calibrated-OrderEntropy pmu_r4 -> pmu_r3 -> cyber_logs -> pmu_r2 -> pmu_r1;H<=2.6094 MSU_ORNL_Power    True
  456  0.301546         0.200000 Calibrated-OrderEntropy pmu_r4 -> cyber_logs -> pmu_r1 -> pmu_r2 -> pmu_r3;H<=2.6094 MSU_ORNL_Power    True
  456  0.301546         0.200000 Calibrated-OrderEntropy pmu_r4 -> cyber_logs -> pmu_r1 -> pmu_r3 -> pmu_r2;H<=2.6094 MSU_ORNL_Power    True
  456  0.301546         0.200000 Calibrated-OrderEntropy pmu_r4 -> cyber_logs -> pmu_r2 -> pmu_r1 -> pmu_r3;H<=2.6094 MSU_ORNL_Power    True
  456  0.301546         0.200000 Calibrated-OrderEntropy pmu_r4 -> cyber_logs -> pmu_r2 -> pmu_r3 -> pmu_r1;H<=2.6094 MSU_ORNL_Power    True
  456  0.301546         0.200000 Calibrated-OrderEntropy pmu_r4 -> cyber_logs -> pmu_r3 -> pmu_r1 -> pmu_r2;H<=2.6094 MSU_ORNL_Power    True
  456  0.301546         0.200000 Calibrated-OrderEntropy pmu_r4 -> cyber_logs -> pmu_r3 -> pmu_r2 -> pmu_r1;H<=2.6094 MSU_ORNL_Power    True
  456  0.303079         0.229026              Greedy-VOI                                                    H<=1.4866 MSU_ORNL_Power    True
  456  0.307562         0.262428              Greedy-VOI                                                    H<=1.4064 MSU_ORNL_Power    True
  456  0.309770         0.263848 Calibrated-OrderEntropy pmu_r4 -> pmu_r1 -> pmu_r3 -> pmu_r2 -> cyber_logs;H<=1.5414 MSU_ORNL_Power    True
  456  0.313926         0.293425 Calibrated-OrderEntropy pmu_r4 -> pmu_r1 -> pmu_r3 -> pmu_r2 -> cyber_logs;H<=1.5086 MSU_ORNL_Power    True
  456  0.319401         0.306025              Greedy-VOI                                                    H<=1.3745 MSU_ORNL_Power    True
  456  0.319729         0.327760 Calibrated-OrderEntropy pmu_r4 -> pmu_r1 -> pmu_r3 -> pmu_r2 -> cyber_logs;H<=1.4769 MSU_ORNL_Power    True
  456  0.329646         0.350825              Greedy-VOI                                                    H<=1.3549 MSU_ORNL_Power    True
  456  0.332416         0.399591 Calibrated-OrderEntropy pmu_r4 -> pmu_r1 -> pmu_r2 -> pmu_r3 -> cyber_logs;H<=1.4247 MSU_ORNL_Power    True
  456  0.332830         0.401420 Calibrated-OrderEntropy pmu_r4 -> pmu_r1 -> pmu_r3 -> pmu_r2 -> cyber_logs;H<=1.4247 MSU_ORNL_Power    True
  456  0.337540         0.409236              Greedy-VOI                                                    H<=1.3326 MSU_ORNL_Power    True
  456  0.340465         0.430702 Calibrated-OrderEntropy pmu_r1 -> pmu_r4 -> pmu_r2 -> pmu_r3 -> cyber_logs;H<=1.4054 MSU_ORNL_Power    True
  456  0.347962         0.445056 Calibrated-OrderEntropy pmu_r4 -> pmu_r1 -> pmu_r2 -> pmu_r3 -> cyber_logs;H<=1.4054 MSU_ORNL_Power    True
  456  0.366345         0.502252 Calibrated-OrderEntropy pmu_r4 -> pmu_r1 -> pmu_r2 -> pmu_r3 -> cyber_logs;H<=1.3872 MSU_ORNL_Power    True
  456  0.370593         0.569067 Calibrated-OrderEntropy pmu_r2 -> pmu_r1 -> pmu_r4 -> pmu_r3 -> cyber_logs;H<=1.3872 MSU_ORNL_Power    True
  456  0.373224         0.621210 Calibrated-OrderEntropy pmu_r2 -> pmu_r1 -> pmu_r4 -> cyber_logs -> pmu_r3;H<=1.3717 MSU_ORNL_Power    True
  456  0.373299         0.621556 Calibrated-OrderEntropy pmu_r2 -> pmu_r1 -> pmu_r4 -> pmu_r3 -> cyber_logs;H<=1.3717 MSU_ORNL_Power    True
  456  0.374508         0.662748 Calibrated-OrderEntropy pmu_r3 -> pmu_r1 -> pmu_r4 -> cyber_logs -> pmu_r2;H<=1.3872 MSU_ORNL_Power    True
  456  0.378293         0.678611 Calibrated-OrderEntropy pmu_r4 -> pmu_r1 -> pmu_r3 -> cyber_logs -> pmu_r2;H<=1.3466 MSU_ORNL_Power    True
  456  0.378753         0.679314 Calibrated-OrderEntropy pmu_r4 -> pmu_r1 -> pmu_r3 -> pmu_r2 -> cyber_logs;H<=1.3466 MSU_ORNL_Power    True
  456  0.383359         0.760215              Greedy-VOI                                                    H<=1.0405 MSU_ORNL_Power    True
  456  0.385991         0.768492 Calibrated-OrderEntropy pmu_r4 -> pmu_r1 -> pmu_r2 -> cyber_logs -> pmu_r3;H<=1.3005 MSU_ORNL_Power    True
  456  0.388309         0.770564 Calibrated-OrderEntropy pmu_r4 -> pmu_r1 -> pmu_r3 -> cyber_logs -> pmu_r2;H<=1.3005 MSU_ORNL_Power    True
  456  0.388995         0.771971 Calibrated-OrderEntropy pmu_r4 -> pmu_r1 -> pmu_r3 -> pmu_r2 -> cyber_logs;H<=1.3005 MSU_ORNL_Power    True
  456  0.390658         0.832557 Calibrated-OrderEntropy pmu_r4 -> pmu_r1 -> pmu_r2 -> cyber_logs -> pmu_r3;H<=1.2117 MSU_ORNL_Power    True
  456  0.391204         0.838621 Calibrated-OrderEntropy pmu_r4 -> pmu_r1 -> pmu_r3 -> pmu_r2 -> cyber_logs;H<=1.2117 MSU_ORNL_Power    True
  456  0.391234         0.854791              Greedy-VOI                                                    H<=0.6983 MSU_ORNL_Power    True
  456  0.392904         0.913381              Greedy-VOI                                                    H<=0.1398 MSU_ORNL_Power    True
  456  0.393036         1.000000              Greedy-VOI                                                   H<=-1.0000 MSU_ORNL_Power    True
```

### Observed-range AUC summary
```
       dataset  seed           method_family  observed_auc  cost_min  cost_max  mean_f1_over_observed_cost_range  n_points
MSU_ORNL_Power    42 Calibrated-OrderEntropy      0.261346  0.200000  0.903109                          0.371701        65
MSU_ORNL_Power    42                     DQN      0.004167  0.761290  0.772291                          0.378782         3
MSU_ORNL_Power    42              Greedy-VOI      0.260608  0.200000  1.000000                          0.325761        17
MSU_ORNL_Power   123 Calibrated-OrderEntropy      0.269008  0.200000  0.941576                          0.362752        70
MSU_ORNL_Power   123                     DQN      0.005927  0.771971  0.788384                          0.361089         3
MSU_ORNL_Power   123              Greedy-VOI      0.282859  0.200000  1.000000                          0.353574        17
MSU_ORNL_Power   456 Calibrated-OrderEntropy      0.225365  0.200000  0.838621                          0.352894        59
MSU_ORNL_Power   456                     DQN      0.019369  0.725598  0.777268                          0.374869         3
MSU_ORNL_Power   456              Greedy-VOI      0.286513  0.200000  1.000000                          0.358141        17
```

## Calibration summary
```
           dataset  n_masks  mean_raw_ece  max_raw_ece  mean_calibrated_ece  max_calibrated_ece
TU_Delft_2026_CPPS        7      0.161668     0.279604             0.099667            0.138664
    MSU_ORNL_Power       31      0.162673     0.240909             0.058778            0.092967
```
