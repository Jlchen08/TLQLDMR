# Complete v2 revision evidence

Complete reporting of all six primary within-farm tasks and all six secondary cross-farm stress tasks; no task, method, seed, comparison, or effect direction is filtered.

## Exact design counts

```json
{
  "scopes": 2,
  "tasks_total": 12,
  "tasks_by_scope": {
    "within": 6,
    "cross": 6
  },
  "methods": 6,
  "task_method_rows": 72,
  "configuration_queries_per_task_method": 54,
  "configuration_queries_total": 3888,
  "q1_selection_fit_records_total": 10368,
  "evaluation_seed_labels_by_method": {
    "Full": [
      0,
      1,
      2
    ],
    "NoVariance": [
      0,
      1,
      2
    ],
    "NoDiscrepancy": [
      0,
      1,
      2
    ],
    "WeightedSVQR": [
      0,
      1,
      2
    ],
    "TargetSVQR": [
      0,
      1,
      2
    ],
    "HistGBR": [
      0
    ]
  },
  "locked_main_evaluation_records": 192,
  "main_evaluation_records_by_method": {
    "Full": 36,
    "NoVariance": 36,
    "NoDiscrepancy": 36,
    "WeightedSVQR": 36,
    "TargetSVQR": 36,
    "HistGBR": 12
  },
  "full_task_comparisons": 60,
  "first_tier_all_pair_tests": 30,
  "first_tier_membership_rows": 12,
  "matched_variants": 4,
  "matched_task_variant_rows": 48,
  "matched_evaluation_seeds_per_variant": 3,
  "matched_evaluation_records": 144,
  "matched_task_comparisons": 36,
  "heldout_evaluation_records_total": 336
}
```

## Primary within-farm: all six methods

| Position | Method | Macro NRMSE (%) | Average rank | Wins | Non-rejection member |
|---:|---|---:|---:|---:|:---:|
| 1 | HistGBR | 13.912667 | 2.000000 | 4 | yes |
| 2 | No variance | 17.347415 | 3.000000 | 1 | yes |
| 3 | TL-QLDMR | 18.090489 | 3.500000 | 1 | no |
| 4 | Weighted SVQR | 18.646195 | 3.833333 | 0 | no |
| 5 | No discrepancy | 19.332891 | 4.166667 | 0 | yes |
| 6 | Target SVQR | 20.050932 | 4.500000 | 0 | no |

Membership records non-rejection of inferiority to the empirical endpoint winner under the predeclared block test. It does not establish equal performance and is not a formal model-confidence set.

## Secondary cross-farm stress: all six methods

| Position | Method | Macro NRMSE (%) | Average rank | Wins | Non-rejection member |
|---:|---|---:|---:|---:|:---:|
| 1 | HistGBR | 14.922525 | 2.833333 | 3 | yes |
| 2 | No discrepancy | 16.823985 | 2.500000 | 1 | yes |
| 3 | No variance | 17.183359 | 3.000000 | 1 | yes |
| 4 | Weighted SVQR | 17.562525 | 4.166667 | 0 | yes |
| 5 | TL-QLDMR | 17.694352 | 3.666667 | 1 | yes |
| 6 | Target SVQR | 20.050932 | 4.833333 | 0 | no |

Membership records non-rejection of inferiority to the empirical endpoint winner under the predeclared block test. It does not establish equal performance and is not a formal model-confidence set.

## Full relations at scope level

| Scope | Comparator | Full (%) | Comparator (%) | Difference (pp) | Direction | Scope Holm p | Scope significant | Significant tasks | Non-significant tasks |
|---|---|---:|---:|---:|---|---:|:---:|---:|---:|
| within | No variance | 18.090489 | 17.347415 | 0.743075 | Full_higher_macro_NRMSE | 0.29897 | no | 1 | 5 |
| within | No discrepancy | 18.090489 | 19.332891 | -1.242401 | Full_lower_macro_NRMSE | 0.8838 | no | 0 | 6 |
| within | Weighted SVQR | 18.090489 | 18.646195 | -0.555706 | Full_lower_macro_NRMSE | 1 | no | 0 | 6 |
| within | Target SVQR | 18.090489 | 20.050932 | -1.960443 | Full_lower_macro_NRMSE | 0.1221 | no | 0 | 6 |
| within | HistGBR | 18.090489 | 13.912667 | 4.177822 | Full_higher_macro_NRMSE | 0.03564 | yes | 2 | 4 |
| cross | No variance | 17.694352 | 17.183359 | 0.510993 | Full_higher_macro_NRMSE | 0.0156 | yes | 1 | 5 |
| cross | No discrepancy | 17.694352 | 16.823985 | 0.870367 | Full_higher_macro_NRMSE | 0.1027 | no | 0 | 6 |
| cross | Weighted SVQR | 17.694352 | 17.562525 | 0.131827 | Full_higher_macro_NRMSE | 0.80824 | no | 0 | 6 |
| cross | Target SVQR | 17.694352 | 20.050932 | -2.356580 | Full_lower_macro_NRMSE | 0.09966 | no | 0 | 6 |
| cross | HistGBR | 17.694352 | 14.922525 | 2.771827 | Full_higher_macro_NRMSE | 0.33642 | no | 1 | 5 |

## All 60 Full task comparisons

| Scope | Task | Comparator | n | Blocks | MSE difference (MW²) | 95% CI | Raw p | Holm p | Significant | Direction |
|---|---|---|---:|---:|---:|---:|---:|---:|:---:|---|
| within | within_site1 | No variance | 973 | 27 | -0.29182164 | [-0.539369433, -0.139449184] | 0.00056 | 0.01568 | yes | Full_lower_loss |
| within | within_site1 | No discrepancy | 973 | 27 | -0.0877108674 | [-0.534539929, 0.304674239] | 0.65826 | 1 | no | Full_lower_loss |
| within | within_site1 | Weighted SVQR | 973 | 27 | -0.295311677 | [-0.803813869, 0.0787655678] | 0.17157 | 1 | no | Full_lower_loss |
| within | within_site1 | Target SVQR | 973 | 27 | 44.7937095 | [-61.8154231, 215.850293] | 0.5165 | 1 | no | Full_higher_loss |
| within | within_site1 | HistGBR | 973 | 27 | 204.912353 | [71.8938281, 425.01171] | 0.00041 | 0.01189 | yes | Full_higher_loss |
| within | within_site2 | No variance | 901 | 15 | -31.2703462 | [-145.378301, 85.6824984] | 0.58534 | 1 | no | Full_lower_loss |
| within | within_site2 | No discrepancy | 901 | 15 | -53.8610836 | [-129.201762, 25.2603898] | 0.16548 | 1 | no | Full_lower_loss |
| within | within_site2 | Weighted SVQR | 901 | 15 | -1.61877993e-06 | [-6.10993963e-06, 4.89247709e-07] | 0.24055 | 1 | no | Full_lower_loss |
| within | within_site2 | Target SVQR | 901 | 15 | -146.629087 | [-326.896742, -1.15633775] | 0.09899 | 1 | no | Full_lower_loss |
| within | within_site2 | HistGBR | 901 | 15 | -196.07054 | [-310.816698, 95.8839773] | 0.24275 | 1 | no | Full_lower_loss |
| within | within_site3 | No variance | 1577 | 23 | 44.2302674 | [-2.69567714, 101.440842] | 0.11428 | 1 | no | Full_higher_loss |
| within | within_site3 | No discrepancy | 1577 | 23 | 27.5752954 | [-4.41971903, 72.6924226] | 0.14236 | 1 | no | Full_higher_loss |
| within | within_site3 | Weighted SVQR | 1577 | 23 | 51.8690145 | [10.6093596, 112.96359] | 0.06263 | 1 | no | Full_higher_loss |
| within | within_site3 | Target SVQR | 1577 | 23 | -63.095421 | [-109.625611, 13.1825958] | 0.26207 | 1 | no | Full_lower_loss |
| within | within_site3 | HistGBR | 1577 | 23 | 71.6233485 | [35.8698672, 114.079743] | 0.00036 | 0.0108 | yes | Full_higher_loss |
| within | within_site4 | No variance | 526 | 9 | 0.0394765964 | [0.0192122996, 0.0545269921] | 0.02391 | 0.54993 | no | Full_higher_loss |
| within | within_site4 | No discrepancy | 526 | 9 | 0.366900963 | [0.219889672, 0.554211636] | 0.00824 | 0.21424 | no | Full_higher_loss |
| within | within_site4 | Weighted SVQR | 526 | 9 | 0.0903784947 | [0.061619762, 0.11158248] | 0.00722 | 0.19494 | no | Full_higher_loss |
| within | within_site4 | Target SVQR | 526 | 9 | -62.62499 | [-84.8071278, -32.2677756] | 0.01562 | 0.37488 | no | Full_lower_loss |
| within | within_site4 | HistGBR | 526 | 9 | 32.4504084 | [10.7452218, 78.0653774] | 0.01166 | 0.2915 | no | Full_higher_loss |
| within | within_site5 | No variance | 720 | 17 | 22.4978283 | [-9.74418348, 55.7661063] | 0.19596 | 1 | no | Full_higher_loss |
| within | within_site5 | No discrepancy | 720 | 17 | -76.1724544 | [-216.163914, 30.7528801] | 0.23447 | 1 | no | Full_lower_loss |
| within | within_site5 | Weighted SVQR | 720 | 17 | -23.7797297 | [-75.7824166, 46.8627006] | 0.54933 | 1 | no | Full_lower_loss |
| within | within_site5 | Target SVQR | 720 | 17 | -40.3901186 | [-88.1260561, 23.6951481] | 0.2994 | 1 | no | Full_lower_loss |
| within | within_site5 | HistGBR | 720 | 17 | 105.180224 | [-18.4565842, 250.036402] | 0.1081 | 1 | no | Full_higher_loss |
| within | within_site6 | No variance | 2233 | 23 | 16.3875541 | [3.44591974, 30.6280303] | 0.02662 | 0.58564 | no | Full_higher_loss |
| within | within_site6 | No discrepancy | 2233 | 23 | -21.7008601 | [-58.0888782, 20.8408025] | 0.33876 | 1 | no | Full_lower_loss |
| within | within_site6 | Weighted SVQR | 2233 | 23 | -86.1150438 | [-151.839096, -17.6229834] | 0.03782 | 0.79422 | no | Full_lower_loss |
| within | within_site6 | Target SVQR | 2233 | 23 | 5.70710569 | [-45.0292394, 42.756455] | 0.79903 | 1 | no | Full_higher_loss |
| within | within_site6 | HistGBR | 2233 | 23 | 9.06693588 | [-19.1062378, 35.1837954] | 0.55049 | 1 | no | Full_higher_loss |
| cross | cross_site1 | No variance | 973 | 27 | -7.5766171e-08 | [-2.06500633e-07, 7.31938543e-09] | 0.13283 | 1 | no | Full_lower_loss |
| cross | cross_site1 | No discrepancy | 973 | 27 | -2.27783593e-07 | [-5.75375773e-07, -1.29264188e-08] | 0.07431 | 1 | no | Full_lower_loss |
| cross | cross_site1 | Weighted SVQR | 973 | 27 | -3.03551474e-07 | [-7.90635704e-07, -2.20157006e-08] | 0.07179 | 1 | no | Full_lower_loss |
| cross | cross_site1 | Target SVQR | 973 | 27 | 68.9629517 | [-24.8973061, 223.2866] | 0.24802 | 1 | no | Full_higher_loss |
| cross | cross_site1 | HistGBR | 973 | 27 | 214.555664 | [91.6278529, 413.791194] | 5e-05 | 0.00145 | yes | Full_higher_loss |
| cross | cross_site2 | No variance | 901 | 15 | -50.1416032 | [-100.39415, 16.4194692] | 0.14866 | 1 | no | Full_lower_loss |
| cross | cross_site2 | No discrepancy | 901 | 15 | -16.7170875 | [-71.2617864, 54.7790156] | 0.58377 | 1 | no | Full_lower_loss |
| cross | cross_site2 | Weighted SVQR | 901 | 15 | -99.4848579 | [-172.116365, -8.82546843] | 0.08395 | 1 | no | Full_lower_loss |
| cross | cross_site2 | Target SVQR | 901 | 15 | -93.6760782 | [-204.325842, 5.65017742] | 0.10163 | 1 | no | Full_lower_loss |
| cross | cross_site2 | HistGBR | 901 | 15 | -196.191409 | [-371.739708, 140.591926] | 0.45282 | 1 | no | Full_lower_loss |
| cross | cross_site3 | No variance | 1577 | 23 | 3.57726218 | [-8.90433539, 12.8568272] | 0.5361 | 1 | no | Full_higher_loss |
| cross | cross_site3 | No discrepancy | 1577 | 23 | 6.8265516 | [0.0963939255, 13.3589408] | 0.09161 | 1 | no | Full_higher_loss |
| cross | cross_site3 | Weighted SVQR | 1577 | 23 | 1.16602331 | [-14.1885382, 10.4374658] | 0.83931 | 1 | no | Full_higher_loss |
| cross | cross_site3 | Target SVQR | 1577 | 23 | -88.6133929 | [-169.974239, -12.7073965] | 0.06904 | 1 | no | Full_lower_loss |
| cross | cross_site3 | HistGBR | 1577 | 23 | 37.4428108 | [-5.16152345, 77.7224971] | 0.17337 | 1 | no | Full_higher_loss |
| cross | cross_site4 | No variance | 526 | 9 | 6.55171311 | [4.08971299, 9.09435762] | 0.00752 | 0.20304 | no | Full_higher_loss |
| cross | cross_site4 | No discrepancy | 526 | 9 | 6.64167033 | [4.1401469, 9.13077432] | 0.00801 | 0.20826 | no | Full_higher_loss |
| cross | cross_site4 | Weighted SVQR | 526 | 9 | 6.55287187 | [4.07890339, 9.05962593] | 0.00874 | 0.2185 | no | Full_higher_loss |
| cross | cross_site4 | Target SVQR | 526 | 9 | -78.1809528 | [-98.1136045, -59.6546732] | 0.00387 | 0.10836 | no | Full_lower_loss |
| cross | cross_site4 | HistGBR | 526 | 9 | 1.81043544 | [-5.53693584, 6.56916173] | 0.59337 | 1 | no | Full_higher_loss |
| cross | cross_site5 | No variance | 720 | 17 | 15.9076043 | [3.28050651, 32.7896803] | 0.0216 | 0.5184 | no | Full_higher_loss |
| cross | cross_site5 | No discrepancy | 720 | 17 | 33.2964069 | [-3.97353663, 77.9320395] | 0.09184 | 1 | no | Full_higher_loss |
| cross | cross_site5 | Weighted SVQR | 720 | 17 | 35.4774848 | [7.0626612, 73.6106705] | 0.02546 | 0.58558 | no | Full_higher_loss |
| cross | cross_site5 | Target SVQR | 720 | 17 | -37.5195468 | [-101.136182, 52.5691552] | 0.42458 | 1 | no | Full_lower_loss |
| cross | cross_site5 | HistGBR | 720 | 17 | 95.0142798 | [-33.4794035, 254.940175] | 0.17325 | 1 | no | Full_higher_loss |
| cross | cross_site6 | No variance | 2233 | 23 | 16.9165231 | [9.14790963, 27.3983054] | 1e-05 | 0.0003 | yes | Full_higher_loss |
| cross | cross_site6 | No discrepancy | 2233 | 23 | 11.3769528 | [1.62023989, 21.1319996] | 0.04234 | 0.93148 | no | Full_higher_loss |
| cross | cross_site6 | Weighted SVQR | 2233 | 23 | -84.7389924 | [-208.358698, 16.6575654] | 0.16282 | 1 | no | Full_lower_loss |
| cross | cross_site6 | Target SVQR | 2233 | 23 | -12.7739558 | [-33.75651, 2.00189448] | 0.15198 | 1 | no | Full_lower_loss |
| cross | cross_site6 | HistGBR | 2233 | 23 | -5.76352764 | [-57.2366998, 42.5400789] | 0.82154 | 1 | no | Full_lower_loss |

## Matched component removal

These component effects are conditional on each task's locked Full configuration, preprocessing, landmarks, and feature map. The removal variants were not independently retuned, and failure to reject does not establish equal performance.

| Scope | Variant | Macro NRMSE (%) | Mean rank |
|---|---|---:|---:|
| within | TL-QLDMR | 18.090489 | 2.000000 |
| within | No variance | 17.869983 | 2.166667 |
| within | No discrepancy | 19.962576 | 2.666667 |
| within | Weighted SVQR | 19.812546 | 3.166667 |
| cross | TL-QLDMR | 17.694352 | 2.166667 |
| cross | No variance | 17.670887 | 2.500000 |
| cross | No discrepancy | 17.714651 | 2.333333 |
| cross | Weighted SVQR | 17.702143 | 3.000000 |

### All 36 matched task comparisons

| Scope | Task | Comparison | n | Blocks | MSE difference (MW²) | 95% CI | Raw p | Holm p | Significant | Direction |
|---|---|---|---:|---:|---:|---:|---:|---:|:---:|---|
| within | within_site1 | Full - NoVariance | 973 | 27 | -0.29182164 | [-0.523820394, -0.142245015] | 0.00048 | 0.00816 | yes | Full_lower_loss |
| within | within_site1 | Full - NoDiscrepancy | 973 | 27 | -0.0877108674 | [-0.51689909, 0.293686356] | 0.65681 | 1 | no | Full_lower_loss |
| within | within_site1 | Full - WeightedSVQR | 973 | 27 | -0.295311677 | [-0.789973373, 0.0830883413] | 0.1718 | 1 | no | Full_lower_loss |
| within | within_site2 | Full - NoVariance | 901 | 15 | -5.24652307e-07 | [-2.8764952e-06, 3.63540936e-07] | 0.43749 | 1 | no | Full_lower_loss |
| within | within_site2 | Full - NoDiscrepancy | 901 | 15 | -1.0941286e-06 | [-3.76464383e-06, 1.17785461e-06] | 0.28593 | 1 | no | Full_lower_loss |
| within | within_site2 | Full - WeightedSVQR | 901 | 15 | -1.61877993e-06 | [-6.04001338e-06, 4.7954507e-07] | 0.23838 | 1 | no | Full_lower_loss |
| within | within_site3 | Full - NoVariance | 1577 | 23 | -4.17466405 | [-10.9411116, -1.03712138] | 0.01307 | 0.18018 | no | Full_lower_loss |
| within | within_site3 | Full - NoDiscrepancy | 1577 | 23 | -0.0154260101 | [-0.0444910131, -0.000333815456] | 0.09103 | 0.72824 | no | Full_lower_loss |
| within | within_site3 | Full - WeightedSVQR | 1577 | 23 | -4.18443471 | [-11.0812693, -1.09633068] | 0.01287 | 0.18018 | no | Full_lower_loss |
| within | within_site4 | Full - NoVariance | 526 | 9 | 6.9835131e-07 | [-2.75077638e-07, 1.25119038e-06] | 0.17503 | 1 | no | Full_higher_loss |
| within | within_site4 | Full - NoDiscrepancy | 526 | 9 | 0.090377273 | [0.0620860247, 0.111537548] | 0.00752 | 0.12032 | no | Full_higher_loss |
| within | within_site4 | Full - WeightedSVQR | 526 | 9 | 0.0903784947 | [0.0621033261, 0.111298484] | 0.00843 | 0.12645 | no | Full_higher_loss |
| within | within_site5 | Full - NoVariance | 720 | 17 | 2.35116624 | [-24.0089248, 33.4269747] | 0.86724 | 1 | no | Full_higher_loss |
| within | within_site5 | Full - NoDiscrepancy | 720 | 17 | -125.533817 | [-268.604671, -15.6346869] | 0.05001 | 0.55011 | no | Full_lower_loss |
| within | within_site5 | Full - WeightedSVQR | 720 | 17 | -127.837507 | [-264.72719, -21.4348266] | 0.03853 | 0.46236 | no | Full_lower_loss |
| within | within_site6 | Full - NoVariance | 2233 | 23 | 30.2504577 | [3.96624585, 59.7493952] | 0.06718 | 0.6716 | no | Full_higher_loss |
| within | within_site6 | Full - NoDiscrepancy | 2233 | 23 | -4.43222429e-05 | [-5.83192905e-05, -3.03029064e-05] | 1e-05 | 0.00018 | yes | Full_lower_loss |
| within | within_site6 | Full - WeightedSVQR | 2233 | 23 | 30.2504389 | [3.93288243, 60.8487512] | 0.06716 | 0.6716 | no | Full_higher_loss |
| cross | cross_site1 | Full - NoVariance | 973 | 27 | -7.5766171e-08 | [-2.04149533e-07, 8.00029496e-09] | 0.13495 | 1 | no | Full_lower_loss |
| cross | cross_site1 | Full - NoDiscrepancy | 973 | 27 | -2.27783593e-07 | [-5.77144826e-07, -1.16338801e-08] | 0.07499 | 1 | no | Full_lower_loss |
| cross | cross_site1 | Full - WeightedSVQR | 973 | 27 | -3.03551474e-07 | [-7.58984789e-07, -2.45902505e-08] | 0.07119 | 1 | no | Full_lower_loss |
| cross | cross_site2 | Full - NoVariance | 901 | 15 | -1.60312114e-06 | [-5.18215694e-06, 1.22363916e-06] | 0.33866 | 1 | no | Full_lower_loss |
| cross | cross_site2 | Full - NoDiscrepancy | 901 | 15 | 5.5951023e-05 | [-0.00011381032, 0.000125563261] | 0.56149 | 1 | no | Full_higher_loss |
| cross | cross_site2 | Full - WeightedSVQR | 901 | 15 | 5.43478435e-05 | [-0.000116017288, 0.000123044125] | 0.5818 | 1 | no | Full_higher_loss |
| cross | cross_site3 | Full - NoVariance | 1577 | 23 | 3.57726218 | [-9.43470677, 12.9491469] | 0.53197 | 1 | no | Full_higher_loss |
| cross | cross_site3 | Full - NoDiscrepancy | 1577 | 23 | -0.866478374 | [-4.66829469, 1.22743117] | 0.45145 | 1 | no | Full_lower_loss |
| cross | cross_site3 | Full - WeightedSVQR | 1577 | 23 | 1.16602331 | [-13.8665, 10.5419436] | 0.83861 | 1 | no | Full_higher_loss |
| cross | cross_site4 | Full - NoVariance | 526 | 9 | -3.12763602e-09 | [-7.25950161e-09, -7.27407519e-10] | 0.03043 | 0.51731 | no | Full_lower_loss |
| cross | cross_site4 | Full - NoDiscrepancy | 526 | 9 | 0.001048667 | [-0.000420474761, 0.0043925673] | 0.74963 | 1 | no | Full_higher_loss |
| cross | cross_site4 | Full - WeightedSVQR | 526 | 9 | 0.00104866401 | [-0.000423999798, 0.00447050089] | 0.75019 | 1 | no | Full_higher_loss |
| cross | cross_site5 | Full - NoVariance | 720 | 17 | -0.0525094462 | [-0.176858108, 0.0445478469] | 0.34442 | 1 | no | Full_lower_loss |
| cross | cross_site5 | Full - NoDiscrepancy | 720 | 17 | -2.29200952e-06 | [-5.27417974e-06, 1.69015212e-07] | 0.07431 | 1 | no | Full_lower_loss |
| cross | cross_site5 | Full - WeightedSVQR | 720 | 17 | -0.0525108364 | [-0.176276074, 0.0447759588] | 0.34288 | 1 | no | Full_lower_loss |
| cross | cross_site6 | Full - NoVariance | 2233 | 23 | 0.0601888366 | [0.0262719386, 0.100625453] | 0.00335 | 0.0603 | no | Full_higher_loss |
| cross | cross_site6 | Full - NoDiscrepancy | 2233 | 23 | -2.10864248 | [-4.5094021, 0.101834449] | 0.08819 | 1 | no | Full_lower_loss |
| cross | cross_site6 | Full - WeightedSVQR | 2233 | 23 | -2.12850383 | [-4.66838798, 0.120534099] | 0.08282 | 1 | no | Full_lower_loss |

## Manuscript-ready numeric sentences

- Across the six primary within-farm tasks, TL-QLDMR recorded a site-equal macro capacity-normalized RMSE of 18.090%, an average task rank of 3.50, and 1 task wins (including ties); it placed 3 of six by the observed macro endpoint and was not in the conservative non-rejection set.
- Across the six secondary cross-farm stress tasks, TL-QLDMR recorded a site-equal macro capacity-normalized RMSE of 17.694%, an average task rank of 3.67, and 1 task wins (including ties); it placed 5 of six by the observed macro endpoint and was in the conservative non-rejection set.
- In the within-farm scope, TL-QLDMR's macro capacity-normalized RMSE was 0.743 percentage points higher than No variance; among the six task-level paired tests, 1 were significant after Holm adjustment and 5 were not.
- In the within-farm scope, TL-QLDMR's macro capacity-normalized RMSE was 1.242 percentage points lower than No discrepancy; among the six task-level paired tests, 0 were significant after Holm adjustment and 6 were not.
- In the within-farm scope, TL-QLDMR's macro capacity-normalized RMSE was 0.556 percentage points lower than Weighted SVQR; among the six task-level paired tests, 0 were significant after Holm adjustment and 6 were not.
- In the within-farm scope, TL-QLDMR's macro capacity-normalized RMSE was 1.960 percentage points lower than Target SVQR; among the six task-level paired tests, 0 were significant after Holm adjustment and 6 were not.
- In the within-farm scope, TL-QLDMR's macro capacity-normalized RMSE was 4.178 percentage points higher than HistGBR; among the six task-level paired tests, 2 were significant after Holm adjustment and 4 were not.
- In the cross-farm stress scope, TL-QLDMR's macro capacity-normalized RMSE was 0.511 percentage points higher than No variance; among the six task-level paired tests, 1 were significant after Holm adjustment and 5 were not.
- In the cross-farm stress scope, TL-QLDMR's macro capacity-normalized RMSE was 0.870 percentage points higher than No discrepancy; among the six task-level paired tests, 0 were significant after Holm adjustment and 6 were not.
- In the cross-farm stress scope, TL-QLDMR's macro capacity-normalized RMSE was 0.132 percentage points higher than Weighted SVQR; among the six task-level paired tests, 0 were significant after Holm adjustment and 6 were not.
- In the cross-farm stress scope, TL-QLDMR's macro capacity-normalized RMSE was 2.357 percentage points lower than Target SVQR; among the six task-level paired tests, 0 were significant after Holm adjustment and 6 were not.
- In the cross-farm stress scope, TL-QLDMR's macro capacity-normalized RMSE was 2.772 percentage points higher than HistGBR; among the six task-level paired tests, 1 were significant after Holm adjustment and 5 were not.
- Under the matched, conditional component-removal protocol in the within-farm scope, Full recorded macro capacity-normalized RMSE 18.090% and mean rank 2.00, compared with No variance 17.870%, No discrepancy 19.963%, Weighted SVQR 19.813%.
- Under the matched, conditional component-removal protocol in the cross-farm stress scope, Full recorded macro capacity-normalized RMSE 17.694% and mean rank 2.17, compared with No variance 17.671%, No discrepancy 17.715%, Weighted SVQR 17.702%.
