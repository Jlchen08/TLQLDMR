# V2 result prose

Completed hash-verified revision evidence, analysis summaries/CSVs, and audited optimization-summary CSV only; no raw result, per-fit, or prediction file read.

## Scope results

### within

In the primary within-site scope, HistGBR was the empirical winner on the macro RMSE endpoint at 13.913\%. TL-QLDMR recorded a site-equal capacity-normalized RMSE of 18.090\%, an average task rank of 3.50, and 1 task win including ties, ranking 3rd of six by the observed macro endpoint.

The conservative first tier in the primary within-site scope contained HistGBR, No variance, and No discrepancy; TL-QLDMR was excluded ($p_{\mathrm{Holm}}=0.036$), meaning that its inferiority to HistGBR was established for the macro RMSE endpoint. Membership means that inferiority to the empirical winner was not established; it does not establish equivalence or equal performance.

### cross

In the secondary cross-site stress scope, HistGBR was the empirical winner on the macro RMSE endpoint at 14.923\%. TL-QLDMR recorded a site-equal capacity-normalized RMSE of 17.694\%, an average task rank of 3.67, and 1 task win including ties, ranking 5th of six by the observed macro endpoint.

The conservative first tier in the secondary cross-site stress scope contained HistGBR, No discrepancy, No variance, Weighted SVQR, and TL-QLDMR; TL-QLDMR was retained ($p_{\mathrm{Holm}}=0.336$), meaning that its inferiority to HistGBR was not established. Membership means that inferiority to the empirical winner was not established; it does not establish equivalence or equal performance.

## Scope-level TL-QLDMR relations

The Holm-significant scope-level TL-QLDMR relations were: In the primary within-site scope, TL-QLDMR's observed macro RMSE was 4.178 percentage points higher than HistGBR; the all-pairs difference was significant after Holm adjustment ($p_{\mathrm{Holm}}=0.036$). In the secondary cross-site stress scope, TL-QLDMR's observed macro RMSE was 0.511 percentage points higher than No variance; the all-pairs difference was significant after Holm adjustment ($p_{\mathrm{Holm}}=0.016$). These tests concern the prespecified macro endpoint; empirical ordering by itself is not a significance claim.

| Scope | Comparator | Difference (percentage points) | Holm p | Significant |
|---|---|---:|---:|:---:|
| within | No variance | 0.743075 | 0.29897 | no |
| within | No discrepancy | -1.242401 | 0.8838 | no |
| within | Weighted SVQR | -0.555706 | 1 | no |
| within | Target SVQR | -1.960443 | 0.1221 | no |
| within | HistGBR | 4.177822 | 0.03564 | yes |
| cross | No variance | 0.510993 | 0.0156 | yes |
| cross | No discrepancy | 0.870367 | 0.1027 | no |
| cross | Weighted SVQR | 0.131827 | 0.80824 | no |
| cross | Target SVQR | -2.356580 | 0.09966 | no |
| cross | HistGBR | 2.771827 | 0.33642 | no |

## Holm-significant TL-QLDMR task comparisons

- Farm~1 (within-site): TL-QLDMR had lower squared error than No variance (MSE difference, TL-QLDMR minus No variance, $-0.292$ MW$^2$; 95\% block-bootstrap CI $[-0.539, -0.139]$, $p_{\mathrm{Holm}}=0.016$).
- Farm~1 (within-site): TL-QLDMR had higher squared error than HistGBR (MSE difference, TL-QLDMR minus HistGBR, $204.912$ MW$^2$; 95\% block-bootstrap CI $[71.894, 425.012]$, $p_{\mathrm{Holm}}=0.012$).
- Farm~3 (within-site): TL-QLDMR had higher squared error than HistGBR (MSE difference, TL-QLDMR minus HistGBR, $71.623$ MW$^2$; 95\% block-bootstrap CI $[35.870, 114.080]$, $p_{\mathrm{Holm}}=0.011$).
- Farm~1 (cross-site): TL-QLDMR had higher squared error than HistGBR (MSE difference, TL-QLDMR minus HistGBR, $214.556$ MW$^2$; 95\% block-bootstrap CI $[91.628, 413.791]$, $p_{\mathrm{Holm}}=0.001$).
- Farm~6 (cross-site): TL-QLDMR had higher squared error than No variance (MSE difference, TL-QLDMR minus No variance, $16.917$ MW$^2$; 95\% block-bootstrap CI $[9.148, 27.398]$, $p_{\mathrm{Holm}}<0.001$).

Favorable to TL-QLDMR: 1; adverse to TL-QLDMR: 4.

The Holm-significant TL-QLDMR comparisons were: Farm~1 (within-site): TL-QLDMR had lower squared error than No variance (MSE difference, TL-QLDMR minus No variance, $-0.292$ MW$^2$; 95\% block-bootstrap CI $[-0.539, -0.139]$, $p_{\mathrm{Holm}}=0.016$). Farm~1 (within-site): TL-QLDMR had higher squared error than HistGBR (MSE difference, TL-QLDMR minus HistGBR, $204.912$ MW$^2$; 95\% block-bootstrap CI $[71.894, 425.012]$, $p_{\mathrm{Holm}}=0.012$). Farm~3 (within-site): TL-QLDMR had higher squared error than HistGBR (MSE difference, TL-QLDMR minus HistGBR, $71.623$ MW$^2$; 95\% block-bootstrap CI $[35.870, 114.080]$, $p_{\mathrm{Holm}}=0.011$). Farm~1 (cross-site): TL-QLDMR had higher squared error than HistGBR (MSE difference, TL-QLDMR minus HistGBR, $214.556$ MW$^2$; 95\% block-bootstrap CI $[91.628, 413.791]$, $p_{\mathrm{Holm}}=0.001$). Farm~6 (cross-site): TL-QLDMR had higher squared error than No variance (MSE difference, TL-QLDMR minus No variance, $16.917$ MW$^2$; 95\% block-bootstrap CI $[9.148, 27.398]$, $p_{\mathrm{Holm}}<0.001$). All other predeclared TL-QLDMR comparisons were not significant after adjustment; this absence of rejection is not evidence of equal performance.

## Matched component removal

Under the conditional matched-removal protocol in the within-site scope, the four aggregate results were No variance 17.870\% (mean rank 2.17); TL-QLDMR 18.090\% (mean rank 2.00); Weighted SVQR 19.813\% (mean rank 3.17); No discrepancy 19.963\% (mean rank 2.67). TL-QLDMR ranked 2nd of four by observed macro RMSE.

Under the conditional matched-removal protocol in the cross-site stress scope, the four aggregate results were No variance 17.671\% (mean rank 2.50); TL-QLDMR 17.694\% (mean rank 2.17); Weighted SVQR 17.702\% (mean rank 3.00); No discrepancy 17.715\% (mean rank 2.33). TL-QLDMR ranked 2nd of four by observed macro RMSE.

All Holm-significant conditional effects:

- Farm~1 (within-site): TL-QLDMR had lower squared error than No variance (MSE difference, TL-QLDMR minus No variance, $-0.292$ MW$^2$; 95\% block-bootstrap CI $[-0.524, -0.142]$, $p_{\mathrm{Holm}}=0.008$).
- Farm~6 (within-site): TL-QLDMR had lower squared error than No discrepancy (MSE difference, TL-QLDMR minus No discrepancy, $-4.432\times 10^{-5}$ MW$^2$; 95\% block-bootstrap CI $[-5.832\times 10^{-5}, -3.030\times 10^{-5}]$, $p_{\mathrm{Holm}}<0.001$). The observed magnitude is practically negligible on the MW$^2$ scale.

The Holm-significant conditional removal effects were: Farm~1 (within-site): TL-QLDMR had lower squared error than No variance (MSE difference, TL-QLDMR minus No variance, $-0.292$ MW$^2$; 95\% block-bootstrap CI $[-0.524, -0.142]$, $p_{\mathrm{Holm}}=0.008$). Farm~6 (within-site): TL-QLDMR had lower squared error than No discrepancy (MSE difference, TL-QLDMR minus No discrepancy, $-4.432\times 10^{-5}$ MW$^2$; 95\% block-bootstrap CI $[-5.832\times 10^{-5}, -3.030\times 10^{-5}]$, $p_{\mathrm{Holm}}<0.001$). The observed magnitude is practically negligible on the MW$^2$ scale. These contrasts hold the TL-QLDMR-selected configuration, fitting rows, preprocessing, landmarks, and feature map fixed; variants were not independently retuned, and a failure to reject does not establish equivalence.

## Solver accounting

The audited Q1 search issued 3,888 distinct within-task--method configuration queries (54 per pair) and 10,368 fit records; 3,455 queried candidates were eligible under the frozen selection rule. Among 9,720 kernel seed fits, 1,214 still missed the fixed-feature primal--dual-gap certificate after the predeclared retry and therefore made their candidate ineligible; 8,506 kernel seed fits met the certificate. Across kernel and HistGBR fits, the records contain 12,691 solver attempts and 32,207,875 reported iterations. Summed per-fit recorded time was 50.70 h. This sum is descriptive because tasks ran in parallel; it is neither elapsed wall-clock time nor inference latency. The certificate applies only to the chosen fixed-feature objective and does not certify Nyström approximation error or predictive generalization.

| Scope | Method | Queries | Eligible | Failed certificate seed fits | Attempts | Iterations | Fit h |
|---|---|---:|---:|---:|---:|---:|---:|
| within | TL-QLDMR | 324 | 280 | 123 | 1287 | 4025325 | 7.072 |
| within | No variance | 324 | 276 | 135 | 1270 | 3941850 | 6.435 |
| within | No discrepancy | 324 | 273 | 146 | 1297 | 4217100 | 6.424 |
| within | Weighted SVQR | 324 | 249 | 215 | 1296 | 4594600 | 5.223 |
| within | Target SVQR | 324 | 324 | 0 | 973 | 137300 | 0.019 |
| within | HistGBR | 324 | 324 | 0 | 324 | 97200 | 2.674 |
| cross | TL-QLDMR | 324 | 275 | 132 | 1230 | 3563825 | 6.140 |
| cross | No variance | 324 | 277 | 132 | 1227 | 3497425 | 4.947 |
| cross | No discrepancy | 324 | 276 | 133 | 1220 | 3514200 | 4.693 |
| cross | Weighted SVQR | 324 | 253 | 198 | 1270 | 4384550 | 4.694 |
| cross | Target SVQR | 324 | 324 | 0 | 973 | 137300 | 0.017 |
| cross | HistGBR | 324 | 324 | 0 | 324 | 97200 | 2.359 |

## Manuscript insertion blocks

### RESULTS_POINT

In the primary within-site scope, HistGBR was the empirical winner on the macro RMSE endpoint at 13.913\%. TL-QLDMR recorded a site-equal capacity-normalized RMSE of 18.090\%, an average task rank of 3.50, and 1 task win including ties, ranking 3rd of six by the observed macro endpoint. In the secondary cross-site stress scope, HistGBR was the empirical winner on the macro RMSE endpoint at 14.923\%. TL-QLDMR recorded a site-equal capacity-normalized RMSE of 17.694\%, an average task rank of 3.67, and 1 task win including ties, ranking 5th of six by the observed macro endpoint.

### RESULTS_TASK_INFERENCE

The Holm-significant TL-QLDMR comparisons were: Farm~1 (within-site): TL-QLDMR had lower squared error than No variance (MSE difference, TL-QLDMR minus No variance, $-0.292$ MW$^2$; 95\% block-bootstrap CI $[-0.539, -0.139]$, $p_{\mathrm{Holm}}=0.016$). Farm~1 (within-site): TL-QLDMR had higher squared error than HistGBR (MSE difference, TL-QLDMR minus HistGBR, $204.912$ MW$^2$; 95\% block-bootstrap CI $[71.894, 425.012]$, $p_{\mathrm{Holm}}=0.012$). Farm~3 (within-site): TL-QLDMR had higher squared error than HistGBR (MSE difference, TL-QLDMR minus HistGBR, $71.623$ MW$^2$; 95\% block-bootstrap CI $[35.870, 114.080]$, $p_{\mathrm{Holm}}=0.011$). Farm~1 (cross-site): TL-QLDMR had higher squared error than HistGBR (MSE difference, TL-QLDMR minus HistGBR, $214.556$ MW$^2$; 95\% block-bootstrap CI $[91.628, 413.791]$, $p_{\mathrm{Holm}}=0.001$). Farm~6 (cross-site): TL-QLDMR had higher squared error than No variance (MSE difference, TL-QLDMR minus No variance, $16.917$ MW$^2$; 95\% block-bootstrap CI $[9.148, 27.398]$, $p_{\mathrm{Holm}}<0.001$). All other predeclared TL-QLDMR comparisons were not significant after adjustment; this absence of rejection is not evidence of equal performance.

### RESULTS_FIRST_TIER

The conservative first tier in the primary within-site scope contained HistGBR, No variance, and No discrepancy; TL-QLDMR was excluded ($p_{\mathrm{Holm}}=0.036$), meaning that its inferiority to HistGBR was established for the macro RMSE endpoint. Membership means that inferiority to the empirical winner was not established; it does not establish equivalence or equal performance. The conservative first tier in the secondary cross-site stress scope contained HistGBR, No discrepancy, No variance, Weighted SVQR, and TL-QLDMR; TL-QLDMR was retained ($p_{\mathrm{Holm}}=0.336$), meaning that its inferiority to HistGBR was not established. Membership means that inferiority to the empirical winner was not established; it does not establish equivalence or equal performance. The Holm-significant scope-level TL-QLDMR relations were: In the primary within-site scope, TL-QLDMR's observed macro RMSE was 4.178 percentage points higher than HistGBR; the all-pairs difference was significant after Holm adjustment ($p_{\mathrm{Holm}}=0.036$). In the secondary cross-site stress scope, TL-QLDMR's observed macro RMSE was 0.511 percentage points higher than No variance; the all-pairs difference was significant after Holm adjustment ($p_{\mathrm{Holm}}=0.016$). These tests concern the prespecified macro endpoint; empirical ordering by itself is not a significance claim.

### RESULTS_MATCHED_ABLATION

Under the conditional matched-removal protocol in the within-site scope, the four aggregate results were No variance 17.870\% (mean rank 2.17); TL-QLDMR 18.090\% (mean rank 2.00); Weighted SVQR 19.813\% (mean rank 3.17); No discrepancy 19.963\% (mean rank 2.67). TL-QLDMR ranked 2nd of four by observed macro RMSE. Under the conditional matched-removal protocol in the cross-site stress scope, the four aggregate results were No variance 17.671\% (mean rank 2.50); TL-QLDMR 17.694\% (mean rank 2.17); Weighted SVQR 17.702\% (mean rank 3.00); No discrepancy 17.715\% (mean rank 2.33). TL-QLDMR ranked 2nd of four by observed macro RMSE. The Holm-significant conditional removal effects were: Farm~1 (within-site): TL-QLDMR had lower squared error than No variance (MSE difference, TL-QLDMR minus No variance, $-0.292$ MW$^2$; 95\% block-bootstrap CI $[-0.524, -0.142]$, $p_{\mathrm{Holm}}=0.008$). Farm~6 (within-site): TL-QLDMR had lower squared error than No discrepancy (MSE difference, TL-QLDMR minus No discrepancy, $-4.432\times 10^{-5}$ MW$^2$; 95\% block-bootstrap CI $[-5.832\times 10^{-5}, -3.030\times 10^{-5}]$, $p_{\mathrm{Holm}}<0.001$). The observed magnitude is practically negligible on the MW$^2$ scale. These contrasts hold the TL-QLDMR-selected configuration, fitting rows, preprocessing, landmarks, and feature map fixed; variants were not independently retuned, and a failure to reject does not establish equivalence.

### RESULTS_OPTIMIZATION

The audited Q1 search issued 3,888 distinct within-task--method configuration queries (54 per pair) and 10,368 fit records; 3,455 queried candidates were eligible under the frozen selection rule. Among 9,720 kernel seed fits, 1,214 still missed the fixed-feature primal--dual-gap certificate after the predeclared retry and therefore made their candidate ineligible; 8,506 kernel seed fits met the certificate. Across kernel and HistGBR fits, the records contain 12,691 solver attempts and 32,207,875 reported iterations. Summed per-fit recorded time was 50.70 h. This sum is descriptive because tasks ran in parallel; it is neither elapsed wall-clock time nor inference latency. The certificate applies only to the chosen fixed-feature objective and does not certify Nyström approximation error or predictive generalization.
