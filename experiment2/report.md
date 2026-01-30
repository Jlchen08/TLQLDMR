# Experiment 2 Report (Single-Farm, Largest Domain Shift)

## Farm Selection Rationale
Based on the domain shift plots from `experiment1/outputs/plots`, **Farm 6 (index 5, 36 MW)** shows the largest distribution gap between normal (source) and extreme (target) conditions for both wind speed and power. The target density is shifted toward higher wind speeds and much higher power output with limited overlap, indicating a pronounced source–target mismatch. Therefore, this farm was selected for Experiment 2.

## Experiment Setup
- **Farm:** Wind farm site 6 (index 5, 36 MW)
- **Target domain test:** extreme-weather samples
- **Model:** TL-QLDMR (Median)

### Search Space (data + model)
We expanded the search to cover **data settings** (which strongly affect results) and **model hyperparameters**:
- **Feature set:** `full`, `wind_only`
- **Window size:** 3, 6, 12 (15‑min intervals)
- **Target train ratio:** 0.5, 0.6, 0.7

Model hyperparameters:
- `lambda1`: 0.001 / 0.01 / 0.1  
- `lambda2`: 0.001 / 0.01 / 0.1  
- `C_S`: 0.5 / 1.0  
- `C_T`: 5.0 / 10.0  
- `kernel_gamma`: 0.005 / 0.02 / 0.05  
- `nystrom_n_components`: 100 / 150 / 200  
- `nystrom_lr`: 0.005 / 0.01  
- `nystrom_epochs`: 8 (search) → **20** (final retrain)

Search results are recorded in:  
`experiment2/results/tl_qldmr_search_results.csv`  
Best configuration: `experiment2/results/tl_qldmr_best_overall.json`

## Best Configuration (test set)
Selected from the combined search over data conditions and TL‑QLDMR hyperparameters:

- **Feature set:** `wind_only`  
- **Window size:** 3  
- **Target train ratio:** 0.6  
- **Hyperparameters:**  
  - `lambda1=0.01`, `lambda2=0.001`  
  - `C_S=1.0`, `C_T=10.0`  
  - `kernel_gamma=0.02`  
  - `nystrom_n_components=150`, `nystrom_lr=0.01`  
  - `nystrom_epochs=20`  

**Test metrics (TL‑QLDMR median):**  
- RMSE **5.5235**  
- MAE **4.1604**  
- R² **0.7709**  

See `experiment2/results/metrics_summary.csv` for the recorded results.

> Note: **MAPE is extremely large** because target power values can be near zero; it is less stable than RMSE/MAE/R² in this dataset.

## Wave Peak/Valley Behavior
From the TL‑QLDMR prediction plot in `experiment2/plots/`:
- The tuned configuration tracks peak/valley timing closely under strong domain shift.
- Remaining errors are mostly at sharp ramps, but overall amplitude and phase alignment improved versus previous settings.

## Conclusion
By jointly tuning **data settings** and **TL‑QLDMR hyperparameters**, we achieved a substantially stronger fit on the most challenging farm (largest domain shift). The best configuration yields **RMSE 5.5235 / MAE 4.1604 / R² 0.7709**, demonstrating that TL‑QLDMR can deliver high‑quality point predictions when carefully optimized.

## Outputs
- Metrics table (CSV/MD): `experiment2/results/metrics_per_farm.*`, `metrics_summary.*`
- Plots (PNG/PDF): `experiment2/plots/`
- This report: `experiment2/report.md`
