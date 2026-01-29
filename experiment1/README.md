# Experiment 1: Extreme Weather Screening & Visualization

This experiment screens extreme-weather samples (target domain) and visualizes:
- time-series snippets containing **cut-out** and **ramp** events
- **KDE** distribution shift between source (normal) vs target (extreme)
- **t-SNE** embedding showing a domain gap in feature space

All code and outputs live under `TL-QLDMR/experiment1/`.

## Run

From repo root (`/mnt/mydisk/zhangxiaohan/lin`):

```bash
python TL-QLDMR/experiment1/experiment1_data_screening_visualization.py --farm-idx 0 --feature-set full
```

Optional args:
- `--farm-idx`: 0..5, sorted by nominal capacity (desc)
- `--feature-set`: `full` | `wind_only` | `simple`
- `--max-tsne-samples`: subsample size for t-SNE (default 2500; t-SNE is O(N^2))
- `--perplexity`, `--tsne-iter`, `--seed`

## Outputs

Saved under `TL-QLDMR/experiment1/outputs/`:
- `farm{idx}_meta.json`: thresholds, counts, detected column names, run config
- `farm{idx}_events.csv`: per-timestep flags (`is_extreme`, `is_ramp`, `is_cutout`)
- `plots/`:
  - `farm{idx}_cutout_seg{k}.png`: time-series snippet with wind cut-out (wind_hub > 25 m/s)
  - `farm{idx}_ramp_seg{k}.png`: time-series snippet with power ramp (|ΔP| > 5% capacity / 15 min)
  - `farm{idx}_kde_wind_hub.png`, `farm{idx}_kde_power.png`: KDE distribution shift plots
  - `farm{idx}_tsne.png`: t-SNE domain gap visualization

## Domain Definition (Source vs Target)

- Source domain: `~is_extreme` (normal samples)
- Target domain: `is_extreme` (union of rules)
  - Statistical extremes: wind_hub > 95th percentile OR power > 95th percentile
  - Ramp: |ΔP(t) - ΔP(t-1)| > 0.05 * nominal_capacity
  - Cut-out: wind_hub > 25 m/s
  - Temperature extreme: T < -10°C or T > 35°C (if temperature exists in the file)

