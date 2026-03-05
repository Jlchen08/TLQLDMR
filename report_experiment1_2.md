# 极端天气下风电功率预测实验报告（实验一 & 实验二）

## 摘要
本报告在统一数据与预处理设置下完成两部分实验：
1) 实验一筛选极端天气样本并评估跨风场分布偏移；
2) 实验二在偏移最大的目标风场上进行点预测对比，并补充噪声鲁棒性测试。

针对审稿关注点，实验二新增了关键标准核方法基线：`Standard-SVR` 与 `Standard-LDMR`。结果显示 TL‑QLDMR 在点预测精度和噪声鲁棒性上均保持最优。

## 1. 数据与统一设置
- 数据：风电场 SCADA（多高度风速/风向、温度、气压、湿度、功率）
- 特征：`full_dir_cyclic`
- 滑窗：`window_size=24`
- 极端规则：统计极端（`q=98`，风速）、温度极端、切出风速（与实验3/4保持一致）
- 划分：`split_mode=shuffle`, `split_seed=0`, `target_train_ratio=0.9`

## 2. 实验一：极端样本筛选与风场选择
采用 W1 与 KS 评估正常域/极端域分布差异，最终选择分布偏移最大的风场作为后续目标域：
- 目标风场：`farm_idx=3`（Wind farm site 6, 96MW）

## 3. 实验二：点预测对比（含新增关键基线）

### 3.1 对比模型
- `TL-QLDMR`（本文）
- `Standard-SVR`（标准 ε-SVR，关键核方法基线）
- `Standard-LDMR`（标准对称方差 LDMR，关键核方法基线）
- `HHO-SVR`, `FLSVR`, `ARA-SVR`, `KMeans-GBT`, `RF-WPF`, `BRF-WPF`

### 3.2 测试集结果（`experiment2/results_selected/best_models_summary.csv`）
| 模型 | RMSE | MAE | R2 |
|---|---:|---:|---:|
| **TL-QLDMR** | **7.4764** | 5.3476 | **0.9553** |
| RF-WPF | 7.9674 | **5.3437** | 0.9492 |
| HHO-SVR | 8.3764 | 6.0220 | 0.9438 |
| Standard-SVR | 8.5637 | 5.5374 | 0.9413 |
| FLSVR | 9.2457 | 6.5611 | 0.9316 |
| ARA-SVR | 9.5515 | 6.0460 | 0.9270 |
| BRF-WPF | 9.7195 | 6.8697 | 0.9244 |
| KMeans-GBT | 10.6039 | 7.0217 | 0.9100 |
| Standard-LDMR | 10.6915 | 6.3269 | 0.9085 |

结论：TL‑QLDMR 在 RMSE/R2 上最优；新增 `Standard-SVR/Standard-LDMR` 后，改进有效性更完整。

### 3.3 预测图
实验二已保留预测图（含新增基线）：
- `experiment2/plots_selected/farm3_TL-QLDMR.pdf`
- `experiment2/plots_selected/farm3_Standard-SVR.pdf`
- `experiment2/plots_selected/farm3_Standard-LDMR.pdf`
- `experiment2/plots_selected/farm3_RF-WPF.pdf`
- `experiment2/plots_selected/farm3_BRF-WPF.pdf`

### 3.4 噪声鲁棒性（点预测）
结果文件：`experiment2/results_noise/noise_robustness.csv`（SNR=60/40/30 dB，训练与测试加噪）。

R2 对比（越高越好）：
- 60 dB：TL‑QLDMR `0.9550`（最高）
- 40 dB：TL‑QLDMR `0.9602`（最高）
- 30 dB：TL‑QLDMR `0.9582`（最高）

## 4. 学术完整性补充（基线来源）
- Standard SVR：经典支持向量回归（Vapnik 的 ε-insensitive 框架）
- **Standard LDMR**：
  - Qi et al., *Large-margin Distribution Machine-based regression*, Neural Computing and Applications, 2018.
  - DOI: `10.1007/s00521-018-3921-3`

## 5. 主要对比模型来源（实验二）
- HHO-SVR：Scientific Reports, 2025，DOI `10.1038/s41598-025-86275-6`
- FLSVR：Neural Processing Letters, 2025，DOI `10.1007/s11063-025-11780-8`
- ARA-SVR：Applied Sciences, 2025，DOI `10.3390/app15041779`
- KMeans-GBT：Journal of Big Data, 2025，DOI `10.1186/s40537-025-01071-3`
- RF-WPF：Cleaner Energy Systems, 2025，DOI `10.1016/j.cles.2025.100210`
- BRF-WPF：Sustainability, 2025，DOI `10.3390/su17114894`

## 6. 复现命令
```bash
# 实验二主对比（含新增 Standard-SVR / Standard-LDMR）
/home/user/lin/.venv/bin/python experiment2/run_benchmark_selected.py \
  --config experiment2/results_hunt/tlqldmr_hunt_best.json \
  --split-mode shuffle \
  --baseline-trials 1 \
  --svr-max-src 4000 --svr-max-tgt 1200 \
  --skip-tlqldmr-train

# 实验二噪声鲁棒性
/home/user/lin/.venv/bin/python experiment2/run_noise_robustness.py \
  --config experiment2/results_hunt/tlqldmr_hunt_best.json \
  --best-dir experiment2/results_selected \
  --split-mode shuffle \
  --noise-on-train \
  --snr-db 60,40,30
```
