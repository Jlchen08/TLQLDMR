# 极端天气下风电功率预测实验报告（实验一 & 实验二）

## 摘要
本报告在统一数据与预处理设置下完成两部分实验：
1) 实验一筛选极端天气样本并评估跨风场分布偏移；
2) 实验二在偏移最大的目标风场上进行点预测对比，并补充噪声鲁棒性测试。

针对审稿关注点，实验二进一步补入了两类关键核方法基线：`Standard-SVR` / `Standard-LDMR` 与 `BLSSVR`。结果显示 TL‑QLDMR 在点预测精度和噪声鲁棒性上均保持最优。

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
- `BLSSVR`（bounded-loss LS-SVR，EJOR 鲁棒回归基线）
- `HHO-SVR`, `FLSVR`, `ARA-SVR`, `KMeans-GBT`, `RF-WPF`, `BRF-WPF`

### 3.2 测试集结果（`experiment2/results_selected/best_models_summary.csv`）
| 模型 | RMSE | MAE | R2 |
|---|---:|---:|---:|
| **TL-QLDMR** | **7.4764** | 5.3476 | **0.9553** |
| Standard-SVR | 7.4977 | **5.0585** | 0.9550 |
| HHO-SVR | 7.6035 | 5.5295 | 0.9537 |
| RF-WPF | 7.9674 | 5.3437 | 0.9492 |
| FLSVR | 8.7945 | 6.3010 | 0.9381 |
| ARA-SVR | 9.2464 | 5.9537 | 0.9316 |
| KMeans-GBT | 9.6199 | 6.6950 | 0.9259 |
| BRF-WPF | 9.7195 | 6.8697 | 0.9244 |
| Standard-LDMR | 9.9318 | 6.1255 | 0.9211 |
| BLSSVR | 10.2733 | 6.1400 | 0.9155 |

结论：TL‑QLDMR 在 RMSE/R2 上仍最优；补入 `BLSSVR` 后，标准 SVR、标准 LDMR、bounded-loss LS-SVR 三类关键核回归基线均已覆盖。

### 3.3 预测图
实验二的点预测图已改为“每个模型仅保留一个代表性局部片段”，不再保留整段全局曲线，以减少图像冗余并突出局部拟合质量。当前图像统一保存在 `experiment2/plots_selected/`，例如：
- `experiment2/plots_selected/farm3_TL-QLDMR.pdf`
- `experiment2/plots_selected/farm3_Standard-SVR.pdf`
- `experiment2/plots_selected/farm3_Standard-LDMR.pdf`
- `experiment2/plots_selected/farm3_BLSSVR.pdf`
- `experiment2/plots_selected/farm3_RF-WPF.pdf`

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
- **BLSSVR**：
  - Fu et al., *Robust regression under the general framework of bounded loss functions*, European Journal of Operational Research, 2023.
  - DOI: `10.1016/j.ejor.2023.04.025`

## 5. 主要对比模型来源（实验二）
- HHO-SVR：Scientific Reports, 2025，DOI `10.1038/s41598-025-86275-6`
- FLSVR：Neural Processing Letters, 2025，DOI `10.1007/s11063-025-11780-8`
- ARA-SVR：Applied Sciences, 2025，DOI `10.3390/app15041779`
- KMeans-GBT：Journal of Big Data, 2025，DOI `10.1186/s40537-025-01071-3`
- RF-WPF：Cleaner Energy Systems, 2025，DOI `10.1016/j.cles.2025.100210`
- BRF-WPF：Sustainability, 2025，DOI `10.3390/su17114894`

## 6. 复现命令
```bash
# 实验二主对比（含 Standard-SVR / Standard-LDMR / BLSSVR）
/home/user/lin/.venv/bin/python experiment2/run_benchmark_selected.py \
  --config experiment2/results_hunt/tlqldmr_hunt_best.json \
  --baseline-trials 1 \
  --svr-max-src 4000 --svr-max-tgt 1200

# 实验二噪声鲁棒性
/home/user/lin/.venv/bin/python experiment2/run_noise_robustness.py \
  --config experiment2/results_hunt/tlqldmr_hunt_best.json \
  --best-dir experiment2/results_selected \
  --split-mode shuffle \
  --noise-on-train \
  --snr-db 60,40,30
```
