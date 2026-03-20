# TL‑QLDMR 全实验复现总报告（实验1–实验5）

## 1. 统一实验约束
- 数据、风场、特征、极端规则在实验2/3/4完全一致
- 目标风场：`farm_idx=3`（Wind farm site 6, 96MW）
- 特征：`full_dir_cyclic`
- 滑窗：`window_size=24`
- 划分：`split_mode=shuffle`, `split_seed=0`, `target_train_ratio=0.9`

## 2. 核心结论总览
- 实验2（点预测）：TL‑QLDMR 的 `RMSE/R2` 最优
- 实验3（区间预测）：TL‑QLDMR 的 `PICP` 最高且 `PINAW/CWC` 最低
- 实验3噪声：TL‑QLDMR 在 60/40/30 dB 下 `PICP` 最高、`CWC` 最低（且在高覆盖模型中区间最窄）
- 实验4消融：去除 Transfer/MMD/AsymVar/TargetWeight 任一模块均退化
- 实验5求解器：Nyström + Fast SGD 是唯一成功扩展到原始训练样本规模 `N=69737` 的求解器

## 3. 关键结果快照

### 3.1 实验2（点预测）
来源：`experiment2/results_selected/best_models_summary.csv`
- **TL‑QLDMR**：RMSE `7.4764`, R2 `0.9553`
- Standard-SVR：RMSE `8.5637`, R2 `0.9413`
- Standard-LDMR：RMSE `10.6915`, R2 `0.9085`

### 3.2 实验3（区间预测）
来源：`experiment3/results_selected/summary_metrics.csv`
| 模型 | PICP | PINAW | CWC |
|---|---:|---:|---:|
| **TL‑QLDMR** | **0.9614** | **0.2999** | **0.2999** |
| Standard-SVQR | 0.9590 | 0.3572 | 0.3572 |
| TSVQR | 0.9590 | 0.3572 | 0.3572 |
| NuSVR-CI | 0.9470 | 0.3324 | 0.4327 |
| NFS-SVQR | 0.9518 | 0.7310 | 0.7310 |
| UQSVM-SSVQR | 0.9566 | 0.8610 | 0.8610 |

### 3.3 实验4（消融）
来源：`experiment4/results_selected/summary_metrics.csv`, `experiment4/results_interval/summary_metrics.csv`
- 点预测：TL‑QLDMR R2 `0.9553`，各消融变体均下降
- 区间预测：TL‑QLDMR `PICP=0.9614`, `PINAW=0.2999`, `CWC=0.2999`，均优于消融变体

### 3.4 实验5（求解器扩展性）
来源：`experiment5/results/scaling_summary.csv`, `experiment5/results/scaling_fit.csv`

- 当前训练样本总量为 `69737`（源域 `66009` + 目标域 `3728`）
- 重新采用更大跨度样本量：`N ∈ {100, 500, 1000, 3000, 10000, 30000, 69737}`
- 只有 `fast_nystrom` 完整跑到了 `N=69737`

| N | cvxopt | torch_gd | batch_sgd | fast_nystrom |
|---:|---:|---:|---:|---:|
| 100 | 0.509 | 2.155 | 2.746 | **0.035** |
| 500 | 1.052 | 2.618 | 4.637 | **1.035** |
| 1000 | 7.655 | 2.979 | 5.699 | **1.038** |
| 3000 | — | 20.395 | 6.418 | **2.468** |
| 10000 | — | — | 10.401 | **2.703** |
| 30000 | — | — | 25.338 | **3.483** |
| 69737 (full) | — | — | — | **5.021** |

- `cvxopt` 在 `N=1000` 已升至 `7.655s`，之后不再扩展
- `torch_gd` 在 `N=3000` 已达 `20.395s`
- `batch_sgd` 虽可扩展到 `N=30000`，但耗时 `25.338s`，显著慢于 `fast_nystrom`
- 对数拟合中，`cvxopt` 的 log-log slope 为 `1.049`，`fast_nystrom` 为 `0.626`，说明 Fast Solver 的增长更平缓

## 4. 新增关键基线与学术来源
- Standard-SVR（标准 ε-SVR）
- Standard-LDMR（标准对称方差 LDMR）
  - Qi et al., 2018, DOI: `10.1007/s00521-018-3921-3`
- Standard-SVQR（标准 pinball loss SVQR）
- TSVQR / UQSVM-SSVQR（来自 `TSVQR.pdf` / `UQSVM.pdf`）

## 5. 结果文件索引
- 实验2主结果：`experiment2/results_selected/`
- 实验2噪声：`experiment2/results_noise/noise_robustness.csv`
- 实验3主结果：`experiment3/results_selected/summary_metrics.csv`
- 实验3噪声：`experiment3/results_noise/noise_robustness.csv`
- 实验4点预测消融：`experiment4/results_selected/summary_metrics.csv`
- 实验4区间消融：`experiment4/results_interval/summary_metrics.csv`
- 实验5扩展性：`experiment5/results/scaling_summary.csv`, `experiment5/results/scaling_fit.csv`, `experiment5/plots/time_vs_samples.pdf`

## 6. 建议复现顺序
1. 实验2：确认点预测最优配置与标准基线
2. 实验3：区间预测主结果 + 噪声鲁棒性
3. 实验4：点预测与区间消融
4. 实验5：求解器扩展性验证
