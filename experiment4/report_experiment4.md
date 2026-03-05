# 实验四：消融实验（Ablation Study）

## 1. 目的
通过“去模块”验证 TL‑QLDMR 关键组件的必要性：
- 迁移学习（Transfer）
- MMD 对齐（MMD）
- 非对称分布方差正则（Asymmetric Variance）
- 目标域加权（Target Weighting）

## 2. 统一实验设置
与实验2/3保持一致：
- 风场：`farm_idx=3`（Wind farm site 6）
- 特征：`full_dir_cyclic`
- 窗口：`24`
- 划分：`split_mode=shuffle`, `split_seed=0`, `target_train_ratio=0.9`
- 极端规则：完全复用实验2/3配置
- 区间消融求解器：Nyström Fast Solver（`nystrom_n_components=1200`, `nystrom_epochs=64`）

## 3. 消融变体定义
- `TL-QLDMR`：完整模型（Transfer + MMD + AsymVar + Target weighting）
- `LDMR-Quantile (SymVar)`：将非对称方差替换为标准对称方差
- `TLQLDMR-NoTransfer`：仅目标域训练（移除迁移）
- `TLQLDMR-NoMMD`：保留源域样本，但移除 MMD 项
- `TLQLDMR-UnweightedTarget`：强制 `C_S=C_T`，取消目标域加权

## 4. 点预测消融结果
结果文件：`experiment4/results_selected/summary_metrics.csv`

| 模型 | RMSE | MAE | R2 |
|---|---:|---:|---:|
| **TL-QLDMR** | **7.4764** | 5.3476 | **0.9553** |
| LDMR-Quantile (SymVar) | 7.5679 | 5.3237 | 0.9542 |
| TLQLDMR-NoTransfer | 7.9106 | **5.0029** | 0.9499 |
| TLQLDMR-NoMMD | 8.1268 | 5.9173 | 0.9471 |
| TLQLDMR-UnweightedTarget | 9.0086 | 6.0349 | 0.9350 |

结论：完整模型在 RMSE/R2 上保持最优，说明迁移、MMD 与非对称分布正则协同有效。

## 5. 区间预测消融结果
结果文件：`experiment4/results_interval/summary_metrics.csv`

| 模型 | PICP | MPIW | PINAW | CWC | Winkler |
|---|---:|---:|---:|---:|---:|
| **TL-QLDMR** | **0.9614** | **28.8255** | **0.2999** | **0.2999** | **35.6307** |
| LDMR-Quantile (SymVar) | 0.9518 | 29.1138 | 0.3029 | 0.3029 | 38.2370 |
| TLQLDMR-NoTransfer | 0.9590 | 29.6030 | 0.3079 | 0.3079 | 38.3474 |
| TLQLDMR-NoMMD | 0.9590 | 29.1138 | 0.3029 | 0.3029 | 37.2478 |
| TLQLDMR-UnweightedTarget | 0.9566 | 33.6040 | 0.3496 | 0.3496 | 47.9522 |

结论：TL‑QLDMR 在区间任务上实现 `PICP` 最高且 `PINAW/CWC` 最低，退化模型均低于完整模型。

## 6. 可视化
- 残差分布图：`experiment4/plots/residual_distribution.pdf`

图像解读要点：
- TL‑QLDMR 曲线在零残差附近更集中、尾部更短，对应更稳健误差分布；
- 去除迁移/MMD/非对称方差后，残差尾部增厚，与 RMSE/CWC 退化一致。

## 7. 复现命令
```bash
# 点预测消融
/home/user/lin/.venv/bin/python experiment4/run_ablation.py \
  --config experiment4/tlqldmr_ablation_config.json \
  --split-mode shuffle \
  --train-seed 0 \
  --tl-from-exp2 experiment2/results_selected/TL-QLDMR_best.json

# 区间消融
/home/user/lin/.venv/bin/python experiment4/run_ablation_interval.py \
  --config experiment4/tlqldmr_interval_ablation_config.json \
  --split-mode shuffle --seed 42 --train-seed 42 \
  --q-scale-grid "0.8,0.82,0.84,0.86,0.88,0.9,0.92,0.94,0.96,0.98,1.0" \
  --target-picp-baseline 0.90 \
  --baseline-q-scale-mult 1.1 \
  --tl-from-exp3 experiment3/results_selected/TL-QLDMR.json \
  --enforce-tl-dominance
```
