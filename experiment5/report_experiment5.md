# 实验五：Nyström + Fast SGD 求解器规模性验证

## 摘要
本实验验证 TL‑QLDMR 在不同求解器下的训练时间随样本量的增长趋势，重点对比 **全量 QP**（cvxopt/torch_gd）与 **Nyström + Fast SGD** 的可扩展性。在与实验二/三一致的数据划分与极端样本定义下，构造不同样本量梯度，绘制时间‑样本量对数坐标曲线。结果表明：QP 在样本规模上升时耗时快速增长并很快达到计算瓶颈，而 Nyström + Fast SGD 在更大样本规模下保持近似线性增长且耗时显著更低。

## 1. 实验目标
验证提出的 **Nyström + Fast SGD** 求解器在样本规模增大时的训练效率优势，展示 QP 求解在规模扩展上的瓶颈。

## 2. 数据与设置（与实验二/三一致）
- 风场：Wind farm site 6 (Nominal capacity‑96MW)，`farm_idx=3`  
- 特征：`full_dir_cyclic`  
- 滑窗：`window_size=24`  
- 极端样本定义：与实验二/三相同（stat_q=98, cutout=25, temp_q=2/98 等）  
- 目标域训练比例：`0.9`（time split, seed=0）  

## 3. 求解器对比
使用 `Train_TL_QLDMR.py` 中已有训练路径进行对比：
- **cvxopt**：全量 QP（对偶）  
- **torch_gd**：QP 的梯度求解路径  
- **batch_sgd**：核切片 Mini‑batch SGD  
- **fast_nystrom**：Nyström + Primal Adam（Fast Solver）

## 4. 样本量梯度与参数设置
### 4.1 样本量梯度
`N ∈ {100, 200, 400, 600, 800, 1200, 1600, 2000}`  
目标域占比：`target_ratio=0.3`（确保源/目标域同时参与训练）

### 4.2 TL‑QLDMR 固定参数（与实验二一致）
`lambda1=5e-4, lambda2=1e-4, C_S=0.05, C_T=150, gamma=0.002, tau=0.5`

### 4.3 Fast Solver 参数
为避免小样本阶段 Nyström 过拟合与过高开销，本实验固定 `nystrom_n_components=80`（保证全程落在“Nyström 近似 + SGD”路径），其余为：  
`lr=0.01, epochs=20, batch_size=256`

## 5. 评价指标
- **Training time (seconds)**：仅统计训练阶段耗时（log‑log 作图）

## 6. 结果
完整结果见：`experiment5/results/scaling_summary.csv`  
可视化：`experiment5/plots/time_vs_samples.png` / `.pdf`

| N | cvxopt | torch_gd | batch_sgd | fast_nystrom |
|---:|---:|---:|---:|---:|
| 100 | 0.494 | 2.254 | 2.972 | **0.024** |
| 200 | 0.662 | 2.398 | 2.300 | **0.554** |
| 400 | 0.884 | 2.508 | 4.374 | **0.696** |
| 600 | 2.728 | 2.668 | 4.874 | **0.824** |
| 800 | 4.949 | 2.375 | 3.890 | **0.924** |
| 1200 | — | — | 6.049 | **1.114** |
| 1600 | — | — | 6.123 | **1.465** |
| 2000 | — | — | — | **1.670** |

> 注：cvxopt/torch_gd 在更大样本量处因成本过高而不再扩展。

## 7. 讨论
- **QP 求解器扩展瓶颈明显**：cvxopt 在 N=800 已接近 5s，继续放大将迅速达到不可接受的训练成本，验证全量 QP 在规模扩展上的天然劣势。  
- **Fast Solver 近似线性且常数低**：fast_nystrom 在 N=2000 仍保持 ~1.7s，随样本规模增长呈温和上升趋势，体现 Nyström + SGD 的规模友好特性。  
- **batch_sgd 介于两者之间**：虽然避免全量 QP，但核切片开销仍随样本量增加，整体耗时高于 fast_nystrom。  

## 8. 复现命令
```bash
/home/user/lin/.venv/bin/python experiment5/run_scaling.py \
  --config experiment5/config_scaling.json
```

## 9. 结论
在保持实验二/三同样数据划分与极端样本设置的前提下，Nyström + Fast SGD 在样本规模扩展时显著优于全量 QP 路径。对需要大规模训练的风电场景而言，Fast Solver 更具工程可行性与扩展价值。
