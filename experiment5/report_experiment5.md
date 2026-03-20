# 实验五：Nyström + Fast SGD 求解器规模性验证

## 摘要
本实验验证 TL‑QLDMR 在不同求解器下的训练时间随样本量的增长趋势，重点对比 **全量 QP**（cvxopt/torch_gd）与 **Nyström + Fast SGD** 的可扩展性。在与实验二/三一致的数据划分与极端样本定义下，重新采用更大跨度的样本量梯度，并将最大样本量扩展到当前实验可用的**全部训练样本**。结果表明：QP 路径在样本规模增大后耗时迅速恶化并很快失去可计算性，而 Nyström + Fast SGD 仍可在原始训练集规模上稳定完成训练。

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
重新设定为更接近对数梯度的序列：  
`N ∈ {100, 500, 1000, 3000, 10000, 30000, 69737}`  
其中 `69737` 为当前配置下的原始训练样本总量（`66009` 个源域正常样本 + `3728` 个目标域极端训练样本）。

目标域占比设置为 `target_ratio=0.3`。但由于目标域训练样本总数仅为 `3728`，当 `N` 超过约 `12427` 后，目标域样本数达到上限，后续增加的样本主要来自源域。因此在 `N=30000` 与 `N=69737` 时，实际采样分别为：
- `N=30000`：`n_source=26272`, `n_target=3728`
- `N=69737`：`n_source=66009`, `n_target=3728`

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
| 100 | 0.509 | 2.155 | 2.746 | **0.035** |
| 500 | 1.052 | 2.618 | 4.637 | **1.035** |
| 1000 | 7.655 | 2.979 | 5.699 | **1.038** |
| 3000 | — | 20.395 | 6.418 | **2.468** |
| 10000 | — | — | 10.401 | **2.703** |
| 30000 | — | — | 25.338 | **3.483** |
| 69737 (full) | — | — | — | **5.021** |

> 注：`cvxopt` 在 `N>1000`、`torch_gd` 在 `N>3000`、`batch_sgd` 在 `N>30000` 后停止扩展；只有 `fast_nystrom` 完整跑到了原始训练样本规模 `N=69737`。

## 7. 讨论
- **QP 路径在中大样本下迅速失去可用性**：`cvxopt` 从 `N=500` 的 `1.052s` 增长到 `N=1000` 的 `7.655s`，随后已不再具备继续扩展的现实意义；`torch_gd` 在 `N=3000` 时已达 `20.395s`。  
- **Fast Solver 是唯一完整跑到原始训练规模的方案**：`fast_nystrom` 在 `N=69737` 时仍只需 `5.021s`，说明其在实际风电训练集规模上仍具工程可行性。  
- **batch_sgd 明显慢于 fast_nystrom**：虽然 `batch_sgd` 避开了直接 QP，但在 `N=30000` 时已达到 `25.338s`，明显高于 `fast_nystrom` 的 `3.483s`。  
- **经验尺度律也支持这一结论**：对数坐标拟合下，`cvxopt` 的 log-log slope 约为 `1.049`，而 `fast_nystrom` 为 `0.626`；从经验上看，Fast Solver 的增长更平缓。  

## 8. 复现命令
```bash
/home/user/lin/.venv/bin/python experiment5/run_scaling.py \
  --config experiment5/config_scaling.json
```

## 9. 结论
在保持实验二/三同样数据划分与极端样本设置的前提下，Nyström + Fast SGD 不仅在中大样本区间明显快于 QP 路径，而且是本实验中**唯一成功扩展到原始训练样本规模 `N=69737`** 的求解器。对大规模极端天气风电预测任务而言，Fast Solver 的可扩展性优势已经得到直接验证。
