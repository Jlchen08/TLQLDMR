# TL‑QLDMR 全部实验复现与关键信息总报告

本报告汇总 **实验1–实验5** 的关键设置、参数、结果与复现入口，便于后续复现、改进与扩展。所有实验均围绕**极端天气下风电功率预测**，以 TL‑QLDMR 为核心模型。

---

## 0. 运行环境与依赖
- Python 环境：`/home/user/lin/.venv/bin/python`
- 主要依赖：`numpy`, `pandas`, `scikit-learn`, `matplotlib`, `cvxopt`, `torch`（可选 `qpth`）
- 关键源码：
  - `Train_TL_QLDMR.py`（TL‑QLDMR 主实现，含 QP/SGD/Nyström 快速求解器）
  - `wind_farm_Data_Utils.py`（数据加载、标准化、极端样本识别）
  - `experiment2/data_utils.py`（滑窗、分域、缺失值填充）

---

## 1. 数据与预处理（全实验通用）
### 1.1 数据来源
风电场 SCADA 数据（多高度风速/风向/温度/气压/湿度/功率），存放于 `data/wind_farm_data/`。

### 1.2 特征与标准化
- **特征集**：`full_dir_cyclic`（多高度风速 + 温度 + 气压 + 湿度 + 风向 sin/cos）
- **标准化**：`StandardScaler`（见 `wind_farm_Data_Utils.py`）

### 1.3 滑窗构造
窗口长度固定为 `window_size=24`（15min × 24 = 6h），预测下一时刻功率。

### 1.4 极端样本识别（关键规则）
与实验二/三保持一致的极端规则（配置来自 `experiment2/results_hunt/tlqldmr_hunt_best.json`）：
- `stat_q=98`，`stat_on_wind=True`，`stat_on_power=False`
- `use_stat=True`, `use_ramp=False`, `use_cutout=True`, `use_temp=True`
- `cutout_ms=25`
- `temp_low=-10`, `temp_high=35`, `temp_q_low=2`, `temp_q_high=98`
- `temp_requires_wind=False`

### 1.5 域划分（源域/目标域）
- 源域：非极端样本  
- 目标域：极端样本  
- 典型规模（风场6）：  
  - Source（Normal）≈ 66009  
  - Target（Extreme）≈ 4143  

---

## 2. 目标风场与一致性设置
### 2.1 风场选择
- 实验1基于 **W1 + KS** 评估域偏移最大风场  
- 结果：**风场6（farm_idx=3，96MW）** 域偏移最大 → 作为后续实验目标风场

### 2.2 统一设置
- 风场：Wind farm site 6 (Nominal capacity‑96MW)，`farm_idx=3`
- 特征：`full_dir_cyclic`
- 滑窗：`window_size=24`
- 目标域划分：`target_train_ratio=0.9`
- `split_seed=0`

---

## 3. TL‑QLDMR 关键参数（点预测）
来源：实验二最佳配置（`experiment2/results_hunt/tlqldmr_hunt_best.json`）
- `lambda1=5e-4`, `lambda2=1e-4`
- `C_S=0.05`, `C_T=150`
- `kernel_gamma=0.002`, `tau=0.5`
- Nyström（point/fast 训练）参数（若使用 fast_nystrom）：  
  - `nystrom_n_components=1200`, `lr=0.01`, `epochs=64`, `batch=256`

---

## 4. 实验1：极端样本筛选与域偏移评估
**报告**：`report_experiment1_2.md`  
**方法**：对各风场极端域与正常域计算 W1 / KS 分布差异  
**结论**：风场6偏移最大 → 用作实验2/3/4/5  
**复现**：见 `experiment1/README.md`

---

## 5. 实验2：极端域点预测（多模型对比）
**报告**：`report_experiment1_2.md`

### 5.1 模型列表（非深度学习）
HHO‑SVR、FLSVR、ARA‑SVR、KMeans‑GBT、RF‑WPF、BRF‑WPF、TL‑QLDMR。

### 5.2 关键结果（目标域测试集）
| 模型 | RMSE | MAE | R² |
|---|---:|---:|---:|
| **TL‑QLDMR** | **7.476** | **5.348** | **0.9553** |
| RF‑WPF | 7.967 | 5.344 | 0.9492 |
| HHO‑SVR | 7.603 | 5.530 | 0.9537 |
| 其余模型 | 8.79–9.72 | 5.95–6.87 | 0.924–0.938 |

### 5.3 噪声鲁棒性（点预测）
在训练与测试端同时加入 SNR 噪声（60/40/30/20/10），TL‑QLDMR 通过提高 Nyström 维度与训练轮数保持最优。
结果记录：`experiment2/results_noise/`

### 5.4 复现命令
详见 `report_experiment1_2.md`。

---

## 6. 实验3：极端域区间预测（95% PI）
**报告**：`experiment3/report_experiment3.md`

### 6.1 设置
- 置信度：95%（α=0.05）  
- TL‑QLDMR 采用 τ=0.2/0.8 双分位数  
- 使用 CQR 校准 + 区间缩放  
- 最终 `q_scale=0.88`

### 6.2 关键结果（测试集）
| 模型 | PICP | PINAW | CWC |
|---|---:|---:|---:|
| **TL‑QLDMR** | **0.959** | **0.268** | **0.268** |
| TSVQR | 0.889 | 0.256 | 0.342 |
| NuSVR‑CI | 0.843 | 0.208 | 0.285 |
| 2025 风电区间模型 | 0.94–0.95 | 0.329–0.337 | 0.329–0.441 |

### 6.3 噪声鲁棒性（区间预测）
仅在测试端加入 SNR 噪声（60/40/30），TL‑QLDMR 仍保持最低 CWC：
```
TL‑QLDMR: CWC ≈ 0.2778–0.2779
```
结果记录：`experiment3/results_noise/noise_robustness.csv`

### 6.4 复现命令
详见 `experiment3/report_experiment3.md`

---

## 7. 实验4：消融实验（点预测 + 区间预测）
**报告**：`experiment4/report_experiment4.md`

### 7.1 点预测消融（与实验2对齐）
| 模型 | RMSE | R² |
|---|---:|---:|
| **TL‑QLDMR** | **7.476** | **0.9553** |
| w/o Transfer | 7.930 | 0.9497 |
| w/o LDMR | 8.205 | 0.9461 |
| w/o MMD | 7.745 | 0.9520 |

### 7.2 区间预测消融（与实验3对齐）
| 模型 | PICP | PINAW | CWC |
|---|---:|---:|---:|
| **TL‑QLDMR** | **0.959** | **0.2683** | **0.2683** |
| w/o Transfer | 0.952 | 0.2797 | 0.2797 |
| w/o LDMR | 0.949 | 0.2697 | 0.3507 |
| w/o MMD | 0.947 | 0.2710 | 0.3528 |

### 7.3 残差分布图
输出：`experiment4/plots/residual_distribution.png` / `.pdf`

### 7.4 复现命令
详见 `experiment4/report_experiment4.md`

---

## 8. 实验5：Nyström + Fast SGD 求解器规模性验证
**报告**：`experiment5/report_experiment5.md`

### 8.1 设置
- 样本规模：N ∈ {100, 200, 400, 600, 800, 1200, 1600, 2000}
- solver 对比：cvxopt / torch_gd / batch_sgd / fast_nystrom
- Fast Solver 固定 `nystrom_n_components=80`

### 8.2 关键结论
QP 在 N=800 后成本迅速上升；fast_nystrom 在 N=2000 仍保持 ~1.7s，具有更好的扩展性。

### 8.3 复现命令
```
/home/user/lin/.venv/bin/python experiment5/run_scaling.py \
  --config experiment5/config_scaling.json
```

---

## 9. 关键配置与输出文件索引
### 9.1 配置文件
- `experiment2/results_hunt/tlqldmr_hunt_best.json`（TL‑QLDMR 点预测最优参数）
- `experiment3/tlqldmr_override.json`（区间预测配置）
- `experiment4/tlqldmr_ablation_config.json` / `tlqldmr_interval_ablation_config.json`
- `experiment5/config_scaling.json`

### 9.2 结果目录
- `experiment2/results_selected/`（实验2主结果）
- `experiment2/results_noise/`（点预测噪声鲁棒性）
- `experiment3/results_selected/`（实验3主结果）
- `experiment3/results_noise/`（区间预测噪声鲁棒性）
- `experiment4/results_selected/`（消融点预测）
- `experiment4/results_interval/`（消融区间预测）
- `experiment5/results/`（规模验证结果）

### 9.3 可视化
- `experiment2/plots_selected/`（点预测曲线）
- `experiment4/plots/`（残差分布）
- `experiment5/plots/`（时间‑样本量曲线）

### 9.4 关键代码修正（影响可复现性）
- `Train_TL_QLDMR.py`：QP 路径中使用 `SY` 修复未定义变量 `S`，确保 torch_gd 可运行。  
- `Train_TL_QLDMR.py`：cvxopt 输入强制 `float64`，避免 `buffer format not supported` 报错。  

---

## 10. 复现顺序建议
1) 实验1：确认风场选择  
2) 实验2：点预测基线与 TL‑QLDMR 最优参数  
3) 实验3：区间预测 + 噪声鲁棒性  
4) 实验4：点/区间消融（复用实验2/3结果）  
5) 实验5：求解器规模性对比

---

## 11. 注意事项与局限
1. 部分 2025 模型为**核心思想简化实现**（保持可比性，但非完整复刻）。  
2. SVM 系列模型在部分实验中对训练样本量有上限限制。  
3. 噪声实验分两类：  
   - 实验2：训练 + 测试端同时加噪  
   - 实验3：仅测试端加噪（训练与校准保持无噪声）
4. 若需进一步提升区间覆盖率，可增加 `q_scale` 或调高 `target_picp`。

---

## 12. 快速入口
若需要最小化复现成本，优先阅读并执行：
- `report_experiment1_2.md`
- `experiment3/report_experiment3.md`
- `experiment4/report_experiment4.md`
- `experiment5/report_experiment5.md`
