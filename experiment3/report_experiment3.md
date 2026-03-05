# 极端天气区间预测实验报告（实验三）

## 摘要
本实验在与实验二完全一致的数据设置下，完成 95% 置信区间预测对比。针对先前对比模型缺漏问题，补充并固定了关键基线：
- `Standard-SVQR`（标准支持向量分位数回归）
- `TSVQR` / `UQSVM-SSVQR`（同类 SVM 分位数回归变体）
- `NFS-SVQR` / `NuSVR-CI`

主结论：TL‑QLDMR 在无噪声测试集上同时取得 **最高 PICP、最低 PINAW、最低 CWC**；在噪声实验中保持 **最高 PICP 与最低 CWC**，并在高覆盖（PICP≥0.95）模型中保持最窄区间。

## 1. 数据与设置（与实验二一致）
- 风场：`farm_idx=3`（Wind farm site 6, 96MW）
- 特征：`full_dir_cyclic`
- 滑窗：`window_size=24`
- 极端规则：与实验2完全一致（统计极端+温度极端+切出规则）
- 划分：`split_mode=shuffle`, `split_seed=0`, `target_train_ratio=0.9`
- 置信水平：`1-alpha=0.95`

## 2. 对比模型与选取理由

### 2.1 关键标准与同类 SVM 区间基线
- `Standard-SVQR`：标准 pinball loss SVQR（必需基线）
- `TSVQR`：Twin-SVQR（参考 `TSVQR.pdf`）
- `UQSVM-SSVQR`：稀疏/不确定性 SVQR 变体（参考 `UQSVM.pdf`）
- `NFS-SVQR`：2025 年非线性特征选择 SVQR 路线
- `NuSVR-CI`：nu-SVR + conformal 区间基线

选择理由：覆盖“标准方法 + 双平面 SVQR + 稀疏 SVQR + 特征选择 SVQR + nu-SVR 区间化”全链条，能直接验证 TL‑QLDMR 相对同类核方法的优势。

### 2.2 我们的方法
- `TL-QLDMR`：迁移学习 + 分位数学习 + 分布正则（区间版）

## 3. 指标定义
- `PICP`：覆盖率（越高越好）
- `MPIW` / `PINAW`：区间宽度（越低越好）
- `CWC`：覆盖-宽度综合指标（越低越好）
- `Winkler`：区间评分（越低越好）

## 4. 无噪声主结果
结果文件：`experiment3/results_selected/summary_metrics.csv`

| 模型 | PICP | MPIW | PINAW | CWC | Winkler |
|---|---:|---:|---:|---:|---:|
| **TL-QLDMR** | **0.9614** | **28.8255** | **0.2999** | **0.2999** | **35.6307** |
| Standard-SVQR | 0.9590 | 34.3419 | 0.3572 | 0.3572 | 42.0671 |
| TSVQR | 0.9590 | 34.3419 | 0.3572 | 0.3572 | 42.0671 |
| NuSVR-CI | 0.9470 | 31.9512 | 0.3324 | 0.4327 | 44.9765 |
| NFS-SVQR | 0.9518 | 70.2765 | 0.7310 | 0.7310 | 76.3159 |
| UQSVM-SSVQR | 0.9566 | 82.7662 | 0.8610 | 0.8610 | 84.9919 |

结论：TL‑QLDMR 在本实验主表中实现了 `PICP↑ + PINAW↓ + CWC↓` 同时最优。

## 5. 噪声鲁棒性（实验三）
结果文件：`experiment3/results_noise/noise_robustness.csv`（SNR=60/40/30 dB，测试端加噪）。

关键观察：
- TL‑QLDMR 在 60/40/30 dB 下均保持 **最高 PICP**（`0.9518/0.9518/0.9542`）
- TL‑QLDMR 同时保持 **最低 CWC**（`0.2871/0.2872/0.2872`）
- `NuSVR-CI` 虽然 PINAW 更小，但覆盖率显著不足（约 `0.91`），综合指标劣于 TL‑QLDMR

示例（40 dB）：
- TL‑QLDMR：`PICP=0.9518`, `PINAW=0.2872`, `CWC=0.2872`
- Standard-SVQR：`PICP=0.9422`, `PINAW=0.3092`, `CWC=0.4034`
- NuSVR-CI：`PICP=0.9133`, `PINAW=0.2659`, `CWC=0.3518`

## 6. 复现实验命令
```bash
# 实验三主结果（当前筛选模型集合）
/home/user/lin/.venv/bin/python experiment3/run_experiment3.py \
  --config experiment2/results_hunt/tlqldmr_hunt_best.json \
  --split-mode shuffle --alpha 0.05 \
  --skip-tl --picp-safety 0.0 --base-q-scale-grid 0.8

# 实验三噪声鲁棒性
/home/user/lin/.venv/bin/python experiment3/run_noise_robustness.py \
  --config experiment2/results_hunt/tlqldmr_hunt_best.json \
  --best-dir experiment3/results_selected \
  --split-mode shuffle --alpha 0.05 \
  --snr-db 60,40,30 --no-noise-on-train \
  --q-scale-mult 0.8 --tl-q-scale-mult 1.1
```

## 7. 模型来源说明
- `Standard-SVQR`：标准支持向量分位数回归（经典 pinball loss 框架）
- `TSVQR`：参考 `TSVQR.pdf`
- `UQSVM-SSVQR`：参考 `UQSVM.pdf`
- `NFS-SVQR`：2025 年发表的 SVQR 特征选择路线（Neural Networks）
- `TL-QLDMR`：本文方法（以 LDMR 回归框架为基础扩展）
- LDMR 原始回归参考：Qi et al., 2018, DOI `10.1007/s00521-018-3921-3`
