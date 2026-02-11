# 极端天气区间预测实验报告（实验三）

## 摘要
本实验围绕“极端天气下风电功率区间预测”开展，目标是在极端域样本上比较多类区间预测模型，并评估 TL-QLDMR 在可靠性（覆盖率）与锐度（区间宽度）之间的取舍。实验采用 **95% 预测区间（PI，α=0.05）**，评价指标包括 PICP、MPIW/PINAW、CWC 与 Winkler Score。对比模型覆盖 2025 年风电功率区间预测文献中的代表方法（神经集成 + 动态 CQR、改进分位数集成、KDE 残差建模等），并补充多类支持向量机分位数回归变体（TSVQR、稀疏 SVQR、非线性特征选择 SVQR、NuSVR-CI）。结果显示，**在极端域 95% PI 场景下，覆盖率与区间宽度存在显著权衡**：NCQ-GBR 与 UQSVM-SSVQR 覆盖率更高但区间更宽；TL-QLDMR 在覆盖率与宽度之间保持相对均衡但未形成绝对优势。若强制要求 PICP ≥ 0.95 且 CWC < 0.5，需进一步放宽置信水平或引入更强的分布漂移校准机制。

## 1. 数据与实验设置
### 1.1 数据来源
风电场 SCADA 数据集（风速、风向、温度、气压、湿度与功率等变量），与实验二保持一致。

### 1.2 特征与滑窗
- 特征：full_dir_cyclic（多高度风速 + 温度 + 气压 + 湿度 + 风向 sin/cos 周期特征）
- 滑窗长度：window_size=24（预测下一时刻功率）
- 目标域划分：极端样本由统计风速极端 + 温度极端 + 切出规则组成（与实验二一致）

### 1.3 目标风场与数据规模
使用风场 6（Nominal capacity 96MW），极端域样本 4143：
- 目标域训练：2983
- 目标域验证：559
- 目标域校准：745
- 目标域测试：415

### 1.4 校准与参数搜索策略
- **校准策略**：先在校准集上进行 CQR（Conformalized Quantile Regression）校准，再在验证集上选择“区间宽度放缩系数（width_scale）”以逼近目标覆盖率，并在测试集上评估。
- **TL‑QLDMR**：采用实验二同范围候选超参数集合，并在验证集上联合选择分位数对与 width_scale，以覆盖率优先的准则确定最优配置。
- **其余模型**：使用轻量候选网格（3 组）或固定配置，保证与文献范式一致并控制计算开销。

## 2. 模型与选取理由
### 2.1 支持向量机分位数回归变体（SVM 系列）
1. **TSVQR**：Twin Support Vector Quantile Regression 基线（见 TSVQR.pdf）。
2. **UQSVM-SSVQR**：稀疏化/近似核的分位数 SVR（见 UQSVM.pdf）。
3. **NFS-SVQR**：非线性特征选择 + SVQR（2025 *Neural Networks*）。
4. **NuSVR-CI**：NuSVR + CQR 校准区间（经典 SVM 回归基线）。

**选取理由**：覆盖分位数回归、稀疏化、特征选择与经典 SVM 变体，体现 SVM 系列在区间预测中的多样化建模路线。

### 2.2 2025 年风电功率区间预测模型（文献代表）
1. **NESCQR**（*Applied Soft Computing*, 2025）：神经集成搜索 + 动态 CQR 的风电区间预测方法。
2. **NCQ-GBR**（*Journal of Renewable and Sustainable Energy*, 2025）：改进分位数集成学习的风电区间预测框架（本文以分位数 GBRT 近似实现非交叉分位数集成思想）。
3. **ENS-KDE**（2025 风电区间预测文献中的 KDE 残差建模思想）：本文使用集成模型 + KDE 残差估计近似实现。

**选取理由**：覆盖 2025 年风电功率区间预测中的三类主流范式：神经集成 + CQR、改进分位数集成、KDE 残差建模。

### 2.3 我们的方法
**TL-QLDMR（区间版）**：在迁移学习框架下训练上下分位数模型（tau 由验证集选择），并使用 CQR 校准 + width_scale 放缩（同时对区间进行 0~额定容量裁剪）。该方法旨在利用源域知识提升极端域小样本预测的稳定性，并在覆盖率优先的前提下尽量压缩区间宽度。

## 3. 区间预测指标
设真实值为 y，预测区间为 [L, U]，置信度 1-α=0.95：
- **PICP**：覆盖率 = mean( y ∈ [L, U] )
- **MPIW**：区间宽度均值 = mean(U − L)
- **PINAW**：标准化区间宽度 = MPIW / (max(y) − min(y))
- **CWC**：综合指标（覆盖率不足时给予惩罚）
- **Winkler Score**：区间评分（宽度 + 覆盖惩罚）

## 4. 结果与分析
### 4.1 数值结果（测试集）
| 模型 | PICP | MPIW | PINAW | CWC | Winkler |
|---|---:|---:|---:|---:|---:|
| NCQ-GBR | **0.954** | 55.669 | 0.628 | **0.628** | 109.413 |
| UQSVM-SSVQR | 0.966 | 91.189 | 1.029 | 1.029 | 94.027 |
| TL-QLDMR | 0.928 | 73.549 | 0.830 | 52.682 | 140.393 |
| TSVQR | 0.908 | 53.599 | 0.605 | 46.426 | 117.055 |
| NuSVR-CI | 0.899 | 48.778 | 0.550 | 46.470 | 114.489 |
| NESCQR | 0.870 | 51.538 | 0.582 | 65.368 | 185.663 |
| ENS-KDE | 0.848 | 56.951 | 0.643 | 89.570 | 137.402 |
| NFS-SVQR | 0.829 | 82.838 | 0.935 | 157.786 | 111.298 |

### 4.2 结论性观察
- **极端域 95% PI 下，覆盖率与区间宽度呈明显权衡**：覆盖率更高的模型往往区间更宽。
- **NCQ-GBR 覆盖率最高且综合指标最好**，但 PINAW 仍显著高于 0.5；
- **TL-QLDMR 的覆盖率（0.928）仍低于 0.95**，其区间宽度中等，表现为“覆盖率与宽度的折中方案”；
- 若要求 **PICP ≥ 0.95 且 CWC < 0.5**，需进一步降低置信度（如 90% PI）或引入更强的分布漂移校准策略。

## 5. 复现实验命令
```bash
.venv/bin/python experiment3/run_experiment3.py --alpha 0.05 --picp-safety 0.02
.venv/bin/python experiment3/recalibrate_all.py --alpha 0.05 --picp-safety 0.02
```

## 6. 局限性
1. 部分 2025 模型采用了**简化实现**（如 NCQ-GBR、ENS-KDE），主要复现其“核心思想”，未完整实现原论文全部结构；
2. 极端样本占比偏小，区间预测在最剧烈波动区间的可靠性仍受样本分布影响；
3. 若强行要求 PICP ≥ 0.95 且 CWC < 0.5，需要更强的校准机制或降低置信度（如 90% PI），目前实现难以同时满足两者。

## 7. 参考文献（模型来源）
[1] Hu J., Deng Y., Che J. *A novel wind power interval prediction method based on neural ensemble search and dynamic conformalized quantile regression*. Applied Soft Computing, 2025, 180:113476. DOI: 10.1016/j.asoc.2025.113476.

[2] Hu L., Liu Y.G. *A day-ahead wind power conformal interval forecasting framework based on numerical weather prediction data and improved quantile ensemble learning*. Journal of Renewable and Sustainable Energy, 2025, 17(4):046104. DOI: 10.1063/5.0233770.

[3] Ye Y.-F., Wang J., Chen W.-J. *Nonlinear feature selection for support vector quantile regression*. Neural Networks, 2025, 185:107136. DOI: 10.1016/j.neunet.2025.107136.

[4] *Twin support vector quantile regression*. Expert Systems with Applications, 2023, 221:121239. DOI: 10.1016/j.eswa.2023.121239.

[5] *A Novel Wind Power Combination Model Interval Prediction Method Based on NWP Wind Speed Error Correction*. 2025. DOI: 10.3969/j.issn.2096-9066.2025.06.014.

[6] TSVQR.pdf, UQSVM.pdf（用户提供的模型来源文件）。
