# 实验二/三样本使用与 TL-QLDMR 快速求解代码定位

## 1. 当前统一数据规模（风场 6，`farm_idx=3`）
- 原始时间点：`70176`
- 建窗后有效样本：`70152`
- 正常域 / 源域样本：`66009`
- 极端域 / 目标域样本：`4143`
- 目标域训练样本：`3728`
- 目标域测试样本：`415`

这些划分来自：
- `experiment2/data_utils.py:146`：滑动窗口构造
- `experiment2/data_utils.py:156`：正常样本进入源域
- `experiment2/data_utils.py:160`：极端样本进入目标域
- `experiment2/data_utils.py:173`：目标域按 `target_train_ratio` 划分训练/测试

## 2. 实验二实际用了多少样本

### 2.1 TL-QLDMR
- TL-QLDMR 在实验二主实验中使用：
  - 源域训练：`66009`
  - 目标域训练：`3728`
  - 目标域测试：`415`
- 也就是说，TL-QLDMR 基本使用了**全部非测试训练样本**，没有再对源域或目标域训练集做额外截断。

对应代码：
- `experiment2/run_tlqldmr_hunt.py:29`
- `experiment2/run_tlqldmr_hunt.py:57`

### 2.2 实验二基线模型
- 实验二多数基线没有使用全部训练样本，而是为了计算成本做了截断。
- 当前主实验脚本里，这些模型会被限制为：
  - 源域最多 `svr_max_src`
  - 目标域训练最多 `svr_max_tgt`
- 你当前实验二主实验常用配置是：
  - `svr_max_src = 4000`
  - `svr_max_tgt = 1200`

对应代码：
- `experiment2/run_benchmark_selected.py:217`
- `experiment2/run_benchmark_selected.py:227`
- `experiment2/run_benchmark_selected.py:229`

因此，实验二可以概括为：
- **TL-QLDMR：基本用满全部非测试训练样本**
- **多数传统基线：没有用满，而是做了训练样本截断**

## 3. 实验三实际用了多少样本

### 3.1 目标域不是全部拿去拟合
- 实验三区间预测会先把目标域训练集 `3728` 再拆成三部分：
  - `train`
  - `val`
  - `calibration`
- 当前划分规则：
  - `val_ratio = 0.15`
  - `cal_ratio = 0.2`

对应代码：
- `experiment3/run_experiment3.py:513`

当前实际数量为：
- `train = 2424`
- `val = 559`
- `calibration = 745`
- `test = 415`

### 3.2 TL-QLDMR 在实验三
- TL-QLDMR 最终训练时使用：
  - 源域全部：`66009`
  - 目标域 `train + val = 2983`
- 不使用 `calibration` 参与模型拟合，因为这部分专门用来做 conformal 校准。

对应代码：
- `experiment3/run_experiment3.py:769`
- `experiment3/run_experiment3.py:815`

当前结果文件中也记录了这一点：
- `experiment3/results_selected/TL-QLDMR.json`
  - `train_samples = 2983`
  - `val_samples = 559`
  - `cal_samples = 745`
  - `test_samples = 415`

### 3.3 实验三区间基线
- 实验三区间基线不仅保留了 calibration，不会把目标域训练样本全部用于拟合；
- 其中很多模型还进一步设置了 `max_target` 上限，所以连 `train+val` 也没有用满。

对应代码：
- 搜索阶段截断：`experiment3/run_experiment3.py:873`
- 最终训练截断：`experiment3/run_experiment3.py:937`

例如当前主实验中：
- `Standard-SVQR`：最终只用了 `700` 个目标域训练样本
- `RLMKL`：最终只用了 `1000` 个目标域训练样本
- `TL-QLDMR`：用了完整的 `2983` 个目标域 `train+val`

可从结果文件看到：
- `experiment3/results_selected/Standard-SVQR.json`
- `experiment3/results_selected/RLMKL.json`
- `experiment3/results_selected/TL-QLDMR.json`

### 3.4 实验三噪声实验
- 噪声实验继承主实验的训练样本组织方式。
- 即：TL-QLDMR 仍然用全部源域 + 完整目标域 `train+val`；
- 其他很多基线仍然受到 `max_target` 限制。

对应代码：
- `experiment3/run_noise_robustness.py:554`
- `experiment3/run_noise_robustness.py:557`
- `experiment3/run_noise_robustness.py:564`

## 4. 结论：实验二和实验三是否“用了所有样本”

严格结论如下：

### 4.1 实验二
- **没有用全部原始样本做训练**，因为必须保留目标域测试集 `415`
- **TL-QLDMR 基本用满了全部非测试训练样本**
- **大多数基线没有用满，做了截断**

### 4.2 实验三
- **没有用全部原始样本做训练**，因为既要保留测试集，也要保留 calibration 集
- **TL-QLDMR 用满了源域样本，但目标域只用到 `train+val`，不使用 calibration**
- **许多基线连 `train+val` 也没用满，因为有 `max_target` 限制**

---

## 5. TL-QLDMR 对 kernel QP 的处理：代码定位

你之前那段描述，对应的核心代码主要在 `Train_TL_QLDMR.py`。

### 5.1 Nyström 低秩近似
- `Train_TL_QLDMR.py:57`
  - `NystromApproximation` 类
- `Train_TL_QLDMR.py:67`
  - `fit()`：选择代表性基点（landmarks）
- `Train_TL_QLDMR.py:97`
  - `transform()`：把输入映射到低维显式特征空间

这部分对应论文表述里的：
- “利用 Nyström 近似选取代表性基点”
- “把依赖完整 Gram 矩阵的隐式核学习问题转为低维显式参数优化问题”

### 5.2 在近似特征空间里统一写目标函数
- `Train_TL_QLDMR.py:147`
  - `_compute_regularizer_matrix()`

这里统一构造了：
- LDMR 方差正则：`Train_TL_QLDMR.py:156`
- 非对称方差版本：`Train_TL_QLDMR.py:159`
- MMD 对齐项：`Train_TL_QLDMR.py:174`

这部分对应论文表述里的：
- “将分位数损失、源/目标域加权、MMD 对齐、LDMR 分布间隔正则统一纳入同一优化目标”

### 5.3 快速一阶优化，而不是完整对偶 QP
- `Train_TL_QLDMR.py:180`
  - `Fast_TL_QLDMR.fit()`
- `Train_TL_QLDMR.py:189`
  - 开始 Nyström 映射
- `Train_TL_QLDMR.py:197`
  - 用样本池拟合特征映射
- `Train_TL_QLDMR.py:213`
  - 源域 / 目标域损失权重 `C_S / C_T`
- `Train_TL_QLDMR.py:252`
  - 分位数损失残差
- `Train_TL_QLDMR.py:255`
  - pinball loss 梯度
- `Train_TL_QLDMR.py:272`
  - Adam 更新

这部分对应论文表述里的：
- “不再精确求解完整对偶二次规划”
- “改为在低秩核特征空间中做快速一阶优化”

### 5.4 主入口如何切到 fast_nystrom
- `Train_TL_QLDMR.py:370`
  - `TL_QLDMR.fit()`
- `Train_TL_QLDMR.py:387`
  - 当 `solver == "fast_nystrom"` 时，切换到快速求解器

### 5.5 fast_nystrom 训练完成后如何回到统一预测接口
- `Train_TL_QLDMR.py:927`
  - `_fit_fast_nystrom()`
- `Train_TL_QLDMR.py:963`
  - 把 Nyström 低维参数恢复成可预测的核系数形式
- `Train_TL_QLDMR.py:969`
  - `X_train` 被设置为 landmarks

### 5.6 预测阶段如何使用这些 landmarks
- `Predict_TL_QLDMR.py:13`
  - `Predictor.predict()`
- `Predict_TL_QLDMR.py:26`
  - 使用 `self.model.X_train`
- `Predict_TL_QLDMR.py:27`
  - 使用 `self.model.beta_primal`
- `Predict_TL_QLDMR.py:49`
  - 预测时计算 `K(x, landmarks)`

也就是说：
- 训练阶段不再围绕“全体训练样本的完整核矩阵”做 QP；
- 预测阶段则使用“测试点到 landmarks 的核值”完成输出。

### 5.7 传统完整 kernel QP 路线在哪
- `Train_TL_QLDMR.py:401`
  - 从这里开始是传统完整核矩阵 + QP 的路径
- `Train_TL_QLDMR.py:981`
  - `_solve_qp_qpth()`
- `Train_TL_QLDMR.py` 中 `cvxopt` / `torch_gd` / `qpth` 相关逻辑，都是对偶 QP 路线

这部分正好可以作为与 `fast_nystrom` 的对照实现。
