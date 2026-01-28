"""
风电场功率预测（TL-QLDMR）
数据来源：国家电网可再生能源发电数据集（s41597-022-01696-6）

参考与用途说明：
- Train_TL_QLDMR.py：核心训练器 TL_QLDMR 的实现，支持对偶 QP、batch_sgd 以及 fast_nystrom
  用途：模型训练与参数配置（核宽、分位数、正则项、solver 等）
- wind_farm_Data_Utils.py：风电场数据读取与处理工具
  用途：加载风场 Excel、构建特征、识别极端天气、滑窗样本与领域划分
"""

import numpy as np
import matplotlib.pyplot as plt
from wind_farm_Data_Utils import WindFarmDataGenerator
from Train_TL_QLDMR import TL_QLDMR
from Predict_TL_QLDMR import Predictor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import os


def run_experiment(farm_idx=0, feature_set='full', window_size=12, solver="batch_sgd"):
    """
    在风电场数据上运行 TL-QLDMR 实验
    
    参数：
        farm_idx: 风场索引（按装机容量降序排列，0-5）
        feature_set: 特征集合，'full'（全部特征）、'wind_only'（仅风速）、'simple'（机舱风速+温度）
        window_size: 滑动窗口长度，默认 12（即 3 小时，15 分钟采样）
        solver: 训练方法，'batch_sgd'（核切片小批量）或 'fast_nystrom'（Nyström+SGD，来自 Train_TL_QLDMR 集成）
    """
    # ===============================
    # 板块 1：数据加载与处理
    # 依赖 wind_farm_Data_Utils.WindFarmDataGenerator 完成：
    # - 加载风电场数据（Excel）
    # - 动态匹配不同风场的列名，选取特征
    # - 识别极端天气样本标签
    # - 归一化与滑窗样本构建、领域划分
    # ===============================
    print(">>> Step 1: Loading Wind Farm Data...")
    
    gen = WindFarmDataGenerator()
    X_scaled, y_scaled, is_extreme = gen.load_and_process_data(farm_idx=farm_idx, feature_set=feature_set)
    
    # 构建滑窗样本
    X, y, labels = gen.create_sliding_window(X_scaled, y_scaled, is_extreme, window_size=window_size)
    
    # 按标签划分领域：源域（正常）与目标域（极端）
    X_S, y_S, X_T, y_T = gen.split_domain_data(X, y, labels)
    
    # 目标域划分：60% 训练、40% 测试（样本过少时调整为 50%）
    n_T_train = int(len(X_T) * 0.6)
    if n_T_train < 10:
        print("Warning: Target samples too few, increasing ratio to 0.5")
        n_T_train = int(len(X_T) * 0.5)
    
    X_T_train = X_T[:n_T_train]
    y_T_train = y_T[:n_T_train]
    
    X_T_test = X_T[n_T_train:]
    y_T_test = y_T[n_T_train:]
    
    print(f"\n[Data Summary]")
    print(f"Source Train (Normal Weather): {X_S.shape[0]}")
    print(f"Target Train (Extreme Weather): {X_T_train.shape[0]}")
    print(f"Target Test (Extreme Weather): {X_T_test.shape[0]}")
    print(f"Feature Dimension: {X_S.shape[1]}")
    
    # ===============================
    # 板块 2：模型训练
    # 依赖 Train_TL_QLDMR.TL_QLDMR 完成：
    # - solver='batch_sgd'：基于核切片的 Mini-batch SGD
    # - solver='fast_nystrom'：Nyström 近似 + Primal Adam（集成自 t 并在 Train_TL_QLDMR 中实现）
    # 模型参数可根据风场与任务调优（lambda1/lambda2/tau/gamma 等）
    # ===============================
    print("\n>>> Step 2: Training TL-QLDMR...")
    
    if solver == "batch_sgd":
        batch_size = 16 if farm_idx == 0 else 8
        print(f"Using Batch Size: {batch_size} (Solver: batch_sgd)")
        model = TL_QLDMR(
            lambda1=0.01,
            lambda2=0.01,
            C_S=1.0,
            C_T=10.0,
            tau=0.95,
            kernel_gamma=0.02,
            solver="batch_sgd",
            torch_lr=1e-3,
            torch_max_iter=50000,
        )
    elif solver == "fast_nystrom":
        batch_size = 256
        print(f"Using Batch Size: {batch_size} (Solver: fast_nystrom)")
        model = TL_QLDMR(
            lambda1=0.01,
            lambda2=0.01,
            C_S=1.0,
            C_T=10.0,
            tau=0.95,
            kernel_gamma=0.02,
            solver="fast_nystrom",
            nystrom_n_components=200,
            nystrom_lr=0.01,
            nystrom_epochs=50,
            nystrom_batch_size=batch_size,
        )
    else:
        raise ValueError(f"Unsupported solver: {solver}")
    
    # 联合训练源域与目标域样本
    if solver == "batch_sgd":
        success = model.fit(X_S, y_S, X_T_train, y_T_train, batch_size=batch_size)
    else:
        success = model.fit(X_S, y_S, X_T_train, y_T_train)
    
    if not success:
        print("Training Failed!")
        return None
    
    # ===============================
    # 板块 3：预测与评估
    # - 使用训练好的模型进行测试集预测
    # - 计算 RMSE/MAE/覆盖率（95%分位的上界）
    # - 注意：TL-QLDMR 是分位回归，上界与中位需分别训练
    # ===============================
    print("\n>>> Step 3: Prediction & Evaluation...")
    predictor = Predictor(model)
    
    # 95% 上界预测（点置信上限）
    y_pred_upper = predictor.predict(X_T_test)
    
    # 50% 中位预测（点估计，用于 RMSE/MAE 评估）
    print("Training Median Model (tau=0.5)...")
    if solver == "batch_sgd":
        model_median = TL_QLDMR(
            lambda1=0.01,
            lambda2=0.01,
            C_S=1.0,
            C_T=10.0,
            tau=0.5,
            kernel_gamma=0.02,
            solver="batch_sgd",
            torch_lr=1e-3,
            torch_max_iter=50000,
        )
        model_median.fit(X_S, y_S, X_T_train, y_T_train, batch_size=batch_size)
    else:
        model_median = TL_QLDMR(
            lambda1=0.01,
            lambda2=0.01,
            C_S=1.0,
            C_T=10.0,
            tau=0.5,
            kernel_gamma=0.02,
            solver="fast_nystrom",
            nystrom_n_components=200,
            nystrom_lr=0.01,
            nystrom_epochs=50,
            nystrom_batch_size=batch_size,
        )
        model_median.fit(X_S, y_S, X_T_train, y_T_train)
    predictor_median = Predictor(model_median)
    y_pred_mean = predictor_median.predict(X_T_test)
    
    # 反归一化到原始功率单位（MW）
    y_true_orig = gen.scaler_y.inverse_transform(y_T_test.reshape(-1, 1)).flatten()
    y_pred_mean_orig = gen.scaler_y.inverse_transform(y_pred_mean.reshape(-1, 1)).flatten()
    y_pred_upper_orig = gen.scaler_y.inverse_transform(y_pred_upper.reshape(-1, 1)).flatten()
    
    # 负值裁剪（功率不可为负）
    y_pred_mean_orig = np.maximum(y_pred_mean_orig, 0)
    y_pred_upper_orig = np.maximum(y_pred_upper_orig, 0)
    
    # 计算评价指标
    rmse = np.sqrt(mean_squared_error(y_true_orig, y_pred_mean_orig))
    mae = mean_absolute_error(y_true_orig, y_pred_mean_orig)
    
    # 覆盖率（PICP）：真实值落在 95% 上界之下的比例
    coverage = np.mean(y_true_orig <= y_pred_upper_orig) * 100
    
    # 归一化 RMSE（相对装机容量的百分比）
    nrmse = rmse / gen.nominal_capacity * 100
    
    print(f"\n{'='*60}")
    print(f"[Results on Wind Farm Data (Extreme Weather)]")
    print(f"{'='*60}")
    print(f"RMSE: {rmse:.4f} MW ({nrmse:.2f}% of capacity)")
    print(f"MAE: {mae:.4f} MW")
    print(f"95% Quantile Coverage (PICP): {coverage:.2f}%")
    print(f"{'='*60}")
    
    # ===============================
    # 板块 4：可视化
    # - 绘制前 200 个时间点的真实值、中位预测与 95% 上界
    
    # 仅绘制前 200 个点，避免过度拥挤
    plot_len = min(200, len(y_true_orig))
    
    plt.figure(figsize=(12, 6))
    
    plt.plot(y_true_orig[:plot_len], label='True Power', color='black', linewidth=1.5)
    plt.plot(y_pred_mean_orig[:plot_len], label='TL-QLDMR (Median)', color='blue', linestyle='--')
    plt.plot(y_pred_upper_orig[:plot_len], label='TL-QLDMR (95% Upper)', color='red', linestyle=':', alpha=0.8)
    
    plt.title(f"Wind Farm Power Prediction (Extreme Weather) - {gen.nominal_capacity}MW Farm")
    plt.xlabel("Time Steps (15 min intervals)")
    plt.ylabel("Power (MW)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 获取脚本所在目录 (TL-QLDMR) 并构建结果路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    result_dir = os.path.join(script_dir, 'results')
    
    # 确保 results 目录存在
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    result_path = os.path.join(result_dir, f'wind_farm_{farm_idx}_result.png')
    plt.savefig(result_path, dpi=150)
    print(f"\nResult plot saved to {result_path}")
    
    return {
        'rmse': rmse,
        'mae': mae,
        'coverage': coverage,
        'nrmse': nrmse,
        'capacity': gen.nominal_capacity
    }


def run_all_farms():
    """Run experiments on all available wind farms"""
    gen = WindFarmDataGenerator()
    farms = gen.get_available_farms()
    
    print("="*70)
    print("Running TL-QLDMR Experiments on All Wind Farms")
    print("="*70)
    
    results = []
    for i, farm in enumerate(farms):
        print(f"\n{'#'*70}")
        print(f"# Farm {i}: {farm['name']} ({farm['capacity_MW']}MW)")
        print(f"{'#'*70}\n")
        
        result = run_experiment(farm_idx=i, feature_set='full', window_size=12)
        if result:
            result['farm_name'] = farm['name']
            results.append(result)
    
    # Summary table
    print("\n" + "="*70)
    print("SUMMARY: All Wind Farms Results")
    print("="*70)
    print(f"{'Farm':<50} {'RMSE(MW)':<12} {'NRMSE(%)':<10} {'PICP(%)':<10}")
    print("-"*70)
    for r in results:
        print(f"{r['farm_name']:<50} {r['rmse']:<12.4f} {r['nrmse']:<10.2f} {r['coverage']:<10.2f}")
    
    return results


if __name__ == "__main__":
    # Run experiment on the largest farm (200MW)
    run_experiment(farm_idx=0, feature_set='full', window_size=6, solver="fast_nystrom")
