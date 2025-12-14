import numpy as np
import matplotlib.pyplot as plt
from Data_Utils import WindDataGenerator
from Train_TL_QLDMR import TL_QLDMR
from Predict_TL_QLDMR import Predictor
from sklearn.metrics import mean_squared_error, mean_absolute_error

def main():
    # ===============================
    # 1. 数据生成与处理
    # ===============================
    print(">>> Step 1: Generating Data...")
    # 生成 3000 个样本，15% 是极端天气
    gen = WindDataGenerator(n_samples=2000, extreme_ratio=0.1)
    X_raw, y_raw, is_extreme = gen.generate_raw_data()
    
    # 构建滑动窗口 (输入展平为长向量)
    # Window=12, 对应过去2小时预测未来
    WINDOW_SIZE = 12
    X, y, labels = gen.create_sliding_window(X_raw, y_raw, is_extreme, window_size=WINDOW_SIZE)
    
    # 划分源域和目标域
    X_S, y_S, X_T, y_T = gen.split_domain_data(X, y, labels)
    
    # 目标域切分：30% 用于迁移训练 (Target Train), 70% 用于测试 (Target Test)
    n_T_train = int(len(X_T) * 0.7)
    X_T_train = X_T[:n_T_train]
    y_T_train = y_T[:n_T_train]
    
    X_T_test = X_T[n_T_train:]
    y_T_test = y_T[n_T_train:]
    
    print(f"Source Train: {X_S.shape[0]}")
    print(f"Target Train: {X_T_train.shape[0]} (Used for Transfer)")
    print(f"Target Test : {X_T_test.shape[0]} (Used for Evaluation)")
    
    # ===============================
    # 2. 模型训练
    # ===============================
    print("\n>>> Step 2: Training TL-QLDMR...")
    
    # 设置参数
    # C_T 设为 C_S 的 10 倍，以平衡样本数量差异 (解决数据淹没问题)
    model = TL_QLDMR(
        lambda1=0.01,   # 方差权重
        lambda2=0.01,   # MMD 权重
        C_S=1.0,       # 源域惩罚
        C_T=100.0,      # 目标域惩罚 (重权)
        tau=0.95,      # 预测 95% 分位点 (上界)
        kernel_gamma=0.1
    )
    
    # 联合训练：源域 + 目标域训练集
    success = model.fit(X_S, y_S, X_T_train, y_T_train)
    
    if not success:
        return

    # ===============================
    # 3. 预测与评估
    # ===============================
    print("\n>>> Step 3: Prediction & Evaluation...")
    predictor = Predictor(model)
    
    # 对极端天气测试集进行预测
    y_pred_upper = predictor.predict(X_T_test)
    
    # 为了对比，我们可以再训练一个 tau=0.5 的模型作为点预测
    print("Training Median Model (tau=0.5)...")
    model_median = TL_QLDMR(lambda1=0.01, lambda2=0.01, C_S=1.0, C_T=10.0, tau=0.5, kernel_gamma=0.01)
    model_median.fit(X_S, y_S, X_T_train, y_T_train)
    predictor_median = Predictor(model_median)
    y_pred_mean = predictor_median.predict(X_T_test)
    
    # 反归一化 (还原到真实功率)
    y_true_orig = gen.scaler_y.inverse_transform(y_T_test.reshape(-1, 1)).flatten()
    y_pred_mean_orig = gen.scaler_y.inverse_transform(y_pred_mean.reshape(-1, 1)).flatten()
    y_pred_upper_orig = gen.scaler_y.inverse_transform(y_pred_upper.reshape(-1, 1)).flatten()
    
    # 计算指标
    rmse = np.sqrt(mean_squared_error(y_true_orig, y_pred_mean_orig))
    mae = mean_absolute_error(y_true_orig, y_pred_mean_orig)
    
    # 计算覆盖率 (PICP) - 仅针对上界
    coverage = np.mean(y_true_orig <= y_pred_upper_orig) * 100
    
    print(f"\n[Result on Target Domain (Extreme Weather)]")
    print(f"RMSE: {rmse:.2f} kW")
    print(f"MAE : {mae:.2f} kW")
    print(f"95% Quantile Coverage (PICP): {coverage:.2f}%")
    
    # ===============================
    # 4. 可视化
    # ===============================
    plt.figure(figsize=(12, 6))
    plt.plot(y_true_orig, label='True Power (Extreme)', color='black', linewidth=1.5)
    plt.plot(y_pred_mean_orig, label='TL-QLDMR (Mean)', color='blue', linestyle='--')
    plt.plot(y_pred_upper_orig, label='TL-QLDMR (95% Upper)', color='red', linestyle=':', alpha=0.8)
    
    plt.title("TL-QLDMR Prediction on Extreme Weather Events")
    plt.xlabel("Time Steps (Test Set)")
    plt.ylabel("Power (kW)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

if __name__ == "__main__":
    main()