import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from real_Data_Utils import RealWindDataGenerator
from Train_TL_QLDMR import TL_QLDMR
from Predict_TL_QLDMR import Predictor
from sklearn.metrics import mean_squared_error, mean_absolute_error

def main():
    # ===============================
    # 1. 数据生成与处理
    # ===============================
    print(">>> Step 1: Loading Real Data...")
    
    # 使用真实数据生成器
    project_dir = Path(__file__).resolve().parent
    data_path = project_dir / "data" / "data1.xlsx"
    gen = RealWindDataGenerator(file_path=str(data_path))
    X_scaled, y_scaled, is_extreme = gen.load_and_process_data()
    
    # 构建滑动窗口
    WINDOW_SIZE = 12 # 增加窗口大小以包含更多历史信息
    X, y, labels = gen.create_sliding_window(X_scaled, y_scaled, is_extreme, window_size=WINDOW_SIZE)
    
    # 划分源域和目标域
    X_S, y_S, X_T, y_T = gen.split_domain_data(X, y, labels)
    
    # 目标域切分：使用 60% 进行训练，40% 测试
    n_T_train = int(len(X_T) * 0.6) 
    if n_T_train < 10:
        print("Warning: Target samples too few, increasing ratio to 0.5")
        n_T_train = int(len(X_T) * 0.5)
        
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
    # 调整说明：
    # 1. lambda 恢复正常 (1000 -> 0.1): 之前的过大正则化导致模型欠拟合(太直)
    # 2. kernel_gamma 适中 (0.02): 保持一定拟合能力的同时平滑
    # 3. C_T 保持 20.0
    model = TL_QLDMR(
        lambda1=0.1,   
        lambda2=0.1,   
        C_S=1.0,       
        C_T=20.0,      
        tau=0.95,      
        kernel_gamma=0.02 
    )
    
    # 联合训练
    success = model.fit(X_S, y_S, X_T_train, y_T_train)
    
    if not success:
        print("Training Failed!")
        return

    # ===============================
    # 3. 预测与评估
    # ===============================
    print("\n>>> Step 3: Prediction & Evaluation...")
    predictor = Predictor(model)
    
    # 1. 上界预测 (95%)
    y_pred_upper = predictor.predict(X_T_test)
    
    # 2. 中位数预测 (50%) - 用于点预测误差评估
    print("Training Median Model (tau=0.5)...")
    model_median = TL_QLDMR(lambda1=0.1, lambda2=0.1, C_S=1.0, C_T=20.0, tau=0.5, kernel_gamma=0.02)
    model_median.fit(X_S, y_S, X_T_train, y_T_train)
    predictor_median = Predictor(model_median)
    y_pred_mean = predictor_median.predict(X_T_test)
    
    # 反归一化
    y_true_orig = gen.scaler_y.inverse_transform(y_T_test.reshape(-1, 1)).flatten()
    y_pred_mean_orig = gen.scaler_y.inverse_transform(y_pred_mean.reshape(-1, 1)).flatten()
    y_pred_upper_orig = gen.scaler_y.inverse_transform(y_pred_upper.reshape(-1, 1)).flatten()
    
    # 截断负值 (功率不能为负)
    y_pred_mean_orig = np.maximum(y_pred_mean_orig, 0)
    y_pred_upper_orig = np.maximum(y_pred_upper_orig, 0)
    
    # 计算指标
    rmse = np.sqrt(mean_squared_error(y_true_orig, y_pred_mean_orig))
    mae = mean_absolute_error(y_true_orig, y_pred_mean_orig)
    
    # 计算覆盖率 (PICP)
    coverage = np.mean(y_true_orig <= y_pred_upper_orig) * 100
    
    print(f"\n[Result on Real Data (Extreme Weather)]")
    print(f"RMSE: {rmse:.2f} kW")
    print(f"MAE : {mae:.2f} kW")
    print(f"95% Quantile Coverage (PICP): {coverage:.2f}%")
    
    # ===============================
    # 4. 可视化
    # ===============================
    # 只画前200个点，避免太密集
    plot_len = min(200, len(y_true_orig))
    
    plt.figure(figsize=(12, 6))
    plt.plot(y_true_orig[:plot_len], label='True Power (Extreme)', color='black', linewidth=1.5)
    plt.plot(y_pred_mean_orig[:plot_len], label='TL-QLDMR (Median)', color='blue', linestyle='--')
    plt.plot(y_pred_upper_orig[:plot_len], label='TL-QLDMR (95% Upper)', color='red', linestyle=':', alpha=0.8)
    
    plt.title("Real Data: TL-QLDMR Prediction on Extreme Weather")
    plt.xlabel("Time Steps")
    plt.ylabel("Power (kW)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 保存图片而不是显示
    results_dir = project_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / "real_data_result.png"
    plt.savefig(str(out_path))
    print(f"Result plot saved to {out_path}")

if __name__ == "__main__":
    main()
