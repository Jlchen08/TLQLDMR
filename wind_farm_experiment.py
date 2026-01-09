"""
Wind Farm Power Prediction using TL-QLDMR
Based on the Chinese State Grid Renewable Energy Generation dataset (s41597-022-01696-6)
"""

import numpy as np
import matplotlib.pyplot as plt
from wind_farm_Data_Utils import WindFarmDataGenerator
from Train_TL_QLDMR import TL_QLDMR
from Predict_TL_QLDMR import Predictor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import os


def run_experiment(farm_idx=0, feature_set='full', window_size=12):
    """
    Run TL-QLDMR experiment on wind farm data
    
    Args:
        farm_idx: Index of wind farm to use (0-5, sorted by capacity)
        feature_set: 'full', 'wind_only', or 'simple'
        window_size: Sliding window size (default 12 = 3 hours)
    """
    # ===============================
    # 1. Data Loading & Processing
    # ===============================
    print(">>> Step 1: Loading Wind Farm Data...")
    
    gen = WindFarmDataGenerator()
    X_scaled, y_scaled, is_extreme = gen.load_and_process_data(farm_idx=farm_idx, feature_set=feature_set)
    
    # Build sliding window samples
    X, y, labels = gen.create_sliding_window(X_scaled, y_scaled, is_extreme, window_size=window_size)
    
    # Split into source (normal) and target (extreme) domains
    X_S, y_S, X_T, y_T = gen.split_domain_data(X, y, labels)
    
    # Target domain split: 60% training, 40% testing
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
    # 2. Model Training
    # ===============================
    print("\n>>> Step 2: Training TL-QLDMR...")
    
    # Model parameters
    model = TL_QLDMR(
        lambda1=0.1,       # L2 regularization
        lambda2=0.1,       # Domain adaptation
        C_S=1.0,           # Source domain weight
        C_T=20.0,          # Target domain weight (higher for extreme weather)
        tau=0.95,          # Quantile level (95% upper bound)
        kernel_gamma=0.02  # RBF kernel width
    )
    
    # Joint training on source and target domains
    success = model.fit(X_S, y_S, X_T_train, y_T_train)
    
    if not success:
        print("Training Failed!")
        return None
    
    # ===============================
    # 3. Prediction & Evaluation
    # ===============================
    print("\n>>> Step 3: Prediction & Evaluation...")
    predictor = Predictor(model)
    
    # Upper bound prediction (95%)
    y_pred_upper = predictor.predict(X_T_test)
    
    # Median prediction (50%) - for point prediction evaluation
    print("Training Median Model (tau=0.5)...")
    model_median = TL_QLDMR(
        lambda1=0.1, lambda2=0.1, C_S=1.0, C_T=20.0, 
        tau=0.5, kernel_gamma=0.02
    )
    model_median.fit(X_S, y_S, X_T_train, y_T_train)
    predictor_median = Predictor(model_median)
    y_pred_mean = predictor_median.predict(X_T_test)
    
    # Inverse transform to original scale
    y_true_orig = gen.scaler_y.inverse_transform(y_T_test.reshape(-1, 1)).flatten()
    y_pred_mean_orig = gen.scaler_y.inverse_transform(y_pred_mean.reshape(-1, 1)).flatten()
    y_pred_upper_orig = gen.scaler_y.inverse_transform(y_pred_upper.reshape(-1, 1)).flatten()
    
    # Clip negative values (power cannot be negative)
    y_pred_mean_orig = np.maximum(y_pred_mean_orig, 0)
    y_pred_upper_orig = np.maximum(y_pred_upper_orig, 0)
    
    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(y_true_orig, y_pred_mean_orig))
    mae = mean_absolute_error(y_true_orig, y_pred_mean_orig)
    
    # Coverage rate (PICP)
    coverage = np.mean(y_true_orig <= y_pred_upper_orig) * 100
    
    # Normalized RMSE (as percentage of capacity)
    nrmse = rmse / gen.nominal_capacity * 100
    
    print(f"\n{'='*60}")
    print(f"[Results on Wind Farm Data (Extreme Weather)]")
    print(f"{'='*60}")
    print(f"RMSE: {rmse:.4f} MW ({nrmse:.2f}% of capacity)")
    print(f"MAE: {mae:.4f} MW")
    print(f"95% Quantile Coverage (PICP): {coverage:.2f}%")
    print(f"{'='*60}")
    
    # ===============================
    # 4. Visualization
    # ===============================
    os.makedirs('results', exist_ok=True)
    
    # Plot first 300 points to avoid clutter
    plot_len = min(300, len(y_true_orig))
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # Plot 1: Time series comparison
    ax1 = axes[0]
    ax1.plot(y_true_orig[:plot_len], label='True Power', color='black', linewidth=1.5)
    ax1.plot(y_pred_mean_orig[:plot_len], label='TL-QLDMR (Median)', color='blue', linestyle='--', alpha=0.8)
    ax1.plot(y_pred_upper_orig[:plot_len], label='TL-QLDMR (95% Upper)', color='red', linestyle=':', alpha=0.7)
    ax1.fill_between(range(plot_len), y_pred_mean_orig[:plot_len], y_pred_upper_orig[:plot_len], 
                     alpha=0.2, color='red', label='Uncertainty Band')
    
    ax1.set_title(f"Wind Farm Power Prediction (Extreme Weather) - {gen.nominal_capacity}MW Farm")
    ax1.set_xlabel("Time Steps (15 min intervals)")
    ax1.set_ylabel("Power (MW)")
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Scatter plot
    ax2 = axes[1]
    ax2.scatter(y_true_orig, y_pred_mean_orig, alpha=0.4, s=10, label='Median Prediction')
    ax2.plot([0, max(y_true_orig)], [0, max(y_true_orig)], 'r--', label='Perfect Prediction')
    ax2.set_xlabel("True Power (MW)")
    ax2.set_ylabel("Predicted Power (MW)")
    ax2.set_title(f"Scatter Plot (RMSE: {rmse:.4f} MW)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    result_path = f'results/wind_farm_{farm_idx}_result.png'
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
    run_experiment(farm_idx=0, feature_set='full', window_size=12)
