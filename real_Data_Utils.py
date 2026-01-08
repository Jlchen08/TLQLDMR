import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

class RealWindDataGenerator:
    """
    真实风电功率数据处理工具
    用于处理 data/data1.xlsx 数据，包含自动极端天气识别
    """
    def __init__(self, file_path='/Users/lin/project/WP/TL-QLDMR/data/data1.xlsx'):
        self.file_path = file_path
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        
    def load_and_process_data(self):
        """
        加载数据并进行预处理
        Returns:
            X_scaled:归一化后的特征 (Wind, Temperature)
            y_scaled: 归一化后的功率
            is_extreme: 极端天气标签 (Boolean Array)
        """
        print(f"Loading data from {self.file_path}...")
        df = pd.read_excel(self.file_path)
        
        # 1. 提取特征和目标
        # 使用 '测风塔70米风速' 和 '温度' (如果有温度列) 作为特征
        # 注意：原代码 manual_data 用的是 temperature, data1.xlsx 有 '温度' 列
        wind_speed = df['测风塔70米风速'].values
        temperature = df['温度'].values
        power = df['实际发电功率'].values
        
        # 2. 极端天气自动识别 (基于物理和统计规则)
        print("Identifying extreme weather events...")
        is_extreme = np.zeros(len(df), dtype=bool)
        
        # 规则 A: 统计阈值 (95% 分位数)
        # 极端大风 or 极端大功率
        q95_wind = np.percentile(wind_speed, 95)
        q95_power = np.percentile(power, 95)
        
        mask_stat = (wind_speed > q95_wind) | (power > q95_power)
        
        # 规则 B: 物理规则 (功率爬坡 Ramp Event)
        # 定义：|P(t) - P(t-1)| > Threshold
        # 根据之前分析，Top 5% 变化量约为 17.6kW，这里取 20kW 作为阈值
        ramp_threshold = 20.0
        power_diff = np.abs(np.diff(power, prepend=power[0]))
        mask_ramp = power_diff > ramp_threshold
        
        # 综合标签
        is_extreme = mask_stat | mask_ramp
        
        print(f"Total Samples: {len(df)}")
        print(f"Extreme Samples: {np.sum(is_extreme)} ({np.mean(is_extreme)*100:.2f}%)")
        print(f"  - Statistical Rule (>95%): {np.sum(mask_stat)}")
        print(f"  - Ramp Rule (>20kW): {np.sum(mask_ramp)}")
        
        # 3. 归一化 (Feature Scaling)
        X_raw = np.stack([wind_speed, temperature], axis=1)
        
        # 使用全量数据进行 fit (或者也可以只用 Source domain fit, 这里为了简单用全量)
        X_scaled = self.scaler_X.fit_transform(X_raw)
        y_scaled = self.scaler_y.fit_transform(power.reshape(-1, 1)).flatten()
        
        return X_scaled, y_scaled, is_extreme

    def create_sliding_window(self, X, y, is_extreme, window_size=12):
        """
        构建滑动窗口样本
        Input: (N, Features)
        Output: (N-window, window*Features) -> Flattened Vector
        """
        X_windows, y_targets, labels = [], [], []
        
        for i in range(len(X) - window_size):
            # 1. 截取窗口内的数据作为特征矩阵
            window_feature = X[i : i+window_size, :]
            
            # 2. 展平
            X_flat = window_feature.flatten()
            
            # 3. 目标值是窗口后的下一个点
            target = y[i + window_size]
            
            # 4. 标签归属判定：如果预测目标时刻是极端天气，则该样本属于目标域
            tag = is_extreme[i + window_size]
            
            X_windows.append(X_flat)
            y_targets.append(target)
            labels.append(tag)
            
        return np.array(X_windows), np.array(y_targets), np.array(labels)

    def split_domain_data(self, X, y, labels):
        """
        根据标签划分为源域和目标域
        """
        # 源域：常规天气 (Tag=False)
        X_S = X[~labels]
        y_S = y[~labels]
        
        # 目标域：极端天气 (Tag=True)
        X_T = X[labels]
        y_T = y[labels]
        
        print(f"[Data Info] Source Samples: {len(X_S)}, Target Samples: {len(X_T)}")
        return X_S, y_S, X_T, y_T
