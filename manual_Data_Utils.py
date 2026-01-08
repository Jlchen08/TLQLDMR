import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

class WindDataGenerator:
    """
    模拟风电功率数据生成器，包含常规天气和极端天气（爬坡/大风切出）
    """
    def __init__(self, n_samples=2000, extreme_ratio=0.1):
        self.n_samples = n_samples
        self.extreme_ratio = extreme_ratio
    
    def generate_raw_data(self):
        """
        生成原始时间序列：风速、温度、功率
        遵循物理公式 P ~ rho * v^3，并加入噪声和极端事件
        """
        np.random.seed(42)
        time_steps = np.arange(self.n_samples)
        
        # 1. 基础风速 (Weibull分布模拟 + 正弦波动)
        wind_speed = np.random.weibull(2, self.n_samples) * 5 + 5 * np.sin(time_steps / 50)
        
        # 2. 温度 (随时间缓慢变化)
        temperature = 20 + 5 * np.sin(time_steps / 200) + np.random.normal(0, 1, self.n_samples)
        
        # 3. 基础功率 (P = 0.5 * rho * v^3 * Cp)
        # 简化物理模型，加入随机噪声
        power = 0.5 * (wind_speed ** 3) * (1 - temperature/100) + np.random.normal(0, 50, self.n_samples)
        power = np.clip(power, 0, 2000) # 额定功率限制
        
        # 4. 注入极端事件 (Extreme Events)
        # 场景A: 大风切出 (High Wind Cut-out) -> 风速极大，功率突降为0
        # 场景B: 爬坡事件 (Ramp Event) -> 功率剧烈波动
        is_extreme = np.zeros(self.n_samples, dtype=bool)
        
        n_extreme_blocks = int(self.n_samples * self.extreme_ratio / 20) # 每个事件持续约20个点
        
        for _ in range(n_extreme_blocks):
            start = np.random.randint(100, self.n_samples - 100)
            end = start + 20
            is_extreme[start:end] = True
            
            event_type = np.random.choice(['cut_out', 'ramp'])
            
            if event_type == 'cut_out':
                wind_speed[start:end] += 15 # 制造极高风速
                power[start:end] = 0        # 触发停机保护
            else:
                wind_speed[start:end] += np.linspace(0, 10, 20) # 快速上升
                power[start:end] += np.linspace(0, 1000, 20)    # 功率剧烈爬坡

        # 归一化处理 (Feature Scaling)
        # 注意：在真实场景中，应该用源域的统计量来归一化目标域
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        
        X_raw = np.stack([wind_speed, temperature], axis=1)
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
            
            # 2. 展平 (Flatten) 成超长向量：这是 SVM/LDMR 的标准输入格式
            # 形状变化: (window_size, n_features) -> (window_size * n_features, )
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