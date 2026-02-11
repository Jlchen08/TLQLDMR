"""
风电场数据工具（供 TL-QLDMR 使用）
数据来源：国家电网可再生能源发电数据集（s41597-022-01696-6）

参考与用途说明：
- 本文件提供 WindFarmDataGenerator，用于把原始/处理后的风电 SCADA 数据转换为可训练样本
- wind_farm_experiment.py 参考本工具完成整套实验流程（数据 -> 划分 -> 训练 -> 评估/画图）

模块板块说明：
- 数据加载：按风场文件名解析装机容量，读取 Excel
- 特征处理：动态匹配列名，按 feature_set 选择特征并做标准化
- 极端样本识别：用统计/物理规则标记极端天气（用于目标域）
- 滑窗构造：把时间序列拼成监督学习样本（窗口特征 + 下一时刻标签）
- 领域划分：按极端标签拆成源域（正常）与目标域（极端）
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from pathlib import Path


class WindFarmDataGenerator:
    """
    风电场数据处理工具
    处理内容包括：
    - 多高度风速（10m/30m/50m/轮毂高度）
    - 多高度风向（不同数据集可能缺失，代码会动态匹配）
    - 气象数据（温度、气压、湿度等）
    - 功率输出（MW）
    """
    
    def __init__(self, data_dir=None):
        """
        初始化数据生成器
        
        参数：
            data_dir: 风电场数据目录路径（默认指向处理后的 wind_farms）
        """
        if data_dir is None:
            # Default to repo-local data directory to avoid hardcoded project root
            base_dir = Path(__file__).resolve().parent
            data_dir = base_dir / 'data' / 'wind_farm_data' / 'data_processed' / 'wind_farms'
        self.data_dir = Path(data_dir)
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self.nominal_capacity = None  # MW
        
        # 列名映射（用于不同文件的列名对齐；实际处理中也支持动态匹配）
        self.col_map = {
            'time': 'Time(year-month-day h:m:s)',
            'wind_10m': 'Wind speed at height of 10 meters (m/s)',
            'wind_dir_10m': 'Wind direction at height of 10 meters (˚)',
            'wind_30m': 'Wind speed at height of 30 meters (m/s)',
            'wind_dir_30m': 'Wind direction at height of 30 meters (˚)',
            'wind_50m': 'Wind speed at height of 50 meters (m/s)',
            'wind_dir_50m': 'Wind direction at height of 50 meters (˚)',
            'wind_hub': 'Wind speed - at the height of wheel hub (m/s)',
            'wind_dir_hub': 'Wind speed - at the height of wheel hub (˚)',
            'temperature': 'Air temperature  (°C) ',
            'pressure': 'Atmosphere (hpa)',
            'humidity': 'Relative humidity (%)',
            'power': 'Power (MW)'
        }
        
    def get_available_farms(self):
        """获取可用风电场文件列表（按装机容量降序）"""
        farms = list(self.data_dir.glob('*.xlsx'))
        farm_info = []
        for f in farms:
            # 从文件名中提取装机容量
            name = f.stem
            capacity = int(name.split('-')[1].replace('MW)', ''))
            farm_info.append({
                'path': f,
                'name': name,
                'capacity_MW': capacity
            })
        return sorted(farm_info, key=lambda x: x['capacity_MW'], reverse=True)
    
    def load_farm_data(self, farm_idx=0, use_processed=True):
        """
        加载指定风电场的数据
        
        参数：
            farm_idx: 风场索引（按装机容量降序）
            use_processed: 是否使用处理后数据（当前实现以 data_dir 为准）
            
        返回：
            df: 风电场数据 DataFrame
        """
        farms = self.get_available_farms()
        if farm_idx >= len(farms):
            raise ValueError(f"Farm index {farm_idx} out of range. Available: 0-{len(farms)-1}")
        
        farm = farms[farm_idx]
        self.nominal_capacity = farm['capacity_MW']
        
        print(f"Loading wind farm: {farm['name']}")
        print(f"Nominal Capacity: {self.nominal_capacity} MW")
        
        df = pd.read_excel(farm['path'])
        print(f"Samples: {len(df)}")
        
        return df
    
    def load_and_process_data(self, farm_idx=0, feature_set='full', extreme_cfg: dict | None = None):
        """
        加载数据并完成预处理
        
        参数：
            farm_idx: 风场索引
            feature_set: 'full'（全部特征）、'wind_only'（仅风速）、'simple'（机舱风速+温度）
                        
        返回：
            X_scaled: 标准化后的特征
            y_scaled: 标准化后的功率
            is_extreme: 极端天气标签（bool 数组）
        """
        df = self.load_farm_data(farm_idx)
        
        # 按 feature_set 抽取特征
        # 不同风场文件的列名可能略有差异，这里做动态匹配
        available_cols = df.columns.tolist()
        
        def find_column(pattern_keywords):
            """在列名中查找同时包含所有关键词的列"""
            for col in available_cols:
                if all(kw.lower() in col.lower() for kw in pattern_keywords):
                    return col
            return None
        
        # 动态匹配各特征列名
        col_wind_hub = find_column(['wind speed', 'wheel hub', 'm/s'])
        col_wind_50m = find_column(['wind speed', '50 meters', 'm/s'])
        col_wind_30m = find_column(['wind speed', '30 meters', 'm/s'])
        col_wind_10m = find_column(['wind speed', '10 meters', 'm/s'])
        col_wind_dir_hub = find_column(['wind speed', 'wheel hub', '˚'])
        col_wind_dir_50m = find_column(['wind direction', '50 meters'])
        col_wind_dir_30m = find_column(['wind direction', '30 meters'])
        col_wind_dir_10m = find_column(['wind direction', '10 meters'])
        col_temp = find_column(['air temperature', '°C'])
        col_pressure = find_column(['atmosphere', 'hpa'])
        col_humidity = find_column(['relative humidity', '%'])
        col_power = find_column(['power', 'MW'])
        
        print(f"Detected columns:")
        print(f"  Wind hub: {col_wind_hub}")
        print(f"  Temperature: {col_temp}")
        print(f"  Power: {col_power}")
        
        if feature_set == 'full':
            # 使用所有可用特征
            feature_cols = [col for col in [
                col_wind_hub, col_wind_50m, col_wind_30m, col_wind_10m,
                col_temp, col_pressure, col_humidity
            ] if col is not None]
        elif feature_set == 'full_dir':
            feature_cols = [col for col in [
                col_wind_hub, col_wind_50m, col_wind_30m, col_wind_10m,
                col_temp, col_pressure, col_humidity,
                col_wind_dir_hub, col_wind_dir_50m, col_wind_dir_30m, col_wind_dir_10m,
            ] if col is not None]
        elif feature_set == 'full_dir_cyclic':
            base_cols = [col for col in [
                col_wind_hub, col_wind_50m, col_wind_30m, col_wind_10m,
                col_temp, col_pressure, col_humidity
            ] if col is not None]
            dir_cols = [col for col in [
                col_wind_dir_hub, col_wind_dir_50m, col_wind_dir_30m, col_wind_dir_10m,
            ] if col is not None]

            features = []
            feature_cols = []
            for col in base_cols:
                features.append(df[col].values)
                feature_cols.append(col)
            for col in dir_cols:
                deg = df[col].values.astype(float)
                rad = np.deg2rad(deg)
                features.append(np.sin(rad))
                feature_cols.append(f"sin({col})")
                features.append(np.cos(rad))
                feature_cols.append(f"cos({col})")

            X_raw = np.column_stack(features) if features else np.empty((len(df), 0))
        elif feature_set == 'wind_only':
            feature_cols = [col for col in [
                col_wind_hub, col_wind_50m, col_wind_30m, col_wind_10m
            ] if col is not None]
        else:  # 'simple'
            feature_cols = [col for col in [col_wind_hub, col_temp] if col is not None]
        
        print(f"Using {len(feature_cols)} features: {feature_cols}")
        
        if feature_set != 'full_dir_cyclic':
            X_raw = df[feature_cols].values
        power = df[col_power].values
        
        # 识别极端天气事件（用于领域划分）
        print("\nIdentifying extreme weather events...")
        is_extreme = self._identify_extreme_weather(
            df,
            power,
            col_wind_hub,
            col_temp,
            extreme_cfg=extreme_cfg,
        )

        
        # 标准化
        X_scaled = self.scaler_X.fit_transform(X_raw)
        y_scaled = self.scaler_y.fit_transform(power.reshape(-1, 1)).flatten()
        
        return X_scaled, y_scaled, is_extreme
    
    def _identify_extreme_weather(self, df, power, col_wind_hub, col_temp, extreme_cfg: dict | None = None):
        """
        使用多条规则识别极端天气样本
        
        规则：
        A. 统计规则：风速或功率超过 95% 分位
        B. 物理规则：功率爬坡事件（|P(t)-P(t-1)| 超过阈值）
        C. 切出规则：风速超过 25 m/s（机组可能停机）
        D. 温度规则：温度过低或过高（< -10°C 或 > 35°C）
        """
        extreme_cfg = extreme_cfg or {}
        is_extreme = np.zeros(len(df), dtype=bool)
        
        wind_hub = df[col_wind_hub].values if col_wind_hub else np.zeros(len(df))
        temperature = df[col_temp].values if col_temp else np.zeros(len(df))
        
        # 规则 A：统计阈值（分位数可配置，默认 95%）
        stat_q = float(extreme_cfg.get("stat_q", 95))
        stat_on_wind = bool(extreme_cfg.get("stat_on_wind", True))
        stat_on_power = bool(extreme_cfg.get("stat_on_power", True))
        q_wind = np.percentile(wind_hub, stat_q) if col_wind_hub else 0
        q_power = np.percentile(power, stat_q)
        mask_stat = np.zeros(len(df), dtype=bool)
        if stat_on_wind and col_wind_hub:
            mask_stat |= wind_hub > q_wind
        if stat_on_power:
            mask_stat |= power > q_power
        
        # 规则 B：功率爬坡事件
        # 阈值：默认每 15 分钟变化超过装机容量的 5%
        ramp_ratio = float(extreme_cfg.get("ramp_ratio", 0.05))
        ramp_q = extreme_cfg.get("ramp_q", None)
        power_diff = np.abs(np.diff(power, prepend=power[0]))
        if ramp_q is not None:
            ramp_threshold = float(np.percentile(power_diff, float(ramp_q)))
        else:
            ramp_threshold = ramp_ratio * self.nominal_capacity
        mask_ramp = power_diff > ramp_threshold
        
        # 规则 C：切出风速（默认 25 m/s）
        cutout_ms = float(extreme_cfg.get("cutout_ms", 25.0))
        mask_cutout = wind_hub > cutout_ms if col_wind_hub else np.zeros(len(df), dtype=bool)
        
        # 规则 D：极端温度（可用分位数阈值）
        temp_q_low = extreme_cfg.get("temp_q_low", None)
        temp_q_high = extreme_cfg.get("temp_q_high", None)
        if temp_q_low is not None and col_temp:
            temp_low = float(np.percentile(temperature, float(temp_q_low)))
        else:
            temp_low = float(extreme_cfg.get("temp_low", -10.0))
        if temp_q_high is not None and col_temp:
            temp_high = float(np.percentile(temperature, float(temp_q_high)))
        else:
            temp_high = float(extreme_cfg.get("temp_high", 35.0))
        mask_temp = ((temperature < temp_low) | (temperature > temp_high)) if col_temp else np.zeros(len(df), dtype=bool)
        temp_requires_wind = bool(extreme_cfg.get("temp_requires_wind", False))
        if temp_requires_wind and col_wind_hub:
            mask_temp = mask_temp & (wind_hub > q_wind)
        
        use_stat = extreme_cfg.get("use_stat", True)
        use_ramp = extreme_cfg.get("use_ramp", True)
        use_cutout = extreme_cfg.get("use_cutout", True)
        use_temp = extreme_cfg.get("use_temp", True)

        masks = []
        if use_stat:
            masks.append(mask_stat)
        if use_ramp:
            masks.append(mask_ramp)
        if use_cutout:
            masks.append(mask_cutout)
        if use_temp:
            masks.append(mask_temp)

        if masks:
            is_extreme = masks[0].copy()
            for m in masks[1:]:
                is_extreme |= m
        else:
            is_extreme = np.zeros(len(df), dtype=bool)
        
        # 输出统计信息
        print(f"Total Samples: {len(df)}")
        print(f"Extreme Samples: {np.sum(is_extreme)} ({np.mean(is_extreme)*100:.2f}%)")
        print(
            f"  - Statistical Rule (>{stat_q:.1f}%, wind={stat_on_wind}, power={stat_on_power}): "
            f"{np.sum(mask_stat)}"
        )
        print(f"  - Ramp Rule (>{ramp_threshold:.2f}MW): {np.sum(mask_ramp)}")
        print(f"  - Cut-out Rule (>{cutout_ms:.1f}m/s): {np.sum(mask_cutout)}")
        print(f"  - Temperature Rule: {np.sum(mask_temp)}")
        
        return is_extreme
    
    def create_sliding_window(self, X, y, is_extreme, window_size=12):
        """
        构建滑动窗口样本
        
        参数：
            X: 特征数组 (N, Features)
            y: 标签数组 (N,)
            is_extreme: 极端天气标签
            window_size: 窗口时间步长度（默认 12 = 3 小时）
            
        返回：
            X_windows: (N-window, window*Features) 将窗口展平后的特征
            y_targets: (N-window,) 预测目标（窗口后的下一时刻）
            labels: (N-window,) 对应目标时刻是否为极端（用于领域划分）
        """
        X_windows, y_targets, labels = [], [], []
        
        for i in range(len(X) - window_size):
            # 提取窗口特征
            window_feature = X[i:i+window_size, :]
            
            # 展平为向量
            X_flat = window_feature.flatten()
            
            # 目标为窗口后的下一时刻
            target = y[i + window_size]
            
            # 标签：若目标时刻为极端天气，则样本划入目标域
            tag = is_extreme[i + window_size]
            
            X_windows.append(X_flat)
            y_targets.append(target)
            labels.append(tag)
        
        return np.array(X_windows), np.array(y_targets), np.array(labels)
    
    def split_domain_data(self, X, y, labels):
        """
        将样本按标签拆分为源域（正常）与目标域（极端）
        
        返回：
            X_S, y_S: 源域数据（正常天气）
            X_T, y_T: 目标域数据（极端天气）
        """
        # 源域：正常天气（Tag=False）
        X_S = X[~labels]
        y_S = y[~labels]
        
        # 目标域：极端天气（Tag=True）
        X_T = X[labels]
        y_T = y[labels]
        
        print(f"\n[Data Split Info]")
        print(f"Source Domain (Normal): {len(X_S)} samples")
        print(f"Target Domain (Extreme): {len(X_T)} samples")
        
        return X_S, y_S, X_T, y_T


def test_data_loading():
    """简单自检：验证数据加载与处理流程是否可用"""
    gen = WindFarmDataGenerator()
    
    # List available farms
    print("Available wind farms:")
    for i, farm in enumerate(gen.get_available_farms()):
        print(f"  [{i}] {farm['name']} - {farm['capacity_MW']}MW")
    
    # Load first farm
    print("\n" + "="*60)
    X_scaled, y_scaled, is_extreme = gen.load_and_process_data(farm_idx=0, feature_set='full')
    
    print(f"\nFeature shape: {X_scaled.shape}")
    print(f"Target shape: {y_scaled.shape}")
    print(f"Extreme samples: {np.sum(is_extreme)}")
    
    # Create sliding windows
    X, y, labels = gen.create_sliding_window(X_scaled, y_scaled, is_extreme, window_size=12)
    print(f"\nWindow samples shape: {X.shape}")
    
    # Split domains
    X_S, y_S, X_T, y_T = gen.split_domain_data(X, y, labels)
    

if __name__ == "__main__":
    test_data_loading()
