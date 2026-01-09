"""
Wind Farm Data Utilities for TL-QLDMR
Based on the Chinese State Grid Renewable Energy Generation dataset (s41597-022-01696-6)
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from pathlib import Path


class WindFarmDataGenerator:
    """
    Wind Farm Data Processing Utility
    Processes wind farm SCADA data including:
    - Wind speed at multiple heights (10m, 30m, 50m, hub height)
    - Wind direction at multiple heights  
    - Meteorological data (temperature, pressure, humidity)
    - Power output (MW)
    """
    
    def __init__(self, data_dir='/Users/lin/project/WP/TL-QLDMR/data/wind_farm_data/data_processed/wind_farms'):
        """
        Initialize the data generator
        
        Args:
            data_dir: Path to the wind farms data directory
        """
        self.data_dir = Path(data_dir)
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self.nominal_capacity = None  # MW
        
        # Column name mapping
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
        """Get list of available wind farm files"""
        farms = list(self.data_dir.glob('*.xlsx'))
        farm_info = []
        for f in farms:
            # Extract capacity from filename
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
        Load data from a specific wind farm
        
        Args:
            farm_idx: Index of the farm to load (0-5, sorted by capacity descending)
            use_processed: Whether to use processed data (True) or original data
            
        Returns:
            DataFrame with wind farm data
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
    
    def load_and_process_data(self, farm_idx=0, feature_set='full'):
        """
        Load data and perform preprocessing
        
        Args:
            farm_idx: Index of the farm to load
            feature_set: 'full' (all features), 'wind_only' (wind speeds only), 
                        'simple' (hub wind speed + temperature)
                        
        Returns:
            X_scaled: Normalized features
            y_scaled: Normalized power output
            is_extreme: Extreme weather labels (Boolean Array)
        """
        df = self.load_farm_data(farm_idx)
        
        # Extract features based on feature_set
        # Handle column name variations across different farm datasets
        available_cols = df.columns.tolist()
        
        def find_column(pattern_keywords):
            """Find column name containing all keywords"""
            for col in available_cols:
                if all(kw.lower() in col.lower() for kw in pattern_keywords):
                    return col
            return None
        
        # Map feature names dynamically
        col_wind_hub = find_column(['wind speed', 'wheel hub', 'm/s'])
        col_wind_50m = find_column(['wind speed', '50 meters', 'm/s'])
        col_wind_30m = find_column(['wind speed', '30 meters', 'm/s'])
        col_wind_10m = find_column(['wind speed', '10 meters', 'm/s'])
        col_temp = find_column(['air temperature', '°C'])
        col_pressure = find_column(['atmosphere', 'hpa'])
        col_humidity = find_column(['relative humidity', '%'])
        col_power = find_column(['power', 'MW'])
        
        print(f"Detected columns:")
        print(f"  Wind hub: {col_wind_hub}")
        print(f"  Temperature: {col_temp}")
        print(f"  Power: {col_power}")
        
        if feature_set == 'full':
            # Use all available features
            feature_cols = [col for col in [
                col_wind_hub, col_wind_50m, col_wind_30m, col_wind_10m,
                col_temp, col_pressure, col_humidity
            ] if col is not None]
        elif feature_set == 'wind_only':
            feature_cols = [col for col in [
                col_wind_hub, col_wind_50m, col_wind_30m, col_wind_10m
            ] if col is not None]
        else:  # 'simple'
            feature_cols = [col for col in [col_wind_hub, col_temp] if col is not None]
        
        print(f"Using {len(feature_cols)} features: {feature_cols}")
        
        X_raw = df[feature_cols].values
        power = df[col_power].values
        
        # Identify extreme weather events
        print("\nIdentifying extreme weather events...")
        is_extreme = self._identify_extreme_weather(df, power, col_wind_hub, col_temp)

        
        # Normalize
        X_scaled = self.scaler_X.fit_transform(X_raw)
        y_scaled = self.scaler_y.fit_transform(power.reshape(-1, 1)).flatten()
        
        return X_scaled, y_scaled, is_extreme
    
    def _identify_extreme_weather(self, df, power, col_wind_hub, col_temp):
        """
        Identify extreme weather conditions using multiple rules
        
        Rules:
        A. Statistical: High wind speed or power (>95th percentile)
        B. Physical: Power ramp events (|P(t) - P(t-1)| > threshold)
        C. Cut-out: Very high wind speed (>25 m/s) - turbine shutdown
        D. Low temperature: < -10°C or > 35°C
        """
        is_extreme = np.zeros(len(df), dtype=bool)
        
        wind_hub = df[col_wind_hub].values if col_wind_hub else np.zeros(len(df))
        temperature = df[col_temp].values if col_temp else np.zeros(len(df))
        
        # Rule A: Statistical threshold (95th percentile)
        q95_wind = np.percentile(wind_hub, 95) if col_wind_hub else 0
        q95_power = np.percentile(power, 95)
        
        mask_stat = (wind_hub > q95_wind) | (power > q95_power)
        
        # Rule B: Power ramp events
        # Threshold: 5% of nominal capacity per 15 minutes
        ramp_threshold = 0.05 * self.nominal_capacity
        power_diff = np.abs(np.diff(power, prepend=power[0]))
        mask_ramp = power_diff > ramp_threshold
        
        # Rule C: Cut-out wind speed (typically 25 m/s)
        mask_cutout = wind_hub > 25.0 if col_wind_hub else np.zeros(len(df), dtype=bool)
        
        # Rule D: Extreme temperature
        mask_temp = ((temperature < -10) | (temperature > 35)) if col_temp else np.zeros(len(df), dtype=bool)
        
        # Combine rules
        is_extreme = mask_stat | mask_ramp | mask_cutout | mask_temp
        
        # Print statistics
        print(f"Total Samples: {len(df)}")
        print(f"Extreme Samples: {np.sum(is_extreme)} ({np.mean(is_extreme)*100:.2f}%)")
        print(f"  - Statistical Rule (>95%): {np.sum(mask_stat)}")
        print(f"  - Ramp Rule (>{ramp_threshold:.2f}MW): {np.sum(mask_ramp)}")
        print(f"  - Cut-out Rule (>25m/s): {np.sum(mask_cutout)}")
        print(f"  - Temperature Rule: {np.sum(mask_temp)}")
        
        return is_extreme
    
    def create_sliding_window(self, X, y, is_extreme, window_size=12):
        """
        Build sliding window samples
        
        Args:
            X: Feature array (N, Features)
            y: Target array (N,)
            is_extreme: Extreme weather labels
            window_size: Number of time steps in the window (default 12 = 3 hours)
            
        Returns:
            X_windows: (N-window, window*Features) - Flattened vectors
            y_targets: (N-window,) - Target values
            labels: (N-window,) - Extreme weather labels
        """
        X_windows, y_targets, labels = [], [], []
        
        for i in range(len(X) - window_size):
            # Extract window features
            window_feature = X[i:i+window_size, :]
            
            # Flatten to vector
            X_flat = window_feature.flatten()
            
            # Target is the next point after the window
            target = y[i + window_size]
            
            # Label assignment: if target point is extreme, sample goes to target domain
            tag = is_extreme[i + window_size]
            
            X_windows.append(X_flat)
            y_targets.append(target)
            labels.append(tag)
        
        return np.array(X_windows), np.array(y_targets), np.array(labels)
    
    def split_domain_data(self, X, y, labels):
        """
        Split data into source domain (normal) and target domain (extreme)
        
        Returns:
            X_S, y_S: Source domain data (normal weather)
            X_T, y_T: Target domain data (extreme weather)
        """
        # Source domain: Normal weather (Tag=False)
        X_S = X[~labels]
        y_S = y[~labels]
        
        # Target domain: Extreme weather (Tag=True)
        X_T = X[labels]
        y_T = y[labels]
        
        print(f"\n[Data Split Info]")
        print(f"Source Domain (Normal): {len(X_S)} samples")
        print(f"Target Domain (Extreme): {len(X_T)} samples")
        
        return X_S, y_S, X_T, y_T


def test_data_loading():
    """Test function to verify data loading works correctly"""
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
