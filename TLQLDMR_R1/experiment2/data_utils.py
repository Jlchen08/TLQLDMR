import random
from typing import Dict, Tuple, Optional

import numpy as np

from wind_farm_Data_Utils import WindFarmDataGenerator


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass


def build_windows(
    X: np.ndarray,
    y: np.ndarray,
    is_extreme: np.ndarray,
    window_size: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build sliding windows.

    Returns:
        X_seq: (N-window, window, features)
        X_flat: (N-window, window*features)
        y_targets: (N-window,)
        labels: (N-window,)
    """
    X_seq, X_flat, y_targets, labels = [], [], [], []
    n_samples = len(X)
    for i in range(n_samples - window_size):
        window = X[i : i + window_size, :]
        X_seq.append(window)
        X_flat.append(window.reshape(-1))
        y_targets.append(y[i + window_size])
        labels.append(is_extreme[i + window_size])
    return (
        np.asarray(X_seq, dtype=np.float32),
        np.asarray(X_flat, dtype=np.float32),
        np.asarray(y_targets, dtype=np.float32),
        np.asarray(labels, dtype=bool),
    )


def _split_target(
    X_T_seq: np.ndarray,
    X_T_flat: np.ndarray,
    y_T: np.ndarray,
    train_ratio: float,
    min_train: int,
    split_mode: str = "time",
    split_seed: int = 42,
) -> Dict[str, np.ndarray]:
    n_total = len(X_T_flat)
    n_T_train = int(n_total * train_ratio)
    if n_T_train < min_train:
        n_T_train = int(n_total * 0.5)

    if split_mode == "shuffle":
        rng = np.random.default_rng(split_seed)
        indices = rng.permutation(n_total)
        train_idx = indices[:n_T_train]
        test_idx = indices[n_T_train:]
        return {
            "X_T_train_seq": X_T_seq[train_idx],
            "X_T_test_seq": X_T_seq[test_idx],
            "X_T_train_flat": X_T_flat[train_idx],
            "X_T_test_flat": X_T_flat[test_idx],
            "y_T_train": y_T[train_idx],
            "y_T_test": y_T[test_idx],
        }

    return {
        "X_T_train_seq": X_T_seq[:n_T_train],
        "X_T_test_seq": X_T_seq[n_T_train:],
        "X_T_train_flat": X_T_flat[:n_T_train],
        "X_T_test_flat": X_T_flat[n_T_train:],
        "y_T_train": y_T[:n_T_train],
        "y_T_test": y_T[n_T_train:],
    }


def _maybe_truncate(arr: np.ndarray, max_samples: Optional[int]) -> np.ndarray:
    if max_samples is None:
        return arr
    if len(arr) <= max_samples:
        return arr
    return arr[:max_samples]


def _fill_nan_seq(X_seq: np.ndarray) -> np.ndarray:
    if not np.isnan(X_seq).any():
        return X_seq
    feat_mean = np.nanmean(X_seq, axis=(0, 1))
    feat_mean = np.where(np.isnan(feat_mean), 0.0, feat_mean)
    filled = np.where(np.isnan(X_seq), feat_mean[None, None, :], X_seq)
    return filled


def _fill_nan_vector(y: np.ndarray) -> np.ndarray:
    if not np.isnan(y).any():
        return y
    mean_val = np.nanmean(y)
    if np.isnan(mean_val):
        mean_val = 0.0
    return np.where(np.isnan(y), mean_val, y)


def prepare_farm_data(
    farm_idx: int,
    feature_set: str = "full",
    window_size: int = 6,
    target_train_ratio: float = 0.6,
    min_target_train: int = 10,
    max_source_samples: Optional[int] = None,
    max_target_train_samples: Optional[int] = None,
    extreme_cfg: Optional[Dict] = None,
    split_mode: str = "time",
    split_seed: int = 42,
) -> Dict[str, np.ndarray]:
    """
    Prepare data for one farm. Uses the same domain split as wind_farm_experiment.py.

    Returns a dict containing:
        - X_S_seq, X_S_flat, y_S
        - X_T_train_seq, X_T_test_seq, X_T_train_flat, X_T_test_flat, y_T_train, y_T_test
        - scaler_y, nominal_capacity, farm_name
    """
    gen = WindFarmDataGenerator()
    farms = gen.get_available_farms()
    farm_name = farms[farm_idx]["name"]

    X_scaled, y_scaled, is_extreme = gen.load_and_process_data(
        farm_idx=farm_idx, feature_set=feature_set, extreme_cfg=extreme_cfg
    )

    X_seq, X_flat, y_targets, labels = build_windows(
        X_scaled, y_scaled, is_extreme, window_size
    )

    # Fill NaNs to keep sklearn models stable
    X_seq = _fill_nan_seq(X_seq)
    y_targets = _fill_nan_vector(y_targets)
    # Rebuild flat from cleaned sequences to keep consistency
    X_flat = X_seq.reshape(X_seq.shape[0], -1)

    X_S_seq = X_seq[~labels]
    X_S_flat = X_flat[~labels]
    y_S = y_targets[~labels]

    X_T_seq = X_seq[labels]
    X_T_flat = X_flat[labels]
    y_T = y_targets[labels]

    print("\n[Data Split Info]")
    print(f"Source Domain (Normal): {len(X_S_flat)} samples")
    print(f"Target Domain (Extreme): {len(X_T_flat)} samples")

    # Optional truncation for efficiency (keeps temporal order)
    X_S_seq = _maybe_truncate(X_S_seq, max_source_samples)
    X_S_flat = _maybe_truncate(X_S_flat, max_source_samples)
    y_S = _maybe_truncate(y_S, max_source_samples)

    split = _split_target(
        X_T_seq,
        X_T_flat,
        y_T,
        target_train_ratio,
        min_target_train,
        split_mode=split_mode,
        split_seed=split_seed,
    )

    # Optional truncation for target train (keeps temporal order)
    split["X_T_train_seq"] = _maybe_truncate(
        split["X_T_train_seq"], max_target_train_samples
    )
    split["X_T_train_flat"] = _maybe_truncate(
        split["X_T_train_flat"], max_target_train_samples
    )
    split["y_T_train"] = _maybe_truncate(
        split["y_T_train"], max_target_train_samples
    )

    return {
        "farm_name": farm_name,
        "nominal_capacity": gen.nominal_capacity,
        "scaler_y": gen.scaler_y,
        "X_S_seq": X_S_seq,
        "X_S_flat": X_S_flat,
        "y_S": y_S,
        **split,
    }
