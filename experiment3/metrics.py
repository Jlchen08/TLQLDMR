import numpy as np


def picp(y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    y_true = np.asarray(y_true)
    lower = np.asarray(lower)
    upper = np.asarray(upper)
    covered = (y_true >= lower) & (y_true <= upper)
    return float(np.mean(covered))


def mpiw(lower: np.ndarray, upper: np.ndarray) -> float:
    lower = np.asarray(lower)
    upper = np.asarray(upper)
    return float(np.mean(upper - lower))


def pinaw(y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    y_true = np.asarray(y_true)
    width = mpiw(lower, upper)
    denom = float(np.max(y_true) - np.min(y_true))
    if denom <= 0:
        return float("nan")
    return float(width / denom)


def cwc(pinaw_value: float, picp_value: float, alpha: float = 0.1, gamma: float = 0.3, eta: float = 2.0) -> float:
    target = 1.0 - alpha
    if picp_value >= target:
        return float(pinaw_value)
    penalty = gamma * np.exp(-eta * (picp_value - target))
    return float(pinaw_value * (1.0 + penalty))


def winkler_score(y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray, alpha: float = 0.1) -> float:
    y_true = np.asarray(y_true)
    lower = np.asarray(lower)
    upper = np.asarray(upper)
    width = upper - lower
    score = width.copy()
    below = y_true < lower
    above = y_true > upper
    score[below] += (2.0 / alpha) * (lower[below] - y_true[below])
    score[above] += (2.0 / alpha) * (y_true[above] - upper[above])
    return float(np.mean(score))


def interval_metrics(
    y_true: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    alpha: float = 0.1,
    gamma: float = 0.3,
    eta: float = 2.0,
) -> dict:
    picp_val = picp(y_true, lower, upper)
    mpiw_val = mpiw(lower, upper)
    pinaw_val = pinaw(y_true, lower, upper)
    cwc_val = cwc(pinaw_val, picp_val, alpha=alpha, gamma=gamma, eta=eta)
    winkler_val = winkler_score(y_true, lower, upper, alpha=alpha)
    return {
        "picp": picp_val,
        "mpiw": mpiw_val,
        "pinaw": pinaw_val,
        "cwc": cwc_val,
        "winkler": winkler_val,
    }
