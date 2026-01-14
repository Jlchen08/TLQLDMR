import numpy as np
from scipy.spatial.distance import cdist

try:
    import cupy as cp
except ImportError:
    cp = None


def _is_cupy_array(x):
    return cp is not None and isinstance(x, cp.ndarray)

def rbf_kernel(X1, X2, gamma=None, xp=None):
    """
    计算 RBF 核矩阵 K(X1, X2) = exp(-gamma * ||x1 - x2||^2)
    
    Args:
        X1: (n1, d) 矩阵
        X2: (n2, d) 矩阵
        gamma: 核宽参数，默认 1/d
        xp: 可选，numpy 或 cupy 模块；未指定时根据输入自动判断
    Returns:
        K: (n1, n2) 核矩阵
    """
    if xp is None:
        xp = cp if _is_cupy_array(X1) or _is_cupy_array(X2) else np

    if xp is cp and cp is None:
        raise ImportError("cupy is required for GPU computation but is not installed.")

    if gamma is None:
        gamma = 1.0 / X1.shape[1]

    if xp is np:
        # 计算欧氏距离的平方
        dists_sq = cdist(X1, X2, metric='sqeuclidean')
        return np.exp(-gamma * dists_sq)

    # GPU path: 使用向量化形式避免 cupyx 依赖
    X1 = xp.asarray(X1)
    X2 = xp.asarray(X2)
    x1_sq = xp.sum(X1 * X1, axis=1, keepdims=True)
    x2_sq = xp.sum(X2 * X2, axis=1, keepdims=True).T
    dists_sq = x1_sq + x2_sq - 2.0 * (X1 @ X2.T)
    return xp.exp(-gamma * dists_sq)
