import numpy as np
from scipy.spatial.distance import cdist

def rbf_kernel(X1, X2, gamma=None):
    """
    计算 RBF 核矩阵 K(X1, X2) = exp(-gamma * ||x1 - x2||^2)
    
    Args:
        X1: (n1, d) 矩阵
        X2: (n2, d) 矩阵
        gamma: 核宽参数，默认 1/d
    Returns:
        K: (n1, n2) 核矩阵
    """
    if gamma is None:
        gamma = 1.0 / X1.shape[1]
        
    # 计算欧氏距离的平方
    dists_sq = cdist(X1, X2, metric='sqeuclidean')
    K = np.exp(-gamma * dists_sq)
    return K