import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.preprocessing import StandardScaler

class NystromApproximation:
    """
    Nyström 方法：将非线性核问题转化为低维线性问题
    """
    def __init__(self, n_components=100, gamma=0.1):
        self.n_components = n_components
        self.gamma = gamma
        self.components_ = None
        self.normalization_ = None

    def fit(self, X):
        """
        使用 K-Means 选择最具代表性的 m 个路标点 (Landmarks)
        比随机采样效果更稳健
        """
        n_samples = X.shape[0]
        # 如果样本数少于组件数，直接用所有样本
        if n_samples <= self.n_components:
            self.components_ = X
        else:
            # 使用 K-Means 聚类中心作为路标点
            kmeans = KMeans(n_clusters=self.n_components, random_state=42, n_init=10)
            kmeans.fit(X)
            self.components_ = kmeans.cluster_centers_
        
        # 计算路标点之间的核矩阵 K_mm
        K_mm = rbf_kernel(self.components_, self.components_, gamma=self.gamma)
        
        # 计算 K_mm 的伪逆平方根，用于归一化映射
        # Z(x) = K(x, landmarks) * (K_mm)^(-1/2)
        try:
            U, S, V = np.linalg.svd(K_mm)
            S_inv_sqrt = np.diag(1.0 / np.sqrt(S + 1e-6)) # 加扰动防除零
            self.normalization_ = np.dot(U, np.dot(S_inv_sqrt, V))
        except np.linalg.LinAlgError:
            # 降级处理：直接使用单位阵
            self.normalization_ = np.eye(self.n_components)
            
        return self

    def transform(self, X):
        """
        将数据映射到 m 维特征空间
        Output shape: (N, n_components)
        """
        K_nm = rbf_kernel(X, self.components_, gamma=self.gamma)
        return np.dot(K_nm, self.normalization_)


class Fast_TL_QLDMR:
    """
    [Fast Solver] 基于 Nyström 近似 + Primal SGD 的快速 TL-QLDMR
    适合大规模风电数据 (N > 10,000)
    """
    def __init__(self, lambda1=1.0, lambda2=1.0, C_S=1.0, C_T=10.0, 
                 tau=0.95, kernel_gamma=0.01, n_components=200, 
                 learning_rate=0.01, epochs=100, batch_size=256):
        
        # 超参数
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.C_S = C_S
        self.C_T = C_T
        self.tau = tau
        self.gamma = kernel_gamma
        
        # 快速求解器参数
        self.n_components = n_components # Nyström 组件数 (m)
        self.lr = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        
        # 模型权重 (在 m 维空间)
        self.w = None
        self.b = 0.0
        self.feature_map = None # Nyström 映射器

    def _compute_regularizer_matrix(self, Z_S, Z_T):
        """
        预计算正则化矩阵 R (m x m 维)
        Original Regularizer: 1/2 w^T w + lambda1 Var + lambda2 MMD
        Mapped to Primal: 1/2 w^T (I + lambda1 * S_z + lambda2 * M_z) w
        """
        m = Z_S.shape[1]
        
        # 1. 基础项 I
        R = np.eye(m)
        
        # 2. 分布方差项 (Source Variance)
        # Var = w^T (Z_S^T J Z_S) w
        # 中心化 Z_S
        Z_S_mean = np.mean(Z_S, axis=0)
        Z_S_centered = Z_S - Z_S_mean
        S_z = np.dot(Z_S_centered.T, Z_S_centered) # (m, m)
        R += self.lambda1 * S_z
        
        # 3. MMD 迁移项
        # MMD^2 = || mean(Z_S) - mean(Z_T) ||^2
        #       = w^T (diff * diff^T) w
        diff = np.mean(Z_S, axis=0) - np.mean(Z_T, axis=0)
        M_z = np.outer(diff, diff) # (m, m)
        R += self.lambda2 * M_z
        
        return R

    def fit(self, X_S, y_S, X_T, y_T):
        """
        使用 Adam/SGD 优化 Primal 问题
        """
        # 1. Nyström 特征映射 (降维)
        # 使用源域数据构建映射基准
        # 考虑到源域数据量大，随机采样部分用于构建映射
        print(f"1. Nyström Mapping (m={self.n_components})...")
        sample_indices = np.random.choice(len(X_S), min(2000, len(X_S)), replace=False)
        self.feature_map = NystromApproximation(n_components=self.n_components, gamma=self.gamma)
        self.feature_map.fit(X_S[sample_indices])
        
        # 映射所有数据
        Z_S = self.feature_map.transform(X_S)
        Z_T = self.feature_map.transform(X_T)
        
        # 2. 预计算正则化矩阵 R (常数矩阵)
        print("2. Computing Regularization Matrix...")
        R_matrix = self._compute_regularizer_matrix(Z_S, Z_T)
        
        # 3. 准备训练数据 (合并源域和目标域)
        Z_train = np.vstack([Z_S, Z_T])
        y_train = np.concatenate([y_S, y_T])
        
        # 生成样本权重向量 C_vec
        C_vec = np.concatenate([
            np.full(len(X_S), self.C_S),
            np.full(len(X_T), self.C_T)
        ])
        
        # 4. SGD 优化 (Adam Optimizer 手写实现)
        print(f"3. Optimizing with Adam (Epochs={self.epochs})...")
        
        n_samples, n_features = Z_train.shape
        self.w = np.zeros(n_features)
        self.b = 0.0
        
        # Adam 参数
        beta1, beta2, epsilon = 0.9, 0.999, 1e-8
        m_w, v_w = np.zeros_like(self.w), np.zeros_like(self.w)
        m_b, v_b = 0.0, 0.0
        t = 0
        
        indices = np.arange(n_samples)
        
        for epoch in range(self.epochs):
            np.random.shuffle(indices)
            
            total_loss = 0
            
            for start_idx in range(0, n_samples, self.batch_size):
                end_idx = min(start_idx + self.batch_size, n_samples)
                batch_idx = indices[start_idx:end_idx]
                
                z_batch = Z_train[batch_idx]
                y_batch = y_train[batch_idx]
                c_batch = C_vec[batch_idx]
                
                # --- Forward ---
                # f(x) = z * w + b
                preds = np.dot(z_batch, self.w) + self.b
                residuals = y_batch - preds
                
                # --- Loss Calculation (Pinball) ---
                # L = tau * u (if u>=0) + (tau-1) * u (if u<0), where u = y - f
                # Gradient w.r.t f: -tau (if u>=0) or -(tau-1) (if u<0)
                # Gradient w.r.t w: grad_f * z
                
                # Subgradient of Pinball Loss w.r.t prediction (f)
                grad_f = np.zeros_like(residuals)
                mask_pos = residuals >= 0 # y >= f, Underestimation
                mask_neg = ~mask_pos      # y < f, Overestimation
                
                # Pinball导数:
                # if y > f (res > 0): grad L / grad f = -tau
                # if y < f (res < 0): grad L / grad f = -(tau - 1) = 1 - tau
                grad_f[mask_pos] = -self.tau
                grad_f[mask_neg] = 1.0 - self.tau
                
                # Apply sample weights C
                grad_f *= c_batch
                
                # --- Backward ---
                # 1. Gradient from Loss
                # grad_w_loss = sum(grad_f * z)
                # grad_b_loss = sum(grad_f)
                grad_w = np.dot(z_batch.T, grad_f)
                grad_b = np.sum(grad_f)
                
                # 2. Gradient from Regularizer (Only for w)
                # J = 1/2 w^T R w  => grad = R w
                # 注意：正则化项是在整个数据集上定义的，但在 SGD 中我们要把它分摊到每个 batch
                # 或者只在最后加。标准做法是 Weight Decay。
                # 这里我们显式计算: grad_reg = R @ w / (n_batches) ?
                # 简化起见，我们直接按比例缩放正则化梯度
                scale = len(batch_idx) / n_samples
                grad_w += scale * np.dot(R_matrix, self.w)
                
                # --- Adam Update ---
                t += 1
                
                # Update w
                m_w = beta1 * m_w + (1 - beta1) * grad_w
                v_w = beta2 * v_w + (1 - beta2) * (grad_w ** 2)
                m_hat_w = m_w / (1 - beta1 ** t)
                v_hat_w = v_w / (1 - beta2 ** t)
                self.w -= self.lr * m_hat_w / (np.sqrt(v_hat_w) + epsilon)
                
                # Update b
                m_b = beta1 * m_b + (1 - beta1) * grad_b
                v_b = beta2 * v_b + (1 - beta2) * (grad_b ** 2)
                m_hat_b = m_b / (1 - beta1 ** t)
                v_hat_b = v_b / (1 - beta2 ** t)
                self.b -= self.lr * m_hat_b / (np.sqrt(v_hat_b) + epsilon)
                
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{self.epochs} complete.")

        print("Fast Training Completed.")
        return True

    def predict(self, X_test):
        """
        预测函数: f(x) = Z(x) * w + b
        """
        if self.w is None:
            raise ValueError("Model untrained.")
            
        Z_test = self.feature_map.transform(X_test)
        return np.dot(Z_test, self.w) + self.b