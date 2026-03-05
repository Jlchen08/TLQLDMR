"""
TL-QLDMR 训练器实现

参考与用途说明：
- Kernel_Function.rbf_kernel：统一的 RBF 核计算（支持 numpy/cupy），用于对偶 QP 与预测
- /mnt/mydisk/zhangxiaohan/lin/TL-QLDMR/t：Nyström 近似 + Primal Adam 的快速训练方法来源
  用途：在 solver='fast_nystrom' 下避免构造完整核矩阵（降低 O(N^2) 内存）

模块板块说明：
- Nyström 近似（NystromApproximation / Fast_TL_QLDMR）：将核问题映射到低维特征空间并训练线性模型
- TL_QLDMR 主类：统一入口，按 solver 选择训练路径
  - cvxopt/torch_gd/qpth：对偶 QP 路径，适合中小规模数据
  - batch_sgd：核切片 Mini-batch SGD（PyTorch），避免显式构造完整核矩阵
  - fast_nystrom：Nyström + Primal Adam，适合更大规模数据与更低内存场景
"""

import time
import numpy as np
import cvxopt
from Kernel_Function import rbf_kernel
import gc

# 训练器 TL_QLDMR 支持以下 solver:
# - cvxopt / torch_gd / qpth: 原有对偶 QP 解法
# - batch_sgd: 现有基于 PyTorch 的核切片 SGD
# - fast_nystrom: 集成自 t 的 Nyström+Primal Adam 快速训练（更省内存）

try:
    import cupy as cp
    from cupy.cuda.memory import OutOfMemoryError
except ImportError:
    cp = None
    OutOfMemoryError = MemoryError  # Fallback

try:
    from sklearn.cluster import KMeans
except ImportError:
    KMeans = None

try:
    import torch
except ImportError:
    torch = None

try:
    from qpth.qp import QPFunction
except ImportError:
    QPFunction = None

# 隐藏 cvxopt 的输出
cvxopt.solvers.options['show_progress'] = False

# ===============================
# 集成训练方法：Nyström + Primal Adam
# 来源：/mnt/mydisk/zhangxiaohan/lin/TL-QLDMR/t
# ===============================
class NystromApproximation:
    """
    Nyström 方法：将非线性核问题转化为低维线性问题
    """
    def __init__(self, n_components=100, gamma=0.1):
        self.n_components = int(n_components)
        self.gamma = gamma
        self.components_ = None
        self.normalization_ = None

    def fit(self, X):
        """
        使用 K-Means 选择最具代表性的 m 个路标点 (Landmarks)
        比随机采样效果更稳健
        """
        X = np.asarray(X)
        n_samples = X.shape[0]
        if n_samples == 0:
            raise ValueError("Nyström fit received empty X.")

        if n_samples <= self.n_components:
            self.components_ = X
        else:
            if KMeans is None:
                raise ImportError("scikit-learn is required for Nyström landmark selection (KMeans).")
            kmeans = KMeans(n_clusters=self.n_components, random_state=42, n_init=10)
            kmeans.fit(X)
            self.components_ = kmeans.cluster_centers_

        K_mm = rbf_kernel(self.components_, self.components_, gamma=self.gamma, xp=np)

        try:
            U, S, V = np.linalg.svd(K_mm)
            S_inv_sqrt = np.diag(1.0 / np.sqrt(S + 1e-6))
            self.normalization_ = np.dot(U, np.dot(S_inv_sqrt, V))
        except np.linalg.LinAlgError:
            self.normalization_ = np.eye(self.components_.shape[0])

        return self

    def transform(self, X):
        """
        将数据映射到 m 维特征空间
        Output shape: (N, m)
        """
        if self.components_ is None or self.normalization_ is None:
            raise ValueError("Nyström mapper is not fitted.")
        K_nm = rbf_kernel(np.asarray(X), self.components_, gamma=self.gamma, xp=np)
        return np.dot(K_nm, self.normalization_)


class Fast_TL_QLDMR:
    """
    [Fast Solver] 基于 Nyström 近似 + Primal SGD 的快速 TL-QLDMR
    适合大规模风电数据 (N > 10,000)
    """
    def __init__(
        self,
        lambda1=1.0,
        lambda2=1.0,
        C_S=1.0,
        C_T=10.0,
        tau=0.95,
        kernel_gamma=0.01,
        n_components=200,
        learning_rate=0.01,
        epochs=100,
        batch_size=256,
        variance_mode="symmetric",
        asym_scale=0.0,
    ):

        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.C_S = C_S
        self.C_T = C_T
        self.tau = tau
        self.gamma = kernel_gamma

        self.n_components = n_components
        self.lr = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.variance_mode = str(variance_mode)
        self.asym_scale = float(asym_scale)

        self.w = None
        self.b = 0.0
        self.feature_map = None

    def _compute_regularizer_matrix(self, Z_S, Z_T, y_var=None):
        """
        预计算正则化矩阵 R (m x m 维)
        Original Regularizer: 1/2 w^T w + lambda1 Var + lambda2 MMD
        Mapped to Primal: 1/2 w^T (I + lambda1 * S_z + lambda2 * M_z) w
        """
        m = Z_S.shape[1]
        R = np.eye(m)

        Z_S_mean = np.mean(Z_S, axis=0)
        Z_S_centered = Z_S - Z_S_mean

        if self.variance_mode == "asymmetric" and y_var is not None and len(y_var) == len(Z_S_centered):
            # Keep weights positive to maintain a stable regularizer while introducing
            # quantile-aware asymmetry around the target quantile.
            q_ref = float(np.quantile(np.asarray(y_var).reshape(-1), self.tau))
            sign_term = np.sign(np.asarray(y_var).reshape(-1) - q_ref)
            w = 1.0 + self.asym_scale * sign_term
            w = np.clip(w.astype(np.float64), 1e-6, None)
            S_z = np.dot(Z_S_centered.T, Z_S_centered * w[:, None])
            S_z = S_z / max(float(np.sum(w)), 1.0)
        else:
            S_z = np.dot(Z_S_centered.T, Z_S_centered)
            S_z = S_z / max(float(len(Z_S_centered)), 1.0)

        R += self.lambda1 * S_z

        diff = np.mean(Z_S, axis=0) - np.mean(Z_T, axis=0)
        M_z = np.outer(diff, diff)
        R += self.lambda2 * M_z

        return R

    def fit(self, X_S, y_S, X_T, y_T):
        """
        使用 Adam/SGD 优化 Primal 问题
        """
        X_S = np.asarray(X_S, dtype=np.float32)
        y_S = np.asarray(y_S, dtype=np.float32).reshape(-1)
        X_T = np.asarray(X_T, dtype=np.float32)
        y_T = np.asarray(y_T, dtype=np.float32).reshape(-1)

        print(f"1. Nyström Mapping (m={self.n_components})...")
        if len(X_T) == 0:
            raise ValueError("Fast_TL_QLDMR requires a non-empty target set.")

        use_source = len(X_S) > 0
        if not use_source:
            print("Fast_TL_QLDMR: target-only mode (no source domain).")

        sample_pool = X_S if use_source else X_T
        sample_count = min(2000, len(sample_pool))
        sample_indices = np.random.choice(len(sample_pool), sample_count, replace=False)
        self.feature_map = NystromApproximation(n_components=self.n_components, gamma=self.gamma)
        self.feature_map.fit(sample_pool[sample_indices])

        if use_source:
            Z_S = self.feature_map.transform(X_S)
            Z_T = self.feature_map.transform(X_T)

            print("2. Computing Regularization Matrix...")
            R_matrix = self._compute_regularizer_matrix(Z_S, Z_T, y_var=y_S)

            Z_train = np.vstack([Z_S, Z_T])
            y_train = np.concatenate([y_S, y_T])

            C_vec = np.concatenate([
                np.full(len(X_S), self.C_S, dtype=np.float32),
                np.full(len(X_T), self.C_T, dtype=np.float32),
            ])
        else:
            Z_T = self.feature_map.transform(X_T)
            Z_S = Z_T

            print("2. Computing Regularization Matrix (target-only)...")
            R_matrix = self._compute_regularizer_matrix(Z_S, Z_T, y_var=y_T)

            Z_train = Z_T
            y_train = y_T
            C_vec = np.full(len(X_T), self.C_T, dtype=np.float32)

        print(f"3. Optimizing with Adam (Epochs={self.epochs})...")

        n_samples, n_features = Z_train.shape
        self.w = np.zeros(n_features, dtype=np.float32)
        self.b = 0.0

        beta1, beta2, epsilon = 0.9, 0.999, 1e-8
        m_w, v_w = np.zeros_like(self.w), np.zeros_like(self.w)
        m_b, v_b = 0.0, 0.0
        t = 0

        indices = np.arange(n_samples)

        for epoch in range(self.epochs):
            np.random.shuffle(indices)

            for start_idx in range(0, n_samples, self.batch_size):
                end_idx = min(start_idx + self.batch_size, n_samples)
                batch_idx = indices[start_idx:end_idx]

                z_batch = Z_train[batch_idx]
                y_batch = y_train[batch_idx]
                c_batch = C_vec[batch_idx]

                preds = np.dot(z_batch, self.w) + self.b
                residuals = y_batch - preds

                grad_f = np.zeros_like(residuals)
                mask_pos = residuals >= 0
                mask_neg = ~mask_pos

                grad_f[mask_pos] = -self.tau
                grad_f[mask_neg] = 1.0 - self.tau

                grad_f *= c_batch

                grad_w = np.dot(z_batch.T, grad_f)
                grad_b = float(np.sum(grad_f))

                scale = len(batch_idx) / n_samples
                grad_w += scale * np.dot(R_matrix, self.w)

                t += 1

                m_w = beta1 * m_w + (1 - beta1) * grad_w
                v_w = beta2 * v_w + (1 - beta2) * (grad_w ** 2)
                m_hat_w = m_w / (1 - beta1 ** t)
                v_hat_w = v_w / (1 - beta2 ** t)
                self.w -= self.lr * m_hat_w / (np.sqrt(v_hat_w) + epsilon)

                m_b = beta1 * m_b + (1 - beta1) * grad_b
                v_b = beta2 * v_b + (1 - beta2) * (grad_b ** 2)
                m_hat_b = m_b / (1 - beta1 ** t)
                v_hat_b = v_b / (1 - beta2 ** t)
                self.b -= self.lr * m_hat_b / (np.sqrt(v_hat_b) + epsilon)

            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{self.epochs} complete.")

        print("Fast Training Completed.")
        return True

class TL_QLDMR:
    def __init__(
        self,
        lambda1=1.0,
        lambda2=1.0,
        C_S=1.0,
        C_T=10.0,
        tau=0.95,
        kernel_gamma=0.1,
        use_gpu=True,
        solver="torch_gd",
        torch_device="auto",
        torch_lr=1e-2,
        torch_lr_decay=0.01,
        torch_max_iter=2000,
        torch_tol=1e-6,
        torch_proj_tol=1e-9,
        torch_proj_max_iter=80,
        nystrom_n_components=200,
        nystrom_lr=0.01,
        nystrom_epochs=100,
        nystrom_batch_size=256,
        variance_mode="symmetric",
        asym_scale=0.0,
    ):
        """
        Args:
            lambda1: 分布方差项 (LDMR Variance) 权重
            lambda2: MMD 迁移项权重
            C_S: 源域 Pinball Loss 权重
            C_T: 目标域 Pinball Loss 权重 (通常设大，防止被源域淹没)
            tau: 分位数 (0.05, 0.5, 0.95 等)
            kernel_gamma: RBF核参数
            use_gpu: 是否启用 GPU (CUPY - CUDA)
            solver: QP 求解器 ("cvxopt", "torch_gd", "qpth") 或 "batch_sgd"/"fast_nystrom" (新增)
            torch_device: "auto"/"cpu"/"cuda" 或 "cuda:0"
            torch_lr: PyTorch 梯度下降初始学习率
            torch_lr_decay: 学习率衰减系数
            torch_max_iter: 最大迭代次数
            torch_tol: 梯度范数停止阈值
            torch_proj_tol: 约束投影的误差阈值
            torch_proj_max_iter: 约束投影的最大迭代次数
            nystrom_n_components: fast_nystrom 的路标点数 (m)
            nystrom_lr: fast_nystrom 的学习率
            nystrom_epochs: fast_nystrom 的训练轮数
            nystrom_batch_size: fast_nystrom 的 batch 大小
            variance_mode: 分布方差模式 ("symmetric" 或 "asymmetric")
            asym_scale: 非对称方差权重幅度（仅 variance_mode='asymmetric' 生效）
        """
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.C_S = C_S
        self.C_T = C_T
        self.tau = tau
        self.gamma = kernel_gamma
        self.use_gpu = use_gpu
        self.solver = solver
        self.torch_device = torch_device
        self.torch_lr = torch_lr
        self.torch_lr_decay = torch_lr_decay
        self.torch_max_iter = torch_max_iter
        self.torch_tol = torch_tol
        self.torch_proj_tol = torch_proj_tol
        self.torch_proj_max_iter = torch_proj_max_iter

        self.nystrom_n_components = nystrom_n_components
        self.nystrom_lr = nystrom_lr
        self.nystrom_epochs = nystrom_epochs
        self.nystrom_batch_size = nystrom_batch_size
        self.variance_mode = str(variance_mode)
        self.asym_scale = float(asym_scale)
        
        self.alpha = None      # 对偶变量 alpha
        self.alpha_star = None # 对偶变量 alpha*
        self.beta = None       # beta = alpha - alpha*
        self.beta_primal = None # Primal coefficients
        self.b = 0.0           # 偏置
        self.X_train = None    # 训练数据 (用于预测时计算核)
        self.qp_stats = None   # QP 求解统计信息

    def fit(self, X_S, y_S, X_T, y_T, batch_size=16):
        """
        训练主函数
        支持全量 QP 求解 (solver='cvxopt'/'torch_gd'/'qpth')、
        Mini-batch SGD (solver='batch_sgd') 与 Nyström 快速训练 (solver='fast_nystrom')。

        fast_nystrom 注意事项:
        - 依赖 scikit-learn (用于 KMeans 选路标点)
        - 训练完成后会将 X_train 设置为路标点，beta_primal 为对应核系数，Predictor 可直接复用
        - 使用方式示例:
          model = TL_QLDMR(solver="fast_nystrom", nystrom_n_components=200, nystrom_epochs=50)
          model.fit(X_S, y_S, X_T, y_T)
        """
        # 如果指定了 batch_sgd，直接跳转到 batch 训练逻辑
        if self.solver == "batch_sgd":
            return self._fit_batch_sgd(X_S, y_S, X_T, y_T, batch_size=batch_size)

        # 集成训练方法：fast_nystrom (来自 t 文件的 Nyström+Primal SGD 版本)
        if self.solver == "fast_nystrom":
            return self._fit_fast_nystrom(X_S, y_S, X_T, y_T)

        n_S = len(X_S)
        n_T = len(X_T)
        N = n_S + n_T
        
        # 尝试使用 float32 以节省内存
        dtype = np.float32
        
        if self.use_gpu and cp is None:
            self.use_gpu = False

        # 1. 数据合并与核矩阵计算
        # 使用 try-except 捕获 GPU OOM，如果失败则回退到 CPU
        try:
            if self.use_gpu:
                # 显式转换为 float32
                X_train_gpu = cp.vstack([cp.asarray(X_S, dtype=dtype), cp.asarray(X_T, dtype=dtype)])
                self.X_train = cp.asnumpy(X_train_gpu)
                
                print("1. Computing Kernel Matrix (GPU)...")
                # 计算 K
                K = rbf_kernel(X_train_gpu, X_train_gpu, self.gamma, xp=cp)
                
                print("2. Constructing Optimization Matrices (GPU Optimized)...")
                # 优化内存：不显式构建 S 和 M
                
                # --- 计算 Variance 项: lambda1 * (KS @ K) ---
                # KS = [K[:, :n_S] @ J_S, 0]
                # J_S = I - 1/n_S
                # K_col_S = K[:, :n_S]
                # K_col_S_centered = K_col_S - mean(K_col_S, axis=1)
                K_col_S = K[:, :n_S]
                K_col_S_mean = cp.mean(K_col_S, axis=1, keepdims=True)
                K_col_S_centered = K_col_S - K_col_S_mean
                
                # Term_Var = lambda1 * (K_col_S_centered @ K_col_S_centered.T)
                # 直接加到 Omega，避免存储 Term_Var
                Omega = cp.array(K, copy=True) # Start with K
                
                # Ensure float32 arithmetic
                lambda1_fp = dtype(self.lambda1)
                lambda2_fp = dtype(self.lambda2)
                
                Omega += lambda1_fp * (K_col_S_centered @ K_col_S_centered.T)
                
                # 释放临时变量
                del K_col_S, K_col_S_mean, K_col_S_centered
                cp.get_default_memory_pool().free_all_blocks()
                
                # --- 计算 MMD 项: lambda2 * (KM @ K) ---
                # M = v @ v.T, v = [1/n_S... -1/n_T...]
                # KM @ K = (K v) @ (K v).T
                v_S = cp.full((n_S, 1), 1.0/n_S, dtype=dtype)
                v_T = cp.full((n_T, 1), -1.0/n_T, dtype=dtype)
                v = cp.vstack([v_S, v_T])
                
                Kv = K @ v
                Omega += lambda2_fp * (Kv @ Kv.T)
                
                del v_S, v_T, v, Kv
                cp.get_default_memory_pool().free_all_blocks()
                
                # --- 构建 H ---
                Omega += 1e-6 * cp.eye(N, dtype=dtype)
                
                # H = K @ solve(Omega, K)
                try:
                    H = K @ cp.linalg.solve(Omega, K)
                except cp.linalg.LinAlgError:
                    print("Warning: Singular matrix, using pseudo-inverse")
                    H = K @ cp.linalg.pinv(Omega) @ K
                    
                H_cpu = cp.asnumpy(H)
                
                # 计算 p_dual 所需项 (需小心内存)
                # p_dual = lambda1 * H @ S @ Y - Y
                # S @ Y = [J_S @ Y_S; 0] = [Y_S - mean(Y_S); 0]
                Y_S_gpu = cp.asarray(y_S, dtype=dtype).reshape(-1, 1)
                Y_S_centered = Y_S_gpu - cp.mean(Y_S_gpu)
                
                # padding with zeros for target domain part
                SY = cp.vstack([Y_S_centered, cp.zeros((n_T, 1), dtype=dtype)])
                
                Y_full_gpu = cp.asarray(np.concatenate([y_S, y_T]), dtype=dtype).reshape(-1, 1)
                
                term1 = self.lambda1 * (H @ SY)
                p_dual = term1 - Y_full_gpu
                p_dual = cp.asnumpy(p_dual)
                
            else:
                raise ImportError("Force CPU fallback") # Trigger CPU block

        except (OutOfMemoryError, ImportError, Exception) as e:
            if isinstance(e, OutOfMemoryError):
                print(f"GPU Out of Memory ({e}). Falling back to CPU...")
            elif str(e) == "Force CPU fallback":
                pass # Normal flow for no-GPU
            else:
                print(f"GPU Error ({e}). Falling back to CPU...")
            
            self.use_gpu = False
            cp_module = np # use numpy as xp
            
            # CPU 实现 (同样使用 float32 和优化矩阵乘法)
            self.X_train = np.vstack([X_S, X_T]).astype(dtype)
            y_train = np.concatenate([y_S, y_T]).astype(dtype)
            
            print("1. Computing Kernel Matrix (CPU)...")
            K = rbf_kernel(self.X_train, self.X_train, self.gamma, xp=np).astype(dtype)
            
            print("2. Constructing Optimization Matrices (CPU)...")
            # --- Variance Term ---
            K_col_S = K[:, :n_S]
            K_col_S_mean = np.mean(K_col_S, axis=1, keepdims=True)
            K_col_S_centered = K_col_S - K_col_S_mean
            
            Omega = K.copy()
            
            lambda1_fp = dtype(self.lambda1)
            lambda2_fp = dtype(self.lambda2)
            
            Omega += lambda1_fp * (K_col_S_centered @ K_col_S_centered.T)
            
            del K_col_S, K_col_S_mean, K_col_S_centered
            gc.collect()
            
            # --- MMD Term ---
            v_S = np.full((n_S, 1), 1.0/n_S, dtype=dtype)
            v_T = np.full((n_T, 1), -1.0/n_T, dtype=dtype)
            v = np.vstack([v_S, v_T])
            
            Kv = K @ v
            Omega += lambda2_fp * (Kv @ Kv.T)
            
            del v_S, v_T, v, Kv
            gc.collect()
            
            Omega += 1e-6 * np.eye(N, dtype=dtype)
            
            try:
                H = K @ np.linalg.solve(Omega, K)
            except np.linalg.LinAlgError:
                print("Warning: Singular matrix, using pseudo-inverse")
                H = K @ np.linalg.pinv(Omega) @ K
                
            H_cpu = H
            
            # p_dual calculation
            Y_S_cpu = y_S.astype(dtype).reshape(-1, 1)
            Y_S_centered = Y_S_cpu - np.mean(Y_S_cpu)
            SY = np.vstack([Y_S_centered, np.zeros((n_T, 1), dtype=dtype)])
            
            Y_full = y_train.reshape(-1, 1)
            term1 = self.lambda1 * (H @ SY)
            p_dual = term1 - Y_full

        # 5. 构建 QP 标准型
        # min 1/2 x^T P x + q^T x
        P_np = np.block([[H_cpu, -H_cpu], [-H_cpu, H_cpu]])
        P_np = 0.5 * (P_np + P_np.T)
        
        q_np = np.vstack([p_dual, -p_dual]).reshape(-1)
        
        # 6. 构造约束条件
        # 不等式约束 Gx <= h (Box Constraints)
        # 0 <= alpha <= C_tau
        # 0 <= alpha* <= C(1-tau)
        
        # 生成 C 向量 (前 n_S 个是 C_S, 后 n_T 个是 C_T)
        C_vec = np.zeros(N)
        C_vec[:n_S] = self.C_S
        C_vec[n_S:] = self.C_T
        
        # 上界向量
        upper_bound_alpha = C_vec * self.tau
        upper_bound_alpha_star = C_vec * (1 - self.tau)
        
        # 构造 G, h
        # -alpha <= 0
        # -alpha* <= 0
        # alpha <= upper
        # alpha* <= upper_star
        diag_I = np.eye(2*N)
        G_np = np.vstack([-diag_I, diag_I])
        
        lower_bound = np.zeros(2*N)
        upper_bound = np.concatenate([upper_bound_alpha, upper_bound_alpha_star])

        h_vec = np.concatenate([-lower_bound, upper_bound])
        
        # 等式约束 Ax = b
        # sum(alpha - alpha*) = 0  => [1, ..., 1, -1, ..., -1] @ x = 0
        A_np = np.hstack([np.ones(N), -np.ones(N)]).reshape(1, -1)
        b_np = np.zeros(1)
        
        # 7. 求解 QP
        print("3. Solving QP...")
        if self.solver == "cvxopt":
            print("3.1. Solving QP with cvxopt...")
            P = cvxopt.matrix(P_np.astype(np.float64))
            q_vec = cvxopt.matrix(q_np.astype(np.float64))
            G = cvxopt.matrix(G_np.astype(np.float64))
            h = cvxopt.matrix(h_vec.astype(np.float64))
            A = cvxopt.matrix(A_np.astype(np.float64))
            b = cvxopt.matrix(b_np.astype(np.float64))
            try:
                start_time = time.perf_counter()
                sol = cvxopt.solvers.qp(P, q_vec, G, h, A, b)
                qp_time = time.perf_counter() - start_time
            except ValueError as e:
                print(f"QP Solver Failed: {e}")
                return False
            x_sol = np.array(sol['x']).flatten()
            self.qp_stats = self._compute_qp_stats(
                x_sol, P_np, q_np, lower_bound, upper_bound, A_np, b_np, qp_time, "cvxopt"
            )
        elif self.solver == "torch_gd":
            print("3.2. Solving QP with torch_gd...")
            x_sol, self.qp_stats = self._solve_qp_torch_gd(
                P_np, q_np, lower_bound, upper_bound, A_np, b_np
            )
        elif self.solver == "qpth":
            print("3.3. Solving QP with qpth...")
            x_sol, self.qp_stats = self._solve_qp_qpth(
                P_np, q_np, G_np, h_vec, A_np, b_np, lower_bound, upper_bound
            )
        else:
            raise ValueError(f"Unsupported solver: {self.solver}")

        # 8. 提取结果
        self.alpha = x_sol[:N]
        self.alpha_star = x_sol[N:]
        
        # 计算系数 beta = alpha - alpha* (用于预测)
        # 但在 TL-LDMR 中，根据推导，真实的系数 beta_primal (核函数的系数)
        # 是 beta_primal = Omega_inv @ (K @ (alpha - alpha*) + ...) 
        # 为了简化，我们可以直接用对偶变量恢复预测函数 f(x)
        # f(x) = K(x, X) @ beta_primal + b
        
        # 恢复 Primal 系数 beta_primal
        gamma_vec = (self.alpha - self.alpha_star).reshape(-1, 1) # gamma = alpha - alpha*
        if self.use_gpu:
            gamma_vec_gpu = cp.asarray(gamma_vec)
            rhs = K @ gamma_vec_gpu + self.lambda1 * (K @ SY)
            beta_primal = cp.linalg.solve(Omega, rhs).flatten()
            self.beta_primal = cp.asnumpy(beta_primal)
        else:
            rhs = K @ gamma_vec + self.lambda1 * (K @ SY)
            self.beta_primal = np.linalg.solve(Omega, rhs).flatten()
        
        # 9. 计算偏置 b
        # 选择支持向量 (0 < alpha < C_tau)
        # 理论上 b = y_k - K_k @ beta_primal (对于支持向量)
        sv_indices = np.where((self.alpha > 1e-5) & (self.alpha < (C_vec * self.tau - 1e-5)))[0]
        
        if len(sv_indices) > 0:
            if self.use_gpu:
                y_train_gpu = cp.asarray(y_train)
                K_sv = K[sv_indices]
                pred_no_b = K_sv @ cp.asarray(self.beta_primal)
                b_values = y_train_gpu[sv_indices] - pred_no_b
                self.b = float(cp.mean(b_values).get())
            else:
                pred_no_b = K[sv_indices] @ self.beta_primal
                b_values = y_train[sv_indices] - pred_no_b
                self.b = float(np.mean(b_values))
        else:
            self.b = 0.0
            print("Warning: No strict support vectors found, bias set to 0.")
            
        if self.qp_stats is not None:
            eq_violation = self.qp_stats.get("eq_violation", 0.0)
            box_violation = self.qp_stats.get("box_violation", 0.0)
            if max(eq_violation, box_violation) > 1e-6:
                print("Warning: Constraint violation exceeds 1e-6.")
            print(
                f"QP Solver [{self.qp_stats.get('solver')}]: "
                f"time={self.qp_stats.get('qp_time'):.4f}s, "
                f"eq={eq_violation:.2e}, box={box_violation:.2e}"
            )

        print("Training Completed.")
        return True

    def _get_torch_device(self):
        if torch is None:
            raise ImportError("PyTorch is required but not installed.")
        if self.torch_device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.torch_device)

    def _project_box_equality(self, v, lower, upper, s, target, max_iter, tol):
        # 投影到 {lower <= x <= upper, s^T x = target}
        low_candidates = torch.where(s > 0, v - upper, lower - v)
        high_candidates = torch.where(s > 0, v - lower, upper - v)
        low = torch.min(low_candidates)
        high = torch.max(high_candidates)

        x = None
        for _ in range(max_iter):
            mid = 0.5 * (low + high)
            x = torch.minimum(torch.maximum(v - mid * s, lower), upper)
            residual = torch.dot(s, x) - target
            if torch.abs(residual) <= tol:
                break
            if residual > 0:
                low = mid
            else:
                high = mid
        return x

    def _compute_qp_stats(self, x, P, q, lower, upper, A, b, qp_time, solver_name):
        x = x.reshape(-1)
        obj = 0.5 * x @ (P @ x) + q @ x
        eq_violation = float(np.max(np.abs(A @ x - b)))
        lower_violation = np.max(lower - x)
        upper_violation = np.max(x - upper)
        box_violation = float(max(0.0, lower_violation, upper_violation))
        return {
            "solver": solver_name,
            "qp_time": float(qp_time),
            "objective": float(obj),
            "eq_violation": eq_violation,
            "box_violation": box_violation,
        }

    def _solve_qp_torch_gd(self, P, q, lower, upper, A, b):
        if torch is None:
            raise ImportError("PyTorch is required for torch_gd solver.")

        device = self._get_torch_device()
        dtype = torch.float64

        P_t = torch.tensor(P, dtype=dtype, device=device)
        q_t = torch.tensor(q, dtype=dtype, device=device)
        lower_t = torch.tensor(lower, dtype=dtype, device=device)
        upper_t = torch.tensor(upper, dtype=dtype, device=device)

        n = q_t.numel()
        n_half = n // 2
        s = torch.cat([
            torch.ones(n_half, dtype=dtype, device=device),
            -torch.ones(n_half, dtype=dtype, device=device),
        ])
        target = torch.tensor(float(b[0]), dtype=dtype, device=device)

        x = torch.zeros_like(q_t)
        x = self._project_box_equality(
            x, lower_t, upper_t, s, target, self.torch_proj_max_iter, self.torch_proj_tol
        )

        start_time = time.perf_counter()
        grad_norm = None
        iterations = 0
        for step in range(self.torch_max_iter):
            iterations = step + 1
            x = x.detach().requires_grad_(True)
            obj = 0.5 * x @ (P_t @ x) + q_t @ x
            grad = torch.autograd.grad(obj, x, create_graph=False)[0]
            grad_norm_val = torch.norm(grad).item()
            grad_norm = grad_norm_val
            if grad_norm_val < self.torch_tol:
                break
            lr = self.torch_lr / (1.0 + self.torch_lr_decay * step)
            with torch.no_grad():
                x = x - lr * grad
                x = self._project_box_equality(
                    x, lower_t, upper_t, s, target, self.torch_proj_max_iter, self.torch_proj_tol
                )

        qp_time = time.perf_counter() - start_time
        x_np = x.detach().cpu().numpy()
        stats = self._compute_qp_stats(x_np, P, q, lower, upper, A, b, qp_time, "torch_gd")
        stats["iterations"] = iterations
        stats["grad_norm"] = float(grad_norm) if grad_norm is not None else None
        return x_np, stats

    def _fit_batch_sgd(self, X_S, y_S, X_T, y_T, batch_size=16):
        """
        TL-QLDMR 的 Mini-batch SGD 实现（核切片训练）。
        使用 PyTorch autograd 在核系数空间直接优化，避免显式构造完整核矩阵（O(N^2)）。
        
        模型形式：f(x) = sum_i w_i * k(x_i, x) + b
        """
        if torch is None:
            raise ImportError("PyTorch is required for batch_sgd solver.")
            
        print(f"Starting Mini-batch SGD Training (Batch Size: {batch_size})...")
        device = self._get_torch_device()
        print(f"Using device: {device}")

        # 1) 数据准备
        # 尽量将 X_train 保持在 GPU 上（若可用），但不预先计算 K（会占用巨大内存）
        X_S_np = np.array(X_S, dtype=np.float32)
        y_S_np = np.array(y_S, dtype=np.float32).reshape(-1, 1)
        X_T_np = np.array(X_T, dtype=np.float32)
        y_T_np = np.array(y_T, dtype=np.float32).reshape(-1, 1)
        
        # 拼接得到核基（用于后续切片计算 K(X_batch, X_train)）
        X_train_np = np.vstack([X_S_np, X_T_np])
        N = X_train_np.shape[0]
        n_S = X_S_np.shape[0]
        
        # 将数据移动到训练设备（cpu/cuda）
        X_train_t = torch.tensor(X_train_np, device=device)
        X_S_t = torch.tensor(X_S_np, device=device)
        y_S_t = torch.tensor(y_S_np, device=device)
        X_T_t = torch.tensor(X_T_np, device=device)
        y_T_t = torch.tensor(y_T_np, device=device)
        
        # 保存 X_train 供预测阶段使用（保持 numpy，兼容 Predictor 的 CPU/GPU 分支）
        self.X_train = X_train_np
        
        # 2) 参数初始化
        # w 是每个训练样本对应的核系数，b 为偏置
        # f(x) = sum_i w_i * k(x_i, x) + b
        # w 用较小随机数初始化，且必须是 leaf tensor 才能被优化器更新
        w = (torch.randn(N, 1, device=device) * 0.001).detach().requires_grad_(True)
        b = torch.zeros(1, device=device, requires_grad=True)
        
        optimizer = torch.optim.Adam([w, b], lr=self.torch_lr)
        # 构建数据集
        ds_S = torch.utils.data.TensorDataset(X_S_t, y_S_t)
        ds_T = torch.utils.data.TensorDataset(X_T_t, y_T_t)
        
        # 构建 DataLoader（源域与目标域分别采样）
        dl_S = torch.utils.data.DataLoader(ds_S, batch_size=batch_size, shuffle=True, drop_last=True)
        dl_T = torch.utils.data.DataLoader(ds_T, batch_size=batch_size, shuffle=True, drop_last=True)
        
        # 每个 epoch 的步数取源域/目标域较小者，保证每步都有两域 batch
        steps_per_epoch = min(len(dl_S), len(dl_T))
        epochs = max(1, self.torch_max_iter // steps_per_epoch)
        
        print(f"Training for {epochs} epochs ({steps_per_epoch} steps/epoch)...")
        
        # 在确定 epochs 后初始化学习率调度器
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
        
        gamma = self.gamma
        
        def pinball_loss(preds, targets, tau):
            diff = targets - preds
            mask = (diff >= 0).float()
            loss = diff * (tau * mask + (tau - 1) * (1 - mask))
            return loss.mean()
            
        def compute_kernel_slice(X_batch, X_all, gamma):
            # 计算 RBF 核切片：K(X_batch, X_all)
            # ||x-y||^2 = ||x||^2 + ||y||^2 - 2xy
            # X_batch: (B, D), X_all: (N, D)
            
            x_sq = torch.sum(X_batch**2, dim=1, keepdim=True) # (B, 1)
            y_sq = torch.sum(X_all**2, dim=1, keepdim=True).t() # (1, N)
            
            # (B, N)
            dist_sq = x_sq + y_sq - 2.0 * torch.matmul(X_batch, X_all.t())
            return torch.exp(-gamma * dist_sq)

        start_time = time.perf_counter()
        
        for epoch in range(epochs):
            iter_S = iter(dl_S)
            iter_T = iter(dl_T)
            
            epoch_loss = 0.0
            
            for _ in range(steps_per_epoch):
                try:
                    x_s_batch, y_s_batch = next(iter_S)
                    x_t_batch, y_t_batch = next(iter_T)
                except StopIteration:
                    break
                
                optimizer.zero_grad()
                
                # --- 1) 核切片计算 ---
                # 计算 K(X_batch, X_train) 得到当前 batch 的预测
                # 复杂度 O(Batch * N)，通常远小于构造完整 K
                K_S = compute_kernel_slice(x_s_batch, X_train_t, gamma) # (B, N)
                K_T = compute_kernel_slice(x_t_batch, X_train_t, gamma) # (B, N)
                
                # --- 2) 预测 ---
                pred_S = K_S @ w + b
                pred_T = K_T @ w + b
                
                # --- 3. Losses ---
                # Pinball Loss
                loss_S = pinball_loss(pred_S, y_s_batch, self.tau)
                loss_T = pinball_loss(pred_T, y_t_batch, self.tau)
                
                # MMD Loss (Batch approximation)
                # Simple empirical MMD between current batches
                # This is a rough approximation but works for SGD
                # MMD = || mean(phi(Xs)) - mean(phi(Xt)) ||^2
                # Here phi(x) is the feature map. In kernel trick:
                # MMD^2 = mean(K(Xs, Xs)) + mean(K(Xt, Xt)) - 2*mean(K(Xs, Xt))
                # We compute this only on the current batch
                K_SS = compute_kernel_slice(x_s_batch, x_s_batch, gamma)
                K_TT = compute_kernel_slice(x_t_batch, x_t_batch, gamma)
                K_ST = compute_kernel_slice(x_s_batch, x_t_batch, gamma)
                
                mmd_loss = K_SS.mean() + K_TT.mean() - 2 * K_ST.mean()
                
                # Regularization (L2 on w)
                # Corresponds to ||f||_H^2 approx ||w||^2
                reg_loss = torch.sum(w**2)
                
                # Total Loss
                loss = self.C_S * loss_S + self.C_T * loss_T + \
                       self.lambda2 * mmd_loss + self.lambda1 * reg_loss
                
                loss.backward()
                
                # Gradient Clipping to prevent explosion
                torch.nn.utils.clip_grad_norm_([w, b], max_norm=1.0)
                
                optimizer.step()
                
                epoch_loss += loss.item()
            
            scheduler.step()
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/steps_per_epoch:.4f}")

        # 4. Finalize
        self.beta_primal = w.detach().cpu().numpy().flatten()
        self.b = b.item()
        self.qp_stats = {
            "solver": "batch_sgd",
            "qp_time": time.perf_counter() - start_time,
            "final_loss": epoch_loss/steps_per_epoch
        }
        
        print("Training Completed (Batch SGD).")
        return True

    def _fit_fast_nystrom(self, X_S, y_S, X_T, y_T):
        """
        fast_nystrom: 使用 Nyström 近似将核映射到低维空间，再用 Primal Adam 训练。

        为了兼容 Predictor:
        - X_train 设为 Nyström 路标点 (landmarks)
        - beta_primal 设为 landmarks 对应的核系数，使得 f(x)=K(x,landmarks)@beta_primal + b
        """
        if KMeans is None:
            raise ImportError("fast_nystrom solver requires scikit-learn.")

        solver = Fast_TL_QLDMR(
            lambda1=self.lambda1,
            lambda2=self.lambda2,
            C_S=self.C_S,
            C_T=self.C_T,
            tau=self.tau,
            kernel_gamma=self.gamma,
            n_components=self.nystrom_n_components,
            learning_rate=self.nystrom_lr,
            epochs=self.nystrom_epochs,
            batch_size=self.nystrom_batch_size,
            variance_mode=self.variance_mode,
            asym_scale=self.asym_scale,
        )

        start_time = time.perf_counter()
        try:
            ok = solver.fit(X_S, y_S, X_T, y_T)
        except Exception as e:
            print(f"fast_nystrom training failed: {e}")
            return False

        if not ok:
            return False

        normalization = solver.feature_map.normalization_
        w = solver.w.reshape(-1, 1)
        beta_landmarks = (normalization @ w).reshape(-1)

        self.beta_primal = beta_landmarks.astype(np.float32)
        self.b = float(solver.b)
        self.X_train = np.asarray(solver.feature_map.components_, dtype=np.float32)
        self.qp_stats = {
            "solver": "fast_nystrom",
            "qp_time": float(time.perf_counter() - start_time),
            "n_components": int(self.X_train.shape[0]),
            "epochs": int(self.nystrom_epochs),
            "batch_size": int(self.nystrom_batch_size),
        }

        print("Training Completed (fast_nystrom).")
        return True

    def _solve_qp_qpth(self, P, q, G, h, A, b, lower, upper):
        if torch is None:
            raise ImportError("PyTorch is required for qpth solver.")
        if QPFunction is None:
            raise ImportError("qpth is required for qpth solver.")

        device = self._get_torch_device()
        dtype = torch.float64

        P_pd = P + 1e-8 * np.eye(P.shape[0])

        P_t = torch.tensor(P_pd, dtype=dtype, device=device)
        q_t = torch.tensor(q, dtype=dtype, device=device)
        G_t = torch.tensor(G, dtype=dtype, device=device)
        h_t = torch.tensor(h, dtype=dtype, device=device)
        A_t = torch.tensor(A, dtype=dtype, device=device)
        b_t = torch.tensor(b, dtype=dtype, device=device)

        start_time = time.perf_counter()
        solver = QPFunction(verbose=False, eps=1e-12, maxIter=200, notImprovedLim=20)
        x = solver(P_t, q_t, G_t, h_t, A_t, b_t)
        qp_time = time.perf_counter() - start_time

        x_np = x.detach().cpu().numpy()
        stats = self._compute_qp_stats(x_np, P, q, lower, upper, A, b, qp_time, "qpth")
        return x_np, stats
