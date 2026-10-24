"""
TL-QLDMR 训练器实现

参考与用途说明：
- Kernel_Function.rbf_kernel：统一的 RBF 核计算（支持 numpy/cupy），用于对偶 QP 与预测
- /mnt/mydisk/zhangxiaohan/lin/TL-QLDMR/t：Nyström 近似 + Primal Adam 的快速训练方法来源
  用途：在 solver='fast_nystrom' 下避免构造完整核矩阵（降低 O(N^2) 内存）

模块板块说明：
- Nyström 近似（NystromApproximation / Fast_TL_QLDMR）：将核问题映射到低维特征空间并训练线性模型
- TL_QLDMR 主类：统一入口，按 solver 选择训练路径
  - cvxopt/qpth：对偶 QP 路径，适合中小规模数据
  - torch_gd：仅保留为未认证的约束 QP 近似迭代，不作为精确求解证据
  - batch_sgd：保留历史代码供追溯；该分支不支持修正后的目标函数
  - fast_nystrom：Nyström + Primal Adam，适合更大规模数据与更低内存场景
"""

import time
import warnings
import numpy as np
import cvxopt
from Kernel_Function import rbf_kernel

# 训练器 TL_QLDMR 支持以下 solver:
# - cvxopt / qpth: 对偶 QP 解法；cvxopt 是默认的未加 ridge 精确求解器
# - torch_gd: 未认证的投影梯度近似迭代
# - batch_sgd: 保留历史代码；修正目标不允许使用此分支
# - fast_nystrom: 集成自 t 的 Nyström+Primal Adam 快速训练（更省内存）

try:
    import cupy as cp
except ImportError:
    cp = None

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


def _validate_objective_parameters(lambda1, lambda2, C_S, C_T, tau, gamma,
                                   variance_mode, asym_scale):
    for name, value in (("lambda1", lambda1), ("lambda2", lambda2),
                        ("C_S", C_S), ("C_T", C_T)):
        if not np.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and nonnegative.")
    if C_S + C_T <= 0:
        raise ValueError("At least one pinball-loss weight must be positive.")
    if not np.isfinite(tau) or not 0 < tau < 1:
        raise ValueError("tau must lie strictly between 0 and 1.")
    if not np.isfinite(gamma) or gamma <= 0:
        raise ValueError("kernel_gamma must be finite and positive.")
    if variance_mode not in {"centered_residual", "symmetric", "asymmetric"}:
        raise ValueError(f"Unsupported variance_mode: {variance_mode}")
    if not np.isfinite(asym_scale) or asym_scale < 0:
        raise ValueError("asym_scale must be finite and nonnegative.")
    if variance_mode != "asymmetric" and asym_scale != 0:
        raise ValueError("asym_scale is only used by the legacy asymmetric mode.")


def _validate_training_data(X_S, y_S, X_T, y_T, dtype=np.float64):
    X_S, X_T = np.asarray(X_S, dtype=dtype), np.asarray(X_T, dtype=dtype)
    y_S = np.asarray(y_S, dtype=dtype).reshape(-1)
    y_T = np.asarray(y_T, dtype=dtype).reshape(-1)
    if X_S.ndim != 2 or X_T.ndim != 2 or X_S.shape[1] != X_T.shape[1]:
        raise ValueError("Source and target features must be 2D with matching columns.")
    if X_T.shape[0] == 0 or X_T.shape[1] == 0:
        raise ValueError("A non-empty target feature matrix is required.")
    if len(X_S) != len(y_S) or len(X_T) != len(y_T):
        raise ValueError("Feature and label counts must match within each domain.")
    if not all(np.isfinite(a).all() for a in (X_S, y_S, X_T, y_T)):
        raise ValueError("Training features and labels must be finite.")
    return X_S, y_S, X_T, y_T


def _solve_psd(matrix, rhs, xp=np):
    """Solve on the numerical range, including singular Gram matrices.

    The same Moore--Penrose solution is used for the dual matrix and primal
    recovery. No diagonal ridge is silently added to the stated objective.
    """
    values, vectors = xp.linalg.eigh((matrix + matrix.T) / 2)
    scale = float(xp.max(xp.abs(values)).item())
    tolerance = np.finfo(np.float64).eps * matrix.shape[0] * max(scale, 1.0) * 10
    if float(xp.min(values).item()) < -tolerance:
        raise ValueError("The kernel objective matrix is not positive semidefinite.")
    keep = values > tolerance
    basis = vectors[:, keep]
    solution = basis @ ((basis.T @ rhs) / values[keep, None])
    return solution, int(xp.sum(keep).item())


def _kernel_objective_system(K, y_S, n_T, lambda1, lambda2, xp=np):
    """Normalized centered-residual kernel quadratic and dual system."""
    n_S = len(y_S)
    source_columns = K[:, :n_S]
    centered = source_columns - xp.mean(source_columns, axis=1, keepdims=True)
    variance_weight = lambda1 / n_S
    mean_weights = xp.concatenate((xp.full(n_S, 1.0 / n_S),
                                   xp.full(n_T, -1.0 / n_T)))
    projected_difference = K @ mean_weights
    Omega = (K + variance_weight * (centered @ centered.T)
             + lambda2 * xp.outer(projected_difference, projected_difference))
    SY = xp.concatenate((y_S - xp.mean(y_S), xp.zeros(n_T)))
    inverse_times_K, rank = _solve_psd(Omega, K, xp=xp)
    H = K @ inverse_times_K
    H = (H + H.T) / 2
    return inverse_times_K, H, variance_weight * SY, rank


def _weighted_quantile(values, weights, tau):
    """Minimize the weighted pinball loss over an unregularized intercept."""
    positive = weights > 0
    values, weights = np.asarray(values)[positive], np.asarray(weights)[positive]
    order = np.argsort(values, kind="stable")
    cumulative = np.cumsum(weights[order])
    index = np.searchsorted(cumulative, tau * cumulative[-1], side="left")
    return float(values[order[min(index, len(order) - 1)]])

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
        # A failed refit must not retain the normalization of earlier landmarks.
        self.normalization_ = None
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
            stabilized = K_mm + 1e-6 * np.eye(len(K_mm))
            U, S, _ = np.linalg.svd(stabilized)
            S_inv_sqrt = np.diag(1.0 / np.sqrt(S))
            self.normalization_ = U @ S_inv_sqrt @ U.T
        except np.linalg.LinAlgError as error:
            raise np.linalg.LinAlgError(
                "Nyström SVD failed; cannot construct the stabilized inverse "
                "square root. No alternative feature map was substituted."
            ) from error

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
    [Fast Solver] 基于 Nyström 近似 + Primal Adam 的快速 TL-QLDMR
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
        variance_mode="centered_residual",
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

        _validate_objective_parameters(lambda1, lambda2, C_S, C_T, tau,
                                       kernel_gamma, self.variance_mode, self.asym_scale)
        for name, value in (("n_components", n_components), ("epochs", epochs),
                            ("batch_size", batch_size)):
            if not isinstance(value, (int, np.integer)) or value <= 0:
                raise ValueError(f"{name} must be a positive integer.")
        if not np.isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive.")
        self.w = None
        self.b = 0.0
        self.feature_map = None

    def _compute_regularizer_terms(self, Z_S, Z_T, y_var=None):
        """
        Return R and c for .5 w^T R w - c^T w (up to a constant).

        centered_residual uses lambda1/(2 n_S) ||H_S(y_S-Z_S w)||^2.
        symmetric/asymmetric retain the historical prediction-variance
        penalties, with c=0; they are not residual-variance models.
        """
        m = Z_S.shape[1]
        if len(Z_S) == 0 or len(Z_T) == 0:
            raise ValueError("Regularizer construction requires non-empty domains.")
        R = np.eye(m)
        c = np.zeros(m)

        if self.variance_mode in {"centered_residual", "asymmetric"}:
            if y_var is None or len(np.asarray(y_var).reshape(-1)) != len(Z_S):
                raise ValueError(f"{self.variance_mode} requires one source label per row.")
            y_var = np.asarray(y_var).reshape(-1)

        Z_S_mean = np.mean(Z_S, axis=0)
        Z_S_centered = Z_S - Z_S_mean

        if self.variance_mode == "asymmetric":
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
        if self.variance_mode == "centered_residual":
            c = (self.lambda1 / len(Z_S)) * (
                Z_S_centered.T @ (y_var - np.mean(y_var)))

        diff = np.mean(Z_S, axis=0) - np.mean(Z_T, axis=0)
        M_z = np.outer(diff, diff)
        R += self.lambda2 * M_z

        return R, c

    def _compute_regularizer_matrix(self, Z_S, Z_T, y_var=None):
        """Compatibility accessor; optimization must also use the linear term."""
        return self._compute_regularizer_terms(Z_S, Z_T, y_var)[0]

    def _batch_gradients(self, z_batch, y_batch, c_batch, R, c, n_samples):
        residuals = y_batch - (z_batch @ self.w + self.b)
        grad_f = np.where(residuals >= 0, -self.tau, 1.0 - self.tau) * c_batch
        # E[g_B] = (B/N) grad J for the SUM-loss objective. This scaling
        # preserves the relative regularization strength for any batch size.
        scale = len(y_batch) / n_samples
        return (z_batch.T @ grad_f + scale * (R @ self.w - c),
                float(np.sum(grad_f)))

    def fit(self, X_S, y_S, X_T, y_T):
        """
        使用固定学习率 Adam 优化显式特征空间目标。
        """
        X_S, y_S, X_T, y_T = _validate_training_data(
            X_S, y_S, X_T, y_T, dtype=np.float32)
        if self.variance_mode != "centered_residual":
            warnings.warn(
                f"variance_mode={self.variance_mode!r} is a legacy prediction-variance "
                "penalty, not the revised centered-residual objective.",
                UserWarning, stacklevel=2)
        if self.C_S * len(X_S) + self.C_T * len(X_T) <= 0:
            raise ValueError("The supplied samples must have positive total loss weight.")

        print(f"1. Nyström Mapping (m={self.n_components})...")
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
            R_matrix, c_vector = self._compute_regularizer_terms(Z_S, Z_T, y_var=y_S)

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
            # Explicit target-only ablation: use target residual variance and
            # identical domain means, so the domain penalty vanishes.
            R_matrix, c_vector = self._compute_regularizer_terms(Z_S, Z_T, y_var=y_T)

            Z_train = Z_T
            y_train = y_T
            C_vec = np.full(len(X_T), self.C_T, dtype=np.float32)

        print(f"3. Optimizing with Adam (Epochs={self.epochs})...")

        n_samples, n_features = Z_train.shape
        self.w = np.zeros(n_features, dtype=np.float32)
        self.b = 0.0
        self.regularizer_matrix_ = R_matrix
        self.regularizer_linear_ = c_vector

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

                grad_w, grad_b = self._batch_gradients(
                    z_batch, y_batch, c_batch, R_matrix, c_vector, n_samples)

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
        solver="cvxopt",
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
        variance_mode="centered_residual",
        asym_scale=0.0,
    ):
        """
        Args:
            lambda1: 分布方差项 (LDMR Variance) 权重
            lambda2: 投影预测均值差异项权重
            C_S: 源域 Pinball Loss 权重
            C_T: 目标域 Pinball Loss 权重 (通常设大，防止被源域淹没)
            tau: 分位数 (0.05, 0.5, 0.95 等)
            kernel_gamma: RBF核参数
            use_gpu: 是否启用 GPU (CUPY - CUDA)
            solver: QP 求解器 ("cvxopt", "qpth")、未认证近似迭代
                "torch_gd"，或 "batch_sgd"/"fast_nystrom"；默认使用 cvxopt
            torch_device: "auto"/"cpu"/"cuda" 或 "cuda:0"
            torch_lr: PyTorch 梯度下降初始学习率
            torch_lr_decay: 学习率衰减系数
            torch_max_iter: 最大迭代次数
            torch_tol: torch_gd 原始梯度范数停止阈值（不是约束 KKT 证书）
            torch_proj_tol: 约束投影的误差阈值
            torch_proj_max_iter: 约束投影的最大迭代次数
            nystrom_n_components: fast_nystrom 的路标点数 (m)
            nystrom_lr: fast_nystrom 的学习率
            nystrom_epochs: fast_nystrom 的训练轮数
            nystrom_batch_size: fast_nystrom 的 batch 大小
            variance_mode: "centered_residual" implements the revised objective.
                "symmetric"/"asymmetric" are legacy prediction-variance modes,
                available only with fast_nystrom for historical reproducibility.
            asym_scale: legacy asymmetric prediction-variance weight amplitude.
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
        _validate_objective_parameters(lambda1, lambda2, C_S, C_T, tau,
                                       kernel_gamma, self.variance_mode, self.asym_scale)
        
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
        支持默认的未加 ridge CVXOPT 对偶 QP、带显式数值 ridge 的 qpth、
        未认证的 torch_gd 近似迭代，以及 Nyström 快速训练
        (solver='fast_nystrom')。
        历史 batch_sgd 不支持修正目标，显式拒绝以避免混淆。

        fast_nystrom 注意事项:
        - 依赖 scikit-learn (用于 KMeans 选路标点)
        - 训练完成后会将 X_train 设置为路标点，beta_primal 为对应核系数，Predictor 可直接复用
        - 使用方式示例:
          model = TL_QLDMR(solver="fast_nystrom", nystrom_n_components=200, nystrom_epochs=50)
          model.fit(X_S, y_S, X_T, y_T)
        """
        X_S, y_S, X_T, y_T = _validate_training_data(X_S, y_S, X_T, y_T)
        if self.solver == "batch_sgd":
            raise ValueError(
                "The historical batch_sgd branch does not implement the revised "
                "centered-residual objective. Use fast_nystrom or an exact QP solver.")
        if self.solver == "fast_nystrom":
            return self._fit_fast_nystrom(X_S, y_S, X_T, y_T)
        if self.solver not in {"cvxopt", "torch_gd", "qpth"}:
            raise ValueError(f"Unsupported solver: {self.solver}")
        if self.variance_mode != "centered_residual":
            raise ValueError(
                "Exact QP solvers support only variance_mode='centered_residual'; "
                "legacy prediction-variance modes require fast_nystrom.")
        if len(X_S) == 0:
            raise ValueError("Exact QP requires a source domain; use fast_nystrom for target-only ablation.")

        n_S, n_T = len(X_S), len(X_T)
        N = n_S + n_T
        self.X_train = np.vstack([X_S, X_T])
        y_train = np.concatenate([y_S, y_T])
        # Float64 and a range-space pseudoinverse keep singular kernels convex
        # without silently adding a coefficient-space ridge to the objective.
        self.use_gpu = self.use_gpu and cp is not None
        if self.use_gpu:
            try:
                print("1. Computing Kernel Objective (GPU)...")
                X_gpu = cp.asarray(self.X_train, dtype=cp.float64)
                K = rbf_kernel(X_gpu, X_gpu, self.gamma, xp=cp)
                inverse_times_K, H, linear_source, rank = _kernel_objective_system(
                    K, cp.asarray(y_S), n_T, self.lambda1, self.lambda2, xp=cp)
                H_cpu = cp.asnumpy(H)
                p_dual = cp.asnumpy(H @ linear_source) - y_train
            except Exception as error:
                print(f"GPU objective construction failed ({error}); falling back to CPU.")
                self.use_gpu = False
        if not self.use_gpu:
            print("1. Computing Kernel Objective (CPU)...")
            K = rbf_kernel(self.X_train, self.X_train, self.gamma, xp=np)
            inverse_times_K, H_cpu, linear_source, rank = _kernel_objective_system(
                K, y_S, n_T, self.lambda1, self.lambda2)
            p_dual = H_cpu @ linear_source - y_train

        # 5. 构建 QP 标准型
        # min 1/2 x^T P x + q^T x
        P_np = np.block([[H_cpu, -H_cpu], [-H_cpu, H_cpu]])
        P_np = 0.5 * (P_np + P_np.T)
        
        q_np = np.concatenate([p_dual, -p_dual])
        
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
            if sol['status'] != 'optimal':
                raise RuntimeError(f"CVXOPT did not reach an optimum: {sol['status']}")
            x_sol = np.array(sol['x']).flatten()
            self.qp_stats = self._compute_qp_stats(
                x_sol, P_np, q_np, lower_bound, upper_bound, A_np, b_np, qp_time, "cvxopt"
            )
        elif self.solver == "torch_gd":
            print("3.2. Solving QP with torch_gd...")
            warnings.warn(
                "torch_gd returns an uncertified approximate constrained-QP iterate; "
                "its raw-gradient stopping rule is not a projected-KKT certificate. "
                "Use solver='cvxopt' for exact-kernel validation claims.",
                RuntimeWarning,
                stacklevel=2,
            )
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
        
        # Reuse exactly the same range-space inverse as the dual system.
        # This also supports duplicate samples / singular Gram matrices.
        gamma_vec = self.alpha - self.alpha_star
        if self.use_gpu:
            beta_primal = inverse_times_K @ (cp.asarray(gamma_vec) + linear_source)
            self.beta_primal = cp.asnumpy(beta_primal)
            pred_no_b = cp.asnumpy(K @ beta_primal)
        else:
            self.beta_primal = inverse_times_K @ (gamma_vec + linear_source)
            pred_no_b = K @ self.beta_primal

        # The centered variance and projected predictive-mean discrepancy do not depend on b. For a
        # fixed coefficient vector, its exact minimizer is a weighted residual
        # quantile, even when no dual variable is strictly inside its box.
        self.b = _weighted_quantile(y_train - pred_no_b, C_vec, self.tau)
        self.qp_stats.update({"variance_mode": self.variance_mode,
                              "objective_version": "centered_residual_v1",
                              "kernel_rank": rank})

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
        stats["raw_gradient_norm"] = float(grad_norm) if grad_norm is not None else None
        stats["converged"] = False
        stats["certified"] = False
        stats["status"] = "uncertified_approximate_iterate"
        stats["stopping_reason"] = (
            "raw_gradient_tol" if grad_norm is not None and grad_norm < self.torch_tol
            else "max_iter"
        )
        return x_np, stats

    def _fit_batch_sgd(self, X_S, y_S, X_T, y_T, batch_size=16):
        """
        TL-QLDMR 的 Mini-batch SGD 实现（核切片训练）。
        Historical implementation only; it is not the revised objective.
        使用 PyTorch autograd 在核系数空间直接优化，避免显式构造完整核矩阵（O(N^2)）。
        
        模型形式：f(x) = sum_i w_i * k(x_i, x) + b
        """
        raise ValueError(
            "Historical batch_sgd is disabled because its regularizers do not "
            "match the centered-residual objective; use fast_nystrom.")
            
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
            "variance_mode": self.variance_mode,
            "objective_version": ("centered_residual_v1" if self.variance_mode == "centered_residual"
                                  else "legacy_prediction_variance"),
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
        warnings.warn(
            "qpth solves a dual with a 1e-8 numerical ridge; use cvxopt for the "
            "unperturbed positive-semidefinite dual.", UserWarning, stacklevel=2)

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
        stats["dual_ridge"] = 1e-8
        return x_np, stats
