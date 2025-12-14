import numpy as np
import cvxopt
from Kernel_Function import rbf_kernel

# 隐藏 cvxopt 的输出
cvxopt.solvers.options['show_progress'] = False

class TL_QLDMR:
    def __init__(self, lambda1=1.0, lambda2=1.0, C_S=1.0, C_T=10.0, tau=0.95, kernel_gamma=0.1):
        """
        Args:
            lambda1: 分布方差项 (LDMR Variance) 权重
            lambda2: MMD 迁移项权重
            C_S: 源域 Pinball Loss 权重
            C_T: 目标域 Pinball Loss 权重 (通常设大，防止被源域淹没)
            tau: 分位数 (0.05, 0.5, 0.95 等)
            kernel_gamma: RBF核参数
        """
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.C_S = C_S
        self.C_T = C_T
        self.tau = tau
        self.gamma = kernel_gamma
        
        self.alpha = None      # 对偶变量 alpha
        self.alpha_star = None # 对偶变量 alpha*
        self.beta = None       # beta = alpha - alpha*
        self.b = 0.0           # 偏置
        self.X_train = None    # 训练数据 (用于预测时计算核)

    def fit(self, X_S, y_S, X_T, y_T):
        """
        训练主函数
        1. 构建核矩阵 K
        2. 构建散度矩阵 S (Variance) 和 MMD 矩阵 M
        3. 构建 QP 问题参数 P, q, G, h, A, b
        4. 求解
        """
        n_S = len(X_S)
        n_T = len(X_T)
        N = n_S + n_T
        
        # 1. 数据合并
        self.X_train = np.vstack([X_S, X_T])
        y_train = np.concatenate([y_S, y_T])
        
        print("1. Computing Kernel Matrix...")
        K = rbf_kernel(self.X_train, self.X_train, self.gamma)
        
        # 2. 构建矩阵 S (Source Domain Variance)
        # S 左上角 n_S x n_S 是中心化矩阵 J，其余为 0
        J_S = np.eye(n_S) - np.ones((n_S, n_S)) / n_S
        S = np.zeros((N, N))
        S[:n_S, :n_S] = J_S
        
        # 3. 构建矩阵 M (MMD Matrix)
        # M_ij = 1/n_S^2 (source), 1/n_T^2 (target), -1/(n_S*n_T) (cross)
        M = np.zeros((N, N))
        M[:n_S, :n_S] = 1.0 / (n_S**2)
        M[n_S:, n_S:] = 1.0 / (n_T**2)
        M[:n_S, n_S:] = -1.0 / (n_S * n_T)
        M[n_S:, :n_S] = -1.0 / (n_S * n_T)
        
        # 4. 计算中间矩阵 Omega 和 H
        # 根据推导: Omega = K + lambda1 * K S K + lambda2 * K M K
        # 技巧：K S K 可以写成 (K S) @ K
        print("2. Constructing Optimization Matrices...")
        KS = K @ S
        KM = K @ M
        
        # Omega 是 Representer Theorem 系数 beta 的二次项系数矩阵
        Omega = K + self.lambda1 * (KS @ K) + self.lambda2 * (KM @ K)
        
        # 为了数值稳定性，加微小扰动
        Omega += 1e-6 * np.eye(N)
        
        # H 是对偶问题中 alpha 的二次项矩阵
        # H = K @ Omega_inv @ K
        # 由于求逆慢，我们用 solve: H = K @ solve(Omega, K)
        try:
            H = K @ np.linalg.solve(Omega, K)
        except np.linalg.LinAlgError:
            print("Warning: Singular matrix, using pseudo-inverse")
            H = K @ np.linalg.pinv(Omega) @ K

        # 5. 构建 QP 标准型
        # min 1/2 x^T P x + q^T x
        # 变量 x = [alpha; alpha*] (长度 2N)
        
        # P = [[H, -H], [-H, H]]
        P = np.block([[H, -H], [-H, H]])
        P = cvxopt.matrix(P)
        
        # 构造一次项 q_vec
        # q = [p_dual; -p_dual]
        # p_dual = lambda1 * K @ Omega_inv @ K @ S @ Y - Y
        # 第一部分 K @ Omega_inv @ K 刚好就是 H
        # 所以 p_dual = lambda1 * H @ S @ Y - Y (注意 Y 的维度对齐)
        # 但这里 S 是 NxN，Y 是 N，直接相乘 S Y 会得到前 n_S 部分中心化的 Y_S
        
        # 构造全量 Y 向量用于计算
        Y_full = y_train.reshape(-1, 1) # (N, 1)
        
        # 计算偏移项
        term1 = self.lambda1 * (H @ S @ Y_full)
        p_dual = term1 - Y_full
        
        q_vec = np.vstack([p_dual, -p_dual])
        q_vec = cvxopt.matrix(q_vec)
        
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
        G = np.vstack([-diag_I, diag_I])
        
        h_vec = np.concatenate([
            np.zeros(2*N),           # 下界 0
            upper_bound_alpha,       # alpha 上界
            upper_bound_alpha_star   # alpha* 上界
        ])
        
        G = cvxopt.matrix(G)
        h = cvxopt.matrix(h_vec)
        
        # 等式约束 Ax = b
        # sum(alpha - alpha*) = 0  => [1, ..., 1, -1, ..., -1] @ x = 0
        A = np.hstack([np.ones(N), -np.ones(N)]).reshape(1, -1)
        b = np.zeros(1)
        
        A = cvxopt.matrix(A)
        b = cvxopt.matrix(b)
        
        # 7. 求解 QP
        print("3. Solving QP...")
        try:
            sol = cvxopt.solvers.qp(P, q_vec, G, h, A, b)
        except ValueError as e:
            print(f"QP Solver Failed: {e}")
            return False

        # 8. 提取结果
        x_sol = np.array(sol['x']).flatten()
        self.alpha = x_sol[:N]
        self.alpha_star = x_sol[N:]
        
        # 计算系数 beta = alpha - alpha* (用于预测)
        # 但在 TL-LDMR 中，根据推导，真实的系数 beta_primal (核函数的系数)
        # 是 beta_primal = Omega_inv @ (K @ (alpha - alpha*) + ...) 
        # 为了简化，我们可以直接用对偶变量恢复预测函数 f(x)
        # f(x) = K(x, X) @ beta_primal + b
        
        # 恢复 Primal 系数 beta_primal
        gamma_vec = (self.alpha - self.alpha_star).reshape(-1, 1) # gamma = alpha - alpha*
        rhs = K @ gamma_vec + self.lambda1 * K @ S @ Y_full
        
        self.beta_primal = np.linalg.solve(Omega, rhs).flatten()
        
        # 9. 计算偏置 b
        # 选择支持向量 (0 < alpha < C_tau)
        # 理论上 b = y_k - K_k @ beta_primal (对于支持向量)
        sv_indices = np.where((self.alpha > 1e-5) & (self.alpha < (C_vec * self.tau - 1e-5)))[0]
        
        if len(sv_indices) > 0:
            b_values = []
            for idx in sv_indices:
                pred_no_b = K[idx] @ self.beta_primal
                b_values.append(y_train[idx] - pred_no_b)
            self.b = np.mean(b_values)
        else:
            self.b = 0.0
            print("Warning: No strict support vectors found, bias set to 0.")
            
        print("Training Completed.")
        return True