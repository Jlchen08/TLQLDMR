import numpy as np
from Kernel_Function import rbf_kernel

class Predictor:
    def __init__(self, model):
        self.model = model

    def predict(self, X_test):
        """
        输入测试集 X_test (n_test, d)
        输出预测值 (n_test, )
        """
        if self.model.beta_primal is None:
            raise ValueError("Model not trained yet!")
            
        # 1. 计算测试集与训练集的核矩阵
        # K_test shape: (n_test, n_train)
        K_test = rbf_kernel(X_test, self.model.X_train, self.model.gamma)
        
        # 2. 预测 f(x) = K * beta + b
        y_pred = K_test @ self.model.beta_primal + self.model.b
        
        return y_pred