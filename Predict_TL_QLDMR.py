import numpy as np
from Kernel_Function import rbf_kernel

try:
    import cupy as cp
except ImportError:
    cp = None

class Predictor:
    def __init__(self, model):
        self.model = model

    def predict(self, X_test, batch_size=100):
        """
        输入测试集 X_test (n_test, d)
        输出预测值 (n_test, )
        """
        if self.model.beta_primal is None:
            raise ValueError("Model not trained yet!")
            
        n_test = len(X_test)
        predictions = []
        
        # 确保 float32
        X_test = np.array(X_test, dtype=np.float32)
        X_train = np.array(self.model.X_train, dtype=np.float32)
        beta = np.array(self.model.beta_primal, dtype=np.float32)
        
        # 预加载 X_train 到 GPU (如果使用 GPU)
        X_train_gpu = None
        beta_gpu = None
        
        if self.model.use_gpu and cp is not None:
            try:
                cp.get_default_memory_pool().free_all_blocks()
                X_train_gpu = cp.asarray(X_train)
                beta_gpu = cp.asarray(beta)
            except Exception as e:
                print(f"Warning: Failed to load model to GPU ({e}), falling back to CPU.")
                self.model.use_gpu = False

        for i in range(0, n_test, batch_size):
            X_batch = X_test[i:i+batch_size]
            
            if self.model.use_gpu and cp is not None:
                try:
                    X_batch_gpu = cp.asarray(X_batch)
                    # K_batch shape: (batch_size, n_train)
                    K_batch = rbf_kernel(X_batch_gpu, X_train_gpu, self.model.gamma, xp=cp)
                    pred_batch = K_batch @ beta_gpu + self.model.b
                    predictions.append(cp.asnumpy(pred_batch))
                    
                    del K_batch, pred_batch, X_batch_gpu
                except Exception as e:
                    print(f"GPU Error during prediction: {e}. Falling back to CPU for this batch.")
                    # Fallback to CPU for this batch
                    K_batch = rbf_kernel(X_batch, X_train, self.model.gamma, xp=np)
                    pred_batch = K_batch @ beta + self.model.b
                    predictions.append(pred_batch)
            else:
                K_batch = rbf_kernel(X_batch, X_train, self.model.gamma, xp=np)
                pred_batch = K_batch @ beta + self.model.b
                predictions.append(pred_batch)
                
        # Clean up GPU memory
        if X_train_gpu is not None:
            del X_train_gpu, beta_gpu
            if cp is not None:
                cp.get_default_memory_pool().free_all_blocks()

        return np.concatenate(predictions)
