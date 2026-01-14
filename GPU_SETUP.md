# TL-QLDMR GPU Setup (CUPY - CUDA)

This project uses a local virtual environment under the `lin` workspace to avoid impacting other projects.

## 1) Create a project venv (once)

```bash
python -m venv /mnt/mydisk/zhangxiaohan/lin/.venv
```

## 2) Activate the venv

```bash
source /mnt/mydisk/zhangxiaohan/lin/.venv/bin/activate
```

## 3) Install dependencies

Install base deps:

```bash
pip install -U pip
pip install numpy scipy cvxopt
pip install matplotlib pandas openpyxl scikit-learn
```

Install the CUPY CUDA build that matches your CUDA runtime:

- CUDA 11.x: `pip install cupy-cuda11x`
- CUDA 12.x: `pip install cupy-cuda12x`

Install PyTorch (CUDA build if available) and a PyTorch QP solver:

- CUDA 12.x (recommended):
  ```bash
  pip install torch --index-url https://download.pytorch.org/whl/cu121
  pip install qpth
  ```
- CPU-only fallback:
  ```bash
  pip install torch
  pip install qpth
  ```

## 4) Enable GPU in code

Pass `use_gpu=True` when constructing the model:

```python
model = TL_QLDMR(
    lambda1=0.1,
    lambda2=0.1,
    C_S=1.0,
    C_T=10.0,
    tau=0.95,
    kernel_gamma=0.1,
    use_gpu=True,
)
```

To use the PyTorch solvers:

```python
# Gradient descent (autograd) solver
model = TL_QLDMR(..., solver="torch_gd", use_gpu=True)

# qpth solver
model = TL_QLDMR(..., solver="qpth", use_gpu=True)
```

Notes:
- GPU is used for kernel/matrix math; CVXOPT QP solver remains CPU-bound.
- Prediction uses GPU if `use_gpu=True` and CUPY is available.
- PyTorch solvers can use CUDA when torch detects a GPU.
