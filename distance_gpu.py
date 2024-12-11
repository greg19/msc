import numpy as np
import cupy as cp
from sklearn.preprocessing import normalize

def _normalize(x: cp.ndarray) -> cp.ndarray:
    norms = cp.sqrt(cp.sum(x * x, axis=1, keepdims=True))
    return x / norms

def jaccard(x1: cp.ndarray, x2: cp.ndarray) -> cp.ndarray:
    assert len(x1.shape) == 2
    assert len(x2.shape) == 2
    assert x1.shape[-1] == x2.shape[-1]
    d = x1.shape[-1]
    x1 = x1.reshape(-1, 1, d)
    a = cp.minimum(x1, x2).sum(axis=-1)
    b = cp.maximum(x1, x2).sum(axis=-1)
    return 1 - a / b

def cosine(x1: cp.ndarray, x2: cp.ndarray) -> cp.ndarray:
    # TODO: write normalization in cupy instead of convering to numpy
    x1 = _normalize(x1) # type: ignore
    x2 = _normalize(x2) # type: ignore
    return cp.maximum(1 - (x1 @ x2.T), 0)

def simrank_both(xs: cp.ndarray, C1=0.8, C2=0.8, max_iter=100) -> tuple[cp.ndarray, cp.ndarray]:
    n, m = xs.shape
    X1 = xs / xs.sum(axis=0, keepdims=True)
    X2 = xs.T / xs.T.sum(axis=0, keepdims=True)
    S1 = cp.eye(n)
    S2 = cp.eye(m)
    for i in range(max_iter):
        S2_ = cp.maximum(C2 * X1.T @ S1 @ X1, cp.eye(m))
        S1_ = cp.maximum(C1 * X2.T @ S2 @ X2, cp.eye(n))
        norm1 = cp.linalg.norm(S1-S1_)
        norm2 = cp.linalg.norm(S2-S2_)
        # print(f"Iteration {i} norm is {norm1, norm2}")
        S1 = S1_
        S2 = S2_
        if norm1 + norm2 < 1e-5:
            break
    return 1 - S1, 1 - S2

def simrank(xs: cp.ndarray, C1=0.8, C2=0.8, max_iter=100) -> cp.ndarray:
    return simrank_both(xs, C1, C2, max_iter)[0]