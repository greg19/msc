import numpy as np
from sklearn.preprocessing import normalize
from scipy.spatial import distance_matrix

def jaccard(x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    assert len(x1.shape) == 2
    assert len(x2.shape) == 2
    assert x1.shape[-1] == x2.shape[-1]
    d = x1.shape[-1]
    x1 = x1.reshape(-1, 1, d)
    a = np.minimum(x1, x2).sum(axis=-1)
    b = np.maximum(x1, x2).sum(axis=-1)
    return 1 - a / b

def cosine(x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    x1 = normalize(x1, norm='l2', axis=1) # type: ignore
    x2 = normalize(x2, norm='l2', axis=1) # type: ignore
    return np.maximum(1 - (x1 @ x2.T), 0)

def chord(x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    x1 = normalize(x1, norm='l2', axis=1) # type: ignore
    x2 = normalize(x2, norm='l2', axis=1) # type: ignore
    return distance_matrix(x1, x2)

def simrank_both(xs: np.ndarray, C1=0.8, C2=0.8, max_iter=100) -> tuple[np.ndarray, np.ndarray]:
    n, m = xs.shape
    X1 = xs / xs.sum(axis=0, keepdims=True)
    X2 = xs.T / xs.T.sum(axis=0, keepdims=True)
    S1 = np.eye(n)
    S2 = np.eye(m)
    for i in range(max_iter):
        S2_ = np.maximum(C2 * X1.T @ S1 @ X1, np.eye(m))
        S1_ = np.maximum(C1 * X2.T @ S2 @ X2, np.eye(n))
        norm1 = np.linalg.norm(S1-S1_)
        norm2 = np.linalg.norm(S2-S2_)
        # print(f"Iteration {i} norm is {norm1, norm2}")
        S1 = S1_
        S2 = S2_
        if norm1 + norm2 < 1e-5:
            break
    return 1 - S1, 1 - S2

def simrank(xs: np.ndarray, C1=0.8, C2=0.8, max_iter=100) -> np.ndarray:
    return simrank_both(xs, C1, C2, max_iter)[0]
