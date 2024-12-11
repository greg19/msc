from typing import Literal
import numpy as np

def jaccard(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    assert len(x.shape) == 2
    assert len(y.shape) == 2
    assert x.shape[-1] == y.shape[-1]
    d = x.shape[-1]
    x = x.reshape(-1, 1, d)
    a = np.minimum(x, y).sum(axis=-1)
    b = np.maximum(x, y).sum(axis=-1)
    return 1 - a / b

def gradient(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Computes dJ(x,y')/dx for all rows y' in y.
    """
    assert len(x.shape) == 1
    assert len(y.shape) == 2
    mn = np.sum(np.minimum(x, y), axis=1)
    mx = np.sum(np.maximum(x, y), axis=1)
    return ((y > x) * mx.reshape(-1, 1) - (y < x) * mn.reshape(-1, 1)) / (mx ** 2).reshape(-1, 1)

def gradient_descend(
        x: np.ndarray,
        y: np.ndarray,
        iter: int,
        lr: float,
        objective: Literal['sum', 'max'],
    ) -> np.ndarray:
    assert len(x.shape) == 1
    assert len(y.shape) == 2
    assert x.shape[-1] == y.shape[-1]
    if objective == 'sum':
        for _ in range(iter):
            grad = gradient(x, y)
            x = np.clip(x + lr * grad.mean(axis=0), 0, 1)
    elif objective == 'max':
        for _ in range(iter):
            grad = gradient(x, y)
            e_dist = np.e ** (10 * jaccard(x.reshape(1, -1), y).T)
            x = np.clip(x + lr * (e_dist / e_dist.sum() * grad).sum(axis=0), 0, 1)
    else:
        raise RuntimeError(f'Invalid objective: {objective}')
    return x
