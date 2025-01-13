import numpy as np
import math
from pulp import *
from collections import Counter

def jaccard(_x: np.ndarray, _y: np.ndarray) -> np.ndarray:
    if _x.shape[-1] != _y.shape[-1]:
        raise ValueError(f"last dimension of x and y is not equal ({_x.shape[-1]} != {_y.shape[-1]})")
    if len(_x.shape) not in {1, 2}:
        raise ValueError(f"x should be one or two dimensional numpy array (it has shape {_x.shape})")
    if len(_y.shape) not in {1, 2}:
        raise ValueError(f"y should be one or two dimensional numpy array (it has shape {_y.shape})")
    d = _x.shape[-1]
    x = _x.reshape(-1, 1, d)
    y = _y.reshape(-1, d)
    a = np.minimum(x, y).sum(axis=-1)
    b = np.maximum(x, y).sum(axis=-1)
    res = 1 - a / b
    match len(_x.shape), len(_y.shape):
        case 1, 1:
            return res[0,0]
        case 1, _:
            return res[0,:]
        case _, 1:
            return res[:,0]
        case _, _:
            return res

def jaccard_to_closest(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if len(y.shape) != 2:
        raise ValueError(f"y should be two dimensional (it has shape {y.shape})")
    d = jaccard(x, y)
    return d.min(axis=1), d.argmin(axis=1)

def _gen_lp_program(x: np.ndarray, dist: np.ndarray) -> tuple[LpProblem, list, list, list]:
    assert len(x.shape) == 2
    assert len(dist.shape) == 1
    eps = 1e-6
    n, d = x.shape
    A = [
        [LpVariable(f"a_{k}_{i}", 0, 1) for i in range(d)]
        for k in range(n)
    ]
    B = [
        [LpVariable(f"b_{k}_{i}", 0, 1) for i in range(d)]
        for k in range(n)
    ]
    C = [LpVariable(f"c_{i}") for i in range(d)]
    prob = LpProblem("myProblem", LpMinimize)
    for k in range(n):
        for i in range(d):
            prob += A[k][i] <= C[i]
            prob += A[k][i] <= x[k][i]
            prob += B[k][i] >= C[i]
            prob += B[k][i] >= x[k][i]
        prob += lpSum(A[k]) >= (1 - dist[k]) * lpSum(B[k]) + eps
    return prob, A, B, C

def is_blocking_coalition(x: np.ndarray, dist: np.ndarray, rho: float = 1.0):
    assert(len(x.shape) == 2)
    assert(len(dist.shape) == 1)
    assert dist.shape[0] == x.shape[0]
    prob, A, B, C = _gen_lp_program(x, dist / rho)
    status = prob.solve(PULP_CBC_CMD(msg=False))
    return A, B, C, status

def find_blocking_coalition(
        V: np.ndarray,
        centers: np.ndarray,
        size: int,
        rho: float
    ) -> tuple[np.ndarray, np.ndarray] | None:
    n = len(V)
    d_c, _ = jaccard_to_closest(V, centers)
    d_v = jaccard(V, V)
    with np.errstate(divide='ignore'):
        G = np.array([
            [(d_c[i] + d_c[j]) / d_v[i,j] for j in range(n)]
            for i in range(n)
        ]) > rho
    G2 = G.copy()

    curr = np.arange(n)
    while True:
        G = G2[np.ix_(curr, curr)]
        which = np.sum(G, axis=0) >= size
        curr = curr[which]
        if np.all(which):
            break
    
    print(f"potential results: {math.comb(len(curr), size)}")
    
    for arr in itertools.combinations(curr, size):
        arr = np.array(arr)
        if np.all(G2[np.ix_(arr, arr)]):
            A, B, C, status = is_blocking_coalition(V[arr], d_c[arr], rho=rho)
            if status == 1:
                return arr, np.array([value(c) for c in C])
    return None

def local_capture_fixed(V: np.ndarray, k: int, rho: float, steps: int) -> np.ndarray | None:
    assert len(V.shape) == 2
    assert k > 0
    n, d = V.shape
    centers = np.eye(k, d)
    for _ in range(steps):
        match find_blocking_coalition(V, centers, math.ceil(n / k), rho):
            case None:
                return centers
            case v, c:
                extended = np.concatenate((centers, c.reshape(1, d)), axis=0)
                _, ass = jaccard_to_closest(V, extended)
                counter = dict(Counter(ass))
                counter = {i: counter.get(i, 0) for i in range(k+1)}
                i = min(counter, key=counter.get)
                assert i != k
                assert (ass[v] == k).all()
                centers[i] = c
    return None

def local_capture(V: np.ndarray, k: int, steps: int) -> tuple[np.ndarray, float]:
    assert len(V.shape) == 2
    assert k > 0
    l = 1
    r = 1
    while local_capture_fixed(V, k, r, steps) is None:
        r += 0.1
    while r - l > 1e-6:
        print(l, r)
        m = (l + r) / 2
        if local_capture_fixed(V, k, m, steps) is None:
            l = m
        else:
            r = m
    res = local_capture_fixed(V, k, r, steps)
    assert res is not None
    return res, r