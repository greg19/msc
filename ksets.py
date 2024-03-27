import numpy as np
import random

def ksets_distances(xs: np.ndarray, hs: np.ndarray) -> np.ndarray:
    assert len(xs.shape) == 2
    assert len(hs.shape) == 2
    assert xs.shape[-1] == hs.shape[-1]
    d = xs.shape[-1]
    xs = xs.reshape(-1, 1, d)
    a = np.minimum(xs, hs).sum(axis=2)
    c = np.maximum(xs, hs).sum(axis=2)
    return 1 - a / c

def group_points(xs: np.ndarray, hs: np.ndarray) -> tuple[float, np.ndarray]:
    d = ksets_distances(xs, hs)
    ids = np.argmin(d, axis=1)
    sdh = d.min(axis=1).sum()
    return sdh, ids

def recompute_clusters(k: int, xs: np.ndarray, ids: np.ndarray) -> np.ndarray:
    return np.array([np.sum(xs[ids == i], axis=0) for i in range(k)])

def ksets_one(
        k: int,
        xs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    assert len(xs.shape) == 2

    ids = np.random.randint(k, size=xs.shape[0])
    sums = np.array([np.sum(xs[ids == i], axis=0) for i in range(k)])
    cnts = np.array([np.sum(ids == i) for i in range(k)]).reshape(-1, 1)

    changed = True
    while changed:
        changed = False
        for i, x in enumerate(xs):
            d = ksets_distances(x.reshape(1, -1), sums / cnts).reshape(-1)
            c = np.argmin(d)
            c_old = ids[i]
            if c != c_old:
                ids[i] = c
                sums[c_old] -= x
                cnts[c_old] -= 1
                sums[c] += x
                cnts[c][0] += 1
                changed = True

    return ids, sums / cnts


def ksets(
        k: int,
        xs: np.ndarray,
        iter: int = 1
) -> np.ndarray:
    return min([ksets_one(k, xs)[0] for _ in range(iter)], key=lambda x: x[0])

"""
def kswaps(
        k: int,
        xs: np.ndarray,
        iter: int = 20,
        override: int = 1
) -> np.ndarray:
    ids = ksets_one(k, xs)
    for i in range(iter):
        hs_swapped = hs.copy()
        for _ in range(override):
            hs_swapped[random.randrange(k)] = random.choice(xs)
        sdh_new, ids_new, hs_new = ksets_one(k, xs, hs_swapped)
        # print(f"Iteration {i:3} sdh is {sdh_new:.3f} (best is {sdh:.3f})")
        if sdh_new < sdh:
            sdh = sdh_new
            ids = ids_new
            hs = hs_new
    return ids
"""