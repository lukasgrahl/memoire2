import numpy as np


def get_VAR_arr(data: np.array, n_lags: int) -> np.array:
    return np.concatenate([data[i:len(data) - (n_lags - i)] for i in range(1, n_lags + 1)], axis=1)


def get_samp(max_dim, size=100):
    return np.random.randint(0, max_dim, min(size, max_dim))
