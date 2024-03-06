import os
import pickle
from datetime import datetime
from settings import NEWS_TEXT_DIR, GRAPHS_DIR
import numba as nb

import numpy as np
import pandas as pd

from settings import DATA_DIR, GRAPHS_DIR


def get_dt_index(df: pd.DataFrame, dt_index_col=None, is_rename_date: bool = True):
    if dt_index_col is None: dt_index_col = 'date'
    df = df.set_index(pd.DatetimeIndex(df[dt_index_col].apply(lambda x: datetime.strptime(x, "%Y-%m-%d"))))
    if dt_index_col in df.columns:
        df = df.drop(dt_index_col, axis=1)
    if is_rename_date: df.index.name = 'date'
    return df


def load_pd_df(file_name, file_path=None, is_replace_nan=True, **kwargs):
    file_type = file_name.split('.')[-1]

    if file_path is None:
        file_path = DATA_DIR

    if file_type == 'csv':
        return pd.read_csv(os.path.join(file_path, file_name), **kwargs)
    elif file_type == 'xlsx':
        return pd.read_excel(os.path.join(file_path, file_name), **kwargs)
    elif file_type == 'feather':
        df = pd.read_feather(os.path.join(file_path, file_name), **kwargs)
        # if is_replace_nan:
        #     warnings.warn("replacing nan for feather format, pass 'is_replace_nan=False' to disable")
        #     df = df.replace({'nan': np.nan})
        return df
    else:
        raise KeyError(f"{file_type} unknown")


def save_pd_df(df, file_name: str, file_path=None):
    file_type = file_name.split('.')[-1]
    if file_path is None:
        file_path = DATA_DIR

    if file_type == "csv":
        df.to_csv(os.path.join(file_path, file_name))
    elif file_type == "feather":
        df.to_feather(os.path.join(file_path, file_name))
    else:
        raise KeyError(f"{file_type} unknown")


from io import StringIO
import sys


class Capturing(list):
    def __init__(self, file_name: str):
        self.file_name = file_name
        pass

    def __enter__(self):
        self._stdout = sys.stdout
        sys.stdout = self._stringio = StringIO()
        return self

    def __exit__(self, *args, **kwargs):
        self.extend(self._stringio.getvalue().splitlines())
        del self._stringio
        sys.stdout = self._stdout
        write_to_txt("\n".join(self), self.file_name, file_dir=os.getcwd())


def write_to_txt(output: str, file_name, file_dir=None):
    if file_dir is None: file_dir = GRAPHS_DIR
    f = open(os.path.join(file_dir, file_name), 'w+')
    f.write(output)
    f.close()
    pass


def save_pkl(file: dict, f_name: str, f_path: str = None):
    if f_path is None:
        f_path = NEWS_TEXT_DIR
    t = open(os.path.join(f_path, f"{f_name}"), "wb+")
    pickle.dump(file, t)
    t.close()
    pass

def save_fig(fig, f_name: str, f_path: str = None):
    if f_path is None:
        f_path = GRAPHS_DIR
    fig.savefig(os.path.join(f_path, f_name))
    pass


def load_pickle(f_name, f_path=None):
    if f_path is None:
        f_path = NEWS_TEXT_DIR
    t = open(os.path.join(f_path, f_name), 'rb')
    file = pickle.load(t)
    t.close()
    return file


@nb.njit()
def frobenius_norm(a):
    norms = np.empty(a.shape[0], dtype=a.dtype)
    for i in nb.prange(a.shape[0]):
        norms[i] = np.sqrt(np.sum(a[0] ** 2))
    return norms


@nb.njit()
def arr_norm(arr, axis=0):
    return np.sqrt(np.sum(arr ** 2, axis=axis))


@nb.njit()
def arr_to_unity(arr):
    # norm = frobenius_norm(arr)
    norm = arr_norm(arr, axis=1)
    is_null = np.abs(norm) == 0
    norm[is_null] = np.ones(is_null.sum()) * 1e-8
    arr = np.divide(arr, norm[:, None])
    return arr


@nb.njit()
def vec_similarity(arr, search_terms):
    arr, search_terms = arr_to_unity(arr), arr_to_unity(search_terms)
    return np.dot(arr, search_terms.T)


def arr_min_max_scale(arr):
    if arr.min() != arr.max():
        return (arr - arr.min()) / (arr.max() - arr.min())
    else:
        return arr


def pd_join_freq(df1, df2, freq: str = 'D', keep_indices: bool = True, **kwargs):
    df1, df2 = df1.copy(), df2.copy()
    
    for d in [df1, df2]:
        assert d.index.name != freq, "pls change index name"
        
    if keep_indices:
        df1[df1.index.name] = df1.index
        df2['index_right'] = df2.index
    
    df1[freq] = df1.index.to_period(freq)
    df2[freq] = df2.index.to_period(freq)
    
    df = pd.merge(df1, df2, on=freq, **kwargs).set_index(freq) #, axis=1)
    df.index = df.index.to_timestamp()
    return df