import os
from datetime import datetime

import numpy as np
import pandas as pd

from settings import DATA_DIR


def get_samp(max_dim, size=100):
    return np.random.randint(0, max_dim, min(size, max_dim))


def get_dt_index(df: pd.DataFrame, dt_index_col: str = 'date'):
    df = df.set_index(pd.DatetimeIndex(df[dt_index_col].apply(lambda x: datetime.strptime(x, "%Y-%m-%d"))))
    if dt_index_col in df.columns:
        df = df.drop(dt_index_col, axis=1)
    return df


def load_pd_df(file_name, file_path=None, **kwargs):
    if file_path is None:
        file_path = DATA_DIR
    if file_name.split('.')[-1] == 'csv':
        return pd.read_csv(os.path.join(file_path, file_name), **kwargs)
    elif file_name.split('.')[-1] == 'xlsx':
        return pd.read_excel(os.path.join(file_path, file_name), **kwargs)
    elif file_name.split('.')[-1] == 'feather':
        return pd.read_feather(os.path.join(file_path, file_name), **kwargs)
    else:
        raise KeyError(f"{file_name.split('.')[-1]} unknonw")
