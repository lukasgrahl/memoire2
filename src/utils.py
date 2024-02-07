import os
from datetime import datetime
import warnings

from io import StringIO
import sys

import numpy as np
import pandas as pd

from settings import DATA_DIR, GRAPHS_DIR


def get_samp(max_dim, size=100):
    return np.random.randint(0, max_dim, min(size, max_dim))


def get_dt_index(df: pd.DataFrame, dt_index_col: str = 'date'):
    df = df.set_index(pd.DatetimeIndex(df[dt_index_col].apply(lambda x: datetime.strptime(x, "%Y-%m-%d"))))
    if dt_index_col in df.columns:
        df = df.drop(dt_index_col, axis=1)
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


class Capturing(list):
    def __enter__(self):
        self._stdout = sys.stdout
        sys.stdout = self._stringio = StringIO()
        return self

    def __exit__(self, *args):
        self.extend(self._stringio.getvalue().splitlines())
        del self._stringio
        sys.stdout = self._stdout


def write_to_txt(output: str, file_name):
    f = open(os.path.join(GRAPHS_DIR, file_name), 'w+')
    f.write(output)
    f.close()
    pass
