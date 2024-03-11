from src.nlp_utils import run_parallel
from src.utils import load_pd_df

import numpy as np
import multiprocessing
import time
import os

if __name__ == "__main__":

    tlda = load_pd_df("lda_topics.feather").reset_index()
    res = run_parallel(tlda.drop(['id'], axis=1), id_col='date')
