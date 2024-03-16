import sys
sys.path.append('..')
from settings import DATA_DIR, GRAPHS_DIR, ECB_TEXT_DIR, NEWS_TEXT_DIR, SPACY_DIR, DICT_PARSE_COLS

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from itertools import chain, compress

import os

from datetime import datetime
from datetime import timedelta

plt.rcParams["figure.figsize"] = (12,3)
