import os

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__)) #os.path.dirname(__file__)

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
TEXT_DIR = os.path.join(DATA_DIR, "texts")
SPACY_DIR = os.path.join(PROJECT_ROOT, 'spacy')
GRAPHS_DIR = os.path.join(PROJECT_ROOT, 'latex', 'graphs')

RANDOM_SEED = 101