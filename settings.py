import os

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__)) #os.path.dirname(__file__)

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
ECB_TEXT_DIR = os.path.join(DATA_DIR, "texts")
NEWS_TEXT_DIR = os.path.join(DATA_DIR, "news_txts")
SPACY_DIR = os.path.join(PROJECT_ROOT, 'spacy')
GRAPHS_DIR = os.path.join(PROJECT_ROOT, 'graphs')
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

RANDOM_SEED = 101