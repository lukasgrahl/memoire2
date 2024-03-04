import pandas as pd
import spacy
from tqdm import tqdm
import os
from uuid import uuid4
import pickle

from settings import DATA_DIR, NEWS_TEXT_DIR
from src.utils import load_pickle, save_pd_df

def get_spacy_NLP(lang: str = 'de'):
    if lang == 'de':
        nlp = spacy.load(
            os.path.abspath("C:\\Users\\LukasGrahl\\Documents\\GIT\\memoire2\\spacy\\de_core_news_lg\\de_core_news_lg-3.7.0")
        )
    elif lang == 'en':
        nlp = spacy.load(
            os.path.abspath("C:\\Users\\LukasGrahl\\Documents\\GIT\\memoire2\\spacy\\en_core_web_lg\\en_core_web_lg-3.7.1")
        )
    else:
        raise KeyError(f"please specify file location of {lang} package")
    return nlp


def load_raw_data(f_name: str):
    data = load_pickle(f_name, f_path=DATA_DIR)

    df = pd.DataFrame(data)
    df['id'] = [str(uuid4()) for i in range(len(df))]
    df = df.set_index('id')
    save_pd_df(df, 'news_data.feather')

    df = df.rename(columns={'datetime': 'date'})
    df = df.sort_values('date')
    df['date'] = df.date.apply(lambda x: x.date())

    for idx, row in tqdm([*df.iterrows()]):
        f = open(os.path.join(NEWS_TEXT_DIR, 'orig', f'{idx}.pkl'), 'wb+')
        pickle.dump(row.to_dict(), f)
        f.close()
    pass