import multiprocessing
import numpy as np
import time
import os
import pickle
from uuid import uuid4
import spacy
from collections import Counter
from itertools import compress, chain
from tqdm import tqdm

from settings import DATA_DIR, NEWS_TEXT_DIR, SPACY_DIR
from src.utils import load_pickle, save_pkl, vec_similarity, arr_min_max_scale

# NLP = spacy.load("en_core_web_lg")
def get_NLP(lang: str = 'en'):
    if lang == 'de':
        return spacy.load(os.path.join(SPACY_DIR, f'de_core_news_lg', 'de_core_news_lg-3.7.0'))
    elif lang == 'en':
        return spacy.load(os.path.join(SPACY_DIR, f'en_core_web_lg', 'en_core_web_lg-3.7.1'))
    else:
        raise KeyError(f'{lang} unknonw')

NLP = get_NLP()

LST_IS_INFL_TOKENS = ['inflation', 'price']
LST_SPACY_POS = ['PROPN', 'NOUN', 'AJD']
LST_FREQUENT_NON_MEANING = ['euro', 'area', 'rate', 'council', 'month', 'year', 'today']

DICT_NARRATIVES = {
    "S_labour": {
        'g1': ["labour"],
        'g2': ["wage", "employment"],
    },
    "S_supply_chain": {
        'g1': ["supply", "chain", "shortage"],
        'g2': ['semi-conductor'],
        'g3': ['trade'],
    },
    # "D_tax_cut": ["government spending", "tax reduction"],
    "D_hh_spend": {
        'g1': ['demand', 'household', 'spending'],
        'g2': ['consumption']
    },
    # "M_policy": ["interest rate", "quantitative easing"],
}
DICT_NARRATIVES_DOC = {
    nkey: {
        gkey: [NLP(term) for term in group] for gkey, group in narrative.items()
    }
    for nkey, narrative in DICT_NARRATIVES.items()
}
DICT_NARRATIVES_VEC = {
    nkey: {
        # gkey: [NLP(term) for term in group]
        gkey: np.concatenate([term.vector[None] for term in group], axis=0)
        for gkey, group in narrative.items()
    }
    for nkey, narrative in DICT_NARRATIVES_DOC.items()
}

is_load = False
if is_load:
    f = open(os.path.join(DATA_DIR, 'ecb_speeches.pickle'), 'rb')
    speeches = pickle.load(f)
    f.close()

    speeches = {
        str(uuid4()): {
            k: speech[k] for k in speech.keys() if k in ['date', 'url', 'title', 'header', 'text']
        }
        for speech in speeches.values()
    }

    for k, v in speeches.items():
        f = open(os.path.join(NEWS_TEXT_DIR, 'orig', f'{k}.pkl'), 'wb+')
        pickle.dump(v, f)
        f.close()


def run(file_names: tuple):
    for file_name in tqdm(file_names):

        try:
            s = load_pickle(file_name, f_path=os.path.join(NEWS_TEXT_DIR, 'orig'))
        except EOFError:
            continue

        # flags
        is_infl = sum([(exp in s['text']) for exp in LST_IS_INFL_TOKENS]) > 0
        s['infl'] = is_infl

        if is_infl:

            # spacy
            s['doc'] = NLP(s['text'])
            s['lst_nouns'] = [
                                i.lemma_.lower() for i in s['doc'] if 
                                (
                                    i.is_alpha
                                    and i.pos_ in LST_SPACY_POS
                                    # and not i.ent_type_ != ""
                                    and not (i.is_stop or i.is_punct or i.is_currency or i.is_bracket)
                                    and i.lemma_.lower() not in LST_FREQUENT_NON_MEANING
                                )
                            ]

            lst_nchunks = [*s['doc'].noun_chunks]
            arr_nchunks = np.concatenate([i.vector[None] for i in lst_nchunks], axis=0)

            # noun counter
            s['counter'] = dict(Counter(s['lst_nouns']))

            # check each narrative
            dict_narratives = {}
            for nkey, narrative in DICT_NARRATIVES_VEC.items():

                # check group in each narrative
                dict_group = {}
                for gkey, arr_group in narrative.items():

                    # find similar noun chunks to narrative group
                    filt_sim = vec_similarity(arr_nchunks, arr_group)

                    # find noun chunks which contain the string literal
                    lst_sterms = [" ".join([t.lemma_.lower() for t in sterm]) for sterm in DICT_NARRATIVES_DOC[nkey][gkey]]
                    filt_det = np.array(
                        [
                            [sterm in " ".join([t.lemma_.lower() for t in chunk]) for chunk in lst_nchunks]
                            for sterm in lst_sterms
                        ]
                    ).T
                    # aggregate similarity and literal filter
                    # filt = arr_min_max_scale(np.sqrt(np.prod((1 + filt_sim) ** 2, axis=1)) * filt_det)

                    _ = ((filt_sim + 1) * (filt_det + 1)).prod(axis=1)
                    # _ = np.prod(((filt_sim + 1) * (filt_det + 1))**2, axis=1)
                    # _ = np.prod((1+filt_sim), axis=1)
                    filt = arr_min_max_scale(_)

                    # consider noun chunks for a score of both
                    lst_narratives_chunks_det = [*compress(lst_nchunks, filt_det.sum(axis=1) > 0)]
                    lst_narratives_chunks_sim = [*compress(lst_nchunks, filt > .85)]

                    # dict for group in narrative
                    dict_group[gkey] = {
                    
                        'sim': {
                                    'group_score': len(lst_narratives_chunks_sim) / len(lst_nchunks),
                                    'group_chunk_txt': [i.text for i in lst_narratives_chunks_sim],
                                    'group_chunk_lem': [i.lemma_.lower() for i in lst_narratives_chunks_sim]
                                },

                        'det': {
                                    'group_score': len(lst_narratives_chunks_det) / len(lst_nchunks),
                                    'group_chunk_txt': [i.text for i in lst_narratives_chunks_det],
                                    'group_chunk_lem': [i.lemma_.lower() for i in lst_narratives_chunks_det]
                                },
                    }

                # dict for narrative in narratives
                dict_narratives[nkey] = {
                    'narrative_score_sim': np.sum([dict_group[k]['sim']['group_score'] for k in dict_group.keys()]),
                    'narrative_score_det': np.sum([dict_group[k]['det']['group_score'] for k in dict_group.keys()]),
                    'narrative_chunk_sim': [*chain(*[dict_group[k]['sim']['group_chunk_txt'] for k in dict_group.keys()])],
                    'narrative_chunk_det': [*chain(*[dict_group[k]['det']['group_chunk_txt'] for k in dict_group.keys()])],
                    'dict_groups': dict_group
                }

            # dict of all narratives and their groups
            s['narratives'] = dict_narratives

            # drop unnecessary information
            s = {
                k: s[k] for k in s.keys() if k in [
                    'date',
                    'is_infl',
                    'title',
                    'narratives',
                    'counter',
                    'lst_nouns',
                    'text'
                ]
            }

            save_pkl(s, file_name, NEWS_TEXT_DIR)

    pass


if __name__ == "__main__":
    inputs = os.listdir(os.path.join(NEWS_TEXT_DIR, 'orig'))
    N = int(np.ceil(len(inputs) / os.cpu_count()))
    inputs = [tuple(inputs[i:i + N]) for i in range(0, len(inputs), N)]
    # inputs = [i[:40] for i in inputs]

    # run(inputs[0])
    with multiprocessing.Pool(processes=os.cpu_count()) as pool:
        start = time.time()
        res = pool.map(run, inputs)

    print(f"\nThis process ran: {time.time() - start:<=.4}")
