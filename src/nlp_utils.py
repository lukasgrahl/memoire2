import spacy
import os

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