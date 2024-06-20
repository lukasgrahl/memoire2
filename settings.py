import os

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))  # os.path.dirname(__file__)

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
ECB_TEXT_DIR = os.path.join(DATA_DIR, "texts")
NEWS_TEXT_DIR = os.path.join(DATA_DIR, "news_txts")
SPACY_DIR = os.path.join(PROJECT_ROOT, 'spacy')
GRAPHS_DIR = os.path.join(PROJECT_ROOT, 'graphs')
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

RANDOM_SEED = 101

DICT_PARSE_COLS = {'pi_perc': float,
                   'pi_perc_WY': float,
                   'id': float,
                   'pinc': float,
                   'hhinc_midpoint': float,
                   'debt': 'category',
                   'is_homeown': bool,
                   'wealth_bank': 'category',
                   'is_unempl': bool,
                   'pi_de_Y': float,
                   'T_sum': float,
                   'T_sum_diff_lag': float,
                   'eduwork': 'category',
                   'profession': 'category',
                   'pinc_midpoint': float,
                   'week_recorded': 'datetime64[ns]',
                   'date_forecast': 'datetime64[ns]',
                   'date_recorded': 'datetime64[ns]',
                   'delta_pe': float,
                   'pi_exp': float,
                   'hhinc': 'category',
                   'employ': 'category',
                   'is_food_shop': bool,
                   'is_invest_shop': bool,
                   'pi_perc_error': float,
                   'debt': 'category', 
                   'eduwork_txt': 'category',
                   'eduwork': 'category',
                   'pi_de_surprise_Y': float,
                   'pi_de_estimate_Y': float,
                    'hhsize': float, 
                    'savings_planned': float,
                    'mir': float,
                    'ecb_rate': float,
                    'delta_pe_MY_error_act': float,
                    'pi_perc_MY_error_act': float,
                    'hhchildren': 'category',
                    'hhsize': 'categoryO', 'hhinc_midpoint': float, 'pinc_midpoint': 'categoryO', 'profession_txt': 'category',
        'riskaversion': 'categoryO',
    'debt_midpoint': float, 'wealth_bank_midpoint': float, 'savings_expend': float, 'is_cash_pay': bool,
    'pi_exp_resid': float, 'pi_exp_resid': float, 'eduschool': 'category', 'familystatus': 'category', 'ecbtrust': 'categoryO', 'is_homeown': bool , 'durables_planned_expend': bool, 'durables_expend': float,
                   }
