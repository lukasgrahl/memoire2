LST_IS_INFL_TOKENS = ['inflation', 'inflationsrate']
LST_SPACY_POS = ['PROPN', 'NOUN', 'AJD']

DICT_NARRATIVES = {
    "inflation": {
        'g1': ["kaufpreis", "preis",],
        'g2': [ "inflation", "inflationsrate"],
        'g3': ['teuerung', 'preissteigerung'],
    },
    "S_labour": {
        'g1': ["beschäftigter", "mitarbeiter", "arbeitnehmer"],
        'g2': ["gehalt", "lohn"],
        'g3': ["arbeitslosigkeit", "arbeitslosenquote", "arbeitslosenzahl"]
    },
    "S_supply_chain": {
        'g1': ["supply", "chain"],
        'g2': ["lieferkette"],
        'g3': ["import", "handel"],
        'g4': ['produktion', 'hersteller'],
        'g5': ['kosten',],
        'g6': ['halbleiter'],
    },
    "S_war": {
        'g1': ['russland', 'ukraine'],
        'g2': ['krieg', 'konflikt'],
        'g3': ['sanktion']
    },
    "S_energy": {
        'g1': ['nord', 'stream', 'pipeline'],
        'g2': ['gas', 'erdgas', 'preis'],
        'g3': ['strom', 'öl', 'preis'],
        'g4': ['energie', 'preis'], 
        
    },    
    "D_hh_spend": {
        'g1': ['nachfrage', 'ausgabe'],
        'g2': ['konsum'],
        'g3': ["haushalt", "verbraucher", "kunde"]
        'g4': ['lebensmittel']
    },
    'S_pandemic': {
        'g1': ['corona', 'covid']
        'g2': ['pandemie', 'virus']
        'g3': ['lockdown', 'lockdowns']
    },
    "M_policy": {
        'g1': ["zins", "zinsrate"],
        'g2': ['geldpolitik'],
        'g3': ['notenbank', 'zentralbank'],
        'g4': ['fed', 'ezb'],
        'g5': ['lagarde']
    },

    "M_crisis": {
        'g1': ['krise'],
        'g2': ['rezession'],
        'g3': ['risiko']
    }

    # "D_tax_cut": ["government spending", "tax reduction"],
}

"""
mensch, markt, folge, welt, ziel, höhe
"""

LST_FREQUENT_NON_MEANING = [
        
        'prozent',
         'euro',
         'million',
         'milliarde',
         'deutschland',
         'deutsch',
         'land',
         'firma',
         'frage',
         'zahl',
         'woche',
         'monat',
         'stadt',
         'bereich',
         'seite',
         'datum',
         'quartal',
         'tonne',
         'zeit',
         'entwicklung',
         'standort',
         'vergleich',
         'jahr',
         'experte',
         'anfangm',
         'lage',
         'thema',
         'anteil',
         'blick',
         'weg',
         'meinungen',
         'punkt',
         'januar',
         'september',
         'maßnahme',
         'juni',
         'situation',
         'hälfte',
         'angabe',
         'jahrzehnt',
         'juli',
         'institut',
         'region',
         'vorjahr',
         'frau',
         'april',
         'kilometer',
         'dienstag',
         'system',
         'familie',
         'mitt',
         'gespräch',
         'mai',
         'niveau',
         'oktober',
         'lösung',
         'modell',
         'idee',
         'dezember',
         'prognose',
         'papier',
         'drittel',
         'schnitt',
         'mal',
         'strategie',
         'november',
         'platz',
         'mittwoch',
         'montag',
         'stelle',
         'august',
         'februar',
         'teil',
         'stunde',
         'hand',
         'name',
         'freitag',
         'halbjahr',
         'umfrage',
         'prozentpunkt',
         'stand',
         'wort',
         'summe'
]