'''This script was used to preprocess client names added to the first and last names lists. 
Accents were removed and names were split at interpunction.'''

import unicodedata
import pandas as pd


def remove_accents(text):
    '''This script removes accents from the names lists.'''
    text = unicodedata.normalize('NFD', text).encode('ascii', 'ignore')
    return str(text.decode("utf-8"))


def firstnames_processing(df):
    '''Client first names added to the firstnames list were split up at spaces. If '.'
    was present in the cell, this meant that only first letters were given and therefore these cells were removed.'''
    split_names = []
    for index, row in df.iterrows():
        cells = row['voornaam']
        if '.' in cells:
            df.drop(index, inplace=True)
        else:
            names = row['voornaam'].split(' ')
            for name in names:
                name_accent = remove_accents(name)
                if '?' in str(name_accent):
                    continue
                else:
                    split_names.append(name_accent)
    split_names_df = pd.DataFrame(split_names, columns=['voornaam'])
    split_names_df.to_csv('firstnames_corrected.csv', index=False)


def lastnames_processing(df):
    '''Client last names added to the lastnames list were split up at '-'.'''
    split_names = []
    for _, row in df.iterrows():
        names = row['achternaam'].split('-')
        for name in names:
            name_accent = remove_accents(name)
            if '?' in str(name_accent):
                continue
            else:
                split_names.append(name_accent)
    split_names_df = pd.DataFrame(split_names, columns=['achternaam'])
    split_names_df.to_csv('lastnames_corrected.csv', index=False)

df_lastnames = pd.read_csv(r'/lastnames.csv', encoding='latin-1')
df_firstnames = pd.read_csv(r'/firstnames.csv', encoding='latin-1')
firstnames_processing(df_firstnames)
