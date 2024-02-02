import unicodedata
import pandas as pd


def remove_accents(text):
    text = unicodedata.normalize('NFD', text).encode('ascii', 'ignore')
    return str(text.decode("utf-8"))


def lastnames_processing(df):
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
    split_names_df.to_csv('last_names_split_goede.csv', index=False)  # or encoding='latin-1'


def firstnames_processing(df):
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
    split_names_df.to_csv('first_names_split_goede.csv', index=False)  # or encoding='latin-1'


df_lastnames = pd.read_csv(r'F:/Documenten/Universiteit/Master_TM+_commissies/Jaar 3/Afstuderen/Thesis/Anonimiseren_data/datasets/lastnames_goede.csv', encoding='latin-1')
df_firstnames = pd.read_csv(r'F:/Documenten/Universiteit/Master_TM+_commissies/Jaar 3/Afstuderen/Thesis/Anonimiseren_data/datasets/firstnames_update.csv', encoding='latin-1')
firstnames_processing(df_firstnames)
