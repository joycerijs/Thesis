import pandas as pd
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
import re
import os
from scipy.stats import chi2_contingency
from sklearn import model_selection
from statistics import mean
import numpy as np
from scipy.stats import mannwhitneyu


def tokenize_and_stem(text):
    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    tokens = [word for sent in nltk.sent_tokenize(text, language='dutch') for word in nltk.word_tokenize(sent, language='dutch')]
    filtered_tokens = []
    for token in tokens:
        if '/' in token:
            token_split = token.split('/')
            for token_ in token_split:
                if re.fullmatch(r'[a-zA-Z]{2,}', token_):
                    filtered_tokens.append(token_)
        if '@' in token:
            token_split = token.split('@')
            for token_ in token_split:
                if re.fullmatch(r'[a-zA-Z]{2,}', token_): 
                    filtered_tokens.append(token_)
        if '.' in token:
            token_split = token.split('.')
            for token_ in token_split:
                if re.fullmatch(r'[a-zA-Z]{2,}', token_): 
                    filtered_tokens.append(token_)
        if '-' in token:
            token_split = token.split('-')
            for token_ in token_split:
                if re.fullmatch(r'[a-zA-Z]{2,}', token_): 
                    filtered_tokens.append(token_)
        if '_' in token:
            token_split = token.split('_')
            for token_ in token_split:
                if re.fullmatch(r'[a-zA-Z]{2,}', token_): 
                    filtered_tokens.append(token_)
        if '~' in token:
            token_split = token.split('~')
            for token_ in token_split:
                if re.fullmatch(r'[a-zA-Z]{2,}', token_): 
                    filtered_tokens.append(token_)
        if "'" in token:
            token_split = token.split("'")
            for token_ in token_split:
                if re.fullmatch(r'[a-zA-Z]{2,}', token_): 
                    filtered_tokens.append(token_)
        if re.fullmatch(r'[a-zA-Z]{2,}', token):
                filtered_tokens.append(token)
    stems = [stemmer.stem(t) for t in filtered_tokens]
    return stems


def vocabulary(text):
    '''This function returns a dataframe with stems and the words mapped to the stems.'''
    filtered_tokens = []
    for text in text:
        tokens = [word for sent in nltk.sent_tokenize(text, language='dutch') for word in nltk.word_tokenize(sent, language='dutch')]
        for token in tokens:
            if '/' in token:
                token_split = token.split('/')
                for token_ in token_split:
                    if re.fullmatch(r'[a-zA-Z]{2,}', token_):
                        filtered_tokens.append(token_)
            if '@' in token:
                token_split = token.split('@')
                for token_ in token_split:
                    if re.fullmatch(r'[a-zA-Z]{2,}', token_): 
                        filtered_tokens.append(token_)
            if '.' in token:
                token_split = token.split('.')
                for token_ in token_split:
                    if re.fullmatch(r'[a-zA-Z]{2,}', token_): 
                        filtered_tokens.append(token_)
            if '-' in token:
                token_split = token.split('-')
                for token_ in token_split:
                    if re.fullmatch(r'[a-zA-Z]{2,}', token_): 
                        filtered_tokens.append(token_)
            if '_' in token:
                token_split = token.split('_')
                for token_ in token_split:
                    if re.fullmatch(r'[a-zA-Z]{2,}', token_): 
                        filtered_tokens.append(token_)
            if '~' in token:
                token_split = token.split('~')
                for token_ in token_split:
                    if re.fullmatch(r'[a-zA-Z]{2,}', token_): 
                        filtered_tokens.append(token_)
            if "'" in token:
                token_split = token.split("'")
                for token_ in token_split:
                    if re.fullmatch(r'[a-zA-Z]{2,}', token_): 
                        filtered_tokens.append(token_)
            if re.fullmatch(r'[a-zA-Z]{2,}', token):
                    filtered_tokens.append(token)
    stems = [stemmer.stem(t) for t in filtered_tokens]
    vocabulary = pd.DataFrame({'words': filtered_tokens}, index=stems)
    return vocabulary


def merge_txt_files(input_path, output_path):
    '''Voeg de txt files per client samen in een nieuw bestand in een nieuw mapje. hoeft maar 1 keer.'''
    for root, dirs, _ in os.walk(input_path):
        for dir_name in dirs:
            text_content = []
            current_dir = os.path.join(root, dir_name)
            for file_name in os.listdir(current_dir):
                file_path = os.path.join(current_dir, file_name)
                if file_name.endswith('.txt') and os.path.isfile(file_path):
                    with open(file_path, 'r', encoding='utf-8') as file:
                        text_content.append(file.read())

            output_file_path = os.path.join(output_path, f'{dir_name}_GHZ_merged.txt')
            with open(output_file_path, 'w', encoding='utf-8') as output_file:
                output_file.write('\n'.join(text_content))


def process_txt_files(input_directory):
    # Create lists to store the content and file names
    all_text_content = []
    file_names = []

    # Loop through each file in the directory
    for file_name in os.listdir(input_directory):
        file_path = os.path.join(input_directory, file_name)
        if file_name.endswith(".txt") and os.path.isfile(file_path):
            with open(file_path, "r", encoding="utf-8") as file:
                all_text_content.append(file.read())
                file_names.append(file_name)
    return all_text_content, file_names


def statistics(GHZ_data, VVT_data, train_data):
    '''In deze functie worden alle variabelen in de gegeven files met elkaar vergeleken middels Student's t-test.
    De variabelen die significant van elkaar verschillen (p<0.05) worden in een dataframe gezet met gemiddelde,
    standaard deviatie en p-waarde. Deze dataframe is de output van de functie'''
    df_p = pd.DataFrame({'Features': GHZ_data.keys()})
    for key in GHZ_data.keys():
        # Chi square binair
        # _, p, _, _ = chi2_contingency(pd.crosstab(train_data['Label'], train_data[key]))
        
        # Mann Whitney U TD-IDF
        _, p = mannwhitneyu(GHZ_data[key], VVT_data[key])

        df_p.loc[df_p['Features'] == key, 'P-value'] = p
        mean_VVT = np.round(VVT_data[key].mean(), decimals=2)
        std_VVT = np.round(VVT_data[key].std(), decimals=2)
        mean_GHZ = np.round(GHZ_data[key].mean(), decimals=2)
        std_GHZ = np.round(GHZ_data[key].std(), decimals=2)
        df_p.loc[df_p['Features'] == key, 'Mean VVT'] = mean_VVT
        df_p.loc[df_p['Features'] == key, 'Std VVT'] = std_VVT
        df_p.loc[df_p['Features'] == key, 'Mean GHZ'] = mean_GHZ
        df_p.loc[df_p['Features'] == key, 'Std GHZ'] = std_GHZ
    df_p_sorted = df_p.sort_values(by=['P-value'])
    df_p_sorted['Rank'] = range(1, len(df_p_sorted)+1)    # Rank the features
    df_p_sorted['Significance level'] = 0.05/(len(df_p_sorted)+1-df_p_sorted['Rank'])    # Calculate the significance level per feature
    df_p_sorted['Significant'] = np.where(df_p_sorted['P-value'] < df_p_sorted['Significance level'], 'Yes', 'No')
    df_p_sign = df_p_sorted.loc[df_p_sorted['Significant'] == 'Yes']
    df_p_for_table = df_p_sign.drop(['Rank', 'Significant'], axis=1)
    return df_p_for_table


def cross_val_stat(cv_dataframe, cv, dict):
    for i, (train_index, _) in enumerate(cv.split(cv_dataframe, labels)):
        train_data = cv_dataframe.iloc[train_index]
        grouped = train_data.groupby('Label')
        df_GHZ = grouped.get_group(1)
        df_GHZ.drop(['Label'], axis=1)
        df_VVT = grouped.get_group(0)
        df_VVT.drop(['Label'], axis=1)
        df_p_for_table = statistics(df_GHZ, df_VVT, train_data)
        dict[f'df_{i}'] = df_p_for_table

    feature_totals = {}
    feature_counts = {}

    # Loop door elke dataframe in de dictionary
    for df in dict.values():
        for _, row in df.iterrows():
            feature = row['Features']
            if feature in feature_totals:
                feature_totals[feature] = {col: feature_totals[feature].get(col, 0) + row[col] for col in df.columns[1:]}
                feature_counts[feature] += 1
            else:
                feature_totals[feature] = {col: row[col] for col in df.columns[1:]}
                feature_counts[feature] = 1
    result_list = []

    # Vul de resultaten dataframe met gemiddelde waarden
    for feature, total in feature_totals.items():
        count = feature_counts[feature]
        if count > 5:
            avg_values = {col: total[col] / count for col in total}
            avg_values['Features'] = feature
            result_list.append(avg_values)
    result_dataframe = pd.DataFrame(result_list)[['Features'] + [col for col in result_list[0] if col != 'Features']]
    return result_dataframe


def characteristics_text(input_directory):
    '''Aantal tokens per bestand en per client berekenen. en mediaan aantal bestanden per client.'''
    num_files_per_client = []

    for _, _, files in os.walk(input_directory):
        num_files = len(files)
        num_files_per_client.append(num_files)

    num_files_per_client.pop(0)  # eerste mapje is 0
    num_files_per_client.sort()

    len_filtered_tokens = []
    # files = glob.glob(os.path.join(input_directory, '**/*'))    #voor losse files van de clienten
    files = os.listdir(input_directory)     # voor de merged client files
    for file_name in files:
        file_path = os.path.join(input_directory, file_name)
        if file_name.endswith(".txt") and os.path.isfile(file_path):
            with open(file_path, "r", encoding="utf-8") as file:
                    text = file.read()

            tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
            filtered_tokens = []

            # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
            for token in tokens:
                if re.fullmatch('[a-zA-Z]+', token): #Hier worden alleen tokens meegnomen waar alleen letters in zitten. getallen worden weggefilterd.
                    filtered_tokens.append(token)
            len_filtered_tokens.append(len(filtered_tokens))
    return num_files_per_client, len_filtered_tokens

# Define input paths
input_directory_GHZ = '/GHZ_a'
input_directory_VVT = '/VVT_a'

# Merge all files per client in one txt file.
merged_path = '/all_merged' # het pad waar alle merged bestanden terecht komen.
merge_txt_files(input_directory_GHZ, merged_path)
merge_txt_files(input_directory_VVT, merged_path)

# Calculate amount of files and words per client
input_directory_GHZ_merged = '/GHZ_merged'
input_directory_VVT_merged = '/VVT_merged'
num_files_per_client_GHZ, _ = characteristics_text(input_directory_GHZ)
num_files_per_client_VVT, _ = characteristics_text(input_directory_VVT)
_, len_filtered_tokens_GHZ = characteristics_text(input_directory_GHZ_merged)
_, len_filtered_tokens_VVT = characteristics_text(input_directory_VVT_merged)
X_GHZ = [i for i in num_files_per_client_GHZ if i != 0]
median_files_GHZ = np.median(X_GHZ)
X_VVT = [i for i in num_files_per_client_VVT if i != 0]
median_files_VVT = np.median(X_VVT)
_, p_files = mannwhitneyu(num_files_per_client_GHZ, num_files_per_client_VVT)
_, p_words = mannwhitneyu(len_filtered_tokens_GHZ, len_filtered_tokens_VVT)

# Create lists to store the text content and file names for further processing
all_text_content, file_names = process_txt_files(merged_path)

# Define ID/ no ID labels
labels = []

for file in file_names:
    if 'VVT' in file:
        labels.append(0)
    if 'GHZ' in file:
        labels.append(1)

# Define stemmer, stopwords, crossvalidation and dictionaries
stemmer = SnowballStemmer('dutch')
stopwords = stopwords.words('dutch') + ['filtered', 'naam', 'plaats', 'adres', 'land', 'email', 'postcode', 'phone', 'bsn', 'meneer', 'mr', 'mevrouw', 'mw']
stemmed_stopwords = [stemmer.stem(word) for word in stopwords]
cv = model_selection.StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
dict_bin = {}
dict_tfidf = {}

# Calcuate TF-IDF features
tfidf = TfidfVectorizer(ngram_range=(1, 1), stop_words=stemmed_stopwords, tokenizer=tokenize_and_stem, max_df=0.8, min_df=0.2, token_pattern=None)
Count_data_tfidf = tfidf.fit_transform(all_text_content)
cv_dataframe_tfidf = pd.DataFrame(Count_data_tfidf.toarray(), columns=tfidf.get_feature_names_out())
cv_dataframe_tfidf.insert(1, 'Label', labels)

# Calculate significant features and save to Excel file for further examination
sign_features_tfidf = cross_val_stat(cv_dataframe_tfidf, cv, dict_tfidf)
sign_features_tfidf.to_excel('Sign_features_tfidf.xlsx')




