'''With this script, all bag-of-words processing steps were performed and a ML model was trained and evaluated. 
Furthermore, client age, gender, amount of files and amount of words were statistically compared.'''

import pandas as pd
import numpy as np
import re
import os
import nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from statistics import mean
from scipy.stats import mannwhitneyu
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection
from sklearn.ensemble import GradientBoostingClassifier
import matplotlib.pyplot as plt
from Statistics_and_ML import cross_val_stat
from Statistics_and_ML import baseline
from Statistics_and_ML import characteristics_text
from Statistics_and_ML import pipeline_model
from Statistics_and_ML import mean_ROC_curves
from Statistics_and_ML import calculate_lc
from Statistics_and_ML import plot_learning_curve


def tokenize_and_stem(text):
    '''This function serves as a tokenizer and stemmer for the calculation of TF-IDF features. Stems are returned.'''
    # First tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
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
    '''In this function, all files per client are combined into one file.'''
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
    '''In this function, the texts and filenames are stored into lists for further processing.'''
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


def main():
    # Define input paths
    input_directory_GHZ = '/GHZ_a'
    input_directory_VVT = '/VVT_a'

    # Merge all files per client in one txt file.
    merged_path = '/all_merged' # Path where all merged files will be put in.
    # merge_txt_files(input_directory_GHZ, merged_path)
    # merge_txt_files(input_directory_VVT, merged_path)

    # Load files with age and gender of clients and calculate statistics
    baseline_VVT = pd.read_excel('/overzicht-clienten-VVT-bewerkt.xlsx')
    baseline_GHZ = pd.read_excel('/overzicht-clienten-GHZ-bewerkt.xlsx')
    baseline_combined = pd.read_excel('/overzicht-clienten-beiden-bewerkt.xlsx')
    df_characteristics = baseline(baseline_VVT, baseline_GHZ, baseline_combined)

    # Calculate amount of files and words per client and calculate statistics
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
    dict_table_txt = {'Files per client (median)': [median_files_GHZ, median_files_VVT, p_files],
                    'Words per client (median)': [len_filtered_tokens_GHZ, len_filtered_tokens_VVT, p_words]}
    df_txt_characteristics = pd.DataFrame.from_dict(dict_table_txt, orient='index', columns=['ID group', 'no ID group', 'P-value'])

    # Create lists to store the text content and file names for further processing
    all_text_content, file_names = process_txt_files(merged_path)

    # Define ID/ no ID labels
    labels = []
    for file in file_names:
        if 'VVT' in file:
            labels.append(0)
        if 'GHZ' in file:
            labels.append(1)

    # Define stemmer, stopwords, cross-validation and dictionary
    stemmer = SnowballStemmer('dutch')
    stopwords = stopwords.words('dutch') + ['filtered', 'naam', 'plaats', 'adres', 'land', 'email', 'postcode', 'phone', 'bsn', 'meneer', 'mr', 'mevrouw', 'mw']
    stemmed_stopwords = [stemmer.stem(word) for word in stopwords]
    cv = model_selection.StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    dict_tfidf = {}

    # Calcuate TF-IDF features
    tfidf = TfidfVectorizer(ngram_range=(1, 1), stop_words=stemmed_stopwords, tokenizer=tokenize_and_stem, max_df=0.8, min_df=0.2, token_pattern=None)
    Count_data_tfidf = tfidf.fit_transform(all_text_content)
    cv_dataframe_tfidf = pd.DataFrame(Count_data_tfidf.toarray(), columns=tfidf.get_feature_names_out())
    cv_dataframe_tfidf.insert(1, 'Label', labels)

    # Calculate significant features and save to Excel file for further examination
    sign_features_tfidf = cross_val_stat(cv_dataframe_tfidf, labels, cv, dict_tfidf, 'TF-IDF')
    # sign_features_tfidf.to_excel('Sign_features_tfidf.xlsx')

    # Unbiased significant words chosen (including labels ID/no ID)
    unbiased_words = ['Label', 'emotionel', 'onrust', 'spanning', 'bos', 'rust', 'ontspann',
                    'bril', 'ogen', 'zien', 'draagt', 'oren', 'hoort', 'gehor',
                    'epilepsie', 'bloedonderzoek', 'licham', 'dochter', 'ouder', 'moeder',
                    'vader','zus', 'broer', 'duidelijk', 'geboort', 'vermoed', 'onderzocht',
                    'afgenom', 'prat', 'sprak']

    ### Uncomment for printing full words mapped to stems of unbiased words
    # vocabulary_list_dupl = vocabulary(all_text_content)
    # vocabulary_list = vocabulary_list_dupl.drop_duplicates()

    # for word in unbiased_words:
    #     if word in vocabulary_list.index:
    #         linked_words_total = []
    #         linked_word = vocabulary_list.loc[word, 'words']
    #         if linked_word not in linked_words_total:
    #             linked_words_total.append(linked_word)
    #         print(f'stem: {word}, full word: {linked_words_total}')

    # Create dataframe of unbiased features
    cv_unbiased_stem = cv_dataframe_tfidf[unbiased_words]

    # Train and evaluate ML model
    tprs = []
    aucs = []
    _, axis = plt.subplots()
    tns = []
    tps = []
    fns = []
    fps = []
    spec = []
    sens = []
    accuracy = []
    train_scores_mean_all= []
    test_scores_mean_all= []

    for i, (train_index, test_index) in enumerate(cv.split(cv_unbiased_stem, labels)):
        train_data_ = cv_unbiased_stem.iloc[train_index]
        train_data = train_data_.drop(['Label'], axis=1)
        test_data_ = cv_unbiased_stem.iloc[test_index]
        test_data = test_data_.drop(['Label'], axis=1)

        train_label = cv_unbiased_stem['Label'].iloc[train_index]
        test_label = cv_unbiased_stem['Label'].iloc[test_index]

        clf_XGB = GradientBoostingClassifier(min_samples_split=10, min_samples_leaf=5, max_depth=3, random_state=42)
        tprs, aucs, tns, tps, fps, fns, spec, sens, accuracy = \
            pipeline_model(train_data, train_label, test_data, test_label, i, clf_XGB, tprs, aucs, tns, tps, fps, fns,
                            spec, sens, accuracy, axis)
        
        # Learning curves
        train_sizes, train_scores_mean, test_scores_mean = calculate_lc(clf_XGB, train_data, train_label, cv)
        train_scores_mean_all.append(list(train_scores_mean))
        test_scores_mean_all.append(list(test_scores_mean))

    # ROC curves
    mean_ROC_curves(tprs, aucs, axis)
    plt.show()
    plt.close()

    # Scoring metrics
    dict_scores = {'Model scores XGB': [f'{np.round(mean(aucs), decimals=2)} ± {np.round(np.std(aucs), decimals=2)}',
                                        f'{np.round(mean(accuracy), decimals=2)} ± {np.round(np.std(accuracy), decimals=2)}',
                                        f'{np.round(mean(sens), decimals=2)} ± {np.round(np.std(sens), decimals=2)}',
                                        f'{np.round(mean(spec), decimals=2)} ± {np.round(np.std(spec), decimals=2)}']}
    df_scores = pd.DataFrame.from_dict(dict_scores, orient='index', columns=['AUC', 'Accuracy', 'Sensitivity', 'Specificity'])
    print(df_scores)

    # Mean learning curve
    fig, ax = plt.subplots()
    title = 'Learning curve TF-IDF'
    plot_learning_curve(ax, title, train_sizes, train_scores_mean_all, test_scores_mean_all)
    plt.show()


if __name__ == "__main__":
    main()