'''This file contains the functions developed for calculating statistics and training and evaluating 
machine learning models.'''

import pandas as pd
import numpy as np
import re
import os
import nltk
from statistics import mean
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from scipy import stats
from scipy.stats import chi2_contingency
from scipy.stats import mannwhitneyu
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import learning_curve
from sklearn.metrics import RocCurveDisplay
import matplotlib.pyplot as plt


def baseline(baseline_VVT, baseline_GHZ, baseline_combined):
    '''In this function, mean and std of age is calculated and the groups are statistically compared.
    Furtermore, gender is statistically compared. A dataframe with characteristics and p-value is returned.'''
    # First calculate the means and stds of age in the two groups, rounded to two decimals
    mean_age_0 = np.round(baseline_VVT['Leeftijd'].mean(), decimals=2)
    std_age_0 = np.round(baseline_VVT['Leeftijd'].std(), decimals=2)
    mean_age_1 = np.round(baseline_GHZ['Leeftijd'].mean(), decimals=2)
    std_age_1 = np.round(baseline_GHZ['Leeftijd'].std(), decimals=2)
    # Next, find the percentage of females per group
    f_gender_0 = (baseline_VVT['Geslacht'].sum())/len(baseline_VVT)
    f_gender_1 = (baseline_GHZ['Geslacht'].sum())/len(baseline_GHZ)
    # Calculate the difference in gender with a Chi-square and the difference in age with a Student's t-test
    chi_table = pd.crosstab(baseline_combined['Label'], baseline_combined['Geslacht'])
    _, p_gender, _, _ = chi2_contingency(chi_table)
    _, p_age = stats.ttest_ind(baseline_VVT['Leeftijd'], baseline_GHZ['Leeftijd'])

    # Combine the calculated values into a dictionary, that is converted to a dataframe for visualisation.
    dict_table = {'Amount of patients': [f'N={len(baseline_GHZ)}', f'N={len(baseline_VVT)}', ' '],
                  'Age': [f'{mean_age_1} ± {std_age_1}', f'{mean_age_0} ± {std_age_0}', p_age],
                  'Gender': [f'{np.round(f_gender_1*100, decimals=0)}% females (N={np.round(f_gender_1*len(baseline_GHZ), decimals=0)})',
                             f'{np.round(f_gender_0*100, decimals=0)}% females (N={np.round(f_gender_0*len(baseline_VVT), decimals=0)})', p_gender]}
    df_characteristics = pd.DataFrame.from_dict(dict_table, orient='index', columns=['ID group', 'no ID group', 'P-value'])
    return df_characteristics


def characteristics_text(input_directory):
    '''Calculate the amount of files and words per client.'''
    num_files_per_client = []

    for _, _, files in os.walk(input_directory):
        num_files = len(files)
        num_files_per_client.append(num_files)

    num_files_per_client.pop(0)
    num_files_per_client.sort()

    len_filtered_tokens = []
    # files = glob.glob(os.path.join(input_directory, '**/*'))    # For separate client files
    files = os.listdir(input_directory)     # For merged client files
    for file_name in files:
        file_path = os.path.join(input_directory, file_name)
        if file_name.endswith(".txt") and os.path.isfile(file_path):
            with open(file_path, "r", encoding="utf-8") as file:
                    text = file.read()

            tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
            filtered_tokens = []

            # Filter out any tokens not containing letters
            for token in tokens:
                if re.fullmatch('[a-zA-Z]+', token):
                    filtered_tokens.append(token)
            len_filtered_tokens.append(len(filtered_tokens))
    return num_files_per_client, len_filtered_tokens


def statistics(GHZ_data, VVT_data, feature_type, train_data):
    '''In this function, features in the input files can be statistically compared between the ID- and no-ID groups.
    For TF-IDF features, a Mann Whitney U test is performed and for clinical concepts, a Chi-square test is performed.
    The significantly different features are placed in a dataframe including mean, std and the p-value. The p-values are corrected
    with Holm-Bonferroni.'''

    df_p = pd.DataFrame({'Features': GHZ_data.keys()})
    for key in GHZ_data.keys():
        if feature_type == 'Clinical concepts':
            _, p, _, _ = chi2_contingency(pd.crosstab(train_data['Label'], train_data[key]))
        
        if feature_type == 'TF-IDF':
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


def cross_val_stat(cv_dataframe, labels, cv, dict, feature_type):
    '''This function allows for cross-validation of the calculated statistics on the trainsets.'''
    for i, (train_index, _) in enumerate(cv.split(cv_dataframe, labels)):
        train_data = cv_dataframe.iloc[train_index]
        grouped = train_data.groupby('Label')
        df_GHZ = grouped.get_group(1)
        df_GHZ.drop(['Label'], axis=1)
        df_VVT = grouped.get_group(0)
        df_VVT.drop(['Label'], axis=1)
        df_p_for_table = statistics(df_GHZ, df_VVT, feature_type, train_data)
        # Dataframes for every fold are stored in a dictionary
        dict[f'df_{i}'] = df_p_for_table

    feature_totals = {}
    feature_counts = {}

    # Loop through every dataframe in de dictionary
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

    # Fill in the results dataframe with mean values
    for feature, total in feature_totals.items():
        count = feature_counts[feature]
        if count > 5:
            avg_values = {col: total[col] / count for col in total}
            avg_values['Features'] = feature
            result_list.append(avg_values)
    result_dataframe = pd.DataFrame(result_list)[['Features'] + [col for col in result_list[0] if col != 'Features']]
    return result_dataframe


def pipeline_model(train_data, train_label, test_data, test_label, i, clf, tprs, aucs, tns, tps, fps, fns, spec, sens, accuracy, axis):
    '''In this function, a ML model is trained and tested. Scoring metrics are returned and appended every fold.'''
    clf.fit(train_data, train_label)
    predicted = clf.predict(test_data)

    # Plot ROC-curve per fold
    mean_fpr = np.linspace(0, 1, 100)
    viz = RocCurveDisplay.from_estimator(clf, test_data, test_label, name='ROC fold {}'.format(i+1), alpha=0.3, lw=1, ax=axis)
    interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
    interp_tpr[0] = 0.0
    tprs.append(interp_tpr)
    aucs.append(viz.roc_auc)

    tn, fp, fn, tp = confusion_matrix(test_label, predicted).ravel()
    tns.append(tn)
    tps.append(tp)
    fps.append(fp)
    fns.append(fn)
    spec.append(tn/(tn+fp))
    sens.append(tp/(tp+fn))
    accuracy.append(metrics.accuracy_score(test_label, predicted))
    return tprs, aucs, tns, tps, fps, fns, spec, sens, accuracy


def mean_ROC_curves(tprs, aucs, axis):
    '''With this function, the mean ROC-curves of the models over a 10-cross-validation are plot.
    The true positive rates, areas under the curve and axes where the mean ROC-curve must be plot
    are given as input.'''
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_fpr = np.linspace(0, 1, 100)
    mean_auc = metrics.auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    std_tpr = np.std(tprs, axis=0)
    axis.plot(mean_fpr, mean_tpr, color='b', label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc), lw=2, alpha=.8)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)    # Set the upper value of the true positive rates
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)    # Set the upper value of the true positive rates
    axis.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2, label=r'$\pm$ 1 std')    # Plot the standard deviations of the ROC-curves
    axis.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05], title=f'ROC-curves clinical concepts')    # Set axes and title
    axis.set_xlabel('False Positive Rate (1 - specificity)')
    axis.set_ylabel('True Positive Rate (sensitivity)')
    axis.legend(loc="lower right")    # Set legend
    return


def calculate_lc(clf, train_data, train_label, train_sizes=np.linspace(.01, 1.0, 20)):
    '''In this function, a learning curve of an estimator is created.'''
    train_sizes, train_scores, test_scores = \
    learning_curve(clf, train_data, train_label, cv=None, n_jobs=None,
                       train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    return train_sizes, train_scores_mean, test_scores_mean


def plot_learning_curve(axes, title, train_sizes, train_scores_mean_all, test_scores_mean_all):
    '''In this function, the mean learning curve of all folds is created and a plot is returned.'''
    train_scores_mean_cros = [np.mean(pos) for pos in zip(*train_scores_mean_all)]
    test_scores_mean_cros = [sum(pos)/len(pos) for pos in zip(*test_scores_mean_all)]
    train_scores_std_cros = [np.std(pos) for pos in zip(*train_scores_mean_all)]
    test_scores_std_cros = [np.std(pos) for pos in zip(*test_scores_mean_all)]

    ylim = (0, 1.01)
    axes.set_title(title)
    if ylim is not None:
        axes.set_ylim(*ylim)

    axes.set_xlabel("Examples trainset")
    axes.set_ylabel("Accuracy")
    axes.grid()

    axes.plot(train_sizes, train_scores_mean_cros, 'o-', color="r",
                label="Mean train score")
    axes.fill_between(train_sizes, np.array(train_scores_mean_cros) - np.array(train_scores_std_cros),
                        np.array(train_scores_mean_cros) + np.array(train_scores_std_cros), alpha=.2,
                        color="r", label=r'$\pm$ 1 std train score')
    axes.plot(train_sizes, test_scores_mean_cros, 'o-', color="g",
                label="Mean cross-validation score")
    axes.fill_between(train_sizes, np.array(test_scores_mean_cros) - np.array(test_scores_std_cros),
                        np.array(test_scores_mean_cros) + np.array(test_scores_std_cros), alpha=.2,
                        color="g", label=r'$\pm$ 1 std cross-validation score')
    axes.legend(loc="lower right")
    return plt