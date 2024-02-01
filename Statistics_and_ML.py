import re
import os
import pandas as pd
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn import model_selection
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from statistics import mean
import numpy as np
from scipy import stats
from sklearn.metrics import confusion_matrix
from scipy.stats import chi2_contingency
from scipy.stats import mannwhitneyu
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import RocCurveDisplay, auc
from matplotlib import pyplot
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import LearningCurveDisplay
import math
import nltk


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


def cross_val_stat(cv_dataframe, labels, cv, dict):
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


def baseline(df_0_baseline, df_1_baseline, df_baseline_combined):
    # First calculate the means and stds of age in the two groups, rounded to two decimals
    mean_age_0 = np.round(df_0_baseline['Leeftijd'].mean(), decimals=2)
    std_age_0 = np.round(df_0_baseline['Leeftijd'].std(), decimals=2)
    mean_age_1 = np.round(df_1_baseline['Leeftijd'].mean(), decimals=2)
    std_age_1 = np.round(df_1_baseline['Leeftijd'].std(), decimals=2)
    # Next, find the percentage of females per group
    f_gender_0 = (df_0_baseline['Geslacht'].sum())/len(df_0_baseline)
    f_gender_1 = (df_1_baseline['Geslacht'].sum())/len(df_1_baseline)
    # Calculate the difference in gender with a Chi-square and the difference in age with a Student's t-test
    chi_table = pd.crosstab(df_baseline_combined['Label'], df_baseline_combined['Geslacht'])
    _, p_gender, _, _ = chi2_contingency(chi_table)
    # print(p_gender)
    _, p_age = stats.ttest_ind(df_0_baseline['Leeftijd'], df_1_baseline['Leeftijd'])

    # Combine the calculated values into a dictionary, that is converted to a dataframe for visualisation.
    dict_table = {'Amount of patients': [f'N={len(df_1_baseline)}', f'N={len(df_0_baseline)}', ' '],
                  'Age': [f'{mean_age_1} ± {std_age_1}', f'{mean_age_0} ± {std_age_0}', p_age],
                  'Gender': [f'{np.round(f_gender_1*100, decimals=0)}% females (N={np.round(f_gender_1*len(df_1_baseline), decimals=0)})',
                             f'{np.round(f_gender_0*100, decimals=0)}% females (N={np.round(f_gender_0*len(df_0_baseline), decimals=0)})', p_gender]}
    df_characteristics = pd.DataFrame.from_dict(dict_table, orient='index', columns=['ID group', 'no ID group', 'P-value'])
    return df_characteristics


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



def pipeline_model(train_data, train_label, test_data, test_label, i, clf, tprs, aucs, tns, tps, fps, fns, spec, sens, accuracy, axis,
                   filename='model.sav'):
    '''In deze functie wordt een machine learning model ontwikkeld en getest. Dataframes met de train data, train
    labels, test data en test labels moeten als input worden gegeven. Indien het model opgeslagen moet worden,
    moet een filename als input worden gegeven. Metrics terecht-positieven (tp),
    terecht-negatieven (tn), fout-positieven (fp), fout-negatieven (fn), sensitiviteit, specificiteit en
    accuraatheid worden als input gegeven, aangevuld bij elke fold van de cross-validatie en als output gegeven.'''
    clf.fit(train_data, train_label)
    # Uncomment deze om het model op te slaan.
    # pickle.dump(clf, open(filename, 'wb'))
    predicted = clf.predict(test_data)

    # plot ROC-curve per fold
    mean_fpr = np.linspace(0, 1, 100)    # Help for plotting the false positive rate
    viz = RocCurveDisplay.from_estimator(clf, test_data, test_label, name='ROC fold {}'.format(i+1), alpha=0.3, lw=1, ax=axis)    # Plot the ROC-curve for this fold on the specified axis.
    interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)    # Interpolate the true positive rate
    interp_tpr[0] = 0.0    # Set the first value of the interpolated true positive rate to 0.0
    tprs.append(interp_tpr)   # Append the interpolated true positive rate to the list
    aucs.append(viz.roc_auc)    # Append the area under the curve to the list

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


def calculate_lc(estimator, X, y, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.01, 1.0, 20)):
    train_sizes, train_scores, test_scores = \
    learning_curve(estimator, X, y, cv=None, n_jobs=n_jobs,
                       train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    return train_sizes, train_scores_mean, test_scores_mean


def plot_learning_curve(axes, title, train_sizes, train_scores_mean_all, test_scores_mean_all):
    train_scores_mean_cros = [np.mean(pos) for pos in zip(*train_scores_mean_all)]  ## sterretje zorgt dat de elementen van lists op dezelfde positie in de lists samen worden gegroepeerd.
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


print('finish')