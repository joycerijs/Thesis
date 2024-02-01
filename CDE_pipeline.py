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
from Train_test_model import pipeline_model
from Train_test_model import mean_ROC_curves
from Train_test_model import calculate_lc
from Train_test_model import plot_learning_curve
import math

print('start')

# Load the UMLS features obtained from Medspacy
# df_GHZ = pd.read_csv('/GHZ_features_UMLS.csv', index_col=0)
# df_VVT = pd.read_csv('/VVT_features_UMLS.csv', index_col=0)
df_GHZ = pd.read_csv('F:/Documenten/Universiteit/Master_TM+_commissies/Jaar 3/Afstuderen/Thesis/MedSpacy/run_GHZ_1312_features_UMLS.csv', index_col=0)
df_VVT = pd.read_csv('F:/Documenten/Universiteit/Master_TM+_commissies/Jaar 3/Afstuderen/Thesis/MedSpacy/run_VVT_1312_features_UMLS.csv', index_col=0)
df_GHZ['Label'] = 1
df_VVT['Label'] = 0

# Ensure all codes are available in both dataframes
all_codes = set(df_GHZ.columns).union(df_VVT.columns)
missing_GHZ = list(all_codes.difference(df_GHZ.columns))
missing_VVT = list(all_codes.difference(df_VVT.columns))
df_GHZ = pd.concat([df_GHZ, pd.DataFrame(0, columns=missing_GHZ, index=df_GHZ.index)], axis=1)
df_VVT = pd.concat([df_VVT, pd.DataFrame(0, columns=missing_VVT, index=df_VVT.index)], axis=1)
df_combined = pd.concat([df_GHZ, df_VVT], ignore_index=True)
labels = df_combined['Label']

# Unbiased UMLS codes
unbiased_codes = ['Label', 'C0014544', 'C0543467', 'C1275743', 'C0030551', 'C0233494', 'C0085631', 'C0337527', 'C0018767', 'C0004352', 'C0337514', 'C0015421', 'C0042812', 'C0026591', 'C0920139']

# Unbiased codes UMLS clustered
cv_unbiased_unclustered = df_combined[unbiased_codes].copy()
cv_unbiased_unclustered.loc[:, 'Familie'] = (df_combined['C0030551'] + df_combined['C0337527'] + df_combined['C0337514']+ df_combined['C0026591']).clip(upper=1)
cv_unbiased_unclustered.loc[:, 'Zicht'] = (df_combined['C0015421'] + df_combined['C0920139'] + df_combined['C0042812']).clip(upper=1)
cv_unbiased_unclustered.loc[:, 'Onrust'] = (df_combined['C0233494'] + df_combined['C0085631']).clip(upper=1)
unbiased_codes_clustered = ['Label', 'C0014544', 'C0543467', 'C1275743', 'C0018767', 'C0004352', 'Familie', 'Zicht', 'Onrust']
cv_unbiased = cv_unbiased_unclustered[unbiased_codes_clustered]

# Define cross-validation and dictionary
cv = model_selection.StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
dict_stat = {}

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

for i, (train_index, test_index) in enumerate(cv.split(cv_unbiased, labels)):
    train_data_ = cv_unbiased.iloc[train_index]
    train_data = train_data_.drop(['Label'], axis=1)
    test_data_ = cv_unbiased.iloc[test_index]
    test_data = test_data_.drop(['Label'], axis=1)

    train_label = cv_unbiased['Label'].iloc[train_index]
    test_label = cv_unbiased['Label'].iloc[test_index]

    clf_XGB = GradientBoostingClassifier()
    tprs, aucs, tns, tps, fps, fns, spec, sens, accuracy = \
        pipeline_model(train_data, train_label, test_data, test_label, i, clf_XGB, tprs, aucs, tns, tps, fps, fns,
                        spec, sens, accuracy, axis)
    
    # train_sizes, train_scores_mean, test_scores_mean = calculate_lc(clf_XGB, train_data, train_label, cv)
    # train_scores_mean_all.append(list(train_scores_mean))
    # test_scores_mean_all.append(list(test_scores_mean))


# mean_ROC_curves(tprs, aucs, axis)
# plt.show()
plt.close()

dict_scores = {'Model scores XGB': [f'{np.round(mean(aucs), decimals=2)} ± {np.round(np.std(aucs), decimals=2)}',
                                    f'{np.round(mean(accuracy), decimals=2)} ± {np.round(np.std(accuracy), decimals=2)}',
                                    f'{np.round(mean(sens), decimals=2)} ± {np.round(np.std(sens), decimals=2)}',
                                    f'{np.round(mean(spec), decimals=2)} ± {np.round(np.std(spec), decimals=2)}',
                                    ]}

df_scores = pd.DataFrame.from_dict(dict_scores, orient='index', columns=['AUC', 'Accuracy', 'Sensitivity',
                                                                            'Specificity'])
print(df_scores)

# fig, ax = plt.subplots()
# title = 'Learning curve clinical concepts'
# plot_learning_curve(ax, title, train_sizes, train_scores_mean_all, test_scores_mean_all)
# plt.show()