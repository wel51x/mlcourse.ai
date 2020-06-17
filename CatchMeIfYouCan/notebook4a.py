# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in
datadir = "/Users/wel51x/Box Sync/MyBox/Courses/mlcourse.ai/data/kaggle_catch-me-if-you-can/"

import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
import matplotlib.pyplot as plt
import seaborn as sns
pd.set_option('display.max_columns', None) # all cols
pd.set_option('display.width', 161)
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
"""
import os
for dirname, _, filenames in os.walk(datadir):
    for filename in filenames:
        print(os.path.join(dirname, filename))
"""
# a helper function for writing predictions to a file
def write_to_submission_file(predicted_labels, out_file,
                             target='target', index_label="session_id"):
    predicted_df = pd.DataFrame(predicted_labels,
                                index = np.arange(1, predicted_labels.shape[0] + 1),
                                columns=[target])
    predicted_df.to_csv(out_file, index_label=index_label)

train_df = pd.read_csv(datadir + '/train_sessions.csv',
                       index_col='session_id')
test_df = pd.read_csv(datadir + '/test_sessions.csv',
                      index_col='session_id')

# Convert time1, ..., time10 columns to datetime type
times = ['time%s' % i for i in range(1, 11)]
train_df[times] = train_df[times].apply(pd.to_datetime)
test_df[times] = test_df[times].apply(pd.to_datetime)

# Sort the data by time
train_df = train_df.sort_values(by='time1')

print(train_df.shape, test_df.shape, '\n', train_df.info())

# Look at the first rows of the training set
#print(train_df)

sites = ['site%s' % i for i in range(1, 11)]
'''
train_df[sites].fillna(0).astype('int').to_csv(datadir + '/train_sessions_text.txt',
                                               sep=' ',
                       index=None, header=None)
test_df[sites].fillna(0).astype('int').to_csv(datadir + '/test_sessions_text.txt',
                                              sep=' ',
                       index=None, header=None)
'''
train_df[sites].fillna(0).astype('int')
test_df[sites].fillna(0).astype('int')

cv = CountVectorizer()

with open(datadir + '/train_sessions_text.txt') as inp_train_file:
    X_train = cv.fit_transform(inp_train_file)
with open(datadir + '/test_sessions_text.txt') as inp_test_file:
    X_test = cv.transform(inp_test_file)

print(X_train.shape, X_test.shape, type(X_train))

y_train = train_df['target'].astype(int)
print(y_train)

logit = LogisticRegression(C = 1, random_state=17, verbose=True)

cv_scores = cross_val_score(logit, X_train, y_train, cv=5, scoring='roc_auc')

print("cv mean =", cv_scores.mean())

x = logit.fit(X_train, y_train)
print(x)

test_pred_logit1 = logit.predict_proba(X_test)[:,1]
print(test_pred_logit1)

write_to_submission_file(test_pred_logit1, datadir + '/logit_sub001a.txt') ## .908 ROC AUC; Kaggle score 0.90804
# Your submission scored 0.90703, which is not an improvement of your best score.

### Time Features

def add_time_features(time1_series, X_sparse):
    hour = time1_series.apply(lambda ts: ts.hour)
    morning = ((hour >= 7) & (hour <= 11)).astype('int')
    day = ((hour >= 12) & (hour <= 18)).astype('int')
    evening = ((hour >= 19) & (hour <= 23)).astype('int')
    night = ((hour >= 0) & (hour <= 6)).astype('int')
    # stack features together
    X = hstack([X_sparse, morning.values.reshape(-1, 1), # why reshape?
                day.values.reshape(-1, 1),
                evening.values.reshape(-1, 1),
                night.values.reshape(-1, 1)])
    return X

test_df.loc[:, 'time1'].fillna(0).apply(lambda ts: ts.hour).head()

X_train_with_time = add_time_features(train_df['time1'].fillna(0), X_train)
X_test_with_time = add_time_features(test_df['time1'].fillna(0), X_test)

logit_with_time = LogisticRegression(C = 1, random_state=17, verbose=True)
cv_scores = cross_val_score(logit_with_time, X_train_with_time, y_train, cv= 5, scoring='roc_auc')

print("cv mean =", cv_scores.mean())

x = logit_with_time.fit(X_train_with_time, y_train)
print(x)

test_pred_logit2 = logit_with_time.predict_proba(X_test_with_time)[:,1]
write_to_submission_file(test_pred_logit2, datadir + '/logit_sub002a.txt') ## .93565 ROC AUC; Kaggle score 0.93565
# Your submission scored 0.93523, which is not an improvement of your best score.

c_values = np.logspace(-2, 2, 10)
c_values = np.arange(0.545, 0.556, .001)
time_split = TimeSeriesSplit(n_splits=10)

logit_grid = GridSearchCV(estimator=logit, param_grid={'C': c_values},
                          scoring="roc_auc", cv=time_split, n_jobs=-1, verbose=True)

x = logit_grid.fit(X_train_with_time, y_train)
print(x)
print(logit_grid.best_score_, logit_grid.best_params_)
# 0.9158322460209616 {'C': 0.5499999999999999}

test_pred_logit3 = logit_grid.predict_proba(X_test_with_time)[:,1]

write_to_submission_file(test_pred_logit3, datadir + '/logit_sub003a.txt') ## 0.9155 ROC AUC; Kaggle score 0.93740
# Your submission scored 0.93770, which is an improvement of your previous score of 0.93740
