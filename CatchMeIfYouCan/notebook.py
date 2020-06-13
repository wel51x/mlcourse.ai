from __future__ import division, print_function
catchmedir = "/Users/wel51x/temp/CatchMeIfYouCan/"
datadir = "/Users/wel51x/Box Sync/MyBox/Courses/mlcourse.ai/data/kaggle_catch-me-if-you-can/"
#import warnings
#warnings.filterwarnings('ignore')
import os
import pickle
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import roc_auc_score
from sklearn import feature_extraction
from sklearn.preprocessing import StandardScaler
from scipy.sparse import hstack
from sklearn.linear_model import LogisticRegression
"""
train = pd.read_csv(datadir + 'train_sessions.csv', index_col='session_id')
test = pd.read_csv(os.path.join(datadir, 'test_sessions.csv'), index_col='session_id')
#exit(2)
# combine the two
sum_data = pd.concat([train, test])

# misc pre-processing
site_cols = ['site%d' % i for i in range(1, 11)]
time_cols = ['time%d' % i for i in range(1, 11)]

sum_data[site_cols] = sum_data[site_cols].fillna(0).astype(np.int).astype(np.str)
sum_data[time_cols] = sum_data[time_cols].apply(pd.to_datetime)

print(sum_data)

# create a discharged matrix in the form of a "word bag" on the sites
def join_str(row):
    return ' '.join(row)

site_text_data = sum_data[site_cols].apply(join_str, axis=1)
print('Number of sessions: {}'.format(site_text_data.shape[0]))

# vectorize with tfidf
vectorizer = feature_extraction.text.TfidfVectorizer()
sum_data_site_sparse = vectorizer.fit_transform(site_text_data)

print('Sparse matrix dimensions: {}'.format(sum_data_site_sparse.shape))

# Create new properties:
# session_timespan
# unique_sites
# day_of_week
# start_hour

def calc_session_timespan(row):
    timestamps = row[time_cols].values
    session_timespan = timestamps.max() - timestamps.min()

    return session_timespan.total_seconds()


def calc_unique_sites(row):
    sites_vals = row[site_cols].values

    return len(np.unique([a for a in sites_vals if int(a) > 0]))


def calc_day_of_week(row):
    timestamps = row[time_cols].values
    return timestamps.min().weekday()


def calc_start_hour(row):
    timestamps = row[time_cols].values
    return timestamps.min().hour


def calc_end_hour(row):
    timestamps = row[time_cols].values
    return timestamps.max().hour


def calc_day_of_month(row):
    timestamps = row[time_cols].values
    return timestamps.min().day


def calc_month(row):
    timestamps = row[time_cols].values
    return timestamps.min().month


def calc_is_weekend(row):
    day_of_week = row['day_of_week']
    if day_of_week == 6 or day_of_week == 5:
        return 1

    return 0

# run long!!
#%%time
sum_data['unique_sites'] = sum_data.apply(calc_unique_sites, axis=1)
sum_data['session_timespan'] = sum_data.apply(calc_session_timespan, axis=1)
sum_data['day_of_week'] = sum_data.apply(calc_day_of_week, axis=1)
sum_data['start_hour'] = sum_data.apply(calc_start_hour, axis=1)
sum_data['end_hour'] = sum_data.apply(calc_end_hour, axis=1)
sum_data['month'] = sum_data.apply(calc_month, axis=1)
sum_data['day_of_month'] = sum_data.apply(calc_day_of_month, axis=1)
sum_data['is_weekend'] = sum_data.apply(calc_is_weekend, axis=1)


# identify bad data
def print_empty_cell(collection, name):
    total_row = collection.shape[0]
    data_count = collection.count().sort_values(ascending=True)

    i = 0
    str_val = []
    for item, value in data_count.items():
        if value < total_row:
            str_val.append("{}:{}".format(item, total_row - value))
            i += 1

    if i > 0:
        print("--> invalid features in {}:".format(name))
        for s in str_val:
            print(s)
    else:
        print("--> success data in {}:".format(name))


print_empty_cell(sum_data, 'sum_data')

# clean it and take a look
sum_data['day_of_week'] = sum_data['day_of_week'].fillna(round(sum_data['day_of_week'].mean())).astype(np.int)
sum_data['start_hour'] = sum_data['start_hour'].fillna(round(sum_data['start_hour'].mean())).astype(np.int)
sum_data['end_hour'] = sum_data['end_hour'].fillna(round(sum_data['end_hour'].mean())).astype(np.int)
sum_data['month'] = sum_data['month'].fillna(round(sum_data['month'].mean())).astype(np.int)
sum_data['day_of_month'] = sum_data['day_of_month'].fillna(round(sum_data['day_of_month'].mean())).astype(np.int)
sum_data['session_timespan'] = sum_data['session_timespan'].fillna(round(sum_data['session_timespan'].mean())).astype(np.int)
sum_data['start_site'] = sum_data['site1'].astype(np.int)
sum_data['is_weekend'] = sum_data['is_weekend'].fillna(round(sum_data['is_weekend'].mean())).astype(np.int)
print(sum_data)

print(sum_data.shape)
sum_data = pd.get_dummies(sum_data, columns=['day_of_week', 'start_hour', 'end_hour', 'month', 'day_of_month'])
print(sum_data.shape)
"""
train = pd.read_csv(datadir + 'train_sessions.csv', index_col='session_id')
test = pd.read_csv(os.path.join(datadir, 'test_sessions.csv'), index_col='session_id')
#exit(2)
# combine the two
sum_data = pd.concat([train, test])

# misc pre-processing
site_cols = ['site%d' % i for i in range(1, 11)]
time_cols = ['time%d' % i for i in range(1, 11)]

sum_data[site_cols] = sum_data[site_cols].fillna(0).astype(np.int).astype(np.str)
sum_data[time_cols] = sum_data[time_cols].apply(pd.to_datetime)

print(sum_data)

# create a discharged matrix in the form of a "word bag" on the sites
def join_str(row):
    return ' '.join(row)

site_text_data = sum_data[site_cols].apply(join_str, axis=1)
print('Number of sessions: {}'.format(site_text_data.shape[0]))

# vectorize with tfidf
vectorizer = feature_extraction.text.TfidfVectorizer()
sum_data_site_sparse = vectorizer.fit_transform(site_text_data)

print('Sparse matrix dimensions: {}'.format(sum_data_site_sparse.shape))

# now read temp file
sum_data = pd.read_csv(catchmedir + "sum_data.csv", index_col=0)
print(sum_data.shape)
#print(sum_data)

day_of_week_cols = sum_data.filter(like='day_of_week').columns
start_hour_cols = sum_data.filter(like='start_hour').columns
end_hour_cols = sum_data.filter(like='end_hour').columns
day_of_month_cols = sum_data.filter(like='day_of_month').columns
month_cols = ['month_1','month_2','month_3','month_4','month_5','month_6',
              'month_7','month_8','month_9','month_10','month_11','month_12']
print("day_of_week_cols\n", day_of_week_cols)
print("start_hour_cols\n", start_hour_cols)
print("end_hour_cols\n", end_hour_cols)
print("day_of_month_cols\n", day_of_month_cols)
print("month_cols\n", month_cols)

# more features
additional_cols = np.hstack((['unique_sites', 'start_site', 'session_timespan', 'is_weekend'],
                             day_of_week_cols,
                             start_hour_cols,
                             end_hour_cols,
                             day_of_month_cols,
                             month_cols
                            ))

# write temp
# sum_data.to_csv("sum_data.csv")

standard_scaler = StandardScaler()
scaler_sum_data_2 = standard_scaler.fit_transform(sum_data[additional_cols])

additional_data = csr_matrix(scaler_sum_data_2)
print('additional_data shape: {}'.format(additional_data.shape))
print('sum_data_site_sparse shape: {}'.format(sum_data_site_sparse.shape))
print('sum_data shape: {}'.format(sum_data.shape))

# Combine additional_data, sum_data_site_sparse and select the training and test sets
temp = hstack((sum_data_site_sparse, additional_data))
print('combined shape: {}'.format(temp.shape))

X_train = temp.tocsc()[:train.shape[0]]
y_train = train['target']
X_test = temp.tocsc()[train.shape[0]:]

print("X_train.shape =", X_train.shape, "y_train.shape =", y_train.shape,
      "X_test.shape =", X_test.shape)

X_train_tmp, X_test_tmp, y_train_tmp, y_test_tmp = train_test_split(X_train, y_train, test_size=0.3, random_state=17)
print("X_train_tmp.shape =", X_train_tmp.shape, "y_train_tmp.shape =", y_train_tmp.shape,
      "X_test_tmp.shape =", X_test_tmp.shape, "y_test_tmp.shape =", y_test_tmp.shape)

sgd_logit = SGDClassifier(loss='log', random_state=17, n_jobs=-1)

sgd_logit.fit(X_train_tmp, y_train_tmp)

y_pred = sgd_logit.predict_proba(X_test_tmp)[:, 1]

roc = roc_auc_score(y_test_tmp, y_pred)
print('SGDClassifier ROC AUC: {}'.format(round(roc, 4)))

sgd_logit.fit(X_train, y_train)
y_pred = sgd_logit.predict_proba(X_test)[:, 1]

def write_to_submission_file(predicted_labels, out_file,
                             target='target', index_label="session_id"):
    predicted_df = pd.DataFrame(predicted_labels,
                                index = np.arange(1, predicted_labels.shape[0] + 1),
                                columns=[target])
    predicted_df.to_csv(out_file, index_label=index_label)

write_to_submission_file(y_pred, catchmedir + 'SGDClassifier_y_pred.CSV') # 0.85692 on kaggle

reg_logit = LogisticRegression(random_state=17, n_jobs=-1, max_iter=200)
reg_logit.fit(X_train_tmp, y_train_tmp)

y_pred = reg_logit.predict_proba(X_test_tmp)[:, 1]

roc = roc_auc_score(y_test_tmp, y_pred)

print('LogisticRegression ROC AUC: {}'.format(round(roc, 4)))

reg_logit.fit(X_train, y_train)
y_pred = reg_logit.predict_proba(X_test)[:, 1]

write_to_submission_file(y_pred, catchmedir + 'LogisticRegression_y_pred.CSV') # 0.88027 on kaggle
