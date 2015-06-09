from collections import defaultdict
import os
import time

import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.cross_validation import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, \
    GradientBoostingClassifier, \
    BaggingClassifier, AdaBoostClassifier
from sklearn.metrics import roc_auc_score
import graphlab as gl
import xgboost as xgb

from tools import *


def generate_timediff():
    bids = pd.read_csv('bids.csv')
    bids_grouped = bids.groupby('auction')
    bds = defaultdict(list)
    last_row = None

    for bids_auc in bids_grouped:
        for i, row in bids_auc[1].iterrows():
            if last_row is None:
                last_row = row
                continue

            time_difference = row['time'] - last_row['time']
            bds[row['bidder_id']].append(time_difference)
            last_row = row

    df = []
    for key in bds.keys():
        df.append({'bidder_id': key, 'mean': np.mean(bds[key]),
                   'min': np.min(bds[key]), 'max': np.max(bds[key])})

    pd.DataFrame(df).to_csv('tdiff.csv', index=False)


def generate_features_auc(group):
    return generate_features(group, auction=True)


def generate_features(group, auction=False):
    time_diff = np.ediff1d(group['time'])

    if len(time_diff) == 0:
        diff_mean = 0
        diff_std = 0
        diff_median = 0
        diff_zeros = 0
    else:
        diff_mean = np.mean(time_diff)
        diff_std = np.std(time_diff)
        diff_median = np.median(time_diff)
        diff_zeros = time_diff.shape[0] - np.count_nonzero(time_diff)

    row = dict.fromkeys(categories, 0)
    row.update(dict.fromkeys(countries_list, 0))

    row['devices_c'] = group['device'].unique().shape[0]
    row['countries_c'] = group['country'].unique().shape[0]
    row['ip_c'] = group['ip'].unique().shape[0]
    row['url_c'] = group['url'].unique().shape[0]
    if not auction:
        row['auction_c'] = group['auction'].unique().shape[0]
        row['auc_mean'] = np.mean(group['auction'].value_counts())
    row['merch_c'] = group['merchandise'].unique().shape[0]
    row['bids_c'] = group.shape[0]
    row['tmean'] = diff_mean
    row['tstd'] = diff_std
    row['tmedian'] = diff_median
    row['tzeros'] = diff_zeros

    for cat, value in group['merchandise'].value_counts().iteritems():
        row[cat] = value

    for c in group['country'].unique():
        row[str(c)] = 1

    row = pd.Series(row)
    return row


def group_by_auction():
    bids = pd.read_csv('bids.csv')
    bidders = bids.groupby(['bidder_id', 'auction']). \
        apply(generate_features_auc)
    bidders.to_csv('bids_auc.csv')


def df_to_vw(df, categorical, name):
    of = open(name, 'w')
    df.drop(['payment_account', 'address', 'bidder_id'], 1, inplace=True)
    columns = list(df.columns)

    if 'outcome' in columns:
        columns.remove('outcome')

    for i, row in df.iterrows():
        if name == 'train.vw':

            if row['outcome'] == 0:
                outcome = -1
            else:
                outcome = 1

            output = '%s |A ' % outcome
        else:
            output = '-1 |A '

        for column in columns:
            if column in categorical:
                if row[column] == 1:
                    output += '%s_%s ' % (column, 1)
            else:
                output += '%s:%s ' % (column, row[column])
        output += '\n'
        of.write(output)


def to_vw():
    train = pd.read_csv('train_full.csv')
    test = pd.read_csv('test_full.csv')

    if not os.path.exists('vw'):
        os.mkdir('vw')

    # Dirty hack
    categorical = []
    for col in train.columns:
        if len(train[col].unique()) == 2:
            categorical.append(col)

    df_to_vw(train, categorical, name='vw/train.vw')
    df_to_vw(test, categorical, name='vw/test.vw')


def merge_data():
    t = time.time()
    train = pd.read_csv('train.csv')
    test = pd.read_csv('test.csv')
    bids = pd.read_csv('bids.csv')

    time_differences = pd.read_csv('tdiff.csv', index_col=0)
    bids_auc = pd.read_csv('bids_auc.csv')

    bids_auc = bids_auc.groupby('bidder_id').mean()
    bidders = bids.groupby('bidder_id').apply(generate_features)

    bidders = bidders.merge(bids_auc, right_index=True, left_index=True)
    bidders = bidders.merge(time_differences, right_index=True,
                            left_index=True)

    train = train.merge(bidders, left_on='bidder_id', right_index=True)
    train.to_csv('train_full.csv', index=False)

    test = test.merge(bidders, left_on='bidder_id', right_index=True)
    test.to_csv('test_full.csv', index=False)
    print time.time() - t


def xgboost_model(X_train, X_test, y_train):
    X_train = xgb.DMatrix(X_train.values, label=y_train.values)
    X_test = xgb.DMatrix(X_test.values)
    params = {'objective': 'binary:logistic', 'nthread': 10,
              'eval_metric': 'auc', 'silent': 1, 'seed': 1111,

              'max_depth': 6, 'gamma': 0, 'base_score': 0.50,
              'min_child_weight': 4, 'subsample': 0.5,
              'colsample_bytree': 1, 'eta': 0.01,
              }
    model = xgb.train(params, X_train, 600)
    predictions = model.predict(X_test)
    return predictions


def graphlab_model(train, test):
    model = gl.boosted_trees_classifier.create(gl.SFrame(train),
                                               target='outcome',
                                               max_iterations=25,
                                               max_depth=4,
                                               verbose=False,
                                               validation_set=None,
                                               step_size=0.5
                                               )
    predictions = model.classify(gl.SFrame(test))['probability']
    predictions = np.array((1 - predictions))
    return predictions


def gradient_model(X_train, X_test, y_train):
    model = GradientBoostingClassifier(n_estimators=200,
                                       random_state=1111,
                                       max_depth=5,
                                       learning_rate=0.03,
                                       max_features=40, )
    model.fit(X_train, y_train)
    predictions = model.predict_proba(X_test)[:, 1]
    return predictions


def forest_model(X_train, X_test, y_train):
    model = RandomForestClassifier(n_estimators=160, max_features=35,
                                   max_depth=8, random_state=1111,
                                   criterion='entropy', )
    model.fit(X_train, y_train)
    predictions = model.predict_proba(X_test)[:, 1]
    return predictions


def forest_ada_model(X_train, X_test, y_train):
    model = RandomForestClassifier(n_estimators=160, max_features=35,
                                   max_depth=8, random_state=1111,
                                   criterion='entropy', )
    model = AdaBoostClassifier(base_estimator=model, n_estimators=25)
    model.fit(X_train, y_train)
    predictions = model.predict_proba(X_test)[:, 1]
    return predictions


def forest_calibrated(X_train, X_test, y_train):
    model = RandomForestClassifier(n_estimators=60, max_features=33,
                                   max_depth=8, random_state=1111,
                                   criterion='entropy', )
    model = CalibratedClassifierCV(model, method='isotonic', cv=5)
    model.fit(X_train, y_train)
    predictions = model.predict_proba(X_test)[:, 1]
    return predictions


def forest_bagging(X_train, X_test, y_train):
    model = RandomForestClassifier(n_estimators=150, max_features=40,
                                   max_depth=8, random_state=1111,
                                   criterion='entropy', )
    model = BaggingClassifier(base_estimator=model, max_features=0.80,
                              n_jobs=-1, n_estimators=50)
    model.fit(X_train, y_train)
    predictions = model.predict_proba(X_test)[:, 1]
    return predictions


def predict(X_train, X_test, y_train, g_train, g_test, vw=False):
    predictors_len = 8 if vw else 7
    predictions = np.zeros([len(X_test), predictors_len], np.float)
    predictions[:, 0] = gradient_model(X_train, X_test, y_train)
    predictions[:, 1] = xgboost_model(X_train, X_test, y_train)
    predictions[:, 2] = forest_model(X_train, X_test, y_train)
    predictions[:, 3] = graphlab_model(g_train, g_test)
    predictions[:, 4] = forest_ada_model(X_train, X_test, y_train)
    predictions[:, 5] = forest_calibrated(X_train, X_test, y_train)
    predictions[:, 6] = forest_bagging(X_train, X_test, y_train)
    if vw:
        predictions[:, 7] = [float(x.strip()) for x in open(
            'vw/preds.txt').readlines()]

    predictions = np.apply_along_axis(np.mean, axis=1, arr=predictions)
    return predictions


def kfold(train, ytrain, gtrain):
    kf = StratifiedKFold(y=ytrain, n_folds=10)
    scores = []
    for train_index, test_index in kf:
        X_train, X_test = train.iloc[train_index], train.iloc[test_index]

        y_train, y_test = ytrain.iloc[train_index], ytrain.iloc[test_index]

        g_train, g_test = gl.SFrame(gtrain.iloc[train_index]), \
                          gl.SFrame(gtrain.iloc[test_index])

        predictions = predict(X_train, X_test, y_train, g_train, g_test)

        scores.append(roc_auc_score(y_test, predictions))

    print np.mean(scores)


def submit(X_train, X_test, y_train, gtrain, gtest, test_ids):
    # vowpal wabbit model
    os.system('vw vw/train.vw -c --passes 3000 -f vw/model.vw '
              '--loss_function=logistic  --link=logistic  '
              '--initial_t 0.001 --power_t 0.6 -l 0.1')
    os.system('vw vw/test.vw  -t -i vw/model.vw -p vw/predictions.txt')

    predictions = predict(X_train, X_test, y_train, gtrain, gtest, vw=True)

    sub = pd.read_csv('sampleSubmission.csv')
    result = pd.DataFrame()
    result['bidder_id'] = test_ids
    result['outcome'] = predictions
    sub = sub.merge(result, on='bidder_id', how='left')

    # Fill missing values with mean
    sub.fillna(0.0511674, inplace=True)

    sub.drop('prediction', 1, inplace=True)
    sub.to_csv('result.csv', index=False, header=['bidder_id', 'prediction'])


if __name__ == "__main__":
    if not os.path.exists('train_full.csv'):
        group_by_auction()
        generate_timediff()
        merge_data()
        to_vw()

    train = pd.read_csv('train_full.csv')
    test = pd.read_csv('test_full.csv')
    train['outcome'] = train['outcome'].astype(int)

    gtrain = train.copy()
    gtest = test.copy()

    ytrain = train['outcome']

    train.drop('outcome', 1, inplace=True)
    test_ids = test['bidder_id']

    labels = ['payment_account', 'address', 'bidder_id']
    train.drop(labels=labels, axis=1, inplace=True)
    test.drop(labels=labels, axis=1, inplace=True)

    # kfold(train, ytrain, gtrain)
    submit(train, test, ytrain, gtrain, gtest, test_ids)
