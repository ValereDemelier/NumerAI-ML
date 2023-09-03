#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 13 15:24:02 2021

@author: valeredemelier
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import xgboost as xgb
from scipy.stats import spearmanr
from sklearn.metrics import mean_squared_error

TRAINING_DATAPATH = 'https://numerai-public-datasets.s3-us-west-2.amazonaws.com/latest_numerai_training_data.csv.xz'
TOURNAMENT_DATAPATH = 'https://numerai-public-datasets.s3-us-west-2.amazonaws.com/latest_numerai_tournament_data.csv.xz'

df_tourn = pd.read_csv(TOURNAMENT_DATAPATH)
df_train = pd.read_csv(TRAINING_DATAPATH).set_index('id')
def score(y_true, y_pred):
    x = np.corrcoef(y_true, y_pred)[0,1]
    print('The Correlation is: {}'.format(x))
    return x    

features = df_train.columns[df_train.columns.str.startswith('feature')]
target = df_train.columns[df_train.columns.str.startswith('target')]

df_train[features] = df_train[features].astype(np.float64)
df_train[target] = df_train[target].astype(np.float64)

'''
data_dmatrix = xgb.DMatrix(data=df_train[features], label=df_train[target])
params = {'objective':'reg:linear', 'max_depth':5, 'learning_rate':.1,
          'n_estimators':200, 'colsample_bytree':.3, 'alpha':.1}

model = xgb.XGBRegressor(objective='reg:linear', max_depth=5, learning_rate=.1, 
                         n_estimators=200, colsample_bytree=.3, alpha=.1)
model.fit(df_train[features], df_train[target])
y_pred = model.predict(df_tourn[features])

cv_results = xgb.cv(params=params, dtrain=data_dmatrix, nfold=5, 
                    early_stopping_rounds=(10), metrics='rmse', as_pandas=True)
'''
model = xgb.XGBRegressor(objective='reg:linear', max_depth=5, learning_rate=.1, 
                         n_estimators=200, colsample_bytree=.35, alpha=.1)
model.fit(df_train[features].values, df_train[target].values)
tournament_predictions = model.predict(df_tourn[features].values)
score(df_tourn[target].values.ravel(), tournament_predictions)


final_predictions = pd.DataFrame(df_tourn['id'], columns=['id'])
final_predictions['prediction'] = tournament_predictions
final_predictions.to_csv('Predictions.csv', index=False)