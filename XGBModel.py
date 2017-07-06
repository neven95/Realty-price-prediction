import pandas as pd
import numpy as np
import scipy
import plotly
from plotly import graph_objs as go
from sklearn import preprocessing, model_selection, ensemble, neural_network
import xgboost as xgb
from xgboost.sklearn import XGBRegressor
import fancyimpute as impute
import copy
import pickle
from Predicting import param_search_res
plotly.offline.init_notebook_mode(connected=True)

# load data
train = pd.read_csv("train_macro.csv")
test = pd.read_csv("test_macro.csv")

# features which will be used
features = [col for col in train.columns if
            col not in ['id', 'timestamp', 'price_doc', 'price_log', 'price_per_sq']]

"""
Train 1 - Fit weak learners first
"""

"""
Model 1 - XGB
"""


# cross-validation
xgb_param_distr = dict(n_estimators=scipy.stats.randint(50, 250 + 1),
                       learning_rate=scipy.stats.uniform(loc=0.0001, scale=0.1),
                       max_depth=scipy.stats.randint(5, 20 + 1),
                       min_child_weight=scipy.stats.randint(0, 10 + 1),
                       colsample_bytree=scipy.stats.uniform(loc=0.1, scale=0.9),
                       subsample=scipy.stats.uniform(loc=0.1, scale=0.9),
                       gamma=scipy.stats.uniform())

xgb_rand_param_search = model_selection.RandomizedSearchCV(estimator=XGBRegressor(objective="reg:linear"),
                                                           param_distributions=xgb_param_distr,
                                                           cv=10,
                                                           n_jobs=2,
                                                           n_iter=100,
                                                           iid=False,
                                                           verbose=20)

xgb_rand_param_search.fit(train[features].values, train.price_doc.values)

# output result

# Let's look at the score by index
g = go.Scatter(x=np.arange(0, 999),
               y=[e[1] for e in xgb_rand_param_search.grid_scores_])

plotly.offline.plot([g])
psr_xgb1 = param_search_res(xgb_rand_param_search.cv_results_)
pickle.dump(psr_xgb1, open("psr_xgb1", "wb"))

# Let's look at the score by index
g = go.Scatter(x=np.arange(0, 999),
               y=[e[1] for e in xgb_rand_param_search.grid_scores_])


