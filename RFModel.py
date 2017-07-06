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
Model 2 - Random Forest
"""
rf_param_distr = dict(n_estimators=scipy.stats.randint(1, 300 + 1),
                      max_features=scipy.stats.uniform(loc=0.1, scale=0.9),
                      max_depth=scipy.stats.randint(1, 20 + 1),
                      min_samples_split=scipy.stats.randint(2, 20 + 1),
                      min_samples_leaf=scipy.stats.randint(1, 30 + 1))

rf_rand_param_search = model_selection.RandomizedSearchCV(estimator=ensemble.RandomForestRegressor(),
                                                          param_distributions=rf_param_distr,
                                                          n_iter=200,
                                                          n_jobs=2,
                                                          cv=5,
                                                          verbose=20)

rf_rand_param_search.fit(train[features].values, train.price_doc.values)
psr_rf = param_search_res(rf_rand_param_search.cv_results_)
pickle.dump(psr_rf, open("psr_rf", "wb"))

rf_rand_param_search.best_params_
# best score = 0.67122194
