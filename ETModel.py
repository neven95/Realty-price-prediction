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
from RFModel import rf_param_distr
plotly.offline.init_notebook_mode(connected=True)

# load data
train = pd.read_csv("train_macro.csv")
test = pd.read_csv("test_macro.csv")

# features which will be used
features = [col for col in train.columns if
            col not in ['id', 'timestamp', 'price_doc', 'price_log', 'price_per_sq']]

"""
Model 3 - Extra Trees
"""
# don't need parameter dictionary for et since we already have it for rf
et_rand_param_search = model_selection.RandomizedSearchCV(estimator=ensemble.ExtraTreesRegressor(),
                                                          param_distributions=rf_param_distr,
                                                          n_iter=200,
                                                          n_jobs=2,
                                                          cv=10,
                                                          verbose=20)

et_rand_param_search.fit(train[features].values, train.price_doc.values)
psr_et = param_search_res(et_rand_param_search.cv_results_)
pickle.dump(psr_et, open("psr_et", "wb"))
et_rand_param_search.best_params_