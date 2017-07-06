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

class param_search_res(object):
    # here you input the sklearn random parameter/grid search CV result and initialize the class
    def __init__(self, cv_result):
        self.cv_result = copy.deepcopy(cv_result)

    def get_n_iter(self):
        return len(self.cv_result["params"])

    def get_cv(self):
        return len(self.cv_result["split0_test_score"])

    def get_model_params(self, model_id):
        return copy.deepcopy(self.cv_result["params"][model_id])

    def get_result_dataframe(self):
        test_scores = list()
        model_id = list()
        split_list = list()
        for key in self.cv_result.keys():
            if "split" in key and "_test_score" in key:
                j = 0
                split = int(key.strip("split_test_score"))
                for e in self.cv_result[key]:
                    test_scores.append(e)
                    model_id.append(j)
                    split_list.append(split)
                    j += 1
        #
        return pd.DataFrame.from_dict(dict(model_id=model_id,
                                           split=split_list,
                                           fold_score=test_scores))

    def get_best_model_ids(self):
        ind = self.get_result_dataframe().groupby("split")["fold_score"].idxmax().values
        return self.get_result_dataframe().loc[ind, "model_id"].values

    def get_best_model_comparison(self):
        ids = self.get_best_model_ids()
        res = self.get_model_params(ids[0])
        res["model_id"] = ids[0]
        res["model_count"] = 1
        res["mean_test_score"] = self.cv_result["mean_test_score"][ids[0]]
        # make values lists so appending is possible
        for key in res:
            res[key] = [res[key]]
        #
        for i in np.arange(1, len(ids)):
            if ids[i] in res["model_id"]:
                res["model_count"][res["model_id"].index(ids[i])] += 1
            else:
                res["model_id"].append(ids[i])
                res["model_count"].append(1)
                res["mean_test_score"].append(self.cv_result["mean_test_score"][ids[i]])
                for key in res:
                    if key not in ["model_id", "model_count", "mean_test_score"]:
                        res[key].append(self.get_model_params(ids[i])[key])
        #
        return pd.DataFrame.from_dict(res).sort_values(by="mean_test_score", ascending=False)

# initialize offline plotting
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
psr_xgb1 = param_search_res(xgb_rand_param_search.cv_results_)
pickle.dump(psr_xgb1, open("psr_xgb1", "wb"))
xgb_models = pickle.load(open("psr_xgb1", "rb"))
n=xgb_models.get_best_model_comparison()

# Let's look at the score by index
g = go.Scatter(x=np.arange(0, 999),
               y=[e[1] for e in xgb_rand_param_search.grid_scores_])

plotly.offline.plot([g])

xgb_rand_param_search.best_params_
#best score 0.673474

# fit model with the best params
xgb1 = xgb.XGBRegressor(colsample_bytree=0.820647,
                        gamma=0.790284,
                        learning_rate=0.069506,
                        max_depth=6,
                        min_child_weight=8,
                        n_estimators=170,
                        subsample=0.732114)

xgb1.fit(train[features].values, train.price_doc.values)

#let's see how does our predictor behaves for XGBOOST

g = go.Scatter(x=[train.price_doc.min(), train.price_doc.max()],
               y=[train.price_doc.min(), train.price_doc.max()],
               name="R2 = 1")
# XGB
g1 = go.Scatter(x=train.price_doc.values,
                y=xgb1.predict(train[features].values),
                mode="markers",
                name="XGB")
plotly.offline.plot([g1, g])

#Random forest model- we got it from another file where we did randomizes search

rf = ensemble.RandomForestRegressor(max_depth=16,
                                    max_features=0.692,
                                    min_samples_leaf=3,
                                    min_samples_split=7,
                                    n_estimators=266)

rf.fit(train[features].values, train.price_doc.values)

#let's see how does our predictor behaves for RandomForest

g = go.Scatter(x=[train.price_doc.min(), train.price_doc.max()],
               y=[train.price_doc.min(), train.price_doc.max()],
               name="R2 = 1")
# RF
g1 = go.Scatter(x=train.price_doc.values,
                y=rf.predict(train[features].values),
                mode="markers",
                name="RF")
plotly.offline.plot([g1, g])

# ExtraTrees
et = ensemble.ExtraTreesRegressor(max_depth=20,
                                  max_features=0.7685,
                                  min_samples_leaf=4,
                                  min_samples_split=13,
                                  n_estimators=248)

et.fit(train[features].values, train.price_doc.values)
#let's see how does our predictor behaves for RandomForest
g = go.Scatter(x=[train.price_doc.min(), train.price_doc.max()],
               y=[train.price_doc.min(), train.price_doc.max()],
               name="R2 = 1")
# ET
g1 = go.Scatter(x=train.price_doc.values,
                y=et.predict(train[features].values),
                mode="markers",
                name="ET")
plotly.offline.plot([g1, g])

# Combining them
# 10-fold cross validation
cv_10 = model_selection.KFold(n_splits=10)
cv_10.get_n_splits(train[features].values)

# DataFrame for storing the predictions of each model
ensemble_y = pd.DataFrame(np.zeros([train.shape[0], 3]),
                          columns=["xgb1_prediction", "rf_prediction", "et_prediction"])

i = 0
for train_ind, test_ind in cv_10.split(train):
    ### XGB
    print("Fitting XGB \n")
    #
    xgb1.fit(X=train.loc[train_ind, features].values,
             y=train.loc[train_ind, "price_doc"].values)
    print("Predicting using XGB \n")
    ensemble_y.loc[test_ind, "xgb1_prediction"] = xgb1.predict(train.loc[test_ind, features].values)
    ### RF
    print("Fitting RF \n")
    #
    rf.fit(X=train.loc[train_ind, features].values,
           y=train.loc[train_ind, "price_doc"].values)
    #
    print("Predicting using RF \n")
    ensemble_y.loc[test_ind, "rf_prediction"] = rf.predict(train.loc[test_ind, features].values)

    ### ET
    print("Fitting ET \n")
    #
    et.fit(X=train.loc[train_ind, features].values,
           y=train.loc[train_ind, "price_doc"].values)
    #
    print("Predicting using ET \n")
    ensemble_y.loc[test_ind, "et_prediction"] = et.predict(train.loc[test_ind, features].values)

    # verbatim
    i += 1
    print("%d / 10" % i)






# let's combine the predictions of the three models
for col in ensemble_y.columns:
    train[col] = ensemble_y[col].values
    features.append(col)

"""
Stack : Using NN
"""
# First you need to normalize
scaler = preprocessing.StandardScaler()
scaler.fit(train[features].values)
train_scaled = scaler.transform(train[features])

# Next, create the random CV dictionary
nn_param_distr = dict(hidden_layer_sizes = [tuple([x]) for x in np.arange(1, 5 + 1)],
                      activation = ["logistic", "tanh"],
                      alpha=scipy.stats.uniform(0.0001, 1))

nn_rand_param_search = model_selection.RandomizedSearchCV(estimator=neural_network.MLPRegressor(max_iter=200000),
                                                          param_distributions=nn_param_distr,
                                                          n_jobs=2,
                                                          verbose=20,
                                                          n_iter=100,
                                                          cv=10,
                                                          iid=False)

nn_rand_param_search.fit(train_scaled, train.price_doc.values)

