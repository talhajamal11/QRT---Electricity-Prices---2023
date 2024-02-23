import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pandasgui as pdg

from sklearn import linear_model
from sklearn.datasets import load_diabetes, make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn import metrics
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LassoCV
from sklearn.metrics import mean_squared_error, root_mean_squared_error, r2_score

X_train_raw = pd.read_csv("E:\秘密基地\看一听\硕士研究生相关\Electives\Big Data in Finance I\Course Project\X_train_NHkHMNU.csv")
y_train_raw = pd.read_csv("E:\秘密基地\看一听\硕士研究生相关\Electives\Big Data in Finance I\Course Project\y_train_ZAN5mwg.csv")
print(X_train_raw.columns)
X_train_raw = X_train_raw.fillna(method="ffill").fillna(method="bfill")

X_train_raw = pd.get_dummies(X_train_raw, columns=["COUNTRY"])
X_train_raw["FR"] = X_train_raw["COUNTRY_FR"].astype(int)
print(X_train_raw)

X_train_raw = X_train_raw.sort_values(by=["DAY_ID", "FR"], ascending=[True, True])

X_train_adjusted = pd.merge(X_train_raw, y_train_raw, on="ID", how="left")
X, y = X_train_adjusted.drop(["ID", "TARGET", "DAY_ID", "COUNTRY_FR", "COUNTRY_DE"], axis=1), X_train_adjusted["TARGET"]

#Principal Component Analysis (PCA) and standardisation
X_pca = pd.DataFrame(PCA().fit_transform(X), columns=X.columns)
X_pca_std = pd.DataFrame(StandardScaler().fit_transform(X_pca), columns=X_pca.columns)
print(X_pca_std.head())


train_size = int(len(X) * 0.8)
X_train, X_test = X_pca_std.iloc[:train_size, :], X_pca_std.iloc[train_size:, :]
y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]
y_train = y_train.values.ravel()
y_test = y_test.values.ravel()

#Lasso regression with time-series cross-validation
lasso_cv = LassoCV(cv=TimeSeriesSplit(n_splits=5)).fit(X_train, y_train)
print(lasso_cv.alpha_)
print(lasso_cv.coef_)

#Testing dataset metrics
y_predict = lasso_cv.predict(X_test)
mse = mean_squared_error(y_test, y_predict)
rmse = root_mean_squared_error(y_test, y_predict)
r2 = r2_score(y_test, y_predict)
print(y_predict)
print(mse)
print(rmse)
print(r2)

