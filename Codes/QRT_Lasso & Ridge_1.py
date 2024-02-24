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
from sklearn.model_selection import KFold, ShuffleSplit, TimeSeriesSplit
from sklearn.linear_model import LassoCV, RidgeCV
from sklearn.metrics import mean_squared_error, root_mean_squared_error, r2_score

X_train_raw = pd.read_csv("E:\秘密基地\看一听\硕士研究生相关\Electives\Big Data in Finance I\Course Project\X_train_NHkHMNU.csv")
y_train_raw = pd.read_csv("E:\秘密基地\看一听\硕士研究生相关\Electives\Big Data in Finance I\Course Project\y_train_ZAN5mwg.csv")
X_train_raw = X_train_raw.fillna(method="ffill").fillna(method="bfill")
X_train_raw = pd.get_dummies(X_train_raw, columns=["COUNTRY"])
X_train_raw["FR"] = X_train_raw["COUNTRY_FR"].astype(int)
X_train_raw = X_train_raw.sort_values(by=["DAY_ID", "FR"], ascending=[True, True])
X_train_raw["DE_NET_NET_EXPORT"] = X_train_raw["DE_NET_EXPORT"] - X_train_raw["DE_NET_IMPORT"]
X_train_raw["FR_NET_NET_EXPORT"] = X_train_raw["FR_NET_EXPORT"] - X_train_raw["FR_NET_IMPORT"]
X_train_adjusted = pd.merge(X_train_raw, y_train_raw, on="ID", how="left")
X, y = X_train_adjusted.drop(["ID", "TARGET", "DAY_ID", "COUNTRY_FR", "COUNTRY_DE", "DE_FR_EXCHANGE", "DE_NET_EXPORT", "DE_NET_IMPORT", "FR_NET_EXPORT", "FR_NET_IMPORT"], axis=1), X_train_adjusted["TARGET"]
print(X.head())

X_test_target = pd.read_csv("E:\秘密基地\看一听\硕士研究生相关\Electives\Big Data in Finance I\Course Project\X_test_final.csv")
X_test_target = X_test_target.fillna(method="ffill").fillna(method="bfill")
X_test_target = pd.get_dummies(X_test_target, columns=["COUNTRY"])
X_test_target["FR"] = X_test_target["COUNTRY_FR"].astype(int)
X_test_target["DE_NET_NET_EXPORT"] = X_test_target["DE_NET_EXPORT"] - X_test_target["DE_NET_IMPORT"]
X_test_target["FR_NET_NET_EXPORT"] = X_test_target["FR_NET_EXPORT"] - X_test_target["FR_NET_IMPORT"]
X_test_ID = X_test_target[["ID"]]
X_test_target = X_test_target.drop(["ID", "DAY_ID", "COUNTRY_FR", "COUNTRY_DE", "DE_FR_EXCHANGE", "DE_NET_EXPORT", "DE_NET_IMPORT", "FR_NET_EXPORT", "FR_NET_IMPORT"], axis=1)
print(X_test_target.head())

#Standardisation
scaler = StandardScaler()
X_std = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
X_test_target_std = pd.DataFrame(scaler.transform(X_test_target), columns=X_test_target.columns)

#Principal Component Analysis
pca = PCA()
X_std_pca = pd.DataFrame(pca.fit_transform(X_std), columns=X_std.columns)
X_test_target_std_pca = pd.DataFrame(pca.transform(X_test_target_std), columns=X_test_target_std.columns)


#Lasso regression with time-series cross-validation
lasso_cv_t = LassoCV(cv=TimeSeriesSplit(n_splits=5), max_iter=10000).fit(X_std_pca, y)
print(f"Alpha: {lasso_cv_t.alpha_}")
#print(f"Coefficients: {lasso_cv_t.coef_}")

#Lasso regression with shuffled normal cross-validation
lasso_cv_ks = LassoCV(cv=KFold(n_splits=5, shuffle=True, random_state=9), max_iter=10000, random_state=9).fit(X_std_pca, y)
print(f"Alpha: {lasso_cv_ks.alpha_}")
#print(f"Coefficients: {lasso_cv_ks.coef_}")

#Lasso predictions
lasso_predict_t = lasso_cv_t.predict(X_test_target_std_pca)
lasso_predict_t_df = pd.DataFrame(lasso_predict_t, columns=["TARGET"])
lasso_predict_t_df["ID"] = X_test_ID["ID"]
lasso_predict_t_df = lasso_predict_t_df[["ID", "TARGET"]]
lasso_predict_t_df.to_csv("E:\秘密基地\看一听\硕士研究生相关\Electives\Big Data in Finance I\Course Project\lasso_predict_t.csv", index=False)

lasso_predict_ks = lasso_cv_ks.predict(X_test_target_std_pca)
lasso_predict_ks_df = pd.DataFrame(lasso_predict_ks, columns=["TARGET"])
lasso_predict_ks_df["ID"] = X_test_ID["ID"]
lasso_predict_ks_df = lasso_predict_ks_df[["ID", "TARGET"]]
lasso_predict_ks_df.to_csv("E:\秘密基地\看一听\硕士研究生相关\Electives\Big Data in Finance I\Course Project\lasso_predict_ks.csv", index=False)


#Ridge regression with time-series cross-validation
ridge_cv_t = RidgeCV(cv=TimeSeriesSplit(n_splits=5), scoring="neg_mean_squared_error").fit(X_std, y)
print(f"Alpha: {ridge_cv_t.alpha_}")
#print(f"Coefficients: {ridge_cv_t.coef_}")

#Ridge regression with shuffled normal cross-validation
ridge_cv_ks = RidgeCV(cv=KFold(n_splits=5, shuffle=True, random_state=9), scoring="neg_mean_squared_error").fit(X_std, y)
print(f"Alpha: {ridge_cv_ks.alpha_}")
#print(f"Coefficients: {ridge_cv_ks.coef_}")

#Ridge predictions
ridge_predict_t = ridge_cv_t.predict(X_test_target_std)
ridge_predict_t_df = pd.DataFrame(ridge_predict_t, columns=["TARGET"])
ridge_predict_t_df["ID"] = X_test_ID["ID"]
ridge_predict_t_df = ridge_predict_t_df[["ID", "TARGET"]]
ridge_predict_t_df.to_csv("E:/秘密基地/看一听/硕士研究生相关/Electives/Big Data in Finance I/Course Project/ridge_predict_t.csv", index=False)

ridge_predict_ks = ridge_cv_ks.predict(X_test_target_std)
ridge_predict_ks_df = pd.DataFrame(ridge_predict_ks, columns=["TARGET"])
ridge_predict_ks_df["ID"] = X_test_ID["ID"]
ridge_predict_ks_df = ridge_predict_ks_df[["ID", "TARGET"]]
ridge_predict_ks_df.to_csv("E:/秘密基地/看一听/硕士研究生相关/Electives/Big Data in Finance I/Course Project/ridge_predict_ks.csv", index=False)




