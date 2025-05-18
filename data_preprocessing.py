import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor

# To use this experimental feature, we need to explicitly ask for it:
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import BayesianRidge, Ridge
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.impute import KNNImputer
from sklearn import preprocessing

rng = np.random.RandomState(0)
x_labels = []
mses_Metoxy=np.zeros(2)
stds_Metoxy=np.zeros(2)

xl = pd.ExcelFile('Standard_Genome.xlsx')
namelist=xl.sheet_names[0]
frame=pd.read_excel('Standard_Genome.xlsx')
model_index=list(frame.index)
frame=frame.loc[model_index,:]
X_full=frame.iloc[:,0:18]
    #x_data=frame[globals()['colindex'+str(i)]]
X_full.index=range(len(X_full))
y_full=frame.iloc[:,18]
y_full.index=range(len(y_full))
x_names=X_full.columns.values.tolist()
colnames=y_full.columns.values.tolist()


# Estimate the score on the entire dataset, with no missing values

NaN=np.nan
missing_samples = np.arange(n_samples)
missing_features = rng.choice(n_features, n_samples, replace=True)
#X_missing['nan'] = np.nan

N_SPLITS = 4
regressor = RandomForestRegressor(random_state=0)
def get_scores_for_imputer(imputer, X_missing, y_missing):
    estimator = make_pipeline(imputer, regressor)

    impute_scores = cross_val_score(
        estimator, X_missing, y_missing, scoring="neg_mean_squared_error", cv=N_SPLITS
    )
    return impute_scores

#Missing value imputation
def get_impute_knn_score(X_missing,y_missing):
    imputer=KNNImputer(missing_values=np.nan, add_indicator=True)#
    n_samples, n_features = X_full.shape
    knn_impute_scores = get_scores_for_imputer(imputer, X_missing, y_missing)
    return knn_impute_scores.mean(), knn_impute_scores.std()


mses_Metoxy[1],stds_Metoxy[1]=get_impute_knn_score(X_missing,y_missing)

x_labels.append("KNN Imputation")
#imputer.fit_transform(X_missing)
print(mses_Metoxy,stds_Metoxy)
