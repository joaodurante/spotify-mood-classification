from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import cross_val_score, cross_val_predict, train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
import constants
from spotifyservice import SpotifyService
import pandas as pd
import os.path
import numpy as np


spotifyService = SpotifyService()
csv_exists = os.path.exists(constants.DATASET_FILE_NAME)
if csv_exists == False:
    spotifyService.export_track_features_to_csv()

df = pd.read_csv(constants.DATASET_FILE_NAME)
X = df[constants.AUDIO_FEATURES_PROPERTIES]
Y = df['mood']
X = MinMaxScaler().fit_transform(X)

trainX, testX, trainY, testY = train_test_split(X, Y, test_size=0.25, random_state = 0, stratify=Y)

def check_learning_gb_rate():
    parameters = {
        'learning_rate': [0.05, 0.075, 0.1],
        'n_estimators' : [100,500,1000],
        'max_depth'    : [2,4,8]
    }

    model = GradientBoostingClassifier()
    grid_GBC = GridSearchCV(estimator=model, param_grid = parameters, cv = 5, n_jobs=-1)
    grid_GBC.fit(trainX, trainY)

    print(" Results from Grid Search " )
    print("\n The best estimator across ALL searched params:\n",grid_GBC.best_estimator_)
    print("\n The best score across ALL searched params:\n",grid_GBC.best_score_)
    print("\n The best parameters across ALL searched params:\n",grid_GBC.best_params_)

def check_learning_lr_rate():
    parameters = {
        'penalty': ['l1', 'l2'],
        'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
        'solver': ['lbfgs', 'liblinear', 'sag', 'saga'],
        'max_iter': [10, 100, 1000, 2000]
    }

    model = LogisticRegression()
    grid_GBC = GridSearchCV(estimator=model, param_grid = parameters, cv = 2, n_jobs=-1)
    grid_GBC.fit(trainX, trainY)

    print(" Results from Grid Search " )
    print("\n The best estimator across ALL searched params:\n",grid_GBC.best_estimator_)
    print("\n The best score across ALL searched params:\n",grid_GBC.best_score_)
    print("\n The best parameters across ALL searched params:\n",grid_GBC.best_params_)



# check_learning_lr_rate()
check_learning_gb_rate()