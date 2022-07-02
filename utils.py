from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
import constants
from spotifyservice import SpotifyService
import pandas as pd
import os.path


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
        'learning_rate': [0.01, 0.025, 0.05, 0.075, 0.1],
        'n_estimators' : [100,500,1000],
        'max_depth'    : [2,4,8]
    }

    model = GradientBoostingClassifier()
    gridGradBoosting = GridSearchCV(estimator=model, param_grid = parameters, cv = 2, n_jobs=-1)
    gridGradBoosting.fit(trainX, trainY)

    print("GridSearch Results (GradientBoosting)")
    print("\nBest estimator:\n",gridGradBoosting.best_estimator_)
    print("\nBest score:\n",gridGradBoosting.best_score_)
    print("\nBest parameters:\n",gridGradBoosting.best_params_)

def check_learning_lr_rate():
    parameters = {
        'penalty': ['l1', 'l2'],
        'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
        'solver': ['lbfgs', 'liblinear', 'sag', 'saga'],
        'max_iter': [10, 100, 1000, 2000]
    }

    model = LogisticRegression()
    gridLogRegression = GridSearchCV(estimator=model, param_grid = parameters, cv = 2, n_jobs=-1)
    gridLogRegression.fit(trainX, trainY)

    print("GridSearch Results (LogisticRegression)")
    print("\nBest estimator:\n",gridLogRegression.best_estimator_)
    print("\nBest score:\n",gridLogRegression.best_score_)
    print("\nBest parameters:\n",gridLogRegression.best_params_)


# check_learning_lr_rate()
check_learning_gb_rate()