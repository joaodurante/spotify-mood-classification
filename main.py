import os.path
import pandas as pd
import numpy as np
from spotifyservice import export_track_features_to_csv
import constants
import math
import matplotlib.pyplot as plt
import seaborn as sns

# sklearn imports
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import cross_val_score, cross_val_predict, train_test_split, GridSearchCV


# check if csv file containing the data already exists, if not, call the function to import it from spotify's API
def import_data():
    csv_exists = os.path.exists(constants.DATASET_FILE_NAME)
    if csv_exists == False:
        export_track_features_to_csv()

    return pd.read_csv(constants.DATASET_FILE_NAME)


# split the dataset into training and test sets (75% training, 25% tests)
def split_in_train_test_data(tracks):
    return train_test_split(
        tracks[constants.AUDIO_FEATURES_PROPERTIES], 
        tracks['mood'], 
        test_size=0.25, 
        random_state = 0, 
        stratify=tracks['mood']
    )

def plotConfusionMatrix(testY, predsY, target):
    confusionMatrix = confusion_matrix(testY, predsY)
    ax = plt.subplot()
    sns.heatmap(confusionMatrix,annot=True,ax=ax)
    labels = target['mood'].tolist()
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title('Confusion Matrix')
    ax.xaxis.set_ticklabels(labels)
    ax.yaxis.set_ticklabels(labels)
    plt.show()

def create_encoded_target_df(Y):
    # encode our labels (moods)
    encoder = LabelEncoder()
    encoder.fit(Y)
    encoded_y = encoder.transform(Y)

    target = pd.DataFrame()
    target['mood'] = Y.tolist()
    target['encode'] = encoded_y
    return target.drop_duplicates().sort_values(['encode'])


# import the data
df = import_data()
X = df[constants.AUDIO_FEATURES_PROPERTIES]
Y = df['mood']

# preprocessing/normalize our data
trainScaled = StandardScaler().fit_transform(X)

# split our data in training and tests sets (75% training, 25% tests)
trainX, testX, trainY, testY = split_in_train_test_data(trainScaled)

# dataframe containing our targets (moods) encoded
target = create_encoded_target_df()

# instantiate and train using LogisticRegression
logRegression = LogisticRegression(max_iter=2000)
logRegression.fit(trainX, trainY)
predsY = logRegression.predict(trainX)
print(accuracy_score(trainY, predsY))

# print cross_val_score mean from the previous (logistic regression) results
scores = cross_val_score(logRegression, trainX, trainY, cv=5)
print(scores.mean())


params = {"C" : np.logspace(-6, 3, 10)}
clf = GridSearchCV(logRegression, params)
clf.fit(trainScaled, trainY)
print (clf.best_estimator_.C)
print (clf.best_score_)

logreg = LogisticRegression(max_iter=2000, C=0.1)
logreg.fit(trainScaled, trainY)
preds = clf.predict(test_scaled)
print (accuracy_score(preds, testY))

fi = pd.DataFrame(clf.best_estimator_.coef_, columns=constants.AUDIO_FEATURES_PROPERTIES)
pow(math.e, fi)
fo = fi.set_axis(logreg.classes_, axis=0)
fo.idxmax(axis=1)

print(fo)