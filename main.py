import os.path
import pandas as pd
from spotifyservice import export_track_features_to_csv
import constants
import matplotlib
matplotlib.use('WebAgg')
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

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
def split_in_train_test_data(X, Y):
    return train_test_split(
        X, 
        Y, 
        test_size=0.25, 
        random_state = 0, 
        stratify=Y
    )

# plot the confusion matrix
def plot_confusion_matrix(testY, predsY, target):
    labels = target.drop_duplicates().tolist()
    confusionMatrix = confusion_matrix(testY, predsY, labels=labels)
    ax = plt.subplot()
    sns.heatmap(confusionMatrix,annot=True,ax=ax)
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title('Confusion Matrix')
    ax.xaxis.set_ticklabels(labels)
    ax.yaxis.set_ticklabels(labels)
    plt.show()

# import the data
df = import_data()
X = df[constants.AUDIO_FEATURES_PROPERTIES]
Y = df['mood']

# preprocessing/normalize our data
scaled = StandardScaler().fit_transform(X)

# split our data in training and tests sets (75% training, 25% tests)
trainX, testX, trainY, testY = split_in_train_test_data(scaled, df['mood'])

# instantiate and train using LogisticRegression
logRegression = LogisticRegression(max_iter=2000)
logRegression.fit(trainX, trainY)
predsY = logRegression.predict(testX)
predsDf = pd.Series(predsY).groupby(predsDf).size().reset_index().values.tolist()
print(predsDf)

# das 50 musicas happy, avaliou-se 45 happy, 1 aggressive e 4 sads
# das 50 agressivas, avaliou-se 34 aggressives, 7 sads e 9 calms
# das 50 sads, 8 happy, 1 aggressive, 39 sads e 2 calms
# das 43 calms, 6 aggressive, 4 sads e 33 calms
print(accuracy_score(testY, predsY))
plot_confusion_matrix(testY, predsY, df['mood'])

