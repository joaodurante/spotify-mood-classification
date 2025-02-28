import os.path
import pandas as pd
from spotifyservice import SpotifyService
import constants
import matplotlib
matplotlib.use('WebAgg')
import matplotlib.pyplot as plt
import seaborn as sns

# sklearn imports
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier

# create SpotifyService instance
spotifyService = SpotifyService()

# check if csv file containing the data already exists, if not, call the function to import it from spotify's API
def import_and_process_data():
    csv_exists = os.path.exists(constants.DATASET_FILE_NAME)
    if csv_exists == False:
        spotifyService.export_track_features_to_csv()

    df = pd.read_csv(constants.DATASET_FILE_NAME)
    X = df[constants.AUDIO_FEATURES_PROPERTIES]
    Y = df['mood']

    # preprocessing/normalize our data
    scaled = MinMaxScaler().fit_transform(X)

    return scaled, Y

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
def plot_confusion_matrix(chartTitle, chartSubtitle, testY, predsY, target):
    labels = target.drop_duplicates().tolist()
    confusionMatrix = confusion_matrix(testY, predsY, labels=labels)
    ax = plt.subplot()
    sns.heatmap(confusionMatrix,annot=True,ax=ax)
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    plt.suptitle(chartTitle, fontsize=18)
    plt.title('Accuracy: {:.6f}'.format(chartSubtitle), fontsize=10)
    ax.xaxis.set_ticklabels(labels)
    ax.yaxis.set_ticklabels(labels)
    plt.show()

# plot the bar chart of predicts results
def plot_results(predsLogReg, predsGradBoost):
    dataLogReg = {i[0]: i[1] for i in predsLogReg}
    dataGradBoost = {i[0]: i[1] for i in predsGradBoost}

    _, ax = plt.subplots(1, 2)

    ax[0].bar(dataLogReg.keys(), dataLogReg.values())
    ax[0].set_title('Logistic Regression Results')
    
    ax[1].bar(dataGradBoost.keys(), dataGradBoost.values())
    ax[1].set_title('Gradient Boosting Results')

    plt.show()


# import the data
X, Y = import_and_process_data()
# split our data in training and tests sets (75% training, 25% tests)
trainX, testX, trainY, testY = split_in_train_test_data(X, Y)


# instantiate and train using LogisticRegression
logRegression = LogisticRegression(C=10, penalty='l2', solver='lbfgs', max_iter=1000)
logRegression.fit(trainX, trainY)
predsY = logRegression.predict(testX)
predsResult = pd.Series(predsY)
predsResult = predsResult.groupby(predsResult).size().reset_index().values.tolist()
plot_confusion_matrix('LogisticRegression Confusion Matrix', accuracy_score(testY, predsY), testY, predsY, Y)


# instantiate and train using GradientBoosting
gradBoosting = GradientBoostingClassifier(learning_rate=0.05, max_depth=4, n_estimators=100) 
gradBoosting.fit(trainX, trainY)
predsY = gradBoosting.predict(testX)
predsResult = pd.Series(predsY)
predsResult = predsResult.groupby(predsResult).size().reset_index().values.tolist()
plot_confusion_matrix('GradientBoosting Confusion Matrix', accuracy_score(testY, predsY), testY, predsY, Y)

while True:
    userPlaylistId = input("Please provide your playlist url: ")
    userTrackFeatures = spotifyService.get_track_feature_from_user_playlist(userPlaylistId)
    userTrackFeaturesList = userTrackFeatures[constants.AUDIO_FEATURES_PROPERTIES].values.tolist()
    scaled = MinMaxScaler().fit_transform(userTrackFeaturesList)

    userPredsY = logRegression.predict(scaled)
    logPredsResult = pd.Series(userPredsY)
    logPredsResult = logPredsResult.groupby(logPredsResult).size().reset_index().values.tolist()

    userPredsY = gradBoosting.predict(scaled)
    gradPredsResult = pd.Series(userPredsY)
    gradPredsResult = gradPredsResult.groupby(gradPredsResult).size().reset_index().values.tolist()


    plot_results(logPredsResult, gradPredsResult)
