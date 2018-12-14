import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from Classification.plot_metrics import plot_metrics
from sklearn import metrics
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
from metrics import compute_metrics

def print_false_positives(predictions, labels, features, labeled_df):
    for i in range(0, len(predictions)):
        if predictions[i] == 1 and labels.iloc[i]["label_pairwise"] != 1:
            print features.iloc[i][["loss_rate_dpr_c0", "loss_rate_dpr_c1", "loss_rate_dpr_w0"]]
            print "Predicted: " + str(predictions[i])
            print "True label: " +str(labels.iloc[i]["label_pairwise"])
            print labeled_df.loc[features.index[i]]["measurement_id"]


def evaluate_classifier(predictions, labels):
    precision, recall, accuracy = compute_metrics(predictions, labels)
    print "Precision: " + str(precision)
    print "Recall: " + str(recall)
    print "Accuracy: " + str(accuracy)

def feature_importance(rf, columns):
    feats = {}  # a dict to hold feature_name: feature_importance
    for feature, importance in zip(columns, rf.feature_importances_):
        feats[feature] = importance  # add the name/value pair

    importances = pd.DataFrame.from_dict(feats, orient='index').rename(columns={0: 'Gini-importance'})
    importances = importances.sort_values(by='Gini-importance')

    return importances

def random_forest_classifier(train_features, train_labels):

    rf = RandomForestClassifier(n_estimators=2000, n_jobs=-1, random_state=0, verbose=0)
    # Train the model on training data
    rf.fit(train_features, train_labels)

    return rf
