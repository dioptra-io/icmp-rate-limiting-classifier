import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from Classification.plot_metrics import plot_metrics
from sklearn import metrics
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np

def print_false_positives(predictions, labels, features):
    for i in range(0, len(predictions)):
        if predictions[i] == 1 and labels.iloc[i]["label_pairwise"] != 1:
            print features.iloc[i][["loss_rate_dpr_c0", "loss_rate_dpr_c1", "loss_rate_dpr_w0"]]
            print "Predicted: " + str(predictions[i])
            print "True label: " +str(labels.iloc[i]["label_pairwise"])

def random_forest_classifier(train_features, train_labels, validation_features, validation_labels):

    rf = RandomForestClassifier(n_estimators=1000, n_jobs=-1, random_state=0, verbose=0)
    # Train the model on training data
    rf.fit(train_features, train_labels)



    print rf.score(train_features, train_labels)
    print rf.score(validation_features, validation_labels)

    validations_predictions = rf.predict(validation_features)
    feats = {}  # a dict to hold feature_name: feature_importance
    for feature, importance in zip(train_features.columns, rf.feature_importances_):
        feats[feature] = importance  # add the name/value pair

    importances = pd.DataFrame.from_dict(feats, orient='index').rename(columns={0: 'Gini-importance'})
    importances = importances.sort_values(by='Gini-importance')

    print importances.to_string()

    print_false_positives(validations_predictions, validation_labels, validation_features)

    # Output a plot of the confusion matrix.
    cm = metrics.confusion_matrix(validation_labels, validations_predictions)
    # Normalize the confusion matrix by row (i.e by the number of samples
    # in each class).
    cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    ax = sns.heatmap(cm_normalized, cmap="bone_r")
    ax.set_aspect(1)
    plt.title("Confusion matrix")
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.show()

    # plot_metrics(training_errors, validation_errors, validation_targets, final_predictions)

    return rf
