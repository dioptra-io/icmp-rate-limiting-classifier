import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from Classification.plot_metrics import plot_metrics
from sklearn import metrics
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import precision_recall_curve
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
from metrics import compute_metrics
from sklearn.utils.fixes import signature


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

    classifier = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=0, verbose=0,
                                class_weight={0:2000000, 1:1}
                                )

    y_scores = cross_val_predict(classifier, train_features, train_labels, cv=3, method="predict_proba")

    y_scores = [y_scores[i][1] for i in range(0, len(y_scores))]
    precision, recall, thresholds = precision_recall_curve(y_true=train_labels, probas_pred=y_scores)

    # # In matplotlib < 1.5, plt.fill_between does not have a 'step' argument
    # step_kwargs = ({'step': 'post'}
    #                if 'step' in signature(plt.fill_between).parameters
    #                else {})
    # plt.step(recall, precision, color='b', alpha=0.2,
    #          where='post')
    # plt.fill_between(recall, precision, alpha=0.2, color='b', **step_kwargs)
    #
    # plt.xlabel('Recall')
    # plt.ylabel('Precision')
    # plt.ylim([0.0, 1.05])
    # plt.xlim([0.0, 1.0])
    # plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(
    #     np.mean(precision)))
    # plt.show()

    # Find the minimum threshold when the precision is 99,5%
    threshold_decision = None
    for i in range(0, len(precision)) :
        if precision[i] > 0.99 :
            if i >= len(thresholds):
                threshold_decision = thresholds[-1]
            else:
                threshold_decision = thresholds[i]
            break

    # Train the model on training data
    classifier.fit(train_features, train_labels)

    return classifier, threshold_decision
