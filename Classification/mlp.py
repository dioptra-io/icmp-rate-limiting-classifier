from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import cross_val_predict
from sklearn.neural_network import MLPClassifier


def mlp_classifier(train_features, train_labels):

    classifier = MLPClassifier(max_iter=1000)

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