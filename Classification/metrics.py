from sklearn import metrics


def compute_metrics(predictions, true_labels):
    precision = metrics.precision_score(y_pred=predictions, y_true=true_labels)
    recall = metrics.recall_score(y_pred=predictions, y_true=true_labels)
    accuracy = metrics.accuracy_score(y_pred=predictions, y_true=true_labels)
    f_score = metrics.f1_score(y_pred=predictions, y_true=true_labels)
    return precision, recall, accuracy, f_score


def evaluate_classifier(predictions, labels):
    precision, recall, accuracy, f_score = compute_metrics(predictions, labels)
    print("Precision: " + str(precision))
    print("Recall: " + str(recall))
    print("Accuracy: " + str(accuracy))
    print("F score: " + str(f_score))
