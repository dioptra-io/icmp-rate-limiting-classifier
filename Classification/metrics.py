from sklearn import metrics

def compute_metrics(predictions, true_labels):
    precision = metrics.precision_score(y_pred=predictions, y_true=true_labels)
    recall = metrics.recall_score(y_pred=predictions, y_true=true_labels)
    accuracy = metrics.accuracy_score(y_pred=predictions, y_true=true_labels)
    f_score = metrics.f1_score(y_pred=predictions, y_true=true_labels)
    return precision, recall, accuracy, f_score

