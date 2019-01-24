from sklearn.svm import SVC
from Classification.meta import compute_threshold_decision

def svm_classifier(train_features, train_labels):
    classifier = SVC(probability=True, class_weight={0:10, 1: 1})

    classifier.fit(train_features, train_labels)
    threshold_decision = compute_threshold_decision(classifier, train_features, train_labels)
    return classifier, threshold_decision
