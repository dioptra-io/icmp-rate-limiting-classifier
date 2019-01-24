from sklearn.neighbors import KNeighborsClassifier
from Classification.meta import compute_threshold_decision

def knn_classifier(train_features, train_labels):
    classifier = KNeighborsClassifier()

    classifier.fit(train_features, train_labels)
    threshold_decision = compute_threshold_decision(classifier, train_features, train_labels)
    return classifier, threshold_decision
