from sklearn.neural_network import MLPClassifier
from Classification.meta import compute_threshold_decision


def mlp_classifier(train_features, train_labels):

    classifier = MLPClassifier(max_iter=1000)
    # Train the model on training data
    classifier.fit(train_features, train_labels)

    threshold_decision = compute_threshold_decision(
        classifier, train_features, train_labels
    )

    return classifier, threshold_decision
