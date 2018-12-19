
from Classification import neural_network as nn
from Classification.random_forest import print_false_positives
from Classification.metrics import compute_metrics

from Validation.midar import transitive_closure


def combin(n, k):
    """Number of combinations C(n,k)"""
    if k > n//2:
        k = n-k
    x = 1
    y = 1
    i = n-k+1
    while i <= n:
        x = (x*i)//y
        y += 1
        i += 1
    return x




def evaluate(classifier, ground_truth_routers, labeled_df, labels_column, feature_columns):
    test_targets, test_examples = nn.parse_labels_and_features(labeled_df, labels_column, feature_columns)
    probabilities = classifier.predict_proba(test_examples)
    predictions = []

    for i in range(0, len(probabilities)):
        if probabilities[i][1] > 0.85:
            predictions.append(1)
        else:
            predictions.append(0)
    print_false_positives(predictions, test_targets, test_examples, labeled_df)

    precision, recall, accuracy, f_score = compute_metrics(predictions, test_targets)
    print ("Precision: " + str(precision))
    print ("Recall: " + str(recall))
    print ("Accuracy: " + str(accuracy))
    print("F score: " + str(f_score))

    # Rebuild routers from the classification
    positives = []
    negatives = []
    for i in range(0, len(labeled_df)):
        ip_address0 = labeled_df.iloc[i]["ip_address_c0"]
        ip_address1 = labeled_df.iloc[i]["ip_address_c1"]
        if predictions[i] == 1 and test_targets.iloc[i][0] == 1:
            positives.append({ip_address0, ip_address1})
            continue

    rate_limiting_routers = transitive_closure(positives)
    for file_name, ips in ground_truth_routers.items():
        ips = set(ips)
        print (file_name)
        for rl_router in rate_limiting_routers:
            if len(ips.intersection(rl_router)) > 0:
                print ("Common: " + str(ips.intersection(rl_router)))
                print ("FP: " + str(rl_router - ips))
                print ("FN: " + str(ips - rl_router))

    """Stats"""
    res_gt = 0

    for router, ips in sorted(ground_truth_routers.items()):
        res_gt += combin(len(ips), 2)

    res = 0

    for router in rate_limiting_routers:
        res += + combin(len(router), 2)

    print ("Ground Truth pairs: ")
    print (res_gt)
    print ("Rate Limiting pairs :")
    print (res)

    print ("RATIO :")
    print (float(res)/res_gt)
