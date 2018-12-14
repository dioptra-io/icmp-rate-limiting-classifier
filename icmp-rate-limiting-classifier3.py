######################## Imports from https://colab.research.google.com/notebooks/mlcc/multi-class_classification_of_handwritten_digits.ipynb?utm_source=mlcc&utm_campaign=colab-external&utm_medium=referral&utm_content=multiclass-colab&hl=fr#scrollTo=4LJ4SD8BWHeh #######################


import glob
import math
import os

#from IPython import display
# from matplotlib import cm
# from matplotlib import gridspec
# from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
# import seaborn as sns
from sklearn import metrics
import tensorflow as tf
from tensorflow.python.data import Dataset
# from Cluster.dbscan import kmeans, DBSCAN_impl
from Classification import neural_network as nn
from Classification.neural_network import print_bad_labels
from Classification import adanet_wrap as adanet_
from Classification import random_forest as rf_
from Classification.random_forest import *
from Classification.metrics import compute_metrics
from Data.preprocess import minmax_scale, minimum_probing_rate
from Validation.midar import extract_routers, set_router_labels, extract_routers_by_node
tf.logging.set_verbosity(tf.logging.ERROR)
pd.options.display.max_rows = 25
pd.options.display.float_format = '{:.3f}'.format
#################################################

import json
from Data.preprocess import *

labels_map = {"P" : 1, "N" : 0, "U": 2}

# Add the type of probing to the columns.
def add_suffix(suffix, columns):
    suffixed_columns = []
    for column in columns:
        suffixed_columns.append("".join([column, "_", suffix]))

    return suffixed_columns

def print_df_labels(df):
    label = "label_pairwise"

    true_negatives_df = df[df[label] == 0]
    true_positives_df = df[df[label] == 1]
    # true_unknown_df = labeled_df[labeled_df[label] == labels_map["U"]]
    print "Number of true negatives: " + str(len(true_negatives_df))
    # print "Number of false negatives: " + str(FN)
    print "Number of true positives: " + str(len(true_positives_df))


debug = [
]


def compute_dataset(is_pairwise, target_loss_rate_window, epsilon, alpha, n_not_triggered_threshold):
    version = "4"
    measurement_prefix = "/srv/icmp-rl-survey/midar/survey/batch2/" + target_loss_rate_window +"/"
    results_dir = measurement_prefix + "results/"
    routers_path = "/home/kevin/mda-lite-v6-survey/resources/midar/batch2/routers/"
    ground_truth_routers = extract_routers_by_node(routers_path)
    max_interfaces = 2

    pd.options.display.float_format = '{:.5f}'.format

    skip_fields = ["Index", "probing_type", "ip_address", "probing_rate", "correlation"]
                   # "transition_0_0", "transition_0_1", "transition_1_0", "transition_1_1"]

    df_computed_result = None

    probing_type_suffix = {"INDIVIDUAL": "ind", "GROUPSPR": "spr", "GROUPDPR": "dpr"}

    # raw_columns are the extracted data from CSV
    global_raw_columns = ["ip_address", "probing_type", "probing_rate", "changing_behaviour", "loss_rate",
                          "transition_0_0", "transition_0_1", "transition_1_0", "transition_1_1"]

    # missing_fields = build_missing_candidates(0, max_interfaces, 0, 1, global_raw_columns, skip_fields,
    #                                           probing_type_suffix, probing_type_rates)



    distinct_ips = set()
    distinct_routers = set()
    # Debug infos
    total = 0
    k = 0
    TN = 0
    FN = 0
    n_not_triggered = 0
    n_not_shared = 0
    n_witness_too_high = 0
    n_not_aliases_lr_too_high = 0
    TN_pairwise = 0
    n_unusable_witness = 0

    n_total = 0
    multiple_interfaces_router = {k: 0 for k in range(2, max_interfaces + 1)}

    correlation_distributions = {
        "correlation_spr": []
        , "correlation_dpr": []
    }

    weak_correlations = []
    strong_correlations = []
    correlations = []


    for result_file in os.listdir(results_dir):

        if result_file.startswith("onelab2"):
            continue
        # Only results file
        if os.path.isdir(results_dir + result_file):
            continue

        #################### DEBUG INFOS ################
        total += 1
        print total, result_file
        # if total != 1440:
        #     continue
        if not recompute_dataset:
            if total == 1:
                break
        #################################################

        new_entry = {}
        split_file_name = result_file.split("_")
        node = split_file_name[0]

        ip_index = 5
        candidates = []
        witnesses = []
        if "TN" in result_file:
            candidates_witness_dir = "/srv/icmp-rl-survey/midar/survey/batch2/lr0.05-0.10/candidates-witness-tn/"
        else:
            candidates_witness_dir = "/srv/icmp-rl-survey/midar/survey/batch2/lr0.05-0.10/candidates-witness/"
        try:
            with open(candidates_witness_dir + result_file) as cw_file:
                for line in cw_file:
                    line = line.replace(" ", "")
                    fields = line.split(",")
                    ip_address = fields[ip_index]
                    if "CANDIDATE" in fields:
                        candidates.append(ip_address)
                    elif "WITNESS" in fields:
                        witnesses.append(ip_address)
        except IOError as e:
            continue

        ######################### SANITIZER #######################
        if len(candidates) > max_interfaces:
            continue

        if len(set(candidates).intersection(set(witnesses))) != 0:
            continue

        if "router" in result_file:
            if len(set(candidates).intersection(distinct_ips)) != 0:
                continue
            for candidate in candidates:
                distinct_ips.add(candidate)
            distinct_routers.add(result_file)
        ###########################################################

        ######################### DEBUG ###########################
        # if candidates != ["27.68.229.146", "27.68.229.218"]:
        #     continue
        ###########################################################

        raw_columns = copy.deepcopy(global_raw_columns)

        # Add correlation columns
        for i in range(1, len(candidates)):
            raw_columns.append("correlation_c" + str(i))

        for i in range(0, len(witnesses)):
            raw_columns.append("correlation_w" + str(i))

        multiple_interfaces_router[len(candidates)] += 1
        print "Routers with more than 2 interfaces: " + str(multiple_interfaces_router)
        # 1 point is represented by different dimensions:
        df_result = pd.read_csv(results_dir + result_file,
                                names=raw_columns,
                                skipinitialspace=True,
                                index_col=False)
        # usecols=[x for x in range(0, 9)])

        # Skip the measurement if all the loss rates are 1.
        not_usable = False
        for candidate in candidates:
            not_usable |= ((df_result["loss_rate"] == 1) & (df_result["ip_address"] == candidates[1])).all()
        if not_usable:
            continue

        # new_entry = copy.deepcopy(missing_fields)
        df_individual = df_result[df_result["probing_type"] == "INDIVIDUAL"]
        # if not (df_individual["probing_rate"] == minimum_probing_rate).all():
        #     df_result = df_result[df_result["probing_rate"] != minimum_probing_rate]


        ind_probing_rate = df_result[df_result["probing_type"] == "INDIVIDUAL"]["probing_rate"].iloc[1]
        group_probing_rate = df_result[df_result["probing_type"] == "GROUPSPR"]["probing_rate"].iloc[1]
        # dpr_probing_rate = df_result[df_result["probing_type"] == "GROUPDPR"]["probing_rate"][0]
        probing_type_rates = {"INDIVIDUAL": [minimum_probing_rate, ind_probing_rate],
                              "GROUPSPR": [minimum_probing_rate, group_probing_rate],
                              "GROUPDPR": [minimum_probing_rate, group_probing_rate]}

        new_row = build_new_row(df_result, candidates, witnesses,
                                skip_fields,
                                probing_type_suffix,
                                probing_type_rates,
                                is_lr_classifier=True)

        correlation_row = parse_correlation(df_result, [group_probing_rate], candidates, witnesses)
        new_row.update(correlation_row)

        label = set_router_labels(new_entry, ground_truth_routers[node], candidates, witnesses)

        if label == "U":
            continue

        new_entry["measurement_id"] = result_file
        new_entry.update(new_row)

        # for i in range(0, len(debug)):
        #     if np.isclose(new_entry["loss_rate_dpr_c0_6000"] , debug[i][1], rtol=1e-05, atol=1e-08) \
        #         and np.isclose(new_entry["loss_rate_dpr_c0_9000"] , debug[i][0], rtol=1e-05, atol=1e-08):
        #         print df_result.to_string()


        for i in range(0, len(candidates)):
            if new_entry["label_c0"] != new_entry["label_c" + str(i)]:
                TN += 1
                print "TN: " + str(TN)

        '''
        Excluded from the classifier:

        '''
        # In a non alias example, if the loss rate for minimum rate is too high, discard it.
        not_aliases_lr_min_rate_too_high = False
        for i in range(1, len(candidates)):
            if new_entry["label_c" + str(i)] == 0:
                df_candidate_not_alias_min_rate = df_individual[(df_individual["ip_address"] == candidates[i])
                                                                & (
                                                                df_individual["probing_rate"] == minimum_probing_rate)]
                if len(df_candidate_not_alias_min_rate) > 1:
                    # The triggering rate was the minimum probing rate.
                    continue
                if (df_candidate_not_alias_min_rate["loss_rate"] > 0.10).all():
                    not_aliases_lr_min_rate_too_high = True
                    break
        if not_aliases_lr_min_rate_too_high:
            n_not_aliases_lr_too_high += 1
            continue

        # RL Not triggered
        df_dpr = df_result[(df_result["probing_type"].isin(["GROUPDPR"]))]

        df_not_triggered = df_dpr[df_dpr["ip_address"] == candidates[0]]
        not_triggered = (df_not_triggered["loss_rate"] < n_not_triggered_threshold).all()
        if not_triggered:
            n_not_triggered += 1
            continue
        # RL not shared but aliases
        alias_but_not_shared = False

        if not not_triggered:
            for i in range(0, len(candidates)):
                df_not_shared = df_dpr[df_dpr["ip_address"] == candidates[i]]
                # Exclude not shared counter from classification:
                not_shared = (df_not_shared["loss_rate"] < epsilon).all()
                if not_shared and new_entry["label_c" + str(i)] == 1:
                    alias_but_not_shared = True
                    break
        if alias_but_not_shared:
            n_not_shared += 1
            continue

        # Check if the loss rate of the witness is too high.
        df_witness_lr = df_dpr[df_dpr["ip_address"] == witnesses[0]]["loss_rate"]
        minimum_lr = min(df_dpr["loss_rate"])

        if df_witness_lr.iloc[0] == 1:
            n_unusable_witness += 1
            continue
        # if df_witness_lr.iloc[0] > minimum_lr and df_witness_lr.iloc[0] >= 0.01:
        #     n_unusable_witness += 1
        #     continue
        if df_witness_lr.iloc[0] > alpha:
            n_witness_too_high += 1
            continue
            # for i in range(0, len(candidates)):
            #     new_entry["label_c" + str(i)] = 0

        # Check different patterns
        # correlation_patterns["correlation_spr"].append(correlation_row["correlation_spr_c1"])
        if float(correlation_row["correlation_c1"]) > 0.99 and new_entry["label_c1"] == 1 and not alias_but_not_shared:
            strong_correlations.append(result_file)

        if float(correlation_row["correlation_c1"]) < 0.3 and new_entry["label_c1"] == 1 and not alias_but_not_shared:
            weak_correlations.append(result_file)
            print "Not correlated but alias and triggered: " + result_file
        if not alias_but_not_shared and new_entry["label_c1"] == 1:
            correlations.append(result_file)
            correlation_distributions["correlation_dpr"].append(correlation_row["correlation_c1"])

        # If the number of candidates is only 2, switch to a one label classification
        if max_interfaces == 2:
            # Map the array of labels to a single label
            # 3 possibilities
            if new_entry["label_c0"] == 0 and new_entry["label_c1"] == 0:
                new_entry["label_pairwise"] = 2
            if new_entry["label_c0"] == 1 and new_entry["label_c1"] == 0:
                TN_pairwise += 1
                print "TN Pairwise: " + str(TN_pairwise)
                new_entry["label_pairwise"] = 0
            if new_entry["label_c0"] == 1 and new_entry["label_c1"] == 1:
                new_entry["label_pairwise"] = 1

        if df_computed_result is None:
            df_computed_result = pd.DataFrame(columns=new_entry.keys())
            df_computed_result.set_index(df_computed_result["measurement_id"])
        if len(new_entry) != df_computed_result.shape[1]:
            print "Bad measurement file " + result_file
            continue
        df_computed_result.loc[len(df_computed_result)] = new_entry

    if recompute_dataset:
        with open("resources/correlation_distributions.json", "w") as correlation_distributions_fp:
            json.dump(correlation_distributions, correlation_distributions_fp)
        with open("resources/weak_correlations.json", "w") as correlations_fp:
            json.dump(weak_correlations, correlations_fp)
        with open("resources/strong_correlations.json", "w") as correlations_fp:
            json.dump(strong_correlations, correlations_fp)

        with open("resources/correlations.json", "w") as correlations_fp:
            json.dump(correlations, correlations_fp)
    df_computed_result.set_index("measurement_id", inplace=True)
    df_computed_result.to_csv("resources/test_set_" + target_loss_rate_window, encoding="utf-8")


    ############################### PRINT METRICS ##########################
    is_pairwise = True
    labeled_df = df_computed_result
    if is_pairwise:

        label = "label_pairwise"

        non_alias_df = labeled_df[labeled_df[label] == 0]
        alias_df = labeled_df[labeled_df[label] == 1]
        unknown_df = labeled_df[labeled_df[label] == 2]
        # true_unknown_df = labeled_df[labeled_df[label] == labels_map["U"]]
        print "Number of alias: " + str(len(alias_df))
        print "Number of alias but not shared: " + str(n_not_shared)
        print "Number of non alias: " + str(len(non_alias_df))
        print "Number of unknown: " + str(len(unknown_df))
        print "Number of not triggered: " + str(n_not_triggered)
        print "Number of unusable witness: " + str(n_unusable_witness)
        print "Number of witness loss rate too high: " + str(n_witness_too_high)
        print "Number of candidate loss rate too high on minimum rate: " + str(n_not_aliases_lr_too_high)
        print "Number of distinct ips: " + str(len(distinct_ips))
        print "Number of distinct routers: " + str(len(distinct_routers))



    return df_computed_result



def compute_classifier(feature_columns,
                       training_examples,
                       training_targets,
                       validation_examples,
                       validation_targets,
                       test_examples,
                       test_targets,
                       use_dnn,
                       use_random_forest):
    ######################## CLASSIFIERS ############################

    #### DEEP NEURAL NETWORK CLASSIFIER ####
    if use_dnn:
        classifier = nn.train_nn_classification_model(
            periods=20,
            n_classes=3,  # Maximum number of interfaces on a router + witness
            is_multilabel=not is_pairwise,
            feature_columns=feature_columns,
            learning_rate=0.05,
            steps=1000,
            batch_size=30,
            hidden_units=[labeled_df.shape[1], labeled_df.shape[1]],
            # hidden_units= [10, 10],
            training_examples=training_examples,
            training_targets=training_targets,
            validation_examples=validation_examples,
            validation_targets=validation_targets)

        predict_test_input_fn = nn.create_predict_input_fn(
            test_examples, test_targets, batch_size=100)

        test_predictions = classifier.predict(input_fn=predict_test_input_fn)
        if not is_pairwise:
            test_pred_labels = np.round([item['probabilities'] for item in test_predictions])
        else:
            test_pred_labels = np.array([item['class_ids'][0] for item in test_predictions])
        print_bad_labels(test_pred_labels, test_targets, test_examples, labeled_df)

        precision, recall, accuracy = compute_metrics(test_predictions, test_targets)
        print "Precision: " + str(precision)
        print "Recall: " + str(recall)
        print "Accuracy: " + str(accuracy)

    ################# RANDOM FOREST ##################

    if use_random_forest:
        classifier = rf_.random_forest_classifier(training_examples,
                                                     training_targets,
                                                     )

        validations_predictions = classifier.predict(validation_examples)
        importances = feature_importance(classifier, training_examples.columns)
        # print importances.to_string()
        # print_false_positives(validations_predictions, validation_targets, validation_examples, labeled_df)

    return classifier

if __name__ == "__main__":
    recompute_dataset = False
    is_pairwise = True
    '''
        Parameters of the classifiers
        target loss rate window:
        epsilon: the lower bound of the "aliases with per-interface rate limiting".
        alpha: the lower bound of the "witness" label.

    '''


    target_loss_rate_window = "lr0.05-0.10"
    epsilon = 0.02
    alpha = 0.02
    n_not_triggered_threshold = 0.05
    if recompute_dataset:
        df_computed_result = compute_dataset(is_pairwise,
                                             target_loss_rate_window = target_loss_rate_window,
                                             epsilon=epsilon,
                                             alpha=alpha,
                                             n_not_triggered_threshold=n_not_triggered_threshold)
    # label = "label"
    else:
        df_computed_result = pd.read_csv("resources/test_set_" + target_loss_rate_window, index_col=0)

    for column in df_computed_result.columns:
        if column.startswith("changing_behaviour"):
            df_computed_result[column] = minmax_scale(np.array(df_computed_result[column]).reshape(-1, 1))
        if column.startswith("label"):
            df_computed_result[column] = df_computed_result[column].apply(np.int64)

    labeled_df = df_computed_result
    labeled_df = labeled_df.reset_index()

    feature_columns, labels_column = extract_feature_labels_columns(df_computed_result, is_pairwise)

    labeled_df = labeled_df.dropna(subset=feature_columns)
    loss_rate_df = labeled_df[["measurement_id","loss_rate_dpr_c0", "loss_rate_dpr_c1", "loss_rate_dpr_w0"]]

    # Now train a classifier on the labeled data.
    #
    # Shuffle the data
    labeled_df = labeled_df.sort_index(axis=1)

    ############################# Evaluate the classifier ##########################
    precisions = []
    recalls = []
    accuracys = []

    use_random_forest = True
    use_dnn = False
    for i in range(0, 20):
        labeled_df = labeled_df.reindex(np.random.permutation(labeled_df.index))
        labeled_df.to_csv("resources/labeled_test_set", encoding='utf-8')
        # Split the training validation and test set
        training_n = int(0.3 * len(labeled_df))
        training_df = labeled_df.iloc[0:training_n]

        cross_validation_n = int(0.3 * len(labeled_df))
        cross_validation_df = labeled_df.iloc[training_n + 1:cross_validation_n + training_n]

        test_df = labeled_df.iloc[cross_validation_n + training_n + 1:]

        training_targets, training_examples = nn.parse_labels_and_features(training_df, labels_column, feature_columns)
        print "Size of the training examples: " + str(training_examples.shape)
        print_df_labels(training_df)

        validation_targets, validation_examples = nn.parse_labels_and_features(cross_validation_df, labels_column, feature_columns)
        print "Size of the validation examples: " + str(validation_examples.shape)
        print_df_labels(cross_validation_df)

        test_targets, test_examples = nn.parse_labels_and_features(test_df, labels_column, feature_columns)
        print "Size of the test examples: " + str(test_examples.shape)
        print_df_labels(test_df)


        classifier = compute_classifier(feature_columns,
                                        training_examples, training_targets,
                                        validation_examples, validation_targets,
                                        test_examples, test_targets,
                                        use_dnn=use_dnn,
                                        use_random_forest=use_random_forest)


        if use_random_forest:
            examples = pd.concat([validation_examples, test_examples])
            targets = pd.concat([validation_targets, test_targets])
            predictions = classifier.predict(examples)

            precision, recall, accuracy = compute_metrics(predictions, targets)
            print "Precision: " + str(precision)
            print "Recall: " + str(recall)
            print "Accuracy: " + str(accuracy)
            precisions.append(precision)
            recalls.append(recall)
            accuracys.append(accuracy)

    with open("resources/results_classifier_"+ target_loss_rate_window + ".json", "w") as results_classifier_fp:
        results = {"precision" : precisions, "recall":recalls, "accuracies":accuracys}
        json.dump(results, results_classifier_fp)
        # from sklearn.tree import export_graphviz

        # Export as dot file
        # export_graphviz(rf_classifier.estimators_[0], out_file='tree.dot',
        #                 feature_names=training_examples.columns,
        #                 class_names=["Non Alias", "Alias"],
        #                 rounded=True, proportion=False,
        #                 precision=2, filled=True)
        #
        # # Convert to png using system command (requires Graphviz)
        # from subprocess import call
        #
        # call(['dot', '-Tpng', 'tree.dot', '-o', 'tree.png', '-Gdpi=600'])
        #
        # # Display in jupyter notebook
        # from IPython.display import Image
        #
        # Image(filename='tree.png')

        # # training_examples = tf.convert_to_tensor(training_examples.values, dtype=tf.float64)
    # # training_targets = tf.convert_to_tensor(training_targets.values, dtype=tf.float64)
    # #
    # # validation_examples = tf.convert_to_tensor(validation_examples.values, dtype=tf.float64)
    # # validation_targets = tf.convert_to_tensor(validation_targets.values, dtype=tf.float64)
    # # test_examples = tf.convert_to_tensor(test_examples.values, dtype=tf.float64)
    # # test_targets = tf.convert_to_tensor(test_targets.values, dtype=tf.float64)
    # results, _ = adanet_.train(periods=10,
    #                            training_examples=training_examples,
    #                            training_targets =training_targets,
    #                            validation_examples = validation_examples,
    #                            validation_targets = validation_targets,
    #                            labels_n = 3,
    #                            learning_rate=0.001,
    #                            steps=1000,
    #                            batch_size = 30
    #                                         )
    # print("Loss:", results["average_loss"])
    # print("Architecture:", adanet_.ensemble_architecture(results))
    # print results
    #
    # # #@test {"skip": true}
    # results, _ = adanet_.train(periods=10,
    #                            training_examples=training_examples,
    #                            training_targets=training_targets,
    #                            validation_examples=validation_examples,
    #                            validation_targets=validation_targets,
    #                            labels_n=3,
    #                            learning_rate=0.001,
    #                            steps=1000,
    #                            batch_size=30,
    #                            learn_mixture_weights=True
    #                            )
    # print("Loss:", results["average_loss"])
    # print("Uniform average loss:", results["average_loss/adanet/uniform_average_ensemble"])
    # print("Architecture:", adanet_.ensemble_architecture(results))
    # print results
    # #
    # #
    # # #@test {"skip": true}
    # results, _ = adanet_.train(periods=10,
    #                            training_examples=training_examples,
    #                            training_targets=training_targets,
    #                            validation_examples=validation_examples,
    #                            validation_targets=validation_targets,
    #                            labels_n=3,
    #                            learning_rate=0.001,
    #                            steps=1000,
    #                            batch_size=30,
    #                            learn_mixture_weights=True,
    #                            adanet_lambda=0.015
    #                            )
    # print("Loss:", results["average_loss"])
    # print("Uniform average loss: ", results["average_loss/adanet/uniform_average_ensemble"])
    # print("Architecture:", adanet_.ensemble_architecture(results))
    # print results



