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
from Classification import adanet_wrap as adanet_
from Classification import random_forest as rf_
from Data.preprocess import minmax_scale
from Validation.midar import extract_routers, set_router_labels, extract_routers_by_node
tf.logging.set_verbosity(tf.logging.ERROR)
pd.options.display.max_rows = 25
pd.options.display.float_format = '{:.3f}'.format
#################################################

import sys
from Data.preprocess import *

labels_map = {"P" : 1, "N" : 0, "U": 2}

# Add the type of probing to the columns.
def add_suffix(suffix, columns):
    suffixed_columns = []
    for column in columns:
        suffixed_columns.append("".join([column, "_", suffix]))

    return suffixed_columns


def find_index(e, l):
    for i in range(0, len(l)):
        if l[i] == e:
            return i
    return None


def parse_correlation(df, rates_dpr, candidates, witnesses):


    row = {}
    high_rate_candidate_ip = candidates[0]

    for rate in rates_dpr:
        df_filter_dpr_rate_high_rate_candidate = df[(df["probing_rate"]== rate) & \
                                                     (df["ip_address"] == high_rate_candidate_ip) & \
                                                      (df["probing_type"] ==  "GROUPDPR")]

        for field in df_filter_dpr_rate_high_rate_candidate.keys():
            if field.startswith("correlation"):
                correlation_field = df_filter_dpr_rate_high_rate_candidate[field].iloc[0]

                # In case candidates correlation are not in the same order
                correlation_split = correlation_field.split(":")
                ip_correlation = correlation_split[0].strip()
                correlation = correlation_split[1].strip()



                ip_corr_index = find_index(ip_correlation, candidates)
                if ip_corr_index is not None:
                    row["correlation_c" + str(ip_corr_index)] = correlation
                else:
                    ip_corr_index = find_index(ip_correlation, witnesses)
                    if ip_corr_index is not None:
                        row["correlation_w" + str(ip_corr_index)] = correlation

    return row


debug = [
]


recompute_dataset = False
if __name__ == "__main__":

    version  = sys.argv[1]

    results_dir = "/srv/icmp-rl-survey/midar/survey/batch1/results/"
    routers_path = "resources/midar/batch1/routers/"
    ground_truth_routers = extract_routers_by_node(routers_path)
    max_interfaces = 2

    pd.options.display.float_format = '{:.5f}'.format

    # Probing rates
    minimum_probing_rate = 500

    skip_fields = ["Index", "probing_type", "ip_address", "probing_rate", "correlation",
                   "transition_0_0", "transition_0_1", "transition_1_0", "transition_1_1"]

    df_computed_result = None

    probing_type_suffix = {"INDIVIDUAL": "ind", "GROUPSPR": "spr", "GROUPDPR": "dpr"}

    # raw_columns are the extracted data from CSV
    global_raw_columns = ["ip_address", "probing_type", "probing_rate", "changing_behaviour", "loss_rate",
                   "transition_0_0", "transition_0_1", "transition_1_0", "transition_1_1"]

    # missing_fields = build_missing_candidates(0, max_interfaces, 0, 1, global_raw_columns, skip_fields,
    #                                           probing_type_suffix, probing_type_rates)


    # Debug infos
    total  = 0
    k  = 0
    TN = 0
    FN = 0
    n_not_triggered = 0
    TN_pairwise = 0
    multiple_interfaces_router = {k:0 for k in range(2, max_interfaces +1)}


    for result_file in os.listdir(results_dir):


        # Only results file
        if os.path.isdir(results_dir + result_file):
            continue

        #################### DEBUG INFOS ################
        total += 1
        print total
        # if total != 5049:
        #     continue
        if not recompute_dataset:
            if k == 1:
                break
        #################################################

        new_entry = {}
        split_file_name = result_file.split("_")
        node = split_file_name[0]

        ip_index = 5
        candidates = []
        witnesses = []
        if "TN" in result_file:
            candidates_witness_dir = "/srv/icmp-rl-survey/midar/survey/batch1/candidates-witness-tn/"
        else:
            candidates_witness_dir = "/srv/icmp-rl-survey/midar/survey/batch1/candidates-witness/"
        with open(candidates_witness_dir + result_file) as cw_file:
            for line in cw_file:
                line = line.replace(" ", "")
                fields = line.split(",")
                if "CANDIDATE" in fields:
                    candidates.append(fields[ip_index])
                elif "WITNESS" in fields:
                    witnesses.append(fields[ip_index])

        ######################### SANITIZER #######################
        if len(candidates) > max_interfaces:
            continue

        if len(set(candidates).intersection(set(witnesses))) != 0:
            continue
        ###########################################################


        raw_columns = copy.deepcopy(global_raw_columns)

        # Add correlation columns
        for i in range(1, len(candidates)):
            raw_columns.append("correlation_c" + str(i))

        for i in range(0, len(witnesses)):
            raw_columns.append("correlation_w" + str(i))

        multiple_interfaces_router [len(candidates)] += 1
        print "Routers with more than 2 interfaces: " + str(multiple_interfaces_router)
        # 1 point is represented by different dimensions:
        df_result = pd.read_csv(results_dir+result_file,
                                names = raw_columns,
                                skipinitialspace=True,
                                index_col=False)
                                # usecols=[x for x in range(0, 9)])

        # Skip the measurement if all the loss rates are 1.
        not_usable = (df_result["loss_rate"] == 1).all()
        if not_usable:
            continue

        # Measurement contains -1, meaning no change point detection method.
        # change_behaviour_column = df_result["changing_behaviour"]
        # if -1 in change_behaviour_column.values:
        #     continue
        # else:
        #     k += 1
        #     print "CPD: " +str(k)
        # # If the measurement is incomplete
        # if (df_result["probing_type"] == "INDIVIDUAL").all():
        #     continue



        # new_entry = copy.deepcopy(missing_fields)
        df_result = df_result[df_result["probing_rate"] != minimum_probing_rate]

        ind_probing_rate = df_result[df_result["probing_type"] == "INDIVIDUAL"]["probing_rate"][0]
        group_probing_rate = df_result[df_result["probing_type"] == "GROUPSPR"]["probing_rate"][0]
        # dpr_probing_rate = df_result[df_result["probing_type"] == "GROUPDPR"]["probing_rate"][0]
        probing_type_rates = {"INDIVIDUAL": ind_probing_rate,
                              "GROUPSPR": [group_probing_rate],
                              "GROUPDPR": [group_probing_rate]}


        new_row = build_new_row(df_result, candidates, witnesses, skip_fields, probing_type_suffix, probing_type_rates)

        correlation_row = parse_correlation(df_result, [group_probing_rate], candidates, witnesses)
        new_row.update(correlation_row)

        label = set_router_labels(new_entry, ground_truth_routers[node], candidates, witnesses)



        if label == "U":
            continue

        # Remove rates where all the interfaces fully respond from the new row.
        # remove_uninteresting_rate(new_row, ind_rates, "ind", default_full_responsiveness_value)
        # remove_uninteresting_rate(new_row, ind_rates, "dpr", default_full_responsiveness_value)
        # remove_uninteresting_rate(new_row, ind_rates, "spr", default_full_responsiveness_value)



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


        # Change the label in the case ICMP RL was not triggered during the group phase or the counter is not shared.
        df_filter_group = df_result[(df_result["probing_type"].isin(["GROUPDPR"])) &  \
                                    (df_result["ip_address"] != candidates[0]) & \
                                    (df_result["ip_address"].isin(candidates))]
        not_triggered =  (df_filter_group["loss_rate"] < 0.03).all()
        if not_triggered:
            n_not_triggered += 1
            print "Not triggered: " + str(n_not_triggered)
            for i in range(0, len(candidates)):
                new_entry["label_c" + str(i)] = 0



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

        df_computed_result.loc[len(df_computed_result)] = new_entry





    # label = "label"
    if recompute_dataset:
        df_computed_result.set_index("measurement_id", inplace=True)
        df_computed_result.to_csv("resources/test_set", encoding="utf-8")
    df_computed_result = pd.read_csv("resources/test_set", index_col=0)
    # df_computed_result.set_index("measurement_id", inplace=True)
    #
    # # labeled_df, cluster_n = DBSCAN_impl(computed_df, feature_columns)
    # cluster_n = 3

    for column in df_computed_result.columns:
        if column.startswith("changing_behaviour"):
            df_computed_result[column] = minmax_scale(np.array(df_computed_result[column]).reshape(-1, 1))
        if column.startswith("label"):
            df_computed_result[column] = df_computed_result[column].apply(np.int64)

    labeled_df = df_computed_result
    labeled_df = labeled_df.reset_index()
    # labeled_df = labeled_df.dropna(subset=["label"])
    is_pairwise = True

    if is_pairwise:

        label = "label_pairwise"

        true_negatives_df = labeled_df[labeled_df[label] == 0]
        true_positives_df = labeled_df[labeled_df[label] == 1]
        # true_unknown_df = labeled_df[labeled_df[label] == labels_map["U"]]
        print "Number of true negatives: " + str(len(true_negatives_df))
        print "Number of false negatives: " + str(FN)
        print "Number of unknown: " + str(n_not_triggered)
        print "Number of true positives: " + str(len(true_positives_df))



    # Select features
    feature_columns = []
    for column in df_computed_result.columns:
        if  not column.startswith("transition") \
            and not column.startswith("label")  \
                :
            feature_columns.append(column)

    labels_column = []
    for column in df_computed_result.columns:
        if is_pairwise:
            if column == ("label_pairwise"):
                labels_column.append(column)
        else:
            if column.startswith("label"):
                labels_column.append(column)

    labeled_df = labeled_df.dropna(subset=feature_columns)

    # Now train a classifier on the labeled data.
    #
    # Shuffle the data
    labeled_df = labeled_df.sort_index(axis=1)
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


    validation_targets, validation_examples = nn.parse_labels_and_features(cross_validation_df, labels_column, feature_columns)
    print "Size of the training examples: " + str(validation_examples.shape)

    test_targets, test_examples = nn.parse_labels_and_features(test_df, labels_column, feature_columns)
    print "Size of the training examples: " + str(test_examples.shape)


    ######################## CLASSIFIERS ############################
    use_dnn = False


    #### DEEP NEURAL NETWORK CLASSIFIER ####
    if use_dnn:
        classifier = nn.train_nn_classification_model(
            periods = 20,
            n_classes = max_interfaces + 1, # Maximum number of interfaces on a router + witness
            feature_columns = feature_columns,
            learning_rate=0.1,
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
        test_pred_labels = np.round([item['probabilities'] for item in test_predictions])

        bad_training_labels = 0
        bad_validation_labels = 0
        # DEBUG Print the labels for which the prediction are incorrect
        for i in range(0, len(test_pred_labels)):
            if not np.array_equal(test_pred_labels[i], np.array(test_targets.iloc[i])):
                bad_training_labels += 1
                true_label = test_examples.iloc[i].sort_index()
                bad_label = test_pred_labels[i]
                # print training_pred_labels
                print "[" + str(true_label["loss_rate_dpr_c0_9000"]) + ", " + str(true_label["loss_rate_dpr_c0_6000"]) + "]"
                print true_label.filter(regex="loss_rate_dpr_c([0-9]+)_9000", axis=0)
                print true_label.filter(regex="label_c([0-9]+)", axis=0)
                print bad_label

        accuracy = metrics.accuracy_score(test_targets, test_pred_labels)
        print("Accuracy on test data: %0.3f" % accuracy)



    ################# RANDOM FOREST ##################
    use_random_forest = True

    if use_random_forest:
        rf_classifier = rf_.random_forest_classifier(training_examples, training_targets,
                                     validation_examples, validation_targets)

        print rf_classifier.predict(test_examples)
        print rf_classifier.score(test_examples, test_targets)

        from sklearn.tree import export_graphviz

        # Export as dot file
        export_graphviz(rf_classifier.estimators_[0], out_file='tree.dot',
                        feature_names=training_examples.columns,
                        class_names=["Non Alias", "Alias"],
                        rounded=True, proportion=False,
                        precision=2, filled=True)

        # Convert to png using system command (requires Graphviz)
        from subprocess import call

        call(['dot', '-Tpng', 'tree.dot', '-o', 'tree.png', '-Gdpi=600'])

        # Display in jupyter notebook
        from IPython.display import Image

        Image(filename='tree.png')

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



