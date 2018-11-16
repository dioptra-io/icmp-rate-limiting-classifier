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
from Data.preprocess import minmax_scale
from Validation.midar import extract_routers, set_router_labels, transitive_closure
tf.logging.set_verbosity(tf.logging.ERROR)
pd.options.display.max_rows = 10
pd.options.display.float_format = '{:.3f}'.format
#################################################

import sys
import copy


labels_map = {"P" : 1, "N" : 0, "U": 2}

# raw_columns are the extracted data from CSV
raw_columns = ["ip_address", "probing_type", "probing_rate", "changing_behaviour", "loss_rate",
           "transition_0_0", "transition_0_1", "transition_1_0", "transition_1_1"]
           # "correlation_1", "correlation_2"]

# Base columns are the different the base columns for the NN. Rate is added further.
base_columns = [
    "cb",
    "lr",
    "tr_0_0", "tr_0_1", "tr_1_0", "tr_1_1"
]

# Add the type of probing to the columns.
def add_suffix(suffix, columns):
    suffixed_columns = []
    for column in columns:
        suffixed_columns.append("".join([column, "_", suffix]))

    return suffixed_columns

def parse_correlation(df, row, correlation_columns, candidates, witnesses):
    candidate_candidate_difference_correlation = {}
    witness_candidate_difference_correlation = {}


    for correlation_column in correlation_columns:
        correlation_value = getattr(row, correlation_column)
        probing_rate_value = getattr(row, "probing_rate")
        ip_address = getattr(row, "ip_address")
        if isinstance(correlation_value, basestring):

            split_cor = correlation_value.split(":")
            ip_address_2 = split_cor[0]
            cor = float(split_cor[1])
            if ip_address in candidates and ip_address_2 in candidates:
                if candidate_candidate_difference_correlation.has_key(probing_rate_value):
                    candidate_candidate_difference_correlation[probing_rate_value].append(cor)
                else:
                    candidate_candidate_difference_correlation[probing_rate_value] = [cor]
            elif ip_address in witnesses:
                if witness_candidate_difference_correlation.has_key(probing_rate_value):
                    witness_candidate_difference_correlation[probing_rate_value].append(cor)
                else:
                    witness_candidate_difference_correlation[probing_rate_value] = [cor]

    return candidate_candidate_difference_correlation, witness_candidate_difference_correlation

def build_missing_candidates_impl(n_min, n_max, default_columns, skip_fields, probing_type_suffix, probing_type_rates, interface_type_suffix):
    new_entry = {}
    for i in range(n_min, n_max):
        for probing_type, probing_rates in probing_type_rates.iteritems():
            for probing_rate in probing_rates:
                for column in default_columns:
                    if column in skip_fields:
                        continue
                    # if default_value_feature.has_key(column):
                    #     value = default_value_feature[column]
                    # else:
                    value = 0.0
                    new_entry["".join([column,
                                       "_",
                                       probing_type_suffix[probing_type],
                                       interface_type_suffix,
                                       str(i),
                                       "_",
                                       str(probing_rate)])] = value
        new_entry["".join(["label", interface_type_suffix,  str(i)])] = 0
    return new_entry


def build_missing_candidates(n_candidates,
                             n_candidates_max,
                             n_witnesses,
                             n_witnesses_max,
                             default_columns,
                             skip_fields,
                             probing_type_suffix,
                             probing_type_rates):

    new_entry = build_missing_candidates_impl(n_candidates, n_candidates_max, default_columns, skip_fields, probing_type_suffix, probing_type_rates, "_c")
    new_entry.update(build_missing_candidates_impl(n_witnesses, n_witnesses_max, default_columns, skip_fields, probing_type_suffix, probing_type_rates, "_w"))
    return new_entry

def build_new_row(df_result, candidates, witnesses, skip_fields, probing_type_suffix, probing_type_rates):

    ips = copy.deepcopy(candidates)
    ips.extend(witnesses)
    df_result["changing_behaviour"] = df_result["changing_behaviour"].replace(-1, 20000)

    #
    # for i in range(0, len(ips)):
    #     if ips[i] in candidates:
    #         additional_suffix = "c"
    #     elif ips[i] in witnesses:
    #         additional_suffix = "w"
    #
    #
    #
    #     # Optimization: perform minimal condition checking
    #     prefilter = (df_result["ip_address"] == ips[i]) & (df_result["probing_type"] == probing_type) & (df_result["probing_rate"].isin(rates))
    #     df_prefiltered = df_result[prefilter]
    #     df_prefiltered = df_prefiltered.drop(["ip_address", "probing_type"], axis=1)
    #
    #     for rate in rates:
    #         filter_rate = (df_prefiltered["probing_rate"] == rate)
    #         # Select only relevant lines
    #         df_filtered = df_prefiltered[filter_rate]
    #
    #         df_filtered = df_filtered.drop(["probing_rate"], axis=1)
    #         # df_filtered = df_filtered.reset_index(drop = True)
    #
    #         # Change the name of the columns
    #         columns_new = add_suffix("".join([probing_type_suffix, "_", additional_suffix, str(i), "_", str(rate)]), df_filtered.columns)
    #         df_filtered.columns = columns_new
    #
    #         for column in df_filtered.columns:
    #             df_feature_columns[column] = df_filtered[column]

    # Dict implementation
    new_entry = {}


    for row in df_result.itertuples():
        probing_rate = row.probing_rate

        ip_address = row.ip_address
        probing_type = row.probing_type

        if probing_rate not in probing_type_rates[probing_type]:
            continue

        for i in range(0, len(row._fields)):
            # Skip some fields
            if row._fields[i] in skip_fields:
                continue
            if ip_address in candidates:
                additional_suffix = "c"
                ips = candidates
            elif ip_address in witnesses:
                additional_suffix = "w"
                ips = witnesses
            new_entry["".join([row._fields[i],
                               "_",
                               probing_type_suffix[probing_type],
                               "_", additional_suffix,
                               str(ips.index(row.ip_address)),
                               "_",
                               str(probing_rate)])] = row[i]


        # print row

    return new_entry

def is_broken_witness(df_result, probing_type, rates, witnesses, candidates):
    broken_witness = False
    for rate in rates:
        df_witness = df_result[
            (df_result["probing_type"] == probing_type) & (df_result["ip_address"].isin(witnesses)) & (
            df_result["probing_rate"] == rate)]
        df_candidates = df_result[
            (df_result["probing_type"] == probing_type) & (df_result["ip_address"].isin(candidates)) & (
            df_result["probing_rate"] == rate)]

        if len(df_candidates) != 0 and len(df_witness) != 0:
            lr_candidates = max(df_candidates["loss_rate"])
            lr_witness = max(df_witness["loss_rate"])

            if lr_witness != 1 and lr_witness > lr_candidates:
                broken_witness = True
                break
    return broken_witness


debug = [
    [0.880167, 0.821792] ,
    [0.945033, 0.893095],
    [0.661235, 0.54353],
    [0.713343, 0.550308],
    [0.705057, 0.559178]]

if __name__ == "__main__":

    version  = sys.argv[1]

    if version == "4":
        results_dir = "/srv/icmp-rl-survey/results/v4/"
    elif version == "6":
        results_dir = "/srv/icmp-rl-survey/results/v6/"

    results_dir = "/srv/icmp-rl-survey/midar/results/"
    candidates_witness_dir = "/srv/icmp-rl-survey/midar/candidates-witness/"
    routers_path = "resources/midar/routers/"
    TN_path = "resources/midar/TN/"
    ground_truth_routers = extract_routers(routers_path)
    TN = extract_routers(TN_path)
    max_interfaces = 12

    pd.options.display.float_format = '{:.5f}'.format

    # Probing rates
    ind_rates = [500, 1000, 2000, 3000]
    spr_and_dpr_rates = [ 3 * x for x in ind_rates]
    # Relevant columns for individual and dpr lines
    ind_and_dpr_columns = [
        "loss_rate",
        "transition_0_0", "transition_0_1", "transition_1_0", "transition_1_1"
    ]
    skip_fields = ["Index", "probing_type", "ip_address", "probing_rate", "correlation_1", "correlation_2"]

    df_computed_result = None

    probing_type_suffix = {"INDIVIDUAL": "ind", "GROUPSPR": "spr", "GROUPDPR": "dpr"}
    probing_type_rates = {"INDIVIDUAL": ind_rates, "GROUPSPR": spr_and_dpr_rates, "GROUPDPR": spr_and_dpr_rates}


    missing_fields = build_missing_candidates(0, max_interfaces, 0, 1, raw_columns, skip_fields,
                                              probing_type_suffix, probing_type_rates)

    k  = 0
    TN = 0
    for result_file in os.listdir(results_dir):
        if os.path.isdir(results_dir + result_file):
            continue
        new_entry = {}

        k += 1
        print k
        # if k ==1 :
        #     break
        # print result_file
        split_file_name = result_file.split("_")
        ip_index = 5
        candidates = []
        witnesses = []
        with open(candidates_witness_dir + result_file) as cw_file:
            for line in cw_file:
                line = line.replace(" ", "")
                fields = line.split(",")
                if "CANDIDATE" in fields:
                    candidates.append(fields[ip_index])
                elif "WITNESS" in fields:
                    witnesses.append(fields[ip_index])

        # 1 point is represented by different dimensions:
        df_result = pd.read_csv(results_dir+result_file,
                                names = raw_columns,
                                skipinitialspace=True,
                                index_col=False,
                                usecols=[x for x in range(0, 9)])

        # Skip the measurement if all the loss rates are 1.
        not_usable = (df_result["loss_rate"] == 1).all()
        if not_usable:
            continue


        new_entry = copy.deepcopy(missing_fields)
        new_row = build_new_row(df_result, candidates, witnesses, skip_fields, probing_type_suffix, probing_type_rates)
        label = set_router_labels(new_entry, ground_truth_routers, TN, candidates, witnesses)

        new_entry["measurement_id"] = result_file
        new_entry.update(new_row)


        for i in range(0, len(debug)):
            if np.isclose(new_entry["loss_rate_dpr_c0_6000"] , debug[i][1], rtol=1e-05, atol=1e-08) \
                and np.isclose(new_entry["loss_rate_dpr_c0_9000"] , debug[i][0], rtol=1e-05, atol=1e-08):
                print df_result.to_string()


        if label == "U":
            continue


        # if label == "P":
        #     df_feature_columns["label"] = 1
            # df_filtered_FP = abs(df_feature_columns["loss_rate_dpr_c0_9000"] - df_feature_columns["loss_rate_dpr_c1_9000"])
            # if df_filtered_FP[0] > 0.4:
            #     print df_result.to_string()
        unknown_indicator = new_entry["loss_rate_dpr_c0_9000"]
        if unknown_indicator == 1.0:
            # print df_result.to_string()
            continue
        # unknown_indicator = new_entry["changing_behaviour_dpr_c0_6000"]
        # if unknown_indicator == 0.0:
        #     # print df_result.to_string()
        #     print "Removed!"
        #     continue


        for i in range(0, len(candidates)):
            if new_entry["label_c0"] != new_entry["label_c" + str(i)]:
                TN += 1
                print "TN: " + str(TN)
            else:

                if abs(new_entry["loss_rate_dpr_c0_9000"] - new_entry["loss_rate_dpr_c"+ str(i) + "_9000"]) > 0.6 \
                        and new_entry["loss_rate_dpr_c" + str(i) + "_9000"] < 0.1:

                    for i in range(0, len(candidates)):
                        new_entry["".join(["label_c", str(i)])] = 0
                    for i in range(0, len(witnesses)):
                        new_entry["".join(["label_w", str(i)])] = 0
                    # print df_result.to_string()
        # elif label == "N":
        #     df_feature_columns["label"] = 0
        #     # print df_result.to_string()
        # elif label == "U":
        #     print "Unable to label: " + result_file
        #     continue

        if df_computed_result is None:
            df_computed_result = pd.DataFrame(columns=new_entry.keys())
            df_computed_result.set_index(df_computed_result["measurement_id"])

        df_computed_result.loc[len(df_computed_result)] = new_entry





    # label = "label"
    # df_computed_result.set_index("measurement_id", inplace=True)
    # df_computed_result.to_csv("resources/test_set", encoding="utf-8")
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


    # true_negatives_df = labeled_df[labeled_df[label] == labels_map["N"]]
    # true_positives_df = labeled_df[labeled_df[label] == labels_map["P"]]
    # true_unknown_df = labeled_df[labeled_df[label] == labels_map["U"]]
    # print "Number of true negatives: " + str(len(true_negatives_df))
    # print "Number of true positives: " + str(len(true_positives_df))
    # print "Number of unknown: " + str(len(true_unknown_df))


    # Select features
    feature_columns = []
    for column in df_computed_result.columns:
        if not column.endswith("500") \
                and not column.endswith("1000") \
                and not column.endswith("1500") \
                and not column.endswith("2000") \
                and not column.startswith("correlation")\
                and not column.startswith("transition")\
                :
            feature_columns.append(column)

    labels_column = []
    for column in df_computed_result.columns:
        if  column.startswith("label"):
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
    test_predictions = np.round([item['probabilities'] for item in test_predictions])

    accuracy = metrics.accuracy_score(test_targets, test_predictions)
    print("Accuracy on test data: %0.3f" % accuracy)
    # roc_auc = metrics.roc_auc_score(test_targets, test_predictions)
    # print("ROC AUC on test data: %0.3f" % roc_auc)
    #
    #
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



