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
from Validation.midar import extract_midar_routers, get_label, transitive_closure
tf.logging.set_verbosity(tf.logging.ERROR)
pd.options.display.max_rows = 10
pd.options.display.float_format = '{:.3f}'.format
#################################################

feature_rate_ind = [1000, 2000, 3000]
feature_rate_spr = [x * 3 for x in feature_rate_ind]
feature_rate_dpr = list(feature_rate_spr)

import sys
import copy


labels_map = {"P" : 1, "N" : 0, "U": 2}

# raw_columns are the extracted data from CSV
raw_columns = ["ip_address", "probing_type", "probing_rate", "changing_behaviour", "loss_rate",
           "transition_0_0", "transition_0_1", "transition_1_0", "transition_1_1",
           "correlation_1", "correlation_2"]

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
        suffixed_columns.append(column + "_" + suffix)

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


def add_columns(df_result, df_feature_columns, candidates, witnesses, rates, probing_type, probing_type_suffix):

    measurement_time = 5
    ratio = 8
    ips = copy.deepcopy(candidates)
    ips.extend(witnesses)
    # df_result["changing_behaviour"] = df_result["changing_behaviour"].replace(-1, 20000)
    for i in range(0, len(ips)):
        if ips[i] in candidates:
            additional_suffix = "c"
        elif ips[i] in witnesses:
            additional_suffix = "w"
        # Optimization: perform minimal condition checking
        prefilter = (df_result["ip_address"] == ips[i]) & (df_result["probing_type"] == probing_type) & (df_result["probing_rate"].isin(rates))
        df_prefiltered = df_result[prefilter]
        df_prefiltered = df_prefiltered.drop(["ip_address", "probing_type"], axis=1)

        for rate in rates:
            filter_rate = (df_prefiltered["probing_rate"] == rate)
            # Select only relevant lines
            df_filtered = df_prefiltered[filter_rate]
            # if probing_type == "INDIDIVUAL":
            #     df_filtered["changing_behaviour"] = df_filtered["changing_behaviour"].replace(-1,
            #                                                                                   rate * measurement_time)
            # elif probing_type == "GROUPSPR":
            #     df_filtered["changing_behaviour"] = df_filtered["changing_behaviour"].replace(-1,
            #                                                                                   rate * measurement_time/len(ips))
            # elif probing_type == "GROUPDPR":
            #     df_filtered["changing_behaviour"] = df_filtered["changing_behaviour"].replace(-1,
            #                                                                                   rate * measurement_time / ratio)

            df_filtered = df_filtered.drop(["probing_rate"], axis=1)
            df_filtered = df_filtered.reset_index(drop = True)


            # Change the name of the columns
            columns_new = add_suffix(probing_type_suffix + "_" + additional_suffix + str(i) + "_" + str(rate), df_filtered.columns)
            df_filtered.columns = columns_new

            df_feature_columns = pd.concat([df_feature_columns, df_filtered], axis=1)
    return df_feature_columns

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

if __name__ == "__main__":

    version  = sys.argv[1]

    if version == "4":
        results_dir = "/srv/icmp-rl-survey/results/v4/"
    elif version == "6":
        results_dir = "/srv/icmp-rl-survey/results/v6/"

    midar_path = "resources/internet2/midar/v4/routers/"
    ground_truth_routers = extract_midar_routers(midar_path)

    # Probing rates
    ind_rates = [1000, 2000, 3000]
    spr_and_dpr_rates = [ 3 * x for x in ind_rates]
    # Relevant columns for individual and dpr lines
    ind_and_dpr_columns = [
        "loss_rate",
        "transition_0_0", "transition_0_1", "transition_1_0", "transition_1_1"
    ]

    df_computed_result = pd.DataFrame()


    i  = 0
    for result_file in os.listdir(results_dir):

        new_entry = {}

        i += 1
        print i
        # if i == 1:
        #     break
        # print result_file
        split_file_name = result_file.split("_")
        candidates = [split_file_name[1], split_file_name[2]]
        witnesses = [split_file_name[3]]

        # 1 point is represented by different dimensions:
        df_result = pd.read_csv(results_dir+result_file, names = raw_columns,  skipinitialspace=True,)


        df_feature_columns = pd.DataFrame()
        df_feature_columns = add_columns(df_result, df_feature_columns, candidates, witnesses, ind_rates, "INDIVIDUAL", "ind")
        df_feature_columns = add_columns(df_result, df_feature_columns, candidates, witnesses, spr_and_dpr_rates, "GROUPSPR", "spr")
        df_feature_columns = add_columns(df_result, df_feature_columns, candidates, witnesses, spr_and_dpr_rates, "GROUPDPR", "dpr")



        df_feature_columns["measurement_id"] = result_file
        label = get_label(ground_truth_routers, candidates)

        if label == "P":
            df_feature_columns["label"] = 1
        elif label == "N":
            df_feature_columns["label"] = 0
            print df_result.to_string()
        elif label == "U":
            print "Unable to label: " + result_file
            continue

        """
            Check manually if the measurement is considered trustable or not according to the witness.
            i.e if the LR of the witness is greater than the LR of the candidates  
        """
        # if is_broken_witness(df_result, "INDIVIDUAL", ind_rates, witnesses, candidates) \
        #     or is_broken_witness(df_result, "GROUPSPR", spr_and_dpr_rates, witnesses, candidates) \
        # if is_broken_witness(df_result, "GROUPDPR", spr_and_dpr_rates, witnesses, candidates):
        #     df_feature_columns["label"] = 2

        df_computed_result = df_computed_result.append(df_feature_columns)



    for column in df_computed_result.columns:
        if column.startswith("changing_behaviour"):
            df_computed_result[column] = minmax_scale(np.array(df_computed_result[column]).reshape(-1,1))

    label = "label"
    df_computed_result.set_index("measurement_id", inplace=True)
    df_computed_result.to_csv("resources/test_set", encoding="utf-8")
    df_computed_result = pd.read_csv("resources/test_set", index_col=0)
    #
    # # labeled_df, cluster_n = DBSCAN_impl(computed_df, feature_columns)
    # cluster_n = 3
    labeled_df = df_computed_result
    labeled_df = labeled_df.dropna(subset=["label"])
    labeled_df[label] = labeled_df[label].apply(np.int64)

    true_negatives_df = labeled_df[labeled_df[label] == labels_map["N"]]
    true_positives_df = labeled_df[labeled_df[label] == labels_map["P"]]
    true_unknown_df = labeled_df[labeled_df[label] == labels_map["U"]]
    print "Number of true negatives: " + str(len(true_negatives_df))
    print "Number of true positives: " + str(len(true_positives_df))
    print "Number of unknown: " + str(len(true_unknown_df))


    # Select features
    feature_columns = []
    for column in df_computed_result.columns:
        if not column.endswith("1000") \
                and not column.endswith("2000") \
                and not column.startswith("correlation")\
                :
            feature_columns.append(column)

    labeled_df = labeled_df.dropna(subset=feature_columns)

    # Now train a classifier on the labeled data.
    #
    # Shuffle the data
    labeled_df = labeled_df.reindex(np.random.permutation(labeled_df.index))
    labeled_df.to_csv("resources/labeled_test_set", encoding='utf-8')
    # Split the training validation and test set
    training_n = int(0.3 * len(labeled_df))
    training_df = labeled_df.iloc[0:training_n]

    cross_validation_n = int(0.3 * len(labeled_df))
    cross_validation_df = labeled_df.iloc[training_n + 1:cross_validation_n + training_n]

    test_df = labeled_df.iloc[cross_validation_n + training_n + 1:]

    training_targets, training_examples = nn.parse_labels_and_features(training_df, label, feature_columns)
    print "Size of the training examples: " + str(training_examples.shape)


    validation_targets, validation_examples = nn.parse_labels_and_features(cross_validation_df, label, feature_columns)
    print "Size of the training examples: " + str(validation_examples.shape)

    test_targets, test_examples = nn.parse_labels_and_features(test_df, label, feature_columns)
    print "Size of the training examples: " + str(test_examples.shape)

    classifier = nn.train_nn_classification_model(
        periods = 10,
        classes_n = 2, # Positive, Negative
        feature_columns = feature_columns,
        learning_rate=0.05,
        steps=1000,
        batch_size=30,
        hidden_units=[20, 20],
        training_examples=training_examples,
        training_targets=training_targets,
        validation_examples=validation_examples,
        validation_targets=validation_targets)

    predict_test_input_fn = nn.create_predict_input_fn(
        test_examples, test_targets, batch_size=100)

    test_predictions = classifier.predict(input_fn=predict_test_input_fn)
    test_predictions = np.array([item['class_ids'][0] for item in test_predictions])

    accuracy = metrics.accuracy_score(test_targets, test_predictions)
    print("Accuracy on test data: %0.3f" % accuracy)
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



