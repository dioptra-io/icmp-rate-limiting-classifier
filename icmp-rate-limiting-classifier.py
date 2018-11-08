######################## Imports from https://colab.research.google.com/notebooks/mlcc/multi-class_classification_of_handwritten_digits.ipynb?utm_source=mlcc&utm_campaign=colab-external&utm_medium=referral&utm_content=multiclass-colab&hl=fr#scrollTo=4LJ4SD8BWHeh #######################


import glob
import math
import os

#from IPython import display
from matplotlib import cm
from matplotlib import gridspec
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import metrics
import tensorflow as tf
from tensorflow.python.data import Dataset
from Cluster.dbscan import kmeans, DBSCAN_impl
from Classification import neural_network as nn
from Classification import adanet_wrap as adanet_
from Validation.midar import compute_midar_routers, get_label, transitive_closure
tf.logging.set_verbosity(tf.logging.ERROR)
pd.options.display.max_rows = 10
pd.options.display.float_format = '{:.3f}'.format
#################################################

feature_rate_ind = [1000, 2000, 3000]
feature_rate_spr = [x * 3 for x in feature_rate_ind]
feature_rate_dpr = list(feature_rate_spr)

import sys

labels_map = {"P" : 1, "N" : 0, "U": 2}

columns = ["ip_address", "probing_type", "probing_rate", "loss_rate", "correlation_1", "correlation_2"]

computed_columns = [
                    "c_c_ind_lr",
                    "w_c_ind_lr_0",
                    "w_c_ind_lr_1",
                    "w_c_ind_lr",
                    "c_c_spr_lr",
                    "w_c_spr_lr_0",
                    "w_c_spr_lr_1",
                    "w_c_spr_lr",
                    "cor_c_c_spr",
                    "cor_w_c_spr_0",
                    "cor_w_c_spr_1",
                    "cor_w_c_spr",
                    "c_c_dpr_lr",
                    "w_c_dpr_lr_0",
                    "w_c_dpr_lr_1",
                    "w_c_dpr_lr_min"
                    ]

feature_columns = []

def create_feature_columns(column, rates):
    feature_columns = []
    for rate in rates:
        feature_columns.append(column + "_" + str(rate))

    return feature_columns

# Add rate to the feature that needs it.
for column in computed_columns:
    if "ind" in column:
        feature_columns.extend(create_feature_columns(column, feature_rate_ind))
    if "spr" in column:
        feature_columns.extend(create_feature_columns(column, feature_rate_spr))
    if "dpr" in column:
        feature_columns.extend(create_feature_columns(column, feature_rate_dpr))

feature_columns.extend(["c_c_dpr_lr","w_c_dpr_lr", "w_dpr_lr", "label"])

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

def compute_diff_loss_rate(df, candidates, witnesses):

    candidate_candidate_difference_loss_rate = {}
    witness_candidate_difference_loss_rate = {}
    for i in range(0, len(df)):
        row_i = df.loc[i]
        ip_address_i = row_i["ip_address"]
        # print ip_address_i
        probing_rate_i = row_i["probing_rate"]
        loss_rate_i = row_i["loss_rate"]
        for j in range(i + 1, len(df)):
            row_j = df.loc[j]
            ip_address_j = row_j["ip_address"]
            probing_rate_j = row_j["probing_rate"]
            loss_rate_j = row_j["loss_rate"]

            if probing_rate_i == probing_rate_j:
                if ip_address_i in candidates and ip_address_j in candidates:
                    if candidate_candidate_difference_loss_rate.has_key(probing_rate_i):
                        candidate_candidate_difference_loss_rate[probing_rate_i].append(abs(loss_rate_i - loss_rate_j))
                    else:
                        candidate_candidate_difference_loss_rate[probing_rate_i] = [abs(loss_rate_i - loss_rate_j)]
                else:
                    if witness_candidate_difference_loss_rate.has_key(probing_rate_i):
                        witness_candidate_difference_loss_rate[probing_rate_i].append(abs(loss_rate_i-loss_rate_j))
                    else:
                        witness_candidate_difference_loss_rate[probing_rate_i] = [abs(loss_rate_i - loss_rate_j)]


    return candidate_candidate_difference_loss_rate, witness_candidate_difference_loss_rate

def build_feature_values(c_c_feature_prefix, w_c_feature_prefix, rates, c_c_values, w_c_values):
    new_entry = {}

    for rate in rates:
        if c_c_values.has_key(rate):
            new_entry[c_c_feature_prefix + str(rate)] = c_c_values[rate][0]
        else:
            new_entry[c_c_feature_prefix + str(rate)] = np.nan
        if w_c_values.has_key(rate):
            w_c_values_rate = w_c_values[rate]
            for i in range (0, len(w_c_values_rate)):
                new_entry[w_c_feature_prefix + str(i) + "_" + str(rate)] = w_c_values_rate[i]
            if len(w_c_values_rate) < 2:
                new_entry[w_c_feature_prefix + str(1) + "_" + str(rate)] = np.nan
                new_entry[w_c_feature_prefix + str(rate)] = np.nan
            else:
                new_entry[w_c_feature_prefix + str(rate)] = min(w_c_values_rate[k] for k in range(0, len(w_c_values_rate)))
        else:
            new_entry[w_c_feature_prefix + "0_" + str(rate)] = np.nan
            new_entry[w_c_feature_prefix + "1_" + str(rate)] = np.nan
            new_entry[w_c_feature_prefix + str(rate)] = np.nan

    return new_entry

if __name__ == "__main__":

    version  = sys.argv[1]

    if version == "4":
        results_dir = "/srv/icmp-rl-survey/results/v4/"
    elif version == "6":
        results_dir = "/srv/icmp-rl-survey/results/v6/"
    # Load dataset

    computed_df = pd.DataFrame(columns=feature_columns)

    midar_path = "resources/internet2/midar/v4/"
    midar_routers = compute_midar_routers(midar_path)
    ground_truth_routers = transitive_closure(midar_routers)
    # Algorithm options
    w_dpr_treshold = 0.3


    i  = 0
    for result_file in os.listdir(results_dir):

        new_entry = {}

        i += 1
        print i
        if i == 1:
            break
        # print result_file
        split_file_name = result_file.split("_")
        candidates = [split_file_name[1], split_file_name[2]]
        witnesses = [split_file_name[3]]

        label = get_label(ground_truth_routers, candidates)

        if label == "U":
            continue
        elif label == "P":
            new_entry["label"] = 1
        elif label == "N":
            new_entry["label"] = 0
        # 1 point is represented by different dimensions:
        df_result = pd.read_csv(results_dir+result_file, names = columns,  skipinitialspace=True,)



        # Remove anomalies of measurement
        df_group_dpr_witness = df_result[df_result["ip_address"].isin(witnesses)]
        df_group_dpr_witness = df_group_dpr_witness[df_group_dpr_witness["probing_type"] == "GROUPDPR"]
        if len(df_group_dpr_witness) == 0:
            continue
        max_lr = df_group_dpr_witness.loc[df_group_dpr_witness["loss_rate"].idxmax()]

        max_witness_lr = max_lr["probing_rate"]
        new_entry["w_dpr_lr"] = max_lr["loss_rate"]

        if new_entry["w_dpr_lr"] > w_dpr_treshold:
            new_entry["label"] = 2

        ################################

        c_c_spr_cor = {}
        w_c_spr_cor = {}


        # Parse correlation
        for row in df_result.itertuples():
            c_c_spr_cor_row, w_c_spr_cor_row = parse_correlation(df_result, row, ["correlation_1", "correlation_2"], candidates, witnesses)
            c_c_spr_cor.update(c_c_spr_cor_row)
            w_c_spr_cor.update(w_c_spr_cor_row)



        # Parse loss rate
        df_individual = df_result[df_result["probing_type"] == "INDIVIDUAL"]
        df_individual.reset_index(drop = True, inplace=True)
        c_c_ind_loss_rate, w_c_ind_loss_rate = compute_diff_loss_rate(df_individual, candidates, witnesses)

        df_group_spr = df_result[df_result["probing_type"] == "GROUPSPR"]
        df_group_spr.reset_index(drop = True, inplace=True)
        c_c_spr_loss_rate, w_c_spr_loss_rate = compute_diff_loss_rate(df_group_spr, candidates, witnesses)

        df_group_dpr = df_result[df_result["probing_type"] == "GROUPDPR"]
        df_group_dpr.reset_index(drop=True, inplace=True)



        c_c_dpr_loss_rate, w_c_dpr_loss_rate = compute_diff_loss_rate(df_group_dpr, candidates, witnesses)


        new_entry.update(
            build_feature_values("c_c_ind_lr_", "w_c_ind_lr_", feature_rate_ind, c_c_ind_loss_rate, w_c_ind_loss_rate))
        new_entry.update(
            build_feature_values("c_c_spr_lr_", "w_c_spr_lr_", feature_rate_spr, c_c_spr_loss_rate, w_c_spr_loss_rate))

        new_entry.update(
            build_feature_values("c_c_spr_cor_", "w_c_spr_cor_", feature_rate_spr, c_c_spr_cor, w_c_spr_cor))

        new_entry.update(
            build_feature_values("c_c_dpr_lr_", "w_c_dpr_lr_", feature_rate_dpr, c_c_dpr_loss_rate, w_c_dpr_loss_rate))

        new_entry["c_c_dpr_lr"] = new_entry["c_c_dpr_lr_" + str(max_witness_lr)]
        new_entry["w_c_dpr_lr"] = new_entry["w_c_dpr_lr_" + str(max_witness_lr)]
        computed_df.loc[result_file] = new_entry



    # print computed_df.to_string()
    # Build labels (Yes, No, Unknown)

    feature_columns = ["c_c_dpr_lr", "w_c_dpr_lr", "w_dpr_lr"]
    label = "label"
    # computed_df.to_csv("resources/test_set", encoding='utf-8')
    computed_df = pd.read_csv("resources/test_set", index_col=0)

    # labeled_df, cluster_n = DBSCAN_impl(computed_df, feature_columns)
    cluster_n = 3
    labeled_df = computed_df
    labeled_df[label] = labeled_df[label].apply(np.int64)
    true_negatives_df = labeled_df[labeled_df[label] == labels_map["N"]]
    true_positives_df = labeled_df[labeled_df[label] == labels_map["P"]]
    true_unknown_df = labeled_df[labeled_df[label] == labels_map["U"]]
    print labeled_df[feature_columns].to_string()
    print "Number of true negatives: " + str(len(true_negatives_df))
    print "Number of true positives: " + str(len(true_positives_df))
    print "Number of unknown: " + str(len(true_unknown_df))
    labeled_df = labeled_df.dropna(subset=feature_columns)




    # Now train a classifier on the labeled data.

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



    # classifier = nn.train_nn_classification_model(
    #     periods = 10,
    #     classes_n = cluster_n,
    #     feature_columns = feature_columns,
    #     learning_rate=0.05,
    #     steps=1000,
    #     batch_size=30,
    #     hidden_units=[10, 10],
    #     training_examples=training_examples,
    #     training_targets=training_targets,
    #     validation_examples=validation_examples,
    #     validation_targets=validation_targets)
    #
    # predict_test_input_fn = nn.create_predict_input_fn(
    #     test_examples, test_targets, batch_size=100)
    #
    # test_predictions = classifier.predict(input_fn=predict_test_input_fn)
    # test_predictions = np.array([item['class_ids'][0] for item in test_predictions])
    #
    # accuracy = metrics.accuracy_score(test_targets, test_predictions)
    # print("Accuracy on test data: %0.3f" % accuracy)


    # training_examples = tf.convert_to_tensor(training_examples.values, dtype=tf.float64)
    # training_targets = tf.convert_to_tensor(training_targets.values, dtype=tf.float64)
    #
    # validation_examples = tf.convert_to_tensor(validation_examples.values, dtype=tf.float64)
    # validation_targets = tf.convert_to_tensor(validation_targets.values, dtype=tf.float64)
    # test_examples = tf.convert_to_tensor(test_examples.values, dtype=tf.float64)
    # test_targets = tf.convert_to_tensor(test_targets.values, dtype=tf.float64)
    results, _ = adanet_.train(periods=10,
                               training_examples=training_examples,
                               training_targets =training_targets,
                               validation_examples = validation_examples,
                               validation_targets = validation_targets,
                               labels_n = 3,
                               learning_rate=0.001,
                               steps=1000,
                               batch_size = 30
                                            )
    print("Loss:", results["average_loss"])
    print("Architecture:", adanet_.ensemble_architecture(results))
    print results

    # #@test {"skip": true}
    results, _ = adanet_.train(periods=10,
                               training_examples=training_examples,
                               training_targets=training_targets,
                               validation_examples=validation_examples,
                               validation_targets=validation_targets,
                               labels_n=3,
                               learning_rate=0.001,
                               steps=1000,
                               batch_size=30,
                               learn_mixture_weights=True
                               )
    print("Loss:", results["average_loss"])
    print("Uniform average loss:", results["average_loss/adanet/uniform_average_ensemble"])
    print("Architecture:", adanet_.ensemble_architecture(results))
    print results
    #
    #
    # #@test {"skip": true}
    results, _ = adanet_.train(periods=10,
                               training_examples=training_examples,
                               training_targets=training_targets,
                               validation_examples=validation_examples,
                               validation_targets=validation_targets,
                               labels_n=3,
                               learning_rate=0.001,
                               steps=1000,
                               batch_size=30,
                               learn_mixture_weights=True,
                               adanet_lambda=0.015
                               )
    print("Loss:", results["average_loss"])
    print("Uniform average loss: ", results["average_loss/adanet/uniform_average_ensemble"])
    print("Architecture:", adanet_.ensemble_architecture(results))
    print results



