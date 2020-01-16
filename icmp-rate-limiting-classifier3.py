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
# import tensorflow as tf
from sklearn.metrics import f1_score
from sklearn import clone
# from Cluster.dbscan import kmeans, DBSCAN_impl
from Algorithms.algorithms import transitive_closure
# from Classification import neural_network as nn
# from Classification.neural_network import print_bad_labels
# from Classification import adanet_wrap as adanet_
from Classification.random_forest import *
from Classification.mlp import *
from Classification.knn import *
from Classification.svm import *
from Classification.metrics import compute_metrics
from Data.preprocess import parse_labels_and_features
from Validation.midar import extract_routers, set_router_labels, extract_routers_by_node, internet2_routers
from Validation.evaluation import evaluate
from joblib import load, dump
# tf.logging.set_verbosity(tf.logging.ERROR)
pd.options.display.max_rows = 25
pd.options.display.float_format = '{:.3f}'.format
#################################################

import json
from Data.preprocess import *
from Algorithms.algorithms import rotate
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
    print ("Number of non aliases: " + str(len(true_negatives_df)))
    # print "Number of false negatives: " + str(FN)
    print ("Number of aliases: " + str(len(true_positives_df)))


debug = [
]




def compute_dataset(version,
                    is_pairwise,
                    ground_truth_routers,
                    alpha, n_not_triggered_threshold,
                    candidates_witness_dir,
                    results_dir,
                    ofile,
                    is_detect_per_interface):

    max_interfaces = 200

    pd.options.display.float_format = '{:.5f}'.format

    skip_fields = ["Index", "probing_type", "ip_address", "correlation"]
                   # "transition_0_0", "transition_0_1", "transition_1_0", "transition_1_1"]

    df_computed_result = None

    probing_type_suffix = {"INDIVIDUAL": "ind", "GROUPSPR": "spr", "GROUPDPR": "dpr"}

    # raw_columns are the extracted data from CSV
    global_raw_columns = ["ip_address", "probing_type", "probing_rate", "changing_behaviour", "loss_rate",
                          "transition_0_0", "transition_0_1", "transition_1_0", "transition_1_1"]

    # missing_fields = build_missing_candidates(0, max_interfaces, 0, 1, global_raw_columns, skip_fields,
    #                                           probing_type_suffix, probing_type_rates)


    router_names = set()

    distinct_ips = set()
    distinct_routers = set()

    n_non_alias = 0
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
    transition_probabilities_distribution = {}
    correlation_distributions = {
        "correlation_spr": []
        , "correlation_dpr": []
    }

    weak_correlations = []
    strong_correlations = []
    correlations_by_ip = {}
    for result_file in os.listdir(results_dir):

        # Only results file
        if os.path.isdir(results_dir + result_file):
            continue

        #################### DISTINCT ROUTERS ###########

        #################### DEBUG INFOS ################
        total += 1
        print (total, result_file)

        if not recompute_dataset:
            if total == 1:
                break
        # if total < 260:
        #     continue
        #################################################

        split_file_name = result_file.split("_")
        if version == "4":
            node = split_file_name[0]
        elif version == "6":
            node = "router"


        router_name = split_file_name[-1]

        ip_index = 5
        candidates = []
        witnesses = []
        if "TN" in result_file:
            if version == "4":
                candidates_witness_dir = "/srv/icmp-rl-survey/midar/survey/batch2/lr0.05-0.10/candidates-witness-tn/"
            elif version == "6":
                candidates_witness_dir = "/srv/icmp-rl-survey/speedtrap/survey/lr0.45-0.50/candidates-witness-tn/"
        elif "router" in result_file:
            if version == "4":
                candidates_witness_dir = "/srv/icmp-rl-survey/midar/survey/batch2/lr0.05-0.10/candidates-witness/"
            elif version == "6":
                candidates_witness_dir = "/srv/icmp-rl-survey/speedtrap/survey/lr0.45-0.50/candidates-witness/"
        else:
            candidates_witness_dir = candidates_witness_dir
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

        # Limit the number of TN to 1000
        if "TN" in result_file:
            if n_non_alias == 1000:
                continue
            n_non_alias += 1

        ###########################################################

        ######################### DEBUG ###########################
        # if candidates != ["27.68.229.146", "27.68.229.218"]:
        #     continue
        #
        # if result_file != "planetlab-1.research.netlab.hut.fi_hous":
        #     continue
        ###########################################################

        raw_columns = copy.deepcopy(global_raw_columns)

        # Add correlation columns
        for i in range(1, len(candidates)):
            raw_columns.append("correlation_c" + str(i))

        for i in range(0, len(witnesses)):
            raw_columns.append("correlation_w" + str(i))

        multiple_interfaces_router[len(candidates)] += 1
        # print "Routers with more than 2 interfaces: " + str(multiple_interfaces_router)
        # 1 point is represented by different dimensions:
        df_raw = pd.read_csv(results_dir + result_file,
                                names=raw_columns,
                                skipinitialspace=True,
                                # encoding="utf-8",
                                engine="python",
                                index_col=False)
        # usecols=[x for x in range(0, 9)])

        if len(df_raw) < 9:
            print("Bad measurement file not enough rows " + result_file)
            continue
        group_spr_probing_rate = df_raw[df_raw["probing_type"] == "GROUPSPR"]["probing_rate"].iloc[0]
        group_dpr_probing_rate = df_raw[df_raw["probing_type"] == "GROUPDPR"]["probing_rate"].iloc[0]
        correlations = parse_correlation(df_raw, [group_dpr_probing_rate], candidates, witnesses)
        ##################################### Split the df in pairwise elements #############################

        df_list_pairwise = []
        pairwise_candidates_list = []
        for i in range(1, len(candidates)):

            df_pairwise = df_raw[df_raw["ip_address"].isin([candidates[0], candidates[i], witnesses[0]])]
            df_list_pairwise.append(df_pairwise)
            pairwise_candidates_list.append([candidates[0], candidates[i]])

        for i in range(0, len(df_list_pairwise)):
            new_entry = {}

            pairwise_candidates = pairwise_candidates_list[i]
            df_result = df_list_pairwise[i]

            for k in range(0, len(pairwise_candidates)):
                new_entry["ip_address_c"+str(k)] = pairwise_candidates[k]

            # Skip the measurement if all the loss rates are 1.
            not_usable = False
            for candidate in pairwise_candidates:
                df_not_usable = df_result[(df_result["loss_rate"] == 1) &
                               (df_result["ip_address"] == candidate) &
                               (df_result["probing_type"] == "GROUPDPR")]
                if len(df_not_usable) > 0:
                    not_usable |= df_not_usable["loss_rate"].iloc[0] == 1
            if not_usable:
                continue


            df_individual = df_result[df_result["probing_type"] == "INDIVIDUAL"]


            ind_probing_rate = df_result[df_result["probing_type"] == "INDIVIDUAL"]["probing_rate"]

            probing_type_rates = {"INDIVIDUAL": list(ind_probing_rate),
                                  "GROUPSPR": [group_spr_probing_rate],
                                  "GROUPDPR": [group_dpr_probing_rate]}

            new_row = build_new_row(df_result, pairwise_candidates, witnesses,
                                    skip_fields,
                                    probing_type_suffix,
                                    probing_type_rates,
                                    is_lr_classifier=True)

            correlation_row = get_pairwise_correlation_row(correlations, pairwise_candidates, witnesses[:1])
            if correlation_row is None:
                continue
            new_row.update(correlation_row)

            label = set_router_labels(new_entry, ground_truth_routers[node], pairwise_candidates, witnesses[:1])

            if label == "U":
                continue

            new_entry["measurement_id"] = result_file
            new_entry.update(new_row)





            for i in range(0, len(pairwise_candidates)):
                if new_entry["label_c0"] != new_entry["label_c" + str(i)]:
                    TN += 1
                    print ("TN: " + str(TN))

            '''
            Excluded from the classifier:
            If the target is unresponsive, skip the data.
            
            '''


            # In a non alias example, if the loss rate for minimum rate is too high, discard it.
            # not_aliases_lr_min_rate_too_high = False
            # for i in range(1, len(pairwise_candidates)):
            #     if new_entry["label_c" + str(i)] == 0:
            #         df_candidate_not_alias_min_rate = df_individual[(df_individual["ip_address"] == pairwise_candidates[i])
            #                                                         & (
            #                                                         df_individual["probing_rate"] == minimum_probing_rate)]
            #         if len(df_candidate_not_alias_min_rate) > 1:
            #             # The triggering rate was the minimum probing rate.
            #             continue
            #         if (df_candidate_not_alias_min_rate["loss_rate"] > 0.10).all():
            #             not_aliases_lr_min_rate_too_high = True
            #             break
            # if not_aliases_lr_min_rate_too_high:
            #     n_not_aliases_lr_too_high += 1
            #     continue

            # RL Not triggered
            df_dpr = df_result[(df_result["probing_type"].isin(["GROUPDPR"]))]
            #
            # df_not_triggered = df_dpr[df_dpr["ip_address"] == pairwise_candidates[0]]
            # not_triggered = (df_not_triggered["loss_rate"] < n_not_triggered_threshold).all()
            # if not_triggered:
            #     n_not_triggered += 1
            #     continue
            # RL not shared but aliases
            if is_detect_per_interface:
                alias_but_not_shared = False
                epsilon = 0
                for i in range(0, len(pairwise_candidates)):
                    df_not_shared = df_dpr[df_dpr["ip_address"] == pairwise_candidates[i]]
                    # Exclude not shared counter from classification:
                    not_shared = (df_not_shared["loss_rate"] <= epsilon).all()
                    if not_shared and new_entry["label_c" + str(i)] == 1:
                        alias_but_not_shared = True
                        break
                if alias_but_not_shared:
                    n_not_shared += 1
                    continue
            #
            # # Check if the loss rate of the witness is too high.
            # df_witness_lr = df_dpr[df_dpr["ip_address"] == witnesses[0]]["loss_rate"]
            # minimum_lr = min(df_dpr["loss_rate"])

            # if df_witness_lr.iloc[0] > alpha:
            #     n_witness_too_high += 1
            #     new_entry["label_c1"] = 0
            #     print ("Witness loss rate too high \n")

            # Check different patterns
            # correlation_patterns["correlation_spr"].append(correlation_row["correlation_spr_c1"])
            # if float(correlation_row["correlation_c1"]) > 0.99 and new_entry["label_c1"] == 1:
            #     strong_correlations.append(result_file)
            #
            # if float(correlation_row["correlation_c1"]) < 0.3 and new_entry["label_c1"] == 1:
            #     weak_correlations.append(result_file)
                # print "Not correlated but alias and triggered: " + result_file
            if not is_detect_per_interface:
                if new_entry["label_c1"] == 1:
                    key = ""
                    for i in range(0, len(pairwise_candidates)):
                        key += pairwise_candidates[i]
                        if i != len(pairwise_candidates) - 1:
                            key += "_"
                    if "correlation_c1" in correlation_row  \
                        and "transition_0_0_dpr_c1" in new_entry \
                        and "transition_0_1_dpr_c1" in new_entry \
                        and "transition_1_0_dpr_c1" in new_entry \
                        and "transition_1_1_dpr_c1" in new_entry:
                        correlations_by_ip[key] = {
                        "correlation":correlation_row["correlation_c1"],
                        "tm_0_0": new_entry["transition_0_0_dpr_c1"],
                        "tm_0_1": new_entry["transition_0_1_dpr_c1"],
                        "tm_1_0": new_entry["transition_1_0_dpr_c1"],
                        "tm_1_1": new_entry["transition_1_1_dpr_c1"]
                    }



            # Map the array of labels to a single label
            # 3 possibilities
            if new_entry["label_c0"] == 0 and new_entry["label_c1"] == 0:
                new_entry["label_pairwise"] = 2
            if new_entry["label_c0"] == 1 and new_entry["label_c1"] == 0:
                TN_pairwise += 1
                print ("TN Pairwise: " + str(TN_pairwise))
                new_entry["label_pairwise"] = 0
            if new_entry["label_c0"] == 1 and new_entry["label_c1"] == 1:
                new_entry["label_pairwise"] = 1

            if df_computed_result is None:
                df_computed_result = pd.DataFrame(columns=new_entry.keys())
                df_computed_result.set_index(df_computed_result["measurement_id"])
            if len(new_entry) != df_computed_result.shape[1]:
                print ("Bad measurement file " + result_file)
                continue
            df_computed_result.loc[len(df_computed_result)] = new_entry
            router_names.add(router_name)


    if recompute_dataset and not is_detect_per_interface:
        with open("resources/transition_probabilities_distributions.json", "w") as correlation_distributions_fp:
            json.dump(correlation_distributions, correlation_distributions_fp)
    #     with open("resources/weak_correlations.json", "w") as correlations_fp:
    #         json.dump(weak_correlations, correlations_fp)
    #     with open("resources/strong_correlations.json", "w") as correlations_fp:
    #         json.dump(strong_correlations, correlations_fp)
    #
        with open("resources/correlations_by_ip.json", "w") as correlations_fp:
            json.dump(correlations_by_ip, correlations_fp)
    df_computed_result.set_index("measurement_id", inplace=True)
    df_computed_result.to_csv(ofile, encoding="utf-8")


    ############################### PRINT METRICS ##########################
    is_pairwise = True
    labeled_df = df_computed_result
    if is_pairwise:

        label = "label_pairwise"

        non_alias_df = labeled_df[labeled_df[label] == 0]
        alias_df = labeled_df[labeled_df[label] == 1]
        unknown_df = labeled_df[labeled_df[label] == 2]
        # true_unknown_df = labeled_df[labeled_df[label] == labels_map["U"]]
        print ("Number of alias: " + str(len(alias_df)))
        print ("Number of alias but not shared: " + str(n_not_shared))
        print ("Number of non alias: " + str(len(non_alias_df)))
        print ("Number of unknown: " + str(len(unknown_df)))
        print ("Number of not triggered: " + str(n_not_triggered))
        print ("Number of unusable witness: " + str(n_unusable_witness))
        print ("Number of witness loss rate too high: " + str(n_witness_too_high))
        print ("Number of candidate loss rate too high on minimum rate: " + str(n_not_aliases_lr_too_high))
        print ("Number of distinct ips: " + str(len(distinct_ips)))
        print ("Number of distinct routers: " + str(len(distinct_routers)))

        print ("Number of routers probed: " +str(len(router_names)))

        # print ("Missing routers: " + str(set(ground_truth_routers["ple41.planet-lab.eu"].keys()) - router_names))



    return df_computed_result



def compute_classifier(feature_columns,
                       training_examples,
                       training_targets,
                       validation_examples,
                       validation_targets,
                       test_examples,
                       test_targets,
                       use_dnn, use_random_forest, use_mlp, use_knn, use_svm):
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
        print ("Precision: " + str(precision))
        print ("Recall: " + str(recall))
        print ("Accuracy: " + str(accuracy))

    ################# RANDOM FOREST ##################

    if use_random_forest:
        classifier, threshold_decision = random_forest_classifier(training_examples,
                                                     training_targets,
                                                     )

        # validations_predictions = classifier.predict_proba(validation_examples)
        importances = feature_importance(classifier, training_examples.columns)
        print (importances.to_string())
        # print_false_positives(validations_predictions, validation_targets, validation_examples, labeled_df)
        return classifier, threshold_decision

    if use_mlp:
        classifier, threshold_decision = mlp_classifier(training_examples, training_targets)
        return classifier, threshold_decision

    if use_knn:
        classifier, threshold_decision = knn_classifier(training_examples, training_targets)
        return classifier, threshold_decision
    if use_svm:
        classifier, threshold_decision = svm_classifier(training_examples, training_targets)
        return classifier, threshold_decision

    return None

if __name__ == "__main__":
    # ple_nodes = []
    # with open("/home/kevin/mda-lite-v6-survey/resources/nodes/v4_nodes") as nodes_f:
    #     for node in nodes_f:
    #         ple_nodes.append(node.strip("\n"))


    '''
        Parameters of the classifiers
        target loss rate window:
        epsilon: the lower bound of the "aliases with per-interface rate limiting".
        alpha: the lower bound of the "witness" label.

    '''


    target_loss_rate_window = "lr0.05-0.10"
    alpha = 0.05
    n_not_triggered_threshold = 0.05
    is_detect_per_interface = True


    '''
        Different paths of data
    '''

    '''
    IPv4
    '''
    version = "4"
    measurement_prefix = "/srv/icmp-rl-survey/midar/survey/batch2/" + target_loss_rate_window + "/"
    candidates_witness_dir = ""
    results_dir = measurement_prefix + "results/"
    routers_path = "/home/kevin/mda-lite-v6-survey/resources/midar/batch2/routers/"
    df_file = "resources/test_set_" + target_loss_rate_window
    # if is_detect_per_interface:
    #     df_file = "resources/test_set_without_per_interface_strict" + target_loss_rate_window
    # ground_truth_routers = extract_routers_by_node(routers_path)
    ground_truth_routers = []

    '''
    IPv6
    '''
    # version = "6"
    # measurement_prefix = "/srv/icmp-rl-survey/speedtrap/survey/" + target_loss_rate_window + "/"
    # candidates_witness_dir = ""
    # results_dir = measurement_prefix + "results/"
    # routers_path = "/home/kevin/mda-lite-v6-survey/resources/speedtrap/routers/"
    # df_file = "resources/test_set6_" + target_loss_rate_window
    # if is_detect_per_interface:
    #     df_file = "resources/test_set6_without_per_interface_strict" + target_loss_rate_window
    # ground_truth_routers = extract_routers_by_node(routers_path)

    '''
    Internet2
    '''
    ground_truth_routers = []

    # measurement_prefix = "/srv/icmp-rl-survey/midar/survey/internet2/"
    # candidates_witness_dir = measurement_prefix +"candidates-witness/"
    # results_dir = measurement_prefix + "results/"
    # routers_path = "/home/kevin/mda-lite-v6-survey/resources/internet2/routers/v4/"
    # df_file = "resources/test_set_internet2"
    # ground_truth_routers = internet2_routers(routers_path, ple_nodes)

    recompute_dataset = False
    is_pairwise = True
    if recompute_dataset:
        df_computed_result = compute_dataset(version,
                                             is_pairwise,
                                             ground_truth_routers=ground_truth_routers,
                                             alpha=alpha,
                                             n_not_triggered_threshold=n_not_triggered_threshold,
                                             candidates_witness_dir=candidates_witness_dir,
                                             results_dir=results_dir,
                                             ofile= df_file,
                                             is_detect_per_interface = is_detect_per_interface)
    # label = "label"
    else:
        df_computed_result = pd.read_csv(df_file, index_col=0)
        # df_computed_result["correlation_c0"].apply(pd.to_numeric)
        df_computed_result["correlation_c1"].apply(pd.to_numeric)
    labeled_df, feature_columns, labels_column = build_labeled_df(is_pairwise, df_computed_result)
    use_saved_classifier = False
    ################################# Use existing classifier##########################
    if use_saved_classifier:
        # features_data = pd.read_csv("resources/features_data.csv", index_col=0)
        classifier_file_name = "resources/random_forest_classifier.joblib"
        classifier = load(classifier_file_name)
        # probabilities = classifier.predict_proba(features_data)
        importances = feature_importance(classifier, feature_columns)
        print (importances.to_string())
        evaluate(classifier, ground_truth_routers["ple41.planet-lab.eu"], labeled_df, labels_column, feature_columns)

    ############################# Train and evaluate the classifier ##########################
    precisions = []
    recalls = []
    accuracys = []
    f_scores = []
    f_scores_99 = []
    train_classifier = not use_saved_classifier
    if train_classifier:
        use_random_forest = True
        use_dnn = False
        use_mlp = False
        use_knn = False
        use_svm = False
        save_results = False
        save_classifier = True
        best_classifier = None


        # 10-fold validation
        k_fold = 10
        labeled_df = labeled_df.reindex(np.random.permutation(labeled_df.index))
        chunks = np.array_split(labeled_df, k_fold)

        for i in range(0, 10):
            i_rotation = rotate(chunks, i)
            # Get 90% of the dataset as training set.
            training_df = i_rotation[0]
            for k in range(0, k_fold - 1):
                training_df = pd.concat([training_df, i_rotation[k]])

            # labeled_df = labeled_df.reindex(np.random.permutation(labeled_df.index))
            # labeled_df.to_csv("resources/labeled_test_set", encoding='utf-8')
            # Split the training validation and test set
            # training_n  = int(0.9 * len(labeled_df))
            # training_df = labeled_df.iloc[0:training_n]

            # cross_validation_n = int(0 * len(labeled_df))
            # cross_validation_df = labeled_df.iloc[training_n + 1:cross_validation_n + training_n]

            # test_df = labeled_df.iloc[cross_validation_n + training_n + 1:]

            cross_validation_df = chunks[-1]

            training_targets, training_examples = parse_labels_and_features(training_df, labels_column, feature_columns)
            print ("Size of the training examples: " + str(training_examples.shape))
            print_df_labels(training_df)

            validation_targets, validation_examples = parse_labels_and_features(cross_validation_df, labels_column, feature_columns)
            print ("Size of the validation examples: " + str(validation_examples.shape))
            print_df_labels(cross_validation_df)

            # test_targets, test_examples = parse_labels_and_features(test_df, labels_column, feature_columns)
            # print ("Size of the test examples: " + str(test_examples.shape))
            # print_df_labels(test_df)


            classifier, threshold_decision = compute_classifier(feature_columns,
                                            training_examples, training_targets,
                                            validation_examples, validation_targets,
                                            None, None,
                                            use_dnn = use_dnn,
                                            use_random_forest = use_random_forest,
                                            use_mlp = use_mlp,
                                            use_knn = use_knn,
                                            use_svm = use_svm)


            if use_random_forest or use_mlp or use_knn or use_svm:
                # examples = pd.concat([validation_examples, None])
                # targets = pd.concat([validation_targets, None])
                examples = validation_examples
                targets = validation_targets
                probabilities = classifier.predict_proba(examples)
                predictions = []
                for probability in probabilities:
                    if probability[1] < threshold_decision:
                        predictions.append(0)
                    else:
                        predictions.append(1)


                precision, recall, accuracy, f_score = compute_metrics(predictions, targets)
                print ("Precision: " + str(precision))
                print ("Recall: " + str(recall))
                print ("Accuracy: " + str(accuracy))
                print("F score: "+ str(f_score))
                precisions.append(precision)
                recalls.append(recall)
                accuracys.append(accuracy)
                f_scores.append(f_score)

                if use_random_forest:
                    importances = feature_importance(classifier, feature_columns)
                    if precision > 0.995:
                        f_scores_99.append(f_score)
                    if len(f_scores_99) > 0:
                        if precision > 0.995 and f_score >= max(f_scores_99):
                            print ("New Best classifier")
                            best_classifier = classifier
                            print(best_classifier)


        if save_classifier:
            if use_random_forest:
                classifier_file_name = 'resources/random_forest_classifier' + version +'.joblib'
                dump(best_classifier, classifier_file_name)
                print("Saved classifier into " + classifier_file_name)
            elif use_dnn:
                pass

        if use_knn:
            classifier_type = "knn"
        elif use_random_forest:
            classifier_type = "random_forest"
        elif use_mlp:
            classifier_type = "mlp"
        elif use_svm:
            classifier_type = "svm"
        print(classifier_type)
        print("Mean Precision: " + str(np.mean(precisions)))
        print("Mean Recall: " + str(np.mean(recalls)))
        print("Mean F-Score: " + str(np.mean(f_scores)))
        latex_table = str(np.mean(precisions)) + " & " + str(np.mean(recalls)) + " & " + str(np.mean(f_scores)) + "\\\\"
        print(latex_table)

        if save_results:

            with open("/home/kevin/icmp-rate-limiting-paper/resources/results/results_classifier_"+ target_loss_rate_window + ".json", "w") as results_classifier_fp:
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



