import sys
import argparse

import json

from Classification.classifier_options import ClassifierOptions
from Cpp.cpp_options import CppOptions
from Cpp.launcher import *
from Data.preprocess import build_classifier_entry_from_csv, parse_labels_and_features, extract_individual
from joblib import load
from Cpp.cpp_files import *
from Algorithms.algorithms import rotate, transitive_closure

def execute_impl(ip_version,
            candidates,
            icmp_install_dir,
            cpp_binary_cmd,
            cpp_options,
            classifier,
            classifier_options,
            iteration):
    # Launch a traceroute to the first candidate to find a witness.
    # If not responsive, pass to the second, etc...
    # if iteration == 1:
    #     candidates = rotate(candidates, candidates.index("198.71.46.178"))

    print("Starting finding witness phase...")
    high_rate_candidate, ip_witness = find_witness(ip_version, candidates)
    if high_rate_candidate is None:
        print("No responsive candidate found. Exiting...")
        exit(0)
    print("Witness is: " + ip_witness)
    print("End of the traceroute phase")
    # ip_witness = "162.252.70.124"
    # Reorder the targets file to put the responsive candidate in the first position.
    candidates = rotate(candidates, candidates.index(high_rate_candidate))

    # Build the cpp candidates files for the CPP tool.

    cpp_candidates_file = icmp_install_dir + "resources/cpp_targets_file_" + str(iteration)
    print("Writing the candidates for the cpp tool in: " + str(cpp_candidates_file) + "...", end="", flush=True)
    write_cpp_candidates_file(ip_version, candidates, ip_witness, cpp_candidates_file)
    print(" done")

    cpp_witness_file = icmp_install_dir + "resources/cpp_witness_file_" + str(iteration)
    print("Writing the witness for the cpp tool in: " + str(cpp_witness_file) + "...", end="", flush=True)
    write_cpp_witness_file(ip_version, ip_witness, cpp_witness_file)
    print(" done")

    # Launch rate limiting algorithm for witness

    # Check if the witness is already present in the pcap individual files.
    cpp_output_file_witness = cpp_options.output_file + "_witness_" + str(iteration)
    if is_witness_already_probed(ip_witness, cpp_options.pcap_dir_individual):
        print("Witness already present in individual files, skipping witness rate limiting algorithm...")
        only_analyse = True
    else:
        print("Executing rate limiting algorithm for witness... ", end="", flush=True)
        only_analyse = False

    output_w, err_w = execute_icmp_rate_limiting_command(cpp_binary_cmd,
                                                         cpp_witness_file,
                                                         cpp_options,
                                                         cpp_output_file_witness,
                                                         is_witness=True,
                                                         only_analyse=only_analyse)
    print(" done")

    # Launch rate limiting algorithm for candidates
    if is_group_already_probed(high_rate_candidate, cpp_options.pcap_dir_groups):
        print("Group already present in individual files, skipping candidates rate limiting algorithm...")
        only_analyse = True
    else:
        print("Executing rate limiting algorithm for candidates... ", end="", flush=True)
        only_analyse = False

    cpp_output_file = cpp_options.output_file + "_" + str(iteration)
    output, err = execute_icmp_rate_limiting_command(cpp_binary_cmd,
                                                     cpp_candidates_file,
                                                     cpp_options,
                                                     cpp_output_file,
                                                     is_witness=False,
                                                     only_analyse=False)
    print(" done")

    cpp_output_file_individual = icmp_install_dir + "internet2_individual"
    df_individual = extract_individual(cpp_output_file_individual, classifier_options.global_raw_columns)
    df_witness_individual = extract_individual(cpp_output_file_witness, classifier_options.global_raw_columns)

    # Transforms the output into features
    print("Processing results...", end="", flush=True)
    unresponsive_candidates, labeled_df, features_columns, labels_column = build_classifier_entry_from_csv(
        cpp_candidates_file,
        classifier_options.global_raw_columns,
        classifier_options.skip_fields,
        classifier_options.probing_type_suffix,
        cpp_output_file,
        df_individual,
        df_witness_individual)

    labels, features_data = parse_labels_and_features(labeled_df, labels_column, features_columns)
    # import pandas as pd
    # features_data = pd.read_csv("resources/features_data.csv", index_col=0)
    # features_data = features_data.reindex(sorted(features_data.columns), axis=1)
    print(" done")
    # Classify
    print("Predicting probabilities...", end="", flush=True)
    probabilities = classifier.predict_proba(features_data)

    predictions = []

    for i in range(0, len(probabilities)):
        if probabilities[i][1] > 0.6:
            predictions.append(1)
        else:
            predictions.append(0)
    print(" done")

    print("Computing remaining set of candidates...", end="", flush=True)
    aliases = []
    to_remove_candidates = set()
    for unresponsive_candidate in unresponsive_candidates:
        to_remove_candidates.add(unresponsive_candidate)

    for i in range(0, len(labeled_df)):
        ip_address0 = labeled_df.iloc[i]["ip_address_c0"]
        ip_address1 = labeled_df.iloc[i]["ip_address_c1"]
        to_remove_candidates.add(ip_address0)
        if predictions[i] == 1:
            aliases.append({ip_address0, ip_address1})
            to_remove_candidates.add(ip_address1)

    remaining_candidates = list(set(candidates) - to_remove_candidates)
    print(" done")
    with open("aliases_" + str(iteration), "w") as alias_fp:
        alias_set = set()
        for pair in aliases:
            for alias in pair:
                alias_set.add(alias)
        for alias in alias_set:
            alias_fp.write(alias + "\n")

    return aliases, remaining_candidates, unresponsive_candidates, high_rate_candidate



def execute(ip_version,
            candidates,
            icmp_install_dir,
            cpp_binary_cmd,
            cpp_options,
            classifier,
            classifier_options,
            iteration):


    aliases, remaining_candidates, new_unresponsive_candidates, high_rate_candidate = execute_impl(ip_version,
            candidates,
            icmp_install_dir,
            cpp_binary_cmd,
            cpp_options,
            classifier,
            classifier_options,
            iteration)

    # Second measurement to ensure no FP
    if len(aliases) > 5:
        aliases_tc = transitive_closure(aliases)
        if len(aliases_tc) == 1:
            aliases_tc = list(aliases_tc[0])
            aliases_tc = rotate(aliases_tc, aliases_tc.index(high_rate_candidate))
            aliases, remaining_candidates_sub, new_unresponsive_candidates_sub, high_rate_candidate = execute_impl(ip_version,
                                                                                              aliases_tc,
                                                                                              icmp_install_dir,
                                                                                              cpp_binary_cmd,
                                                                                              cpp_options,
                                                                                              classifier,
                                                                                              classifier_options,
                                                                                              iteration)

            remaining_candidates.extend(remaining_candidates_sub)
            new_unresponsive_candidates.extend(new_unresponsive_candidates_sub)

        else:
            print("Error, found multiple subsets for iteration " + str(iteration))

    return aliases, remaining_candidates, new_unresponsive_candidates


if __name__ == "__main__":

    '''
        The algorithm is O(n).
    '''

    categorized_candidates = []

    # targets_file = "/home/kevin/mda-lite-v6-survey/resources/internet2/ips"
    targets_file = "resources/internet2/ips"
    # icmp_rl_install_dir = "/root/ICMPRateLimiting/"
    # cpp_binary_cmd = "cd /root/ICMPRateLimiting; " \
    #               "sudo env LD_LIBRARY_PATH=/usr/local/lib64/:$LD_LIBRARY_PATH " \
    #               "./build/ICMPEndToEnd "




    # Command on venus (centos)
    # icmp_rl_install_dir = "/home/kevin/ICMPRateLimiting/"
    # cpp_binary_cmd = "cd " + icmp_rl_install_dir + "; " \
    #                  "sudo env LD_LIBRARY_PATH=/usr/local/lib64/:$LD_LIBRARY_PATH " \
    #                  "./build/ICMPEndToEnd "

    # Command on plenodes (Fedora)
    icmp_rl_install_dir = "/root/ICMPRateLimiting/"
    cpp_binary_cmd = "cd " + icmp_rl_install_dir + "; " \
                                                   "./build/ICMPEndToEnd "
    '''
        Set options
    '''
    ip_version = "4"
    cpp_options = CppOptions()
    # Venus individual dir
    # cpp_options.pcap_dir_individual = "/srv/icmp-rl-survey/internet2/pcap/individual/"
    # cpp_options.pcap_dir_groups = "/srv/icmp-rl-survey/internet2/pcap/groups/"

    # Ple nodes
    node = sys.argv[1]
    cpp_options.pcap_dir_individual = icmp_rl_install_dir + "resources/pcap/individual/"
    cpp_options.pcap_dir_groups = icmp_rl_install_dir + "resources/pcap/groups/"
    cpp_options.pcap_prefix = node

    cpp_options.output_file = icmp_rl_install_dir + "test"
    cpp_options.target_loss_rate_interval = "[0.45,0.50]"
    '''
        Serialiazed classifier
    '''
    classifier_file_name = "resources/random_forest_classifier.joblib"
    classifier = load(filename=classifier_file_name)

    # classifier_file_name = "resources/random_forest_classifier.json"
    # with open(classifier_file_name) as fp:
    #     classifier = json.load(fp)
    #     classifier = jsonpickle.decode(classifier)
    # Build the cpp target_file

    '''
        Global parameters of the classifier
    '''
    classifier_options = ClassifierOptions()

    aliases = []

    candidates = []
    unresponsive_candidates = []
    with open(targets_file) as targets_file_fp:
        for line in targets_file_fp:
            line = line.strip("\n")
            if re.match(ipv4_regex, line) or re.match(ipv6_regex, line):
                if not line in candidates:
                    candidates.append(line)

    iteration = 0
    while len(candidates) > 1:
        print("Performing iteration " + str(iteration) + " on candidates: " + str(len(candidates)))
        aliases_found, remaining_candidates, new_unresponsive_candidates = execute(ip_version,
                                                      candidates,
                                                      icmp_rl_install_dir,
                                                      cpp_binary_cmd,
                                                      cpp_options,
                                                      classifier,
                                                      classifier_options,
                                                      iteration)
        unresponsive_candidates.extend(new_unresponsive_candidates)
        aliases.extend(aliases_found)
        candidates = remaining_candidates
        iteration += 1


    final_aliases = transitive_closure(aliases)

    with open("aliases.json", "w") as fp:
        serializable_aliases = [list(router) for router in final_aliases]
        json.dump(serializable_aliases, fp)
    with open("unresponsive.json", "w") as fp:
        json.dump(unresponsive_candidates, fp)
    print ("Aliases : " + str(final_aliases))
    print ("Unresponsive: " + str(unresponsive_candidates))