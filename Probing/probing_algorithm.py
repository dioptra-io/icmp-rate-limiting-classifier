import json
import pandas as pd
from Cpp.launcher import *
from Data.preprocess import build_classifier_entry_from_csv, parse_labels_and_features, extract_individual
from Cpp.cpp_files import *
from Algorithms.algorithms import rotate, transitive_closure

def execute_impl(ip_version,
                 candidates,
                 icmp_install_dir,
                 cpp_binary_cmd,
                 cpp_options,
                 classifier,
                 classifier_options,
                 iteration,
                 witness_by_candidate_file,
                 cpp_individual_file):
    # Launch a traceroute to the first candidate to find a witness.
    # If not responsive, pass to the second, etc...
    # if iteration == 1:
    #     candidates = rotate(candidates, candidates.index("198.71.46.178"))

    print("Starting finding witness phase...")
    if os.path.isfile(witness_by_candidate_file):
        with open(witness_by_candidate_file) as witness_by_candidate_fp:
            witness_by_candidate = json.load(witness_by_candidate_fp)
    else:
        witness_by_candidate = find_witness(ip_version, candidates)
        with open(witness_by_candidate_file, "w") as witness_by_candidate_fp:
            json.dump(witness_by_candidate, witness_by_candidate_fp)
    high_rate_candidate = None

    unresponsive_candidates = []
    for candidate in candidates:
        if candidate in witness_by_candidate:
            if high_rate_candidate is None:
                high_rate_candidate = candidate

        else:
            unresponsive_candidates.append(candidate)
    for candidate in unresponsive_candidates:
        if candidate in candidates:
            candidates.remove(candidate)
    if high_rate_candidate is None:
        print("No witness for any of the candidates found. Exiting...")
        exit(0)
    print("End of the traceroute phase")
    witnesses = set()
    witnesses_not_candidates = set()
    for candidate in candidates:
        if candidate in witness_by_candidate:
            witness = witness_by_candidate[candidate]
            if witness not in candidates:
                witnesses_not_candidates.add(witness)
                if not is_witness_already_probed(witness, cpp_options.pcap_dir_individual):
                    witnesses.add(witness)

    witnesses = sorted(list(witnesses))
    witnesses_not_candidates = sorted(list(witnesses_not_candidates))

    cpp_witness_file = icmp_install_dir + "resources/cpp_witness_file"
    print("Writing the witness for the cpp tool in: " + str(cpp_witness_file) + "...", end="", flush=True)
    write_cpp_witness_file(ip_version, witnesses_not_candidates, cpp_witness_file)
    print(" done")
    cpp_output_file_witness = icmp_install_dir + "test_witness"

    if not os.path.isfile(cpp_output_file_witness):
        if len(witnesses_not_candidates) > 0:
            if len(witnesses) == 0:
                only_analyse = True
            elif len(witnesses) > 0:
                only_analyse = False
            # Launch rate limiting algorithm for witness
            output_w, err_w = execute_icmp_rate_limiting_command(cpp_binary_cmd,
                                                                 cpp_witness_file,
                                                                 cpp_options,
                                                                 cpp_output_file_witness,
                                                                 is_witness=True,
                                                                 only_analyse=only_analyse)
            print(" done")

    # Reorder the targets file to put the responsive candidate in the first position.
    candidates = rotate(candidates, candidates.index(high_rate_candidate))


    # Build the cpp candidates files for the CPP tool.

    cpp_candidates_file = icmp_install_dir + "resources/cpp_targets_file_" + str(iteration)
    print("Writing the candidates for the cpp tool in: " + str(cpp_candidates_file) + "...", end="", flush=True)
    write_cpp_candidates_file(ip_version, candidates, witnesses_not_candidates,  cpp_candidates_file)
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


    df_individual = extract_individual(cpp_individual_file, classifier_options.global_raw_columns)
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
        df_witness_individual,
        witness_by_candidate)

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
    with open(cpp_options.pcap_prefix + "_aliases_" + str(iteration), "w") as alias_fp:
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
            iteration,
            witness_by_candidate_file,
            cpp_individual_file
            ):
    aliases, remaining_candidates, new_unresponsive_candidates, high_rate_candidate = execute_impl(ip_version,
            candidates,
            icmp_install_dir,
            cpp_binary_cmd,
            cpp_options,
            classifier,
            classifier_options,
            iteration,
            witness_by_candidate_file,
            cpp_individual_file)

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
                                                                                              iteration,
                                                                                              witness_by_candidate_file,
                                                                                              cpp_individual_file)

            remaining_candidates.extend(remaining_candidates_sub)
            new_unresponsive_candidates.extend(new_unresponsive_candidates_sub)

        else:
            print("Error, found multiple subsets for iteration " + str(iteration))

    return aliases, remaining_candidates, new_unresponsive_candidates


def execute_individual(
            ip_version,
            node,
            candidates,
            icmp_install_dir,
            cpp_binary_cmd,
            cpp_options,
            classifier_options,
            cpp_output_individual_file):

    for candidate in candidates:
        cpp_candidates_file = icmp_install_dir + "resources/cpp_targets_file_" + candidate
        print("Writing the candidates file for the cpp individual tool in: " + str(cpp_candidates_file) + "...", end="", flush=True)
        write_cpp_candidates_file(ip_version, [candidate], [], cpp_candidates_file)
        print(" done")
        cpp_output_file = icmp_install_dir + "resources/results/individual/" + node + "_individual_" + candidate
        if os.path.isfile(cpp_output_file):
            continue

        output, err = execute_icmp_rate_limiting_command(cpp_binary_cmd,
                                                         cpp_candidates_file,
                                                         cpp_options,
                                                         cpp_output_file,
                                                         is_individual=True,
                                                         only_analyse=False)
        print(" done")

    # Merge the individual file into one single dataframe
    df_candidates = []
    for candidate in candidates:
        cpp_output_file = icmp_install_dir + "resources/results/individual/" + node + "_individual_" + candidate
        df_candidate = pd.read_csv(cpp_output_file,
                         names=classifier_options.global_raw_columns,
                         skipinitialspace=True,
                         # encoding="utf-8",
                         engine="python",
                         index_col=False)
        df_candidates.append(df_candidate)
    df_individual = pd.concat(df_candidates)
    df_individual.to_csv(icmp_install_dir + "resources/results/individual/" + cpp_output_individual_file, index=False)

def merge_results_file(icmp_install_dir, node, results_dir, columns):
    df_candidates = []
    i = 0
    for result_file in os.listdir(results_dir):
        print(i, result_file)
        i += 1
        candidate = result_file.split("_")[2]
        cpp_output_file = icmp_install_dir + "resources/results/individual/" + node + "_individual_" + candidate
        df_candidate = pd.read_csv(cpp_output_file,
                                   names= columns,
                                   skipinitialspace=True,
                                   # encoding="utf-8",
                                   engine="python",
                                   index_col=False)
        df_candidates.append(df_candidate)
    df_individual = pd.concat(df_candidates)
    df_individual.to_csv(icmp_install_dir + "resources/results/survey_individual", index=False)

if __name__ == "__main__":
    icmp_install_dir = "/root/ICMPRateLimiting/"
    node = "ple41.planet-lab.eu"
    results_dir = icmp_install_dir + "resources/results/individual/"
    columns = ["ip_address", "probing_type", "probing_rate", "changing_behaviour", "loss_rate",
                          "transition_0_0", "transition_0_1", "transition_1_0", "transition_1_1"]

    merge_results_file(icmp_install_dir, node, results_dir, columns)

