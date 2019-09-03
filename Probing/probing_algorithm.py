import json
import pandas as pd
import random
import numpy as np
from Cpp.launcher import *
from Data.preprocess import build_classifier_entry_from_csv, parse_labels_and_features, extract_individual
from Cpp.cpp_files import *
from Algorithms.algorithms import rotate, transitive_closure
from csv import reader


def simultaneous_phase_impl(ip_version,
                            node,
                            candidates,
                            icmp_install_dir,
                            cpp_binary_cmd,
                            cpp_options,
                            classifier,
                            classifier_options,
                            iteration,
                            step_stable,
                            cluster_triggering_rate,
                            witness_by_candidate,
                            hop_by_candidate,
                            df_individual,
                            df_witness_individual,
                            use_fine_grained_classifier):

    high_rate_candidate = candidates[0]

    hop_high_rate_candidate = hop_by_candidate[high_rate_candidate]

    close_candidates = []
    # Hop based optimization
    for i in range(0, len(candidates)):
        low_rate_candidate = candidates[i]
        hop_low_rate_candidate = hop_by_candidate[low_rate_candidate]
        if abs(hop_high_rate_candidate - hop_low_rate_candidate) <= 30:
            close_candidates.append(low_rate_candidate)

    print("Number of candidates after hop based optimization: " + str(len(close_candidates)))

    if len(close_candidates) == 1:
        close_candidates = candidates

    witnesses_not_candidates = set()
    for candidate in close_candidates:
        witness = witness_by_candidate[candidate]
        if witness not in close_candidates:
            witnesses_not_candidates.add(witness)




    # Build the cpp candidates files for the CPP tool.
    cpp_candidates_file = icmp_install_dir + "resources/cpp_targets_file_cluster" + \
                          str(cluster_triggering_rate) + "_" + str(iteration) + "_" + str(step_stable)
    print("Writing the candidates for the cpp tool in: " + str(cpp_candidates_file) + "...", end="", flush=True)
    write_cpp_candidates_file(ip_version, close_candidates, witnesses_not_candidates, cpp_candidates_file)
    print(" done")

    # Launch rate limiting algorithm for candidates
    if is_group_already_probed(high_rate_candidate, cpp_options.pcap_dir_groups):
        print("Group already present in individual files, skipping candidates rate limiting algorithm...")
        only_analyse = True
    else:
        print("Executing rate limiting algorithm for candidates... ", end="", flush=True)
        only_analyse = False

    cpp_output_file = cpp_options.output_file + "_cluster" + str(cluster_triggering_rate) + \
                      "_" + str(iteration) + "_" + str(step_stable)
    only_analyse = False
    if not only_analyse:
        execute_ping_and_icmp_rate_limiting_command(high_rate_candidate,
                                                         cpp_binary_cmd,
                                                         cpp_candidates_file,
                                                         cpp_options,
                                                         cpp_output_file,
                                                         is_individual=False,
                                                         only_analyse=False)


    print(" done")

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
    predictions = []
    try:
        probabilities = classifier.predict_proba(features_data)
        for i in range(0, len(probabilities)):
            if probabilities[i][1] > classifier_options.probability_threshold:
                # Check that the witness has a low loss rate to consider that the loss is due to ICMP Rate Limiting.
                if use_fine_grained_classifier:
                    if labeled_df.iloc[i]["loss_rate_dpr_w0"] > 0.02:
                        predictions.append(0)
                    else:
                        predictions.append(1)
                # if use_fine_grained_classifier:
                #     change_point_high = labeled_df.iloc[i]["changing_behaviour_dpr_c0"]
                #     change_point_low = labeled_df.iloc[i]["changing_behaviour_dpr_c1"]
                #     if change_point_high >= 0.7 * cpp_options.measurement_time * cpp_options.low_rate_dpr:
                #         # Typically not due to ICMP rate limiting.
                #         predictions.append(0)
                #     else:
                #         change_point_ratio = change_point_high / change_point_low
                else:
                    predictions.append(1)
            else:
                predictions.append(0)
    except ValueError as e:
        predictions = [0 for x in range(0, len(labeled_df))]


    # Custom binary tree inspired from the classifier. Does it perform better?


        # Very simple formula at the end.
        # for i in range(0, len(labeled_df)):
        #     candidate_features_i = labeled_df.iloc[i]
        #     if candidate_features_i["correlation_c1"] < 0.15:


        # # First determine if the high rate candidate is ON/OFF or something else:
        # on_off_threshold = 0.85
        # high_rate_is_on_off = False
        # if labeled_df.iloc[0]["transition_1_1_dpr_c0"] > on_off_threshold \
        #         and labeled_df.iloc[0]["transition_0_0_dpr_c0"] > on_off_threshold:
        #     high_rate_is_on_off = True
        # for i in range(0, len(labeled_df)):
        #     candidate_features_i = labeled_df.iloc[i]
        #     low_rate_is_on_off = False
        #     if candidate_features_i["transition_1_1_dpr_c1"] > on_off_threshold  \
        #         and candidate_features_i["transition_0_0_dpr_c1"] > on_off_threshold:
        #         # Detect if they have the same pattern. ON/OFF or else
        #             low_rate_is_on_off = True
        #     if low_rate_is_on_off != high_rate_is_on_off:
        #         predictions.append(0)
        #         continue
        #     # Then if ON/OFF, the correlation should be very high.
        #     if high_rate_is_on_off:
        #         if candidate_features_i["correlation_c1"] < 0.8:
        #             predictions.append(0)
        #             continue
        #     # Then if not ON/OFF, the correlation should still not be 0
        #     if candidate_features_i["correlation_c1"] > 0.10:
        #         # The loss rate should be abnormal.
        #         if candidate_features_i["loss_rate_dpr_c1"] < 0.10:
        #             predictions.append(0)
        #             continue
        #     # Finally check that the loss is due to ICMP Rate limiting
        #     if candidate_features_i["loss_rate_dpr_w0"] > 0.02:
        #         predictions.append(0)
        #         continue
        #
        #     # If we pass all the tests, we are in the case where the pair is an alias
        #     predictions.append(1)



    print(" done")


    aliases = []
    to_remove_candidates = set()

    print("Computing remaining set of candidates...", end="", flush=True)
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
    with open(cpp_options.pcap_prefix + "_aliases_cluster" + str(cluster_triggering_rate) + "_"  + str(iteration)
              + "_" + str(step_stable), "w") as alias_fp:
        alias_set = set()
        for pair in aliases:
            for alias in pair:
                alias_set.add(alias)
        for alias in alias_set:
            alias_fp.write(alias + "\n")

    return aliases, remaining_candidates, unresponsive_candidates, high_rate_candidate





def find_witness_phase(ip_version,
                       node,
                       candidates,
                       icmp_install_dir,
                       cpp_binary_cmd,
                       cpp_options,
                       classifier_options,
                       witness_by_candidate_file,
                       hop_by_candidate_file,
                       cpp_individual_witness_file):
    # Launch a traceroute to the first candidate to find a witness.
    # If not responsive, pass to the second, etc...
    # if iteration == 1:
    #     candidates = rotate(candidates, candidates.index("198.71.46.178"))

    print("Starting finding witness phase...")
    # if os.path.isfile(witness_by_candidate_file):
    #     with open(witness_by_candidate_file) as witness_by_candidate_fp:
    #         witness_by_candidate = json.load(witness_by_candidate_fp)
    #     with open(hop_by_candidate_file) as hop_by_candidate_fp:
    #         hop_by_candidate = json.load(hop_by_candidate_fp)

    witness_by_candidate, hop_by_candidate = find_witness(ip_version, candidates)
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
        return {}, {}
    print("End of the traceroute phase")
    witnesses = set()
    witnesses_not_candidates = set()

    individual_results_dir = icmp_install_dir + "resources/results/individual/"

    cpp_output_file_witness = cpp_individual_witness_file

    if os.path.isfile(cpp_output_file_witness):
        return witness_by_candidate, hop_by_candidate

    for candidate in candidates:
        if candidate in witness_by_candidate:
            witness = witness_by_candidate[candidate]
            if witness not in candidates:
                witnesses_not_candidates.add(witness)
                if not is_witness_already_probed(witness, individual_results_dir):
                    witnesses.add(witness)

    witnesses_not_candidates = sorted(list(witnesses_not_candidates))

    cpp_witness_file = icmp_install_dir + "resources/cpp_witness_file"
    print("Writing the witness for the cpp tool in: " + str(cpp_witness_file) + "...", end="", flush=True)
    write_cpp_witness_file(ip_version, witnesses_not_candidates, cpp_witness_file)
    print(" done")
    if len(witnesses_not_candidates) > 0:
        # Launch individual phase for witness

        create_df_witness_individual(ip_version, icmp_install_dir, witnesses_not_candidates, classifier_options.global_raw_columns)

        # execute_individual(ip_version, node,
        #                    witnesses_not_candidates,
        #                    icmp_install_dir,
        #                    cpp_binary_cmd,
        #                    cpp_options,
        #                    classifier_options,
        #                    cpp_output_file_witness)
        print(" done")

    return witness_by_candidate, hop_by_candidate



def simultaneous_phase(ip_version,
                       node,
                       candidates,
                       icmp_install_dir,
                       cpp_binary_cmd,
                       cpp_options,
                       classifier,
                       classifier_options,
                       iteration,
                       cluster_triggering_rate,
                       witness_by_candidate,
                       hop_by_candidate,
                       df_individual,
                       df_individual_witness
                       ):

    # Keep track of the remaining candidates
    with open("resources/survey/remaining_candidates" + str(cluster_triggering_rate), "w") as remaining_candidates_fp:
        for candidate in candidates:
            remaining_candidates_fp.write(candidate + "\n")
    print("Remaining candidates:" + str(len(candidates)))

    classifier_options.probability_threshold = 0.6
    step_stable = 0
    aliases, remaining_candidates, new_unresponsive_candidates, high_rate_candidate = simultaneous_phase_impl(ip_version, node,
                                                                                                              candidates,
                                                                                                              icmp_install_dir,
                                                                                                              cpp_binary_cmd,
                                                                                                              cpp_options,
                                                                                                              classifier,
                                                                                                              classifier_options,
                                                                                                              iteration,
                                                                                                              step_stable,
                                                                                                              cluster_triggering_rate,
                                                                                                              witness_by_candidate,
                                                                                                              hop_by_candidate,
                                                                                                              df_individual,
                                                                                                              df_individual_witness,
                                                                                                              use_fine_grained_classifier=False)

    classifier_options.probability_threshold = 0.6

    step_stable += 1
    # Loop until the algorithm self stabilize

    # old_len = -1
    #
    # while old_len != len(aliases):
    rho = 2
    for i in range(0, rho):
        if len(aliases) == 0:
            break
        old_len = len(aliases)
        cpp_options.low_rate_dpr = max(10, min(int(cluster_triggering_rate /(2 * (len(aliases)))), 20))
        aliases_tc = transitive_closure(aliases)
        if len(aliases_tc) == 1:
            aliases_tc = list(aliases_tc[0])
            aliases_tc = rotate(aliases_tc, aliases_tc.index(high_rate_candidate))

            if len(aliases_tc) > 100:
                use_fine_grained_classifier = False
            else:
                use_fine_grained_classifier = True
            aliases, remaining_candidates_sub, new_unresponsive_candidates_sub, high_rate_candidate = simultaneous_phase_impl(ip_version, node,
                                                                                                                              aliases_tc,
                                                                                                                              icmp_install_dir,
                                                                                                                              cpp_binary_cmd,
                                                                                                                              cpp_options,
                                                                                                                              classifier,
                                                                                                                              classifier_options,
                                                                                                                              iteration,
                                                                                                                              step_stable,
                                                                                                                              cluster_triggering_rate,
                                                                                                                              witness_by_candidate,
                                                                                                                              hop_by_candidate,
                                                                                                                              df_individual,
                                                                                                                              df_individual_witness,
                                                                                                                              use_fine_grained_classifier=use_fine_grained_classifier)

            remaining_candidates.extend(remaining_candidates_sub)
            new_unresponsive_candidates.extend(new_unresponsive_candidates_sub)
            step_stable += 1
        else:
            break

    if len(aliases) > 0:
        # Last step, try another high rate to be sure that the rate limiting was triggered by the high rate candidate.
        aliases_tc = transitive_closure(aliases)
        aliases_tc = list(aliases_tc[0])

        n_alias = len(aliases_tc)


        # phi parameter of the paper here
        # phi = len(aliases_tc)
        phi = 0
        # Force to change the high rate candidate
        final_alias_sets = []
        alias_by_high = {}
        for i in range(1, phi):

        # high_rate_candidate_index = aliases_tc.index(high_rate_candidate)
        # new_high_rate_candidate_index = random.randint(0, len(aliases_tc)-1)
        # while high_rate_candidate_index == new_high_rate_candidate_index:
        #     new_high_rate_candidate_index = random.randint(0, len(aliases_tc)-1)
            new_high_rate_candidate = aliases_tc[i]
            aliases_tc = rotate(aliases_tc, aliases_tc.index(new_high_rate_candidate))

            aliases, remaining_candidates_sub, new_unresponsive_candidates_sub, new_high_rate_candidate = simultaneous_phase_impl(
                ip_version, node,
                aliases_tc,
                icmp_install_dir,
                cpp_binary_cmd,
                cpp_options,
                classifier,
                classifier_options,
                iteration,
                step_stable,
                cluster_triggering_rate,
                witness_by_candidate,
                hop_by_candidate,
                df_individual,
                df_individual_witness,
                use_fine_grained_classifier=True)

            if len(aliases) > 0:
                alias_sets = transitive_closure(aliases)
                if len(alias_sets) > 1:
                    print("Error, alias sets of 1 high rate candidate can not have a length > 1")
                else:
                    alias_by_high[new_high_rate_candidate] = alias_sets[0]

            step_stable += 1


        # Keep a matrix of decisions.

        decisions_matrix = np.ones(shape=(n_alias, n_alias))

        for c, aliases_of_c in alias_by_high.items():
            i = aliases_tc.index(c)
            for alias in aliases_tc:
                if alias not in aliases_of_c:
                    j = aliases_tc.index(alias)
                    decisions_matrix[i][j] = 0
                    decisions_matrix[j][i] = 0

        for i in range(0, decisions_matrix.shape[0]):
            for j in range(i + 1, decisions_matrix.shape[1]):
                if decisions_matrix[i][j] == 1:
                    final_alias_sets.append({aliases_tc[i], aliases_tc[j]})

        final_alias_sets = transitive_closure(final_alias_sets)
        aliases = final_alias_sets
        # Output the final file of alias
        for i in range(0, len(final_alias_sets)):
            with open(cpp_options.pcap_prefix + "_aliases_cluster" + str(cluster_triggering_rate) + "_" + str(iteration)
                      + "_final_"  + str(i), "w") as alias_fp:
                for ip in final_alias_sets[i]:
                    alias_fp.write(ip + "\n")

        #     if high_rate_candidate not in aliases_tc[0]:
            #         # remaining_candidates_sub.remove(high_rate_candidate)
            #         # if n_alias > 2:
            #         #     # Reput the candidate in the set if it has a chance to be aliases with other interfaces.
            #         #     remaining_candidates.append(new_high_rate_candidate)
            #         # remaining_candidates.extend(remaining_candidates_sub)
            #         # remaining_candidates.extend(aliases_tc[0])
            #
            #         # Likely due to unstable interfaces
            #
            #         aliases = []
            # else:
            #     # In case the candidate has become unresponsive...
            #     if high_rate_candidate in remaining_candidates_sub:
            #         remaining_candidates_sub.remove(high_rate_candidate)
            #     # Remove the new high rate if it triggered
            #     # if n_alias > 2:
            #     #     remaining_candidates.append(new_high_rate_candidate)
            #     # remaining_candidates.extend(remaining_candidates_sub)







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

    if os.path.isfile(cpp_output_individual_file):
        return

    for candidate in candidates:
        cpp_candidates_file = icmp_install_dir + "resources/cpp_targets_file_" + candidate
        print("Writing the candidates file for the cpp individual tool in: " + str(cpp_candidates_file) + "...", end="", flush=True)
        write_cpp_candidates_file(ip_version, [candidate], [], cpp_candidates_file)
        print(" done")
        cpp_output_file = icmp_install_dir + "resources/results/individual/" + node + "_individual_" + candidate
        if os.path.isfile(cpp_output_file):
            continue

        cpp_options.starting_probing_rate = 512
        # execute_ping_and_icmp_rate_limiting_command(candidate,cpp_binary_cmd,
        #                                                  cpp_candidates_file,
        #                                                  cpp_options,
        #                                                  cpp_output_file,
        #                                                  is_individual=True,
        #                                                  only_analyse=False)
        execute_icmp_rate_limiting_command(cpp_binary_cmd,
                                                    cpp_candidates_file,
                                                    cpp_options,
                                                    cpp_output_file,
                                                    is_individual=True,
                                                    only_analyse=True)
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
    df_individual.to_csv(cpp_output_individual_file, index=False)

def merge_results_file(icmp_install_dir, node, results_dir, columns):
    df_candidates = []
    i = 0
    for result_file in os.listdir(results_dir):
        print(i, result_file)
        i += 1
        if i == 10531:
            break
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
    df_individual.to_csv(icmp_install_dir + "resources/results/survey_individual6_paper", index=False)


def create_witness_features(ip_witness):
    line = ip_witness + ",INDIVIDUAL, 32768, 163840, 0.505804, 0.797142, 0.202858, 0.1982, 0.8018"
    return line

def create_df_witness_individual(ip_version, icmp_install_dir, witnesses, columns):
    df_witnesses = []
    for witness in witnesses:

        df_witness = pd.DataFrame(list(reader([create_witness_features(witness)])),
                                 columns=columns
                                 # encoding="utf-8",
                                 )
        df_witnesses.append(df_witness)
    df_individual = pd.concat(df_witnesses)
    df_individual.to_csv(icmp_install_dir + "resources/results/switch_individual_witness" + ip_version, index=False)


if __name__ == "__main__":
    icmp_install_dir = "/root/ICMPRateLimiting/"
    node = "ple2.planet-lab.eu"
    results_dir = icmp_install_dir + "resources/results/individual/"
    columns = ["ip_address", "probing_type", "probing_rate", "changing_behaviour", "loss_rate",
                          "transition_0_0", "transition_0_1", "transition_1_0", "transition_1_1"]

    merge_results_file(icmp_install_dir, node, results_dir, columns)


    ########################### Analyze hop in traceroutes #####################
    # import json
    # routers_file = "resources/survey/cluster_ips_by_hop.json"
    # with open(routers_file) as f:
    #     routers = json.load(f)
    #
    #     for router_name, ips in routers.items():
    #         witness_by_candidate , hop_by_candidate = find_witness("4", ips)
    #         hops = set()
    #         for candidate, hop in hop_by_candidate.items():
    #             hops.add(hop)
    #
    #
    #         if len(hops) > 1:
    #             max_diff = 0
    #             for hop in hops:
    #                 for hop2 in hops:
    #                     if abs(hop2 - hop) > max_diff:
    #                         max_diff = abs(hop2 - hop)
    #             if max_diff > 6:
    #                 print(hop_by_candidate)
    #
    # import pandas as pd
    #
    # individual_file = "resources"


    ######################################## DEBUG #########################################

    # Compute remaining candidates to not redo the experiment.
    if False:
        results_dir = "./"
        node = "ple41.planet-lab.eu"

        # remaining_candidates = []
        removed_candidates = set()
        for filename in os.listdir(results_dir):
            alias_prefix = node + "__aliases_cluster2048"
            if not filename.startswith(alias_prefix):
                continue

            split = filename.split("_")
            iter = split[4]
            stable_iter = split[5]
            # Extract high rate candidate

            if stable_iter == "0":
                cpp_output_file = "/root/ICMPRateLimiting/test_cluster2048_" + iter + "_" + stable_iter
                with open(cpp_output_file) as router_fp:
                    for line in router_fp:
                        high_rate_candidate = line.split(",")[0]
                        removed_candidates.add(high_rate_candidate)
                        break
            else:
                next_stable_iter = int(stable_iter) + 1
                next_alias_file = alias_prefix + "_" + iter + "_" + str(next_stable_iter)
                if os.path.exists(next_alias_file):
                    continue
                else:
                    previous_stable_iter = int(stable_iter) - 1
                    previous_previous_stable_iter = int(stable_iter) - 2
                    previous_alias_file = alias_prefix + "_" + iter + "_" + str(previous_stable_iter)
                    previous_previous_alias_file = alias_prefix + "_" + iter + "_" + str(previous_previous_stable_iter)
                    if os.path.isfile(previous_previous_alias_file) and os.path.isfile(previous_alias_file):

                        if os.stat(previous_alias_file).st_size == os.stat(previous_previous_alias_file).st_size:
                            # We need to check if the algorithm passed the stabilization step
                            # We can look at the size of the two previous iter.

                            # Remove all the candidates from the previous set
                            with open(previous_alias_file) as fp:
                                for ip in fp:
                                    ip = ip.strip()
                                    removed_candidates.add(ip)
        print(len(removed_candidates))
        with open("resources/survey/already_computed_candidates4", "w") as fp:
            for c in removed_candidates:
                fp.write(c + "\n")

