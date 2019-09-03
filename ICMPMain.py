import sys
import argparse

import json

from Classification.classifier_options import ClassifierOptions
from Cpp.cpp_options import CppOptions
from Cpp.launcher import *
from joblib import load
from Cpp.cpp_files import *
from Algorithms.algorithms import rotate, transitive_closure
from Probing.probing_algorithm import *


def main(targets_file, router_id):
    '''
            The algorithm is O(n).
        '''

    categorized_candidates = []

    ip_version = "6"

    # targets_file = "/home/kevin/mda-lite-v6-survey/resources/internet2/ips"


    # targets_file = "resources/survey/ips" + ip_version
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
    icmp_install_dir = "/root/ICMPRateLimiting/"
    cpp_binary_cmd = "cd " + icmp_install_dir + "; " \
                                                "./build/ICMPEndToEnd "
    '''
        Set options
    '''

    ################################ Algorithm Options ##################################################

    cpp_options = CppOptions()
    # Venus individual dir
    # cpp_options.pcap_dir_individual = "/srv/icmp-rl-survey/internet2/pcap/individual/"
    # cpp_options.pcap_dir_groups = "/srv/icmp-rl-survey/internet2/pcap/groups/"

    # Ple nodes
    node = sys.argv[1]
    cpp_options.pcap_dir_individual = icmp_install_dir + "resources/pcap/individual-paper/"
    cpp_options.pcap_dir_groups = icmp_install_dir + "resources/pcap/groups/"
    cpp_options.pcap_prefix = node + "_"
    cpp_options.low_rate_dpr = 10
    cpp_options.measurement_time = 5
    cpp_options.output_file = icmp_install_dir + "test"
    cpp_options.target_loss_rate_interval = "[0.05,0.10]"
    cpp_options.exponential_ratio = 2
    '''
        Survey stuff
    '''
    cpp_options.individual_result_file = icmp_install_dir + "resources/results/survey_individual_paper_" + ip_version

    cpp_individual_file = icmp_install_dir + "resources/results/survey_individual_paper_" + ip_version
    cpp_individual_file_witness = icmp_install_dir + "resources/results/survey_individual_witness" + ip_version
    witness_by_candidate_file = "resources/witness_by_candidate" + ip_version + ".json"
    hop_by_candidate_file = "resources/hop_by_candidate" + ip_version + ".json"

    '''
        Internet2
    '''
    # cpp_options.individual_result_file = icmp_install_dir + "resources/results/internet2_individual" + ip_version + "_0.05"
    #
    # cpp_individual_file = icmp_install_dir + "resources/results/internet2_individual" + ip_version + "_0.05"
    # cpp_individual_file_witness = icmp_install_dir + "resources/results/internet2_individual_witness" + ip_version
    # witness_by_candidate_file = "resources/witness_by_candidate_internet2" + ip_version + ".json"
    # hop_by_candidate_file = "resources/hop_by_candidate_internet2" + ip_version + ".json"

    '''
        SWITCH
    '''
    # cpp_options.individual_result_file = icmp_install_dir + "resources/results/switch_individual" + ip_version
    #
    # cpp_individual_file = icmp_install_dir + "resources/results/switch_individual" + ip_version
    # cpp_individual_file_witness = icmp_install_dir + "resources/results/switch_individual_witness" + ip_version
    # witness_by_candidate_file = "resources/witness_by_candidate_switch" + ip_version + ".json"
    # hop_by_candidate_file = "resources/hop_by_candidate_switch" + ip_version + ".json"
    '''
        Serialiazed classifier
    '''
    classifier_file_name = "resources/random_forest_classifier4" + ".joblib"
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
                    if len(candidates) < 10460:
                        candidates.append(line)


    # Individual phase
    # Batch candidates into 1 candidate because of RAM issues.
    do_individual = True
    if do_individual:
        execute_individual(ip_version,
                           node,
                           candidates,
                           icmp_install_dir,
                           cpp_binary_cmd,
                           cpp_options,
                           classifier_options,
                           cpp_individual_file
                           )
    else:
        # Take the intersection of the candidates list and the individual file.
        individual_ips = set(
            extract_individual(cpp_individual_file, classifier_options.global_raw_columns)["ip_address"])
        # individual_ips_old = set(extract_individual(cpp_individual_file+"-old", classifier_options.global_raw_columns)["ip_address"])

        candidates = list(set(candidates).intersection(individual_ips))#.intersection(individual_ips_old))
        print("Only " + str(len(candidates)) + " found in the individual file. Will use the algorithm on these.")

    # Witness phase
    # Be careful, this function remove candidates with no witness.
    do_witness = True
    if do_witness:
        witness_by_candidate, hop_by_candidate = find_witness_phase(ip_version,
                                                                    node,
                                                                    candidates,
                                                                    icmp_install_dir,
                                                                    cpp_binary_cmd,
                                                                    cpp_options,
                                                                    classifier_options,
                                                                    witness_by_candidate_file,
                                                                    hop_by_candidate_file,
                                                                    cpp_individual_file_witness)

        if len(witness_by_candidate) == 0 or len(hop_by_candidate) == 0:
            return


        df_individual_witness = extract_individual(cpp_individual_file_witness, classifier_options.global_raw_columns)
    df_individual = extract_individual(cpp_individual_file, classifier_options.global_raw_columns)
    # Remove unresponsive addresses
    df_individual = df_individual.apply(pd.to_numeric, errors='ignore')
    df_individual = df_individual[df_individual["loss_rate"].apply(pd.to_numeric) < 1]

    # HACK to remove already computed candidates:
    # computed_candidates_file = "resources/survey/already_computed_candidates4"
    # computed_candidates = set()
    # if os.path.isfile(computed_candidates_file):
    #     with open(computed_candidates_file) as fp:
    #         for ip in fp:
    #             computed_candidates.add(ip.strip())

    # Cluster the results by triggering rates
    do_groups = True
    if do_groups:
        use_cluster = True
        if use_cluster:
            clusters = []
            for i in range(7, 16):
                cluster_triggering_rate = 2 ** i
                probing_rate_column = df_individual["probing_rate"]
                # remaining_candidates_file = "resources/survey/remaining_candidates" + str(cluster_triggering_rate)
                # if os.path.isfile(remaining_candidates_file):
                #     with open(remaining_candidates_file) as fp:
                #         cluster = []
                #         for ip in fp:
                #             cluster.append(ip.strip())
                # else:
                cluster = list(
                    df_individual[(2 ** i <= probing_rate_column) & (probing_rate_column < 2 ** (i + 1))]["ip_address"])

                clusters.append(
                    (sorted(list(set(cluster).intersection(candidates))), cluster_triggering_rate))
        else:
            clusters = [(candidates, router_id)]
        for cluster, cluster_triggering_rate in clusters:
            print("Performing alias resolution on cluster with triggering rate " + str(cluster_triggering_rate))
            cpp_options.low_rate_dpr = 10
            iteration = 0
            #         cluster = [
            # "5.178.43.130",
            # "195.22.199.7",
            # "195.22.199.65",
            #             "195.22.196.213",
            #
            #                    ]
            while len(cluster) > 1:
                print("Performing iteration " + str(iteration) + " on candidates: " + str(len(cluster)))
                aliases_found, remaining_candidates, new_unresponsive_candidates = \
                    simultaneous_phase(ip_version,
                                       node,
                                       cluster,
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
                                       df_individual_witness)
                unresponsive_candidates.extend(new_unresponsive_candidates)
                aliases.extend(aliases_found)
                cluster = remaining_candidates
                iteration += 1
                if not use_cluster:
                    break
        if len(aliases) > 0:
            final_aliases = transitive_closure(aliases)
            with open("aliases.json", "w") as fp:
                serializable_aliases = [list(router) for router in final_aliases]
                json.dump(serializable_aliases, fp)
            with open("unresponsive.json", "w") as fp:
                json.dump(unresponsive_candidates, fp)
            print("Aliases : " + str(final_aliases))
            print("Unresponsive: " + str(unresponsive_candidates))

if __name__ == "__main__":
    #Internet2targets

    # targets_file = "resources/internet2/ips4"

    # SWITCH targets

    # targets_file = "resources/SWITCH/ips6"

    #Survey targets

    targets_file = "resources/survey/ips6"


    main(targets_file, router_id=0)
