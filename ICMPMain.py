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

if __name__ == "__main__":

    '''
        The algorithm is O(n).
    '''

    categorized_candidates = []

    # targets_file = "/home/kevin/mda-lite-v6-survey/resources/internet2/ips"
    targets_file = "resources/internet2/ips6"
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
    ip_version = "6"
    cpp_options = CppOptions()
    # Venus individual dir
    # cpp_options.pcap_dir_individual = "/srv/icmp-rl-survey/internet2/pcap/individual/"
    # cpp_options.pcap_dir_groups = "/srv/icmp-rl-survey/internet2/pcap/groups/"

    # Ple nodes
    node = sys.argv[1]
    cpp_options.pcap_dir_individual = icmp_install_dir + "resources/pcap/individual/"
    cpp_options.pcap_dir_groups = icmp_install_dir + "resources/pcap/groups/"
    cpp_options.pcap_prefix = node + "_"
    cpp_options.low_rate_dpr = 2
    cpp_options.output_file = icmp_install_dir + "test"
    cpp_options.target_loss_rate_interval = "[0.30,0.50]"

    cpp_individual_file = icmp_install_dir + "internet2_individual_6"
    witness_by_candidate_file = "resources/witness_by_candidate.json"

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
                    candidates.append(line)

    iteration = 0

    # Individual phase
    # Batch candidates into 1 candidate because of RAM issues.
    execute_individual(ip_version,
                       node,
                       candidates,
                       icmp_install_dir,
                       cpp_binary_cmd,
                       cpp_options,
                       classifier_options,
                       cpp_individual_file
                       )


    while len(candidates) > 1:
        print("Performing iteration " + str(iteration) + " on candidates: " + str(len(candidates)))
        aliases_found, remaining_candidates, new_unresponsive_candidates = execute(ip_version,
                                                      candidates,
                                                      icmp_install_dir,
                                                      cpp_binary_cmd,
                                                      cpp_options,
                                                      classifier,
                                                      classifier_options,
                                                      iteration,
                                                      witness_by_candidate_file,
                                                      cpp_individual_file)
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