from Validation.midar import (
    extract_routers,
    extract_routers_by_node,
    extract_routers_evaluation,
)
from Algorithms.algorithms import transitive_closure
import pandas as pd
import copy
import numpy as np
import os

import socket
import ipaddress
import json
import random

base_columns = [
    "ip_address",
    "probing_type",
    "probing_rate",
    "changing_behaviour",
    "loss_rate",
    "transition_0_0",
    "transition_0_1",
    "transition_1_0",
    "transition_1_1",
]
targets_columns = [
    "GROUP_ID",
    "AF_FAMILY",
    "PROBING_TYPE",
    "PROTOCOL",
    "INTERFACE_TYPE",
    "IP_TARGET",
    "REAL_IP",
]


def find_corresponding_router(ip, ground_truth_routers):
    matches = []
    for router_name, router in ground_truth_routers.items():
        for ip_router in router:
            if ip_router == ip:
                matches.append((router_name, router))
    return matches


def compare_routers_set(routers_set1, routers_set2):
    # Now compare the common pairs.
    common_pairs = set()
    uncommon_pairs = set()
    disagreement_pairs = set()

    n_router = 0
    for router_name, router in routers_set1.items():
        n_router += 1
        for i in range(0, len(router)):
            # Find the corresponding router in midar
            corresponding_router_i = find_corresponding_router(router[i], routers_set2)
            for j in range(i + 1, len(router)):
                corresponding_router_j = find_corresponding_router(
                    router[j], routers_set2
                )
                if (
                    corresponding_router_i is not None
                    and corresponding_router_j is not None
                ):
                    if corresponding_router_i == corresponding_router_j:
                        common_pairs.add(tuple({router[i], router[j]}))
                    else:
                        disagreement_pairs.add(tuple({router[i], router[j]}))
                else:
                    uncommon_pairs.add(tuple({router[i], router[j]}))

    return common_pairs, uncommon_pairs, disagreement_pairs


def combin(n, k):
    """Number of combinations C(n,k)"""
    if k > n // 2:
        k = n - k
    x = 1
    y = 1
    i = n - k + 1
    while i <= n:
        x = (x * i) // y
        y += 1
        i += 1
    return x


def precision(tp, fp):
    return tp / (tp + fp)


def recall(tp, fn):
    return tp / (tp + fn)


def clean_false_positives(routers, ground_truth_routers):
    tp = 0
    fp = 0
    to_remove_ips = {}
    for router_name, router in routers.items():
        to_remove_ips[router_name] = set()
        for i in range(0, len(router)):
            matches_i = find_corresponding_router(router[i], ground_truth_routers)
            for j in range(i + 1, len(router)):
                matches_j = find_corresponding_router(router[j], ground_truth_routers)
                found_match = False
                for match_router_i, ips_i in matches_i:
                    for match_router_j, ips_j in matches_j:
                        if match_router_i == match_router_j:
                            found_match = True
                            break
                    if found_match:
                        break
                if not found_match:
                    fp += 1
                    print(router[i], router[j])
                    if len(set(router).intersection(set(ips_i))) > len(
                        set(router).intersection(set(ips_j))
                    ):
                        to_remove_ips[router_name].add(router[j])
                    else:
                        to_remove_ips[router_name].add(router[i])
                else:
                    tp += 1
    for router_name, ips in to_remove_ips.items():
        for ip in ips:
            routers[router_name].remove(ip)
    return routers, tp, fp


def write_uncommon_routers(
    uncommon_directory, uncommon_pairs, do_transitive_closure=True
):
    # Uncommon IPs
    routers = [set(x) for x in uncommon_pairs]
    if do_transitive_closure:
        uncommon_routers = transitive_closure(routers)
    else:
        uncommon_routers = routers
    print("Routers found by Lovebirds but not MIDAR: " + str(len(uncommon_routers)))
    n_uncommon = 0
    for uncommom_router in uncommon_routers:
        with open(
            uncommon_directory
            + "uncommon_router_lovebirds_not_midar"
            + str(n_uncommon),
            "w",
        ) as fp:
            for ip in uncommom_router:
                fp.write(ip + "\n")
        n_uncommon += 1


def evaluate(routers, ground_truth_routers, n_ground_truth_pairs):

    # pirate_ips = ["64.57.25.107", "64.57.25.106"]

    ground_truth_routers = copy.deepcopy(ground_truth_routers)
    routers, tp, fp = clean_false_positives(routers, ground_truth_routers)
    print("True positives", tp)
    print("False positives", fp)

    # with open("resources/results/internet2/unresponsive/ple41.planet-lab.eu_internet2_unresponsive6.json") as unresponsive_ips_fp:
    #     unresponsive_ips = set(json.load(unresponsive_ips_fp))
    #     # unresponsive_ips.update(set(pirate_ips))
    # # Remove the unresponsive addresses from all vantage points
    # for file_name, ips in ground_truth_routers.items():
    #     for ip in unresponsive_ips:
    #         if ip in ips:
    #             ips.remove(ip)

    """Stats"""
    res_gt = n_ground_truth_pairs

    # for router, ips in sorted(ground_truth_routers.items()):
    #     res_gt += combin(len(set(ips)), 2)

    print("Ground Truth pairs: ")
    print(res_gt)
    print("Rate Limiting pairs :")
    print(tp)

    print("Precision:")
    print(float(tp) / (tp + fp))

    print("Recall :")
    print(float(tp) / res_gt)


def compare_tools():
    is_internet2 = True
    is_survey = False
    ip_version = "4"
    """
    Internet2 V4 evaluation
    """
    if is_internet2 and ip_version == "4":
        ground_truth_routers = extract_routers(
            "/home/kevin/mda-lite-v6-survey/resources/internet2/routers/v4/"
        )

        rate_limiting_nodes = [
            "cse-yellow.cse.chalmers.se",
            "ple1.cesnet.cz",
            "planetlab1.cs.vu.nl",
            "ple41.planet-lab.eu",
        ]

        routers = []
        precisions = []
        for node in rate_limiting_nodes:
            routers_node = extract_routers_evaluation(
                "resources/results/internet2/aliases/v4/" + node + "/"
            )
            routers_node, tp, fp = clean_false_positives(
                routers_node, ground_truth_routers
            )
            print(node, precision(tp, fp))
            precisions.append(precision(tp, fp))

            # routers.update(extract_routers("resources/results/internet2/aliases/ple41.planet-lab.eu/"))
            # routers_tc = []
            for router_name, router in routers_node.items():
                routers.append(set(router))
        routers = transitive_closure(routers)
        routers_tc_dic = {i: list(routers[i]) for i in range(0, len(routers))}

        print("Mean precision over ple nodes:" + str(np.mean(precisions)))

        midar_nodes = [
            "ple1.cesnet.cz",
            "ple41.planet-lab.eu",
            "planetlab2.informatik.uni-goettingen.de",
            "cse-yellow.chalmers.cse.se",
            "planetlab1.cs.vu.nl",
        ]
        midar_routers = []
        for node in midar_nodes:
            midar_routers_node = extract_routers(
                "/home/kevin/mda-lite-v6-survey/resources/internet2/midar/v4/"
                + node
                + "/"
            )
            for router_name, router in midar_routers_node.items():
                midar_routers.append(set(router))
        midar_routers = transitive_closure(midar_routers)
        midar_routers_tc_dic = {
            i: list(midar_routers[i]) for i in range(0, len(midar_routers))
        }

        evaluate(midar_routers_tc_dic, ground_truth_routers)
        evaluate(routers_tc_dic, ground_truth_routers)

        common_pairs, uncommon_pairs, disagreement_pairs = compare_routers_set(
            routers_tc_dic, midar_routers_tc_dic
        )
        total_pairs = len(common_pairs) + len(uncommon_pairs) + len(disagreement_pairs)
        print("Common pairs: " + str(len(common_pairs)))
        print("Uncommon pairs: " + str(len(uncommon_pairs)))
        print("Disagreement pairs: " + str(len(disagreement_pairs)))
        print("Total pairs: " + str(total_pairs))

        print("Ratio common: " + str(len(common_pairs) / total_pairs))
        print("Ratio uncommon: " + str(len(uncommon_pairs) / total_pairs))
        print("Ratio disagree: " + str(len(disagreement_pairs) / total_pairs))

        # uncommon_directory = "resources/results/internet2/uncommon/v4/lovebirds-not-midar/"
        # write_uncommon_routers(uncommon_directory, uncommon_pairs, do_transitive_closure=False)

        uncommon_directory = (
            "resources/results/internet2/uncommon/v4/lovebirds-not-midar/"
        )
        classification_file = (
            "resources/results/internet2/midar-analysis/target-summary.txt"
        )
        comparable_directory = (
            "resources/results/internet2/uncommon/v4/comparable-lovebirds-not-midar/"
        )

        comparable_pairs = lovebirds_not_midar(
            uncommon_directory, classification_file, comparable_directory
        )
        # uncommon_directory = "resources/results/internet2/uncommon/v4/comparable-lovebirds-not-midar/"
        # write_uncommon_routers(uncommon_directory, comparable_pairs, do_transitive_closure=False)

        # ####### Evaluate the rerun ################
        # rerun_directory = "resources/results/internet2/uncommon/v4/comparable-lovebirds-not-midar-rerun/"
        #
        # n_found_after_rerun = 0
        # for file in os.listdir(rerun_directory):
        #     if file.endswith(".tc"):
        #         if os.path.getsize(rerun_directory + file) > 100:
        #             n_found_after_rerun += 1
        #
        # print("Pairs found by MIDAR after rerun: " + str(n_found_after_rerun))

        common_pairs, uncommon_pairs, disagreement_pairs = compare_routers_set(
            midar_routers_tc_dic, routers_tc_dic
        )
        total_pairs = len(common_pairs) + len(uncommon_pairs) + len(disagreement_pairs)
        print("Common pairs: " + str(len(common_pairs)))
        print("Uncommon pairs: " + str(len(uncommon_pairs)))
        print("Disagreement pairs: " + str(len(disagreement_pairs)))
        print("Total pairs: " + str(total_pairs))

        print("Ratio common: " + str(len(common_pairs) / total_pairs))
        print("Ratio uncommon: " + str(len(uncommon_pairs) / total_pairs))
        print("Ratio disagree: " + str(len(disagreement_pairs) / total_pairs))

        witness_file = "resources/witness_by_candidate_internet24.json"
        with open(witness_file) as f:
            import json

            witness_by_candidate = json.load(f)

        n_findable = 0
        n_not_findable = 0
        for ip1, ip2 in uncommon_pairs:
            if ip1 in witness_by_candidate and ip2 in witness_by_candidate:
                n_findable += 1
                print(ip1, ip2)
            else:
                n_not_findable += 1
        print("Pairs findable by LB: " + str(n_findable))
        print("Pairs not findable by LB: " + str(n_not_findable))
        # uncommon_directory = "resources/results/internet2/uncommon/v4/midar-not-lovebirds/"
        # write_uncommon_routers(uncommon_directory, uncommon_pairs)

    if is_internet2 and ip_version == "6":
        ground_truth_routers = extract_routers(
            "/home/kevin/mda-lite-v6-survey/resources/internet2/routers/v6/"
        )

        rate_limiting_nodes = ["ple41.planet-lab.eu"]

        routers = []
        precisions = []
        for node in rate_limiting_nodes:
            routers_node = extract_routers(
                "resources/results/internet2/aliases/v6/" + node + "/"
            )
            routers_node, tp, fp = clean_false_positives(
                routers_node, ground_truth_routers
            )
            print(node, precision(tp, fp))
            precisions.append(precision(tp, fp))

            # routers.update(extract_routers("resources/results/internet2/aliases/ple41.planet-lab.eu/"))
            # routers_tc = []
            for router_name, router in routers_node.items():
                routers.append(set(router))
        routers = transitive_closure(routers)
        routers_tc_dic = {i: list(routers[i]) for i in range(0, len(routers))}

        print("Mean precision over ple nodes:" + str(np.mean(precisions)))

        evaluate(routers_tc_dic, ground_truth_routers)

    """
        D4 Evaluation
    """

    if is_survey and ip_version == "4":
        result_individual_file = "resources/results/survey/survey_individual4"

        node = "ple41.planet-lab.eu"

        rate_limiting_routers_dir = "resources/results/survey/aliases/v4/" + node + "/"
        rate_limiting_routers = extract_routers_evaluation(rate_limiting_routers_dir)
        midar_routers = extract_routers_by_node(
            "/home/kevin/mda-lite-v6-survey/resources/midar/batch2/routers/"
        )

        # Test on ple41
        midar_routers_node = midar_routers[node]

        # Compare only IP's that were seen by both experience. i.e remove those tried by MIDAR and not by Lovebirds.
        df_individual = pd.read_csv(
            result_individual_file,
            names=base_columns,
            skipinitialspace=True,
            # encoding="utf-8",
            engine="python",
            index_col=False,
        )

        df_unresponsive = df_individual[df_individual["loss_rate"] == 1]
        unresponsive_addresses = list(df_unresponsive["ip_address"])
        unresponsives = set(unresponsive_addresses)

        rate_limiting_ip_addresses = list(
            df_individual[
                (df_individual["probing_rate"] >= 256)
                & (df_individual["probing_rate"] < 50000)
            ]["ip_address"]
        )

        common_ips = []
        for router_name, router in midar_routers_node.items():
            to_remove = []
            for ip in router:
                if ip not in rate_limiting_ip_addresses:
                    to_remove.append(ip)
                else:
                    common_ips.append(ip)
            for ip in to_remove:
                router.remove(ip)
        # print("Common IPs tested: " + str(len(common_ips)))
        # with open("resources/survey/cluster_ips_by_hop.json", "w") as fp:
        #     import json
        #     json.dump(midar_routers_node, fp,sort_keys=True,indent=4, separators=(',', ': '))

        common_pairs, uncommon_pairs, disagreement_pairs = compare_routers_set(
            rate_limiting_routers, midar_routers_node
        )
        total_pairs = len(common_pairs) + len(uncommon_pairs) + len(disagreement_pairs)
        print("Common pairs: " + str(len(common_pairs)))
        print("Uncommon pairs: " + str(len(uncommon_pairs)))
        print("Disagreement pairs: " + str(len(disagreement_pairs)))
        print("Total pairs: " + str(total_pairs))

        print("Ratio common: " + str(len(common_pairs) / total_pairs))
        print("Ratio uncommon: " + str(len(uncommon_pairs) / total_pairs))
        print("Ratio disagree: " + str(len(disagreement_pairs) / total_pairs))

        # uncommon_directory = "resources/results/survey/uncommon/v4/lovebirds-not-midar-disagreement/"
        # write_uncommon_routers(uncommon_directory, disagreement_pairs)

        # uncommon_ips = set()
        # for router in routers:
        #     for ip in router:
        #         uncommon_ips.add(ip)

        # n_disagreement = 0
        # for disagreement_pair in disagreement_pairs:
        #     with open("resources/results/survey/disagreement/v4/disagreement_pair" + str(n_disagreement), "w") as fp:
        #         for ip in disagreement_pair:
        #             fp.write(ip + "\n")
        #     n_disagreement += 1

        common_pairs, uncommon_pairs, disagreement_pairs = compare_routers_set(
            midar_routers_node, rate_limiting_routers
        )
        total_pairs = len(common_pairs) + len(uncommon_pairs) + len(disagreement_pairs)
        print("Common pairs: " + str(len(common_pairs)))
        print("Uncommon pairs: " + str(len(uncommon_pairs)))
        print("Disagreement pairs: " + str(len(disagreement_pairs)))
        print("Total pairs: " + str(total_pairs))

        print("Ratio common: " + str(len(common_pairs) / total_pairs))
        print("Ratio uncommon: " + str(len(uncommon_pairs) / total_pairs))
        print("Ratio disagree: " + str(len(disagreement_pairs) / total_pairs))

        routers = [set(x) for x in uncommon_pairs]
        uncommon_routers = transitive_closure(routers)
        print("Routers found by MIDAR but not Lovebirds: " + str(len(uncommon_routers)))
        # n_uncommon = 0
        # for uncommom_router in uncommon_routers:
        #     with open("resources/results/survey/uncommon/v4/uncommon_router_midar_not_lovebirds" + str(n_uncommon), "w") as fp:
        #         for ip in uncommom_router:
        #             fp.write(ip + "\n")
        #     n_uncommon += 1
        uncommon_directory = (
            "resources/results/survey/uncommon/v4/midar-not-lovebirds-disagreement/"
        )
        write_uncommon_routers(uncommon_directory, disagreement_pairs)

        # Extract correlation from the routers set for printing figure on paper
        # for router_name, router in rate_limiting_routers.items():
        #     if len(router) > 1:
        #         columns = list(base_columns)
        #         for i in range(1, len(router)):
        #             columns.append("correlation_c" + str(i))
        #         split = router_name.split(node + "__aliases_")
        #         identifier = split[1]
        #         corresponding_cpp_output = "/root/ICMPRateLimiting/test_" + identifier
        #         # print (corresponding_cpp_output)
        #         with open(rate_limiting_routers_dir + router_name) as router_fp:
        #             line_count = 0
        #             ips = []
        #             for line in router_fp:
        #                 line = line.strip()
        #                 split = line.split(",")
        #                 ip = split[0]
        #                 ips.append(ip)
        #                 # probing_rate = split[2]
        #                 line_count += 1
        #                 if line_count == 2:
        #                     break
        #         # print("scp root@"+node+ ":/root/ICMPRateLimiting/resources/pcap/groups/" + node + "_icmp_echo_reply_" + \
        #       ips[0] + "_" + ips[1] + "_*_GROUPDPR.pcap resources/pcap/groups/")

        # Generate a command to scp the relevant pcap files.
        # df_alias = pd.read_csv(corresponding_cpp_output,
        #                        names=columns,
        #                        skipinitialspace=True,
        #                        # encoding="utf-8",
        #                        engine="python",
        #                        index_col=False)
        #
        # print(df_alias["correlation_c1"])


def rerun_stable(rerun_directory, results_rerun_directory, rerun_pairs=None):
    # Count number of confirmed pairs after rerun.
    n_stable_pairs = 0
    n_rerun_pairs = 0

    for router_file in os.listdir(rerun_directory):
        with open(rerun_directory + router_file) as f:
            router = []
            for ip in f:
                router.append(ip.strip())
            for i in range(0, len(router)):
                for j in range(i + 1, len(router)):
                    if rerun_pairs is not None:
                        if tuple({router[i], router[j]}) in rerun_pairs:
                            n_rerun_pairs += 1
                    else:
                        n_rerun_pairs += 1
    routers = extract_routers_evaluation(results_rerun_directory)
    for router_name, router in routers.items():
        for i in range(0, len(router)):
            for j in range(i + 1, len(router)):
                if rerun_pairs is not None:
                    if tuple({router[i], router[j]}) in rerun_pairs:
                        n_stable_pairs += 1
                else:
                    n_stable_pairs += 1

    print("Stable pairs: " + str(n_stable_pairs))
    print("Rerun pairs: " + str(n_rerun_pairs))


def compute_midar_comparables_pairs(ips, midar_classification):
    comparable_pairs = set()
    uncomparable_pairs = set()
    for i in range(0, len(ips)):
        classification_ip_i = midar_classification[ips[i]]
        udp_i_class = classification_ip_i["udp"]
        tcp_i_class = classification_ip_i["tcp"]
        icmp_i_class = classification_ip_i["icmp"]

        for j in range(i + 1, len(ips)):
            classification_ip_j = midar_classification[ips[j]]
            udp_j_class = classification_ip_j["udp"]
            tcp_j_class = classification_ip_j["tcp"]
            icmp_j_class = classification_ip_j["icmp"]

            is_comparable = False
            # if udp_i_class == "monotonic" and udp_j_class == "monotonic":
            #     is_comparable = True
            # elif tcp_i_class == "monotonic" and tcp_j_class == "monotonic":
            #     is_comparable = True
            # elif icmp_i_class == "monotonic" and icmp_j_class == "monotonic":
            #     is_comparable = True

            if (
                udp_i_class == "monotonic"
                or tcp_i_class == "monotonic"
                or icmp_i_class == "monotonic"
            ) and (
                tcp_j_class == "monotonic"
                or udp_j_class == "monotonic"
                or icmp_j_class == "monotonic"
            ):
                is_comparable = True

            if is_comparable:
                comparable_pairs.add(frozenset({ips[i], ips[j]}))
                # copyfile(uncommon_directory + file, comparable_directory + file)

            if not is_comparable:
                uncomparable_pairs.add(frozenset({ips[i], ips[j]}))

    return comparable_pairs, uncomparable_pairs


def compute_midar_classification(ips, classification_file):
    midar_classification = {}
    usable = set()

    df_target_summary = pd.read_csv(
        classification_file,
        # header=0,
        # skipinitialspace=True,
        skiprows=[0],
        # encoding="utf-8",
        engine="python",
        sep="\s+",
        index_col=False,
        usecols=[x for x in range(0, 15)],
    )

    targets = list(df_target_summary["target_addr"])
    for target in targets:
        midar_classification[target] = {}

    for row in df_target_summary.itertuples(index=True):
        target = getattr(row, "target_addr")
        method = getattr(row, "method")
        if method.endswith("*"):
            method = method[:-1]
            midar_classification[target]["probing_method"] = method
        classification_row = getattr(row, "classification")
        if classification_row is not None:
            classification_row = classification_row.split(";")[0]
            midar_classification[target][method] = classification_row
            if classification_row == "monotonic" and target in ips:
                usable.add(target)
        else:
            print(target, method)
    non_monotonic = set()
    for target, probing_classification in midar_classification.items():
        if "udp" not in probing_classification:
            midar_classification[target]["udp"] = "unknown"
        if "icmp" not in probing_classification:
            midar_classification[target]["icmp"] = "unknown"
        if "tcp" not in probing_classification:
            midar_classification[target]["tcp"] = "unknown"

        if target not in usable:
            for proto, classification in probing_classification.items():
                if classification == "nonmonotonic":
                    non_monotonic.add(target)
    print("Midar nonmonotonic: " + str(len(non_monotonic)))
    print("Midar classification: " + str(len(midar_classification)))

    for ip in ips:
        if ip not in midar_classification:
            midar_classification[ip] = {
                "udp": "unknown",
                "tcp": "unknown",
                "icmp": "unkown",
            }

    print("Number of usable: " + str(len(usable)))

    return midar_classification


def extract_speedtrap_alias_pairs(ips, speedtrap_pairs_file):
    speedtrap_pairs = set()
    with open(speedtrap_pairs_file) as f:
        for line in f:
            split = line.split()
            alias_0 = split[0]
            alias_1 = split[1]
            if alias_0 in ips and alias_1 in ips:
                speedtrap_pairs.add(frozenset({alias_0, alias_1}))
    return speedtrap_pairs


def compute_speedtrap_comparable_pairs(ips, speedtrap_classification):
    comparable_pairs = set()
    uncomparable_pairs = set()

    for i in range(0, len(ips)):
        classification_i = speedtrap_classification[ips[i]]
        for j in range(i + 1, len(ips)):
            classification_j = speedtrap_classification[ips[j]]
            if classification_i == "incr" and classification_j == "incr":
                comparable_pairs.add(frozenset({ips[i], ips[j]}))
            else:
                uncomparable_pairs.add(frozenset({ips[i], ips[j]}))

    return comparable_pairs, uncomparable_pairs


def compute_speedtrap_classification(ips, speedtrap_classification_file):
    speedtrap_classification = {}
    printable_classification = {}

    with open(speedtrap_classification_file) as f:
        for line in f:
            split = line.split()
            ip = split[0]
            classification = split[1]
            speedtrap_classification[ip] = classification
            if classification not in printable_classification:
                printable_classification[classification] = 0
            printable_classification[classification] += 1

    for ip in ips:
        if ip not in speedtrap_classification:
            print(ip)
            speedtrap_classification[ip] = "unknown"

    print(printable_classification)

    return speedtrap_classification


def lovebirds_not_midar(uncommon_directory, classification_file, comparable_directory):
    ###### Lovebirds not midar #########
    from shutil import copyfile

    n_not_comparable = 0
    n_fp = 0

    comparable_pairs = set()

    classification = {}

    midar_classification = compute_midar_classification(classification_file)

    for file in os.listdir(uncommon_directory):
        router = []
        with open(uncommon_directory + file) as fp:
            for ip in fp:
                router.append(ip.strip())
        for i in range(0, len(router)):
            classification_ip_i = midar_classification[router[i]]
            udp_i_class = classification_ip_i["udp"]
            tcp_i_class = classification_ip_i["tcp"]
            icmp_i_class = classification_ip_i["icmp"]

            if udp_i_class not in classification:
                classification[udp_i_class] = 1
            else:
                classification[tcp_i_class] += 1
            if tcp_i_class not in classification:
                classification[tcp_i_class] = 1
            else:
                classification[tcp_i_class] += 1
            if icmp_i_class not in classification:
                classification[icmp_i_class] = 1
            else:
                classification[icmp_i_class] += 1

            for j in range(i + 1, len(router)):
                classification_ip_j = midar_classification[router[j]]
                udp_j_class = classification_ip_j["udp"]
                tcp_j_class = classification_ip_j["tcp"]
                icmp_j_class = classification_ip_j["icmp"]

                is_comparable = False
                if udp_i_class == "monotonic" and udp_j_class == "monotonic":
                    is_comparable = True
                elif tcp_i_class == "monotonic" and tcp_j_class == "monotonic":
                    is_comparable = True
                elif icmp_i_class == "monotonic" and icmp_j_class == "monotonic":
                    is_comparable = True

                if is_comparable:
                    n_fp += 1
                    comparable_pairs.add(tuple({router[i], router[j]}))
                    # copyfile(uncommon_directory + file, comparable_directory + file)

                if not is_comparable:
                    n_not_comparable += 1

    total = n_not_comparable + n_fp
    print("Not comparable: " + str(n_not_comparable))
    print("Not comparable: " + str(n_not_comparable / total))
    print("Comparable: " + str(n_fp))
    print("Comparable: " + str(n_fp / total))
    print(classification)
    return comparable_pairs


def analyse_rerun():
    """
        Analyze the rerun of both tools on uncommon routers.
    :return:
    """

    # Midar not Lovebirds.
    is_midar_not_lovebirds = True
    if is_midar_not_lovebirds:
        rerun_directory = "resources/results/survey/uncommon/v4/midar-not-lovebirds-disagreement-rerun/"

        import json

        witness_file = "resources/witness_by_candidate4.json"
        with open(witness_file) as f:
            witness_by_candidate = json.load(f)

        # First extract routers.
        # midar_not_lovebirds_routers = []
        # for router_file in os.listdir(uncommon_directory):
        #     router = set()
        #     with open(uncommon_directory + router_file) as f:
        #         for ip in f:
        #             router.add(ip.strip())
        #     midar_not_lovebirds_routers.append(router)

        # Stats
        n_not_triggered = 0
        n_not_shared_counter = 0
        n_witness_too_high = 0
        n_missed = 0
        n_else = 0
        n_multiple_behaviour = 0
        for uncommon_file in os.listdir(rerun_directory):

            if not "test_cluster" in uncommon_file:
                continue

            if not uncommon_file.endswith("_0"):
                continue
            split = uncommon_file.split("_")
            id = split[1].split("cluster")[1]
            # Find the corresponding target file
            targets_file = "cpp_targets_file_cluster" + str(id) + "_0_0"
            try:
                df_targets = pd.read_csv(
                    rerun_directory + targets_file,
                    names=targets_columns,
                    skipinitialspace=True,
                    # encoding="utf-8",
                    engine="python",
                    index_col=False,
                )

                df_alias = pd.read_csv(
                    rerun_directory + uncommon_file,
                    names=base_columns,
                    skipinitialspace=True,
                    # encoding="utf-8",
                    engine="python",
                    index_col=False,
                )
            except:
                continue
            candidates = list(
                df_targets[df_targets["INTERFACE_TYPE"] == "CANDIDATE"]["IP_TARGET"]
            )

            ###### Start pattern pair classification ##########

            is_not_triggered = False
            has_shared_icmp_counter = False
            has_not_shared_icmp_counter = False

            for i in range(0, len(candidates)):
                if is_not_triggered:
                    n_not_triggered += 1
                    continue
                candidate_row = df_alias[df_alias["ip_address"] == candidates[i]]
                if len(candidate_row) == 0:
                    n_not_triggered += 1
                    continue
                loss_rate = float(candidate_row["loss_rate"])
                witness = witness_by_candidate[candidates[i]]
                witness_row = df_alias[df_alias["ip_address"] == witness]
                loss_rate_witness = float(witness_row["loss_rate"])
                if i == 0:
                    if loss_rate < 0.05:
                        n_not_triggered += 1
                        is_not_triggered = True
                    continue
                if loss_rate == 0:
                    n_not_shared_counter += 1
                    has_not_shared_icmp_counter = True
                elif loss_rate_witness > 0:
                    n_witness_too_high += 1
                elif loss_rate > 0.05:
                    n_missed += 1
                    has_shared_icmp_counter = True
            if has_not_shared_icmp_counter and has_shared_icmp_counter:
                n_multiple_behaviour += 1

        total = n_not_shared_counter + n_missed + n_witness_too_high + n_not_triggered
        print("Not shared: " + str(n_not_shared_counter / total))
        print("Witness too high: " + str(n_witness_too_high / total))
        print("Missed: " + str(n_missed / total))
        print("Not triggered" + str(n_not_triggered / total))
        print("Else: " + str(n_else / total))
        print("Multiple: " + str(n_multiple_behaviour))
        print(total)

        uncommon_directory = "resources/results/survey/uncommon/v4/lovebirds-not-midar/"
        classification_file = "resources/results/survey/target-summary.txt"
        comparable_directory = (
            "resources/results/survey/uncommon/v4/comparable-lovebirds-not-midar/"
        )

        comparable_pairs = lovebirds_not_midar(
            uncommon_directory, classification_file, comparable_directory
        )

        comparable_directory_rerun = (
            "resources/results/survey/uncommon/v4/comparable-lovebirds-not-midar-rerun/"
        )
        rerun_stable(comparable_directory, comparable_directory_rerun, comparable_pairs)

        ####### Disagreement analysis ###########
        disagreement_directory = (
            "resources/results/survey/uncommon/v4/lovebirds-not-midar-disagreement/"
        )
        disagreement_directory_rerun = "resources/results/survey/uncommon/v4/lovebirds-not-midar-disagreement-rerun/"
        rerun_stable(disagreement_directory, disagreement_directory_rerun)


def get_lovebirds_comparables_pairs(responsive_addresses, unresponsive_addresses):
    comparable_pairs = set()
    uncomparable_pairs = set()
    unresponsive_index = len(responsive_addresses)
    total_address = copy.deepcopy(responsive_addresses)
    total_address.extend(unresponsive_addresses)
    for i in range(0, len(total_address)):
        for j in range(i + 1, len(total_address)):
            if i < unresponsive_index and j < unresponsive_index:
                comparable_pairs.add(frozenset({total_address[i], total_address[j]}))
            else:
                uncomparable_pairs.add(frozenset({total_address[i], total_address[j]}))
    return comparable_pairs, uncomparable_pairs


def get_alias_common_pairs_and_comparable(
    lovebirds_pairs, midar_pairs, comparable_pairs
):
    common_pairs = lovebirds_pairs.intersection(midar_pairs).intersection(
        comparable_pairs
    )
    lovebirds_not_midar_pair = (lovebirds_pairs - midar_pairs).intersection(
        comparable_pairs
    )
    midar_not_lovebirds_pairs = (midar_pairs - lovebirds_pairs).intersection(
        comparable_pairs
    )

    return common_pairs, lovebirds_not_midar_pair, midar_not_lovebirds_pairs


def get_alias_pairs(routers):
    alias_pairs = set()
    for router_name, router in routers.items():
        for i in range(0, len(router)):
            for j in range(i + 1, len(router)):
                alias_pairs.add(frozenset({router[i], router[j]}))
    return alias_pairs


def extract_rate_limiting_addresses(individual_file):
    # Compare only IP's that were seen by both experience. i.e remove those tried by MIDAR and not by Lovebirds.
    df_individual = pd.read_csv(
        individual_file,
        names=base_columns,
        skipinitialspace=True,
        # encoding="utf-8",
        engine="python",
        index_col=False,
    )

    df_evaluated = df_individual[
        (df_individual["probing_rate"] >= 128)
        & (df_individual["probing_rate"] <= 32768)
    ]
    rate_limiting_ip_addresses = list(set(df_evaluated["ip_address"]))
    print("Total IP addrsses: " + str(len(rate_limiting_ip_addresses)))
    print("Total pairs: " + str(combin(len(rate_limiting_ip_addresses), 2)))

    df_responsive = df_evaluated[df_evaluated["loss_rate"] < 1]

    responsive_addresses = list(set(list(df_responsive["ip_address"])))
    df_unresponsive = df_evaluated[df_evaluated["loss_rate"] == 1]
    unresponsive_addresses = list(set(list(df_unresponsive["ip_address"])))

    return rate_limiting_ip_addresses, responsive_addresses, unresponsive_addresses


def compute_comparable_matrix(
    lovebirds_comparable_pairs,
    lovebirds_uncomparable_pairs,
    else_comparable_pairs,
    else_uncomparable_pairs,
):
    # Get the full picture of comparable
    midar_not_lovebird_comparable = else_comparable_pairs.intersection(
        lovebirds_uncomparable_pairs
    )
    lovebirds_not_midar_comparable = lovebirds_comparable_pairs.intersection(
        else_uncomparable_pairs
    )

    midar_lovebirds_comparable = else_comparable_pairs.intersection(
        lovebirds_comparable_pairs
    )

    midar_lovebirds_not_comparable = else_uncomparable_pairs.intersection(
        lovebirds_uncomparable_pairs
    )

    n_total = (
        len(midar_lovebirds_comparable)
        + len(midar_lovebirds_not_comparable)
        + len(lovebirds_not_midar_comparable)
        + len(midar_not_lovebird_comparable)
    )

    print(
        "Midar and Lovebirds comparable: "
        + str(len(midar_lovebirds_comparable))
        + ", "
        + str(len(midar_lovebirds_comparable) / n_total)
    )
    print(
        "Midar and Lovebirds not comparable: "
        + str(len(midar_lovebirds_not_comparable))
        + ", "
        + str(len(midar_lovebirds_not_comparable) / n_total)
    )
    print(
        "Midar not Lovebirds comparable: "
        + str(len(midar_not_lovebird_comparable))
        + ", "
        + str(len(midar_not_lovebird_comparable) / n_total)
    )
    print(
        "Not Midar but Lovebirds comparable: "
        + str(len(lovebirds_not_midar_comparable))
        + ", "
        + str(len(lovebirds_not_midar_comparable) / n_total)
    )

    return midar_lovebirds_comparable


def compare_v6(
    speedtrap_classification_file,
    individual_file,
    rate_limiting_routers,
    speedtrap_alias_pairs_file,
    is_only_responsive,
):
    rate_limiting_ip_addresses, responsive_addresses, unresponsive_addresses = extract_rate_limiting_addresses(
        individual_file
    )

    #
    # prefix_difference(random_pairs, socket.AF_INET6, ofile="/Users/kevinvermeulen/PycharmProjects/"
    #                                                       "icmp-rate-limiting-paper/resources/"
    #                                                       "prefix_difference/prefix_difference_random_v6")
    # Extract paris from midar and lovebirds results.
    lovebirds_alias_pairs = get_alias_pairs(rate_limiting_routers)
    speedtrap_alias_pairs = extract_speedtrap_alias_pairs(
        rate_limiting_ip_addresses, speedtrap_alias_pairs_file
    )

    # fingerprinting_speedtrap_pairs = {}
    # i = 0
    # for pair in speedtrap_alias_pairs:
    #     fingerprinting_speedtrap_pairs[i] = list(pair)
    #     i += 1

    fingerprinting_return_ttl(lovebirds_alias_pairs, "6")
    fingerprinting_return_ttl(speedtrap_alias_pairs, "6")

    union_alias_pair = lovebirds_alias_pairs.union(speedtrap_alias_pairs)
    print(
        "Total number of alias pairs found by Lovebirds U Speedtrap: "
        + str(len(union_alias_pair))
    )
    print(
        "Total number of alias pairs found by Lovebirds: "
        + str(len(lovebirds_alias_pairs))
    )
    print(
        "Total number of alias pairs found by speedtrap: "
        + str(len(speedtrap_alias_pairs))
    )

    if is_only_responsive:
        unresponsive_addresses = []
    speedtrap_classification = compute_speedtrap_classification(
        rate_limiting_ip_addresses, speedtrap_classification_file
    )
    speedtrap_comparable_pairs, speedtrap_uncomparable_pairs = compute_speedtrap_comparable_pairs(
        rate_limiting_ip_addresses, speedtrap_classification
    )
    print("Comparable pairs for Speedtrap: " + str(len(speedtrap_comparable_pairs)))
    print(
        "Not comparable pairs for Speedtrap: " + str(len(speedtrap_uncomparable_pairs))
    )
    print(
        "Total pairs for Speedtrap: "
        + str(len(speedtrap_comparable_pairs) + len(speedtrap_uncomparable_pairs))
    )

    lovebirds_comparable_pairs, lovebirds_uncomparable_pairs = get_lovebirds_comparables_pairs(
        responsive_addresses, unresponsive_addresses
    )

    print("Comparable pairs for Lovebirds: " + str(len(lovebirds_comparable_pairs)))
    print(
        "Not comparable pairs for Lovebirds: " + str(len(lovebirds_uncomparable_pairs))
    )
    print(
        "Total pairs for Lovebirds: "
        + str(len(lovebirds_comparable_pairs) + len(lovebirds_uncomparable_pairs))
    )

    speedtrap_lovebirds_comparable = compute_comparable_matrix(
        lovebirds_comparable_pairs,
        lovebirds_uncomparable_pairs,
        speedtrap_comparable_pairs,
        speedtrap_uncomparable_pairs,
    )

    alias_common_pairs, alias_lovebirds_not_midar_pairs, alias_midar_not_lovebird_pairs = get_alias_common_pairs_and_comparable(
        lovebirds_alias_pairs, speedtrap_alias_pairs, speedtrap_lovebirds_comparable
    )

    # with open("resources/results/survey/uncommon/v6/lovebirds_not_speedtrap_pairs.pairs","w") as f:
    #     for pair in alias_lovebirds_not_midar_pairs:
    #         for ip in pair:
    #             f.write(ip + " ")
    #         f.write("\n")

    print(
        "Alias common: "
        + str(len(alias_common_pairs))
        + ", "
        + str(len(alias_common_pairs) / len(speedtrap_lovebirds_comparable))
    )
    print(
        "Speedtrap not lovebirds alias: "
        + str(len(alias_midar_not_lovebird_pairs))
        + ", "
        + str(len(alias_midar_not_lovebird_pairs) / len(speedtrap_lovebirds_comparable))
    )
    print(
        "Lovebirds not Speedtrap alias: "
        + str(len(alias_lovebirds_not_midar_pairs))
        + ", "
        + str(
            len(alias_lovebirds_not_midar_pairs) / len(speedtrap_lovebirds_comparable)
        )
    )


def fingerprinting_return_ttl(pairs, ip_version):
    # Compare the fingerprinting
    not_same_fingerprinting = 0
    not_in_classification = 0
    with open("resources/results/survey/fingerprinting_" + ip_version + ".json") as f:
        fingerprinting = json.load(f)

        for p in pairs:
            # Look for the closest power of 2
            ping_fingerprints = []
            ttl_exceeded_fingerprints = []
            skip = False
            for ip in p:
                if ip not in fingerprinting or len(fingerprinting[ip]) < 2:
                    skip = True
                    break
                else:
                    ping_fingerprints.append(fingerprinting[ip][0])
                    ttl_exceeded_fingerprints.append(fingerprinting[ip][1])
            if skip:
                not_in_classification += 1
                continue

            initial_ping_ttl = []
            initial_ttl_exceeded_ttl = []
            for i in range(5, 8):

                for p_fing in ping_fingerprints:
                    if 2 ** i < p_fing <= 2 ** (i + 1):
                        initial_ping_ttl.append(i)
                        break
                for ttl_ex_fing in ttl_exceeded_fingerprints:
                    if 2 ** i < ttl_ex_fing <= 2 ** (i + 1):
                        initial_ttl_exceeded_ttl.append(i)
                        break

            if len(set(initial_ping_ttl)) > 1 or len(set(initial_ttl_exceeded_ttl)) > 1:
                not_same_fingerprinting += 1
    print("Pairs not in  fingerprinting: " + str(not_in_classification))
    print("Pairs with not same fingerprinting: " + str(not_same_fingerprinting))


import json


def compare_v4(
    midar_classification_file,
    individual_file,
    rate_limiting_routers,
    midar_routers,
    is_only_responsive,
):
    """
    This function compares the number of pairs that are comparable from MIDAR, Lovebirds and Speedtrap point of view.
    :return:
    """

    rate_limiting_ip_addresses, responsive_addresses, unresponsive_addresses = extract_rate_limiting_addresses(
        individual_file
    )

    random_pairs = []
    while len(random_pairs) < 4130:
        index1 = random.randint(0, len(rate_limiting_ip_addresses) - 1)
        index2 = random.randint(0, len(rate_limiting_ip_addresses) - 1)
        random_pairs.append(
            [rate_limiting_ip_addresses[index1], rate_limiting_ip_addresses[index2]]
        )
    fingerprinting_return_ttl(random_pairs, "4")
    # random_pairs = {}
    # while len(random_pairs) < 4130:
    #     index1 = random.randint(0, len(rate_limiting_ip_addresses) - 1)
    #     index2 = random.randint(0, len(rate_limiting_ip_addresses) - 1)
    #     random_pairs[len(random_pairs)] = [rate_limiting_ip_addresses[index1], rate_limiting_ip_addresses[index2]]

    # prefix_difference(random_pairs, socket.AF_INET, ofile="/Users/kevinvermeulen/PycharmProjects/"
    #                                                       "icmp-rate-limiting-paper/resources/"
    #                                                       "prefix_difference/prefix_difference_random_v4")
    if is_only_responsive:
        unresponsive_addresses = []
    # Two possibilities. Alias, or not alias.

    # Extract paris from midar and lovebirds results.
    lovebirds_alias_pairs = get_alias_pairs(rate_limiting_routers)
    fingerprinting_return_ttl(lovebirds_alias_pairs, "4")

    full_midar_alias_pairs = get_alias_pairs(midar_routers)
    print(
        "Total number of alias pairs found by MIDAR test set: "
        + str(len(full_midar_alias_pairs))
    )
    midar_alias_pairs = set()
    for pair in full_midar_alias_pairs:
        if len(pair.intersection(rate_limiting_ip_addresses)) == 2:
            midar_alias_pairs.add(pair)

    # fingerprinting_return_ttl(midar_alias_pairs)

    union_alias_pair = lovebirds_alias_pairs.union(midar_alias_pairs)

    print(
        "Total number of alias pairs found by Lovebirds U MIDAR: "
        + str(len(union_alias_pair))
    )
    print(
        "Total number of alias pairs found by Lovebirds: "
        + str(len(lovebirds_alias_pairs))
    )
    print("Total number of alias pairs found by Midar: " + str(len(midar_alias_pairs)))
    unique_alias_ip = set()
    for pair in union_alias_pair:
        for ip in pair:
            unique_alias_ip.add(ip)

    print("Total number of ip in alias sets:" + str(len(unique_alias_ip)))
    """
    Compute Lovebirds comparable
    """
    lovebirds_comparable_pairs, lovebirds_uncomparable_pairs = get_lovebirds_comparables_pairs(
        responsive_addresses, unresponsive_addresses
    )

    print("Comparable pairs for Lovebirds: " + str(len(lovebirds_comparable_pairs)))
    print(
        "Not comparable pairs for Lovebirds: " + str(len(lovebirds_uncomparable_pairs))
    )
    print(
        "Total pairs for Lovebirds: "
        + str(len(lovebirds_comparable_pairs) + len(lovebirds_uncomparable_pairs))
    )

    """
    Compute MIDAR comparable 
    """
    # Compute the number of pairs that are comparable

    midar_classification = compute_midar_classification(
        rate_limiting_ip_addresses, midar_classification_file
    )

    midar_comparable_pairs, midar_uncomparable_pairs = compute_midar_comparables_pairs(
        rate_limiting_ip_addresses, midar_classification
    )

    print("Comparable pairs for MIDAR: " + str(len(midar_comparable_pairs)))
    print("Not comparable pairs for MIDAR: " + str(len(midar_uncomparable_pairs)))
    print(
        "Total pairs for MIDAR: "
        + str(len(midar_uncomparable_pairs) + len(midar_comparable_pairs))
    )

    """
    Compute Matrix of comparable
    """
    # Get the full picture of comparable
    midar_lovebirds_comparable = compute_comparable_matrix(
        lovebirds_comparable_pairs,
        lovebirds_uncomparable_pairs,
        midar_comparable_pairs,
        midar_uncomparable_pairs,
    )

    """
    Now focus on common pairs.
    """

    alias_common_pairs, alias_lovebirds_not_midar_pairs, alias_midar_not_lovebird_pairs = get_alias_common_pairs_and_comparable(
        lovebirds_alias_pairs, midar_alias_pairs, midar_lovebirds_comparable
    )

    reasons = {"udp": 0, "icmp": 0, "multiple_proto": 0}
    for pair in alias_midar_not_lovebird_pairs:
        probing_methods = set()
        for e in pair:
            probing_method = midar_classification[e]["probing_method"]
            probing_methods.add(probing_method)
        if len(probing_methods) == 1:
            reasons[probing_methods.pop()] += 1
        else:
            reasons["multiple_proto"] += 1
    print(reasons)
    print(
        "Alias common: "
        + str(len(alias_common_pairs))
        + ", "
        + str(len(alias_common_pairs) / len(midar_lovebirds_comparable))
    )
    print(
        "Midar not lovebirds alias: "
        + str(len(alias_midar_not_lovebird_pairs))
        + ", "
        + str(len(alias_midar_not_lovebird_pairs) / len(midar_lovebirds_comparable))
    )
    print(
        "Lovebirds not Midar alias: "
        + str(len(alias_lovebirds_not_midar_pairs))
        + ", "
        + str(len(alias_lovebirds_not_midar_pairs) / len(midar_lovebirds_comparable))
    )


def clear_unresponsive(routers, individual_file):
    rate_limiting_ip_addresses, responsive_addresses, unresponsive_addresses = extract_rate_limiting_addresses(
        individual_file
    )
    n_unresponsive = 0
    for ip in unresponsive_addresses:
        for router_name, router in routers.items():
            if ip in router:
                router.remove(ip)
                n_unresponsive += 1
    print("Number of unrepsonsive addresses", n_unresponsive)


def extract_speedtrap_routers(speedtrap_file):

    routers = []
    with open(speedtrap_file) as f:
        for line in f:
            router = []
            ips = line.split()
            for ip in ips:
                router.append(ip.upper())
            routers.append(router)
    return routers


def extract_midar_routers(midar_file):
    routers = []
    with open(midar_file) as alias_file:
        # Parse the file
        router = []
        for line in alias_file.readlines():
            if line.startswith("#"):
                if len(router) > 0:
                    routers.append(copy.deepcopy(router))
                    del router[:]
                continue
            line = line.strip("\n")
            ip = line
            router.append(ip)
        if len(router) > 0:
            routers.append(copy.deepcopy(router))
            del router[:]

    return {str(i): routers[i] for i in range(0, len(routers))}


def prefix_difference(routers, af_family, ofile):

    prefix_difference_list = []

    for name, router in routers.items():

        for i in range(0, len(router)):
            ip1_s = ipaddress.ip_address(router[i].strip())
            ip1_string = ip1_s.exploded
            ip1 = socket.inet_pton(af_family, ip1_string)
            bit_list = []
            for byte in ip1:
                b = format(byte, "#010b")
                for bit in b[2:]:
                    bit_list.append(bit)
            for j in range(i + 1, len(router)):
                ip2_s = ipaddress.ip_address(router[j].strip())
                ip2_string = ip2_s.exploded
                ip2 = socket.inet_pton(af_family, ip2_string)
                bit_list2 = []
                for byte in ip2:

                    b = format(byte, "#010b")
                    for bit in b[2:]:
                        bit_list2.append(bit)

                for k in range(0, len(bit_list)):
                    if bit_list[k] != bit_list2[k]:
                        prefix_difference_list.append(k)

    with open(ofile, "w") as f:
        for prefix_diff in prefix_difference_list:
            f.write(str(prefix_diff) + "\n")

    return prefix_difference_list


if __name__ == "__main__":
    # compare_tools()
    # analyse_rerun()
    imc_paper_root = (
        "/Users/kevinvermeulen/PycharmProjects/icmp-rate-limiting-paper/resources/"
    )
    # Only consider responsive addresses
    is_only_responsive = True
    is_v4 = True
    is_v6 = False
    """
    Survey
    """
    is_survey = False
    if is_survey:
        if is_v4:
            node = "ple41.planet-lab.eu"
            rate_limiting_routers_dir = (
                "resources/results/survey/aliases/v4/" + node + "/"
            )
            midar_classification_file = "resources/results/survey/target-summary.txt"
            individual_file = "resources/results/survey/survey_individual4"

            rate_limiting_routers = extract_routers_evaluation(
                rate_limiting_routers_dir
            )
            # prefix_difference(rate_limiting_routers, af_family=socket.AF_INET, ofile=imc_paper_root + "prefix_difference/prefix_difference_ltdltd_v4")
            midar_routers = extract_routers_by_node(
                "/home/kevin/mda-lite-v6-survey/resources/midar/batch2/routers/"
            )[node]
            # prefix_difference(midar_routers, af_family=socket.AF_INET, ofile=imc_paper_root + "prefix_difference/prefix_difference_midar")
            compare_v4(
                midar_classification_file,
                individual_file,
                rate_limiting_routers,
                midar_routers,
                is_only_responsive,
            )

        if is_v6:
            node = "ple2.planet-lab.eu"
            rate_limiting_routers_dir = (
                "resources/results/survey/aliases/v6/" + node + "/"
            )
            speedtrap_classification_file = (
                "resources/results/survey/speedtrap.classification"
            )
            individual_file = "resources/results/survey/survey_individual6"

            speedtrap_pairs_file = "resources/results/survey/speedtrap.pairs"

            rate_limiting_routers = extract_routers_evaluation(
                rate_limiting_routers_dir
            )

            compare_v6(
                speedtrap_classification_file,
                individual_file,
                rate_limiting_routers,
                speedtrap_pairs_file,
                is_only_responsive,
            )

            # Select 4130 IPv6 addresses
    """
        Internet 2
    """

    is_internet2 = True
    if is_internet2:
        if is_v4:
            # rate_limiting_nodes = ["cse-yellow.cse.chalmers.se", "ple1.cesnet.cz",
            #                        "planetlab1.cs.vu.nl",
            #                        "ple41.planet-lab.eu"]
            rate_limiting_nodes = ["ple2.planet-lab.eu"]
            midar_classification_file = (
                "resources/results/internet2/midar-analysis/target-summary.txt"
            )
            individual_file = "resources/results/internet2/individual/ple2.planet-lab.eu_internet2_individual4"

            ground_truth_routers = extract_routers(
                "resources/internet2/ground-truth/routers/v4/"
            )

            if is_only_responsive:
                clear_unresponsive(ground_truth_routers, individual_file)
            ground_truth_alias_pairs = get_alias_pairs(ground_truth_routers)
            print("Ground truth alias pairs: " + str(len(ground_truth_alias_pairs)))
            routers = []
            for node in rate_limiting_nodes:
                routers_node = extract_routers(
                    "resources/results/internet2/aliases/v4/" + node + "/"
                )
                copy_router_node = copy.deepcopy(routers_node)
                routers_node_final, tp, fp = clean_false_positives(
                    copy_router_node, ground_truth_routers
                )
                fn = len(ground_truth_alias_pairs) - tp
                print("True positives", tp)
                print("False positives", fp)
                print(node, precision(tp, fp))
                print(node, recall(tp, fn))
                # routers.update(extract_routers("resources/results/internet2/aliases/ple41.planet-lab.eu/"))
                # routers_tc = []
                print(node, len(routers_node))
                for router_name, router in routers_node.items():
                    routers.append(set(router))
            routers = transitive_closure(routers)
            print("Number of distinct routers: " + str(len(routers)))
            routers_tc_dic = {i: list(routers[i]) for i in range(0, len(routers))}
            routers_tc_dic_final, tp, fp = clean_false_positives(
                routers_tc_dic, ground_truth_routers
            )
            fn = len(ground_truth_alias_pairs) - tp
            print(precision(tp, fp))
            print(recall(tp, fn))

            # midar_nodes = ["ple1.cesnet.cz",
            #                "ple41.planet-lab.eu",
            #                "cse-yellow.chalmers.cse.se",
            #                "planetlab1.cs.vu.nl",
            #                # "planetlab2.informatik.uni-goettingen.de",
            #                ]
            midar_nodes = ["ple2.planet-lab.eu"]
            midar_routers = []
            for node in midar_nodes:
                midar_routers_node = extract_midar_routers(
                    "resources/internet2/midar/tc"
                )
                for router_name, router in midar_routers_node.items():
                    midar_routers.append(set(router))
            midar_routers = transitive_closure(midar_routers)
            print("Number of distinct routers: " + str(len(midar_routers)))
            midar_routers_tc_dic = {
                i: list(midar_routers[i]) for i in range(0, len(midar_routers))
            }
            midar_routers_tc_dic, tp, fp = clean_false_positives(
                midar_routers_tc_dic, ground_truth_routers
            )
            fn = len(ground_truth_alias_pairs) - tp
            print(precision(tp, fp))
            print(recall(tp, fn))

            compare_v4(
                midar_classification_file,
                individual_file,
                routers_tc_dic,
                midar_routers_tc_dic,
                is_only_responsive,
            )

            union_routers = copy.deepcopy(routers)
            union_routers.extend(midar_routers)
            union_routers = transitive_closure(union_routers)
            print("Number of distinct routers: " + str(len(union_routers)))
            midar_routers_tc_dic = {
                i: list(union_routers[i]) for i in range(0, len(union_routers))
            }
            midar_routers_tc_dic, tp, fp = clean_false_positives(
                midar_routers_tc_dic, ground_truth_routers
            )
            fn = len(ground_truth_alias_pairs) - tp
            print(precision(tp, fp))
            print(recall(tp, fn))

        if is_v6:
            individual_file = "resources/results/internet2/individual/ple2.planet-lab.eu_internet2_individual6"

            ground_truth_routers = extract_routers(
                "resources/internet2/ground-truth/routers/v6/"
            )

            speedtrap_classification_file = (
                "resources/internet2/speedtrap/speedtrap_classification_internet2"
            )

            if is_only_responsive:
                clear_unresponsive(ground_truth_routers, individual_file)

            rate_limiting_nodes = ["ple2.planet-lab.eu"]

            ground_truth_alias_pairs = get_alias_pairs(ground_truth_routers)
            print("Ground truth alias pairs: " + str(len(ground_truth_alias_pairs)))
            routers = []
            precisions = []
            for node in rate_limiting_nodes:
                routers_node = extract_routers(
                    "resources/results/internet2/aliases/v6/" + node + "/"
                )
                routers_node, tp, fp = clean_false_positives(
                    routers_node, ground_truth_routers
                )
                print(node, precision(tp, fp))
                print("True positives", tp)
                print("False positives", fp)
                precisions.append(precision(tp, fp))

                # routers.update(extract_routers("resources/results/internet2/aliases/ple41.planet-lab.eu/"))
                # routers_tc = []
                for router_name, router in routers_node.items():
                    routers.append(set(router))
            routers = transitive_closure(routers)
            routers_tc_dic = {i: list(routers[i]) for i in range(0, len(routers))}

            print("Mean precision over ple nodes:" + str(np.mean(precisions)))

            evaluate(
                routers_tc_dic, ground_truth_routers, len(ground_truth_alias_pairs)
            )

            compute_speedtrap_classification([], speedtrap_classification_file)

    """
    Switch
    """
    from Validation.switch_ground_truth import (
        extract_evaluation_gt_switch,
        extract_gt_from_yaml,
    )

    is_switch = False
    if is_switch:
        switch_gt_file = "resources/SWITCH/ground-truth.yml"
        ground_truth_routers_v4, ground_truth_routers_v6 = extract_evaluation_gt_switch(
            extract_gt_from_yaml(switch_gt_file)
        )
        if is_v4:
            # rate_limiting_nodes = ["cse-yellow.cse.chalmers.se", "ple1.cesnet.cz",
            #                        "planetlab1.cs.vu.nl",
            #                        "ple41.planet-lab.eu"]
            rate_limiting_nodes = ["ple2.planet-lab.eu"]
            midar_classification_file = (
                "resources/results/switch/midar/target-summary.txt"
            )
            individual_file = "resources/results/switch/individual/ple2.planet-lab.eu_switch_individual4"

            if is_only_responsive:
                clear_unresponsive(ground_truth_routers_v4, individual_file)
            ground_truth_alias_pairs = get_alias_pairs(ground_truth_routers_v4)
            print("Ground truth alias pairs: " + str(len(ground_truth_alias_pairs)))
            routers = []
            for node in rate_limiting_nodes:
                routers_node = extract_routers(
                    "resources/results/switch/aliases/v4/" + node + "/"
                )
                copy_router_node = copy.deepcopy(routers_node)
                routers_node_final, tp, fp = clean_false_positives(
                    copy_router_node, ground_truth_routers_v4
                )
                fn = len(ground_truth_alias_pairs) - tp
                print("True positives", tp)
                print("False positives", fp)
                print(node, precision(tp, fp))
                print(node, recall(tp, fn))
                # routers.update(extract_routers("resources/results/internet2/aliases/ple41.planet-lab.eu/"))
                # routers_tc = []
                print(node, len(routers_node))
                for router_name, router in routers_node.items():
                    routers.append(set(router))
            routers = transitive_closure(routers)
            print("Number of distinct routers: " + str(len(routers)))
            routers_tc_dic = {i: list(routers[i]) for i in range(0, len(routers))}
            routers_tc_dic_final, tp, fp = clean_false_positives(
                routers_tc_dic, ground_truth_routers_v4
            )
            fn = len(ground_truth_alias_pairs) - tp
            print("LtdLtd", precision(tp, fp))
            print("LtdLtd", recall(tp, fn))

            # midar_nodes = ["ple1.cesnet.cz",
            #                "ple41.planet-lab.eu",
            #                "cse-yellow.chalmers.cse.se",
            #                "planetlab1.cs.vu.nl",
            #                # "planetlab2.informatik.uni-goettingen.de",
            #                ]
            midar_nodes = ["ple2.planet-lab.eu"]
            midar_routers = []
            for node in midar_nodes:
                midar_routers_node = extract_midar_routers(
                    "resources/results/switch/midar/tc"
                )
                for router_name, router in midar_routers_node.items():
                    midar_routers.append(set(router))
            midar_routers = transitive_closure(midar_routers)
            print("Number of distinct routers: " + str(len(midar_routers)))
            midar_routers_tc_dic = {
                i: list(midar_routers[i]) for i in range(0, len(midar_routers))
            }
            midar_routers_tc_dic, tp, fp = clean_false_positives(
                midar_routers_tc_dic, ground_truth_routers_v4
            )
            fn = len(ground_truth_alias_pairs) - tp

            print("Midar", precision(tp, fp))
            print("Midar", recall(tp, fn))
            print("True positives", tp)
            print("False positives", fp)

            # compare_v4(midar_classification_file, individual_file, routers_tc_dic, midar_routers_tc_dic, is_only_responsive)

            union_routers = copy.deepcopy(routers)
            union_routers.extend(midar_routers)
            union_routers = transitive_closure(union_routers)
            print("Number of distinct routers: " + str(len(union_routers)))
            midar_routers_tc_dic = {
                i: list(union_routers[i]) for i in range(0, len(union_routers))
            }
            midar_routers_tc_dic, tp, fp = clean_false_positives(
                midar_routers_tc_dic, ground_truth_routers_v4
            )
            fn = len(ground_truth_alias_pairs) - tp
            print("Union", precision(tp, fp))
            print("Union", recall(tp, fn))

        if is_v6:
            individual_file = "resources/results/switch/individual/ple2.planet-lab.eu_switch_individual6"

            speedtrap_classification_file = (
                "resources/internet2/speedtrap/speedtrap_classification_switch"
            )
            speedtrap_tc_file = "resources/results/switch/speedtrap/switch_speedtrap.tc"
            if is_only_responsive:
                clear_unresponsive(ground_truth_routers_v6, individual_file)

            rate_limiting_nodes = ["ple2.planet-lab.eu"]

            ground_truth_alias_pairs = get_alias_pairs(ground_truth_routers_v6)
            print("Ground truth alias pairs: " + str(len(ground_truth_alias_pairs)))
            routers = []
            # precisions = []
            for node in rate_limiting_nodes:
                routers_node = extract_routers(
                    "resources/results/switch/aliases/v6/" + node + "/"
                )
                routers_node, tp, fp = clean_false_positives(
                    routers_node, ground_truth_routers_v6
                )
                print("LtdLtd", precision(tp, fp))
                print("LtdLtd", recall(tp, len(ground_truth_alias_pairs) - tp))
                print("True positives", tp)
                print("False positives", fp)
                for router_name, router in routers_node.items():
                    routers.append(set(router))
                routers = transitive_closure(routers)
                routers_tc_dic = {i: list(routers[i]) for i in range(0, len(routers))}

            #
            #     # routers.update(extract_routers("resources/results/internet2/aliases/ple41.planet-lab.eu/"))
            #     # routers_tc = []

            #

            #
            # # print("Mean precision over ple nodes:" + str(np.mean(precisions)))
            #

            # Extract speedtrap routers
            speedtrap_routers = extract_speedtrap_routers(speedtrap_tc_file)
            speedtrap_routers_tc_dic = {
                i: speedtrap_routers[i] for i in range(0, len(speedtrap_routers))
            }
            evaluate(
                speedtrap_routers_tc_dic,
                ground_truth_routers_v6,
                len(ground_truth_alias_pairs),
            )

            union_routers = copy.deepcopy(routers)
            union_routers.extend([set(router) for router in speedtrap_routers])
            union_routers = transitive_closure(union_routers)
            print("Number of distinct routers: " + str(len(union_routers)))
            union_routers_tc_dic = {
                i: list(union_routers[i]) for i in range(0, len(union_routers))
            }
            union_routers_tc_dic, tp, fp = clean_false_positives(
                union_routers_tc_dic, ground_truth_routers_v6
            )
            fn = len(ground_truth_alias_pairs) - tp
            print("Union", precision(tp, fp))
            print("Union", recall(tp, fn))
            # compute_speedtrap_classification([], speedtrap_classification_file)
