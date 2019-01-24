from Validation.midar import extract_routers
from Algorithms.algorithms import transitive_closure
import json
import copy
import numpy as np

def find_corresponding_router(ip, ground_truth_routers):
    for router_name, router in ground_truth_routers.items():
        for ip_router in router:
            if ip_router == ip:
                return router_name, router
    return None

def combin(n, k):
    """Number of combinations C(n,k)"""
    if k > n//2:
        k = n-k
    x = 1
    y = 1
    i = n-k+1
    while i <= n:
        x = (x*i)//y
        y += 1
        i += 1
    return x

def precision(tp, fp):
    return tp / (tp + fp)

def clean_false_positives(routers, ground_truth_routers):
    tp = 0
    fp = 0
    to_remove_ips = {}
    for router_name, router in routers.items():
        to_remove_ips[router_name] = set()
        for i in range(0, len(router)):
            router_i, ips_i = find_corresponding_router(router[i], ground_truth_routers)
            for j in range(i+1, len(router)):
                router_j, ips_j = find_corresponding_router(router[j], ground_truth_routers)
                if router_i != router_j:
                    fp += 1
                    if len(set(router).intersection(set(ips_i))) > len(set(router).intersection(set(ips_j))):
                        to_remove_ips[router_name].add(router[j])
                    else:
                        to_remove_ips[router_name].add(router[i])
                else:
                    tp += 1
    for router_name, ips in to_remove_ips.items():
        for ip in ips:
            routers[router_name].remove(ip)
    return routers, tp, fp

def evaluate(routers, ground_truth_routers):

    # pirate_ips = ["64.57.25.107", "64.57.25.106"]

    ground_truth_routers = copy.deepcopy(ground_truth_routers)
    routers, tp, fp = clean_false_positives(routers, ground_truth_routers)



    # with open("resources/results/internet2/unresponsive/ple41.planet-lab.eu_internet2_unresponsive6.json") as unresponsive_ips_fp:
    #     unresponsive_ips = set(json.load(unresponsive_ips_fp))
    #     # unresponsive_ips.update(set(pirate_ips))
    # # Remove the unresponsive addresses from all vantage points
    # for file_name, ips in ground_truth_routers.items():
    #     for ip in unresponsive_ips:
    #         if ip in ips:
    #             ips.remove(ip)




    """Stats"""
    res_gt = 0

    for router, ips in sorted(ground_truth_routers.items()):
        res_gt += combin(len(ips), 2)


    print ("Ground Truth pairs: ")
    print (res_gt)
    print ("Rate Limiting pairs :")
    print (tp)

    print ("Recall :")
    print (float(tp)/res_gt)

if __name__ == "__main__":

    is_internet2 = True
    is_survey = False
    ip_version = "6"
    '''
    Internet2 V4 evaluation
    '''
    if is_internet2 and ip_version == "4":
        ground_truth_routers = extract_routers("/home/kevin/mda-lite-v6-survey/resources/internet2/routers/v4/")



        rate_limiting_nodes = ["ple41.planet-lab.eu", "cse-yellow.cse.chalmers.se", "ple1.cesnet.cz", "planetlab1.cs.vu.nl"]

        routers = []
        precisions = []
        for node in rate_limiting_nodes:
            routers_node = extract_routers("resources/results/internet2/aliases/v4/" + node + "/")
            routers_node, tp, fp = clean_false_positives(routers_node, ground_truth_routers)
            print (node, precision(tp, fp))
            precisions.append(precision(tp, fp))


        # routers.update(extract_routers("resources/results/internet2/aliases/ple41.planet-lab.eu/"))
        # routers_tc = []
            for router_name, router in routers_node.items():
                routers.append(set(router))
        routers = transitive_closure(routers)
        routers_tc_dic = {i:list(routers[i]) for i in range(0, len(routers))}

        print("Mean precision over ple nodes:" + str(np.mean(precisions)))

        midar_nodes = ["ple1.cesnet.cz",
                       "ple41.planet-lab.eu",
                       # "planetlab2.informatik.uni-goettingen.de",
                       "cse-yellow.chalmers.cse.se",
                       "planetlab1.cs.vu.nl"]
        midar_routers = []
        for node in midar_nodes:
            midar_routers_node = extract_routers("/home/kevin/mda-lite-v6-survey/resources/internet2/midar/v4/" + node + "/")
            for router_name, router in midar_routers_node.items():
                midar_routers.append(set(router))
        midar_routers = transitive_closure(midar_routers)
        midar_routers_tc_dic = {i:list(midar_routers[i]) for i in range(0, len(midar_routers))}



        evaluate(midar_routers_tc_dic, ground_truth_routers)
        evaluate(routers_tc_dic, ground_truth_routers)

    if is_internet2 and ip_version == "6":
        ground_truth_routers = extract_routers("/home/kevin/mda-lite-v6-survey/resources/internet2/routers/v6/")

        rate_limiting_nodes = ["ple41.planet-lab.eu"]

        routers = []
        precisions = []
        for node in rate_limiting_nodes:
            routers_node = extract_routers("resources/results/internet2/aliases/v6/" + node + "/")
            routers_node, tp, fp = clean_false_positives(routers_node, ground_truth_routers)
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
