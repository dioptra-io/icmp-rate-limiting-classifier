import re
import os

def compute_midar_routers(midar_dir):
    """ Parsing midar-noconflicts.sets file(s)
        Returns a list of sets
        Each set corresponds to the IPs of one router
    """

    list_routers = []

    for hitlist in os.listdir(midar_dir):

        if os.path.isdir(midar_dir + hitlist):
            continue

        router = set()

        with open(midar_dir + hitlist, "r") as f:
            lines = f.readlines()

        for line in lines:
            if re.match('# end', line) is None:
                if re.match('# set', line):
                    if len(router) > 0:
                        list_routers.append(router)
                    router = set()
                if re.match('^[0-9]', line):
                    router.add(line.strip())
            else:
                list_routers.append(router)

    return list_routers

def extract_midar_routers(midar_path):
    routers = []
    for router_file in os.listdir(midar_path):
        router = []
        with open(midar_path + router_file) as r:
            for ip in r.readlines():
                ip = ip.strip("\n")
                router.append(ip)
        routers.append(router)
    return routers

def transitive_closure(routers):
    # Apply transitive closure
    old_len_tc = len(routers)
    new_len = 0
    # End condition
    while old_len_tc != new_len:
        old_len_tc = new_len
        tc_routers = []
        # Now apply transitive closure
        merged = []
        for i in range(0, len(routers)):
            if routers[i] in merged:
                continue
            for j in range(i + 1, len(routers)):
                if len(routers[i].intersection(routers[j])) > 0:
                    routers[i].update(routers[j])
                    if routers[j] not in merged:
                        merged.append(routers[j])
            tc_routers.append(routers[i])

        routers = tc_routers
        new_len = len(tc_routers)
    final_routers = tc_routers
    return final_routers


def get_label(midar_routers, candidates):

    found_candidate_in_routers_n = 0

    for router in midar_routers:
        intersection = set(candidates).intersection(set(router))

        if len(intersection) == len(set(candidates)):
            # If the candidates is within a router, positive label
            return "P"

        if len(intersection) == 1:
            found_candidate_in_routers_n += 1

    if found_candidate_in_routers_n == 2:
        # If the candidates are in two different sets, negative label
        return "N"

    if found_candidate_in_routers_n == 1:
        # If only one of the candidates is found in sets, unknown label
        return "U"

    return "U"
