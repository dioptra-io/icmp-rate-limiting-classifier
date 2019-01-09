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

def extract_routers(routers_dir):
    routers = {}
    for router_file in os.listdir(routers_dir):
        routers[router_file] = []
        with open(routers_dir + router_file) as router_fd:
            for ip in router_fd.readlines():
                ip = ip.strip()
                if ip.startswith("#"):
                    continue
                routers[router_file].append(ip)
    return routers


def extract_routers_by_node(routers_dir):
    all_routers = extract_routers(routers_dir)

    routers_by_node = {}
    for router_file, router in all_routers.items():
        node = router_file.split("_")[0]
        if not node in routers_by_node:
            routers_by_node[node] = {}
        routers_by_node[node][router_file] = router
    return routers_by_node


def internet2_routers(routers_dir, nodes):
    all_routers = extract_routers(routers_dir)
    routers_by_node = {}
    for node in nodes:
        routers_by_node[node] = {}
        for router_file, router in all_routers.items():
            routers_by_node[node][router_file] = router
    return routers_by_node


def find_router_from_ip(ip, routers):
    for router_name, ips in routers.items():
        if ip in ips:
            return router_name
    return None
def set_router_labels(df, ground_truth_routers, candidates, witnesses):

    # First candidate is router 1
    used_routers = []

    router_0 = find_router_from_ip(candidates[0], ground_truth_routers)
    used_routers.append(router_0)
    if router_0 is None:
        return "U"
    else:
        df["".join(["label_", "c", str(0)])] = 1

    for i in range(1, len(candidates)):
        router_i = find_router_from_ip(candidates[i], ground_truth_routers)
        if router_i is not None:
            used_routers.append(router_i)
        if router_i != router_0:
            df["".join(["label_", "c", str(i)])] = 0
        else:
            df["".join(["label_", "c", str(i)])] = 1

    for i in range(0, len(witnesses)):
        router_i = find_router_from_ip(witnesses[i], ground_truth_routers)
        if router_i in used_routers:
            return "U"
        if router_i != router_0:
            df["".join(["label_", "w", str(i)])] = 0
