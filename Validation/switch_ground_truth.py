import yaml
import re
import pprint
from Files.utils import ipv4_regex, ipv6_regex

def extract_gt_from_yaml(gt_file):
    with open(gt_file) as f:
        switch_gt = yaml.load(f)

    return switch_gt

def extract_evaluation_gt_switch(switch_gt):
    ips4 = set()

    ips6 = set()
    routers_v4 = {}
    routers_v6 = {}
    for router_id, infos in switch_gt.items():
        routers_v4[router_id] = set()
        routers_v6[router_id] = set()
        vendor = infos["kind"]
        addresses = infos["addrs"]

        for address in addresses:
            if ipv4_regex.match(address):
                ips4.add(address)
                routers_v4[router_id].add(address)
            elif ipv6_regex.match(address):
                ips6.add(address)
                routers_v6[router_id].add(address)
    for router_id, ips in routers_v4.items():
        routers_v4[router_id] = list(ips)

    for router_id, ips in routers_v6.items():
        routers_v6[router_id] = list(ips)

    # print("IPv4 addresses:", len(ips4))
    # print("IPv6 addresses:", len(ips6))
    #
    # pp = pprint.PrettyPrinter(indent=4)
    #
    # print(len(routers_v4))
    # print(len(routers_v6))
    # pp.pprint(routers_v4)
    # pp.pprint(routers_v6)

    return routers_v4, routers_v6


if __name__ == "__main__":

    switch_dir = "resources/SWITCH/"
    gt_file = switch_dir + "ground-truth.yml"

    switch_gt = extract_gt_from_yaml(gt_file)

    routers_v4, router_v6 = extract_evaluation_gt_switch(switch_gt)





    # with open(switch_dir + "ips4", "w") as f:
    #     for ip in ips4:
    #         f.write(ip + "\n")
    #
    # with open(switch_dir + "ips6", "w") as f:
    #     for ip in ips6:
    #         f.write(ip + "\n")




