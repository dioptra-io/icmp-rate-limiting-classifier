from Files.utils import ipv4_regex, ipv6_regex
import re

if __name__ == "__main__":

    """
    This script parse the Internet2 interfaces files and generates router files 
    """

    router_id_regex = re.compile('<th id=".*?">(.*)</th>')

    gt_interface_addresses = (
        "resources/internet2/ground-truth/Internet2-interfaces.html"
    )

    ground_truth_routers_v4 = {}
    ground_truth_routers_v6 = {}

    with open(gt_interface_addresses) as f:

        for line in f:
            # Match the id between <>
            m_router_id = re.search(router_id_regex, line)
            if m_router_id is not None:
                m_interface_v4 = re.search(ipv4_regex, line)
                m_interface_v6 = re.search(ipv6_regex, line)
                router_id = m_router_id.group(1)

                if router_id not in ground_truth_routers_v4:
                    ground_truth_routers_v4[router_id] = set()
                if router_id not in ground_truth_routers_v6:
                    ground_truth_routers_v6[router_id] = set()
                if m_interface_v4 is not None:
                    ground_truth_routers_v4[router_id].add(m_interface_v4.group(0))
                if m_interface_v6 is not None:
                    ground_truth_routers_v6[router_id].add(m_interface_v6.group(0))

    gt_routers_dir = "resources/internet2/ground-truth/routers/"

    n_ipv4_ips = 0

    discard_interfaces = {
        "198.32.11.6",
        "172.31.254.2",
        "64.57.22.233",
        "64.57.22.225",
        "64.57.22.65",
        "2001:468:fc:3a::1",
    }

    ipv4_candidates = set()
    ipv6_candidates = set()

    for router_id, ips in ground_truth_routers_v4.items():
        with open(gt_routers_dir + "v4/" + router_id, "w") as f:
            for ip in ips:
                f.write(ip + "\n")
                if ip in ipv4_candidates:
                    print("Duplicata", router_id, ip)
                if ip not in discard_interfaces:
                    ipv4_candidates.add(ip)
                    n_ipv4_ips += 1

    n_ipv6_ips = 0
    for router_id, ips in ground_truth_routers_v6.items():

        with open(gt_routers_dir + "v6/" + router_id, "w") as f:
            for ip in ips:
                f.write(ip + "\n")
                if ip in ipv6_candidates:
                    print("Duplicata", router_id, ip)
                if ip not in discard_interfaces:
                    ipv6_candidates.add(ip)
                    n_ipv6_ips += 1

    gt_dir = "resources/internet2/ground-truth/"
    with open(gt_dir + "ips4", "w") as f:
        for ip in ipv4_candidates:
            f.write(ip + "\n")
    with open(gt_dir + "ips6", "w") as f:
        for ip in ipv6_candidates:
            f.write(ip + "\n")
    print("IPv4 addresses: " + str(n_ipv4_ips))
    print("IPv6 addresses: " + str(n_ipv6_ips))
