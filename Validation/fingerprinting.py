import os
from graph_tool.all import *
import json
from subprocess import Popen, PIPE
import re


def execute_ping_command(ip, output_file):
    ping_cmd = "ping -6 -w1 " + ip + " > " + output_file + ".ping"
    ping_process = Popen(ping_cmd, stdout=PIPE, stderr=PIPE, shell=True)
    out, err = ping_process.communicate()


def extract_fingerprinting(ips):
    """
        Extract fingerprinting TTL-Exceeded, then run a ping to the ip.
        """
    ip_version = "6"
    # multilevel_traceroutes_dir = "/srv/icmp-rl-survey/midar/multilevel-traceroutes/"
    multilevel_traceroutes_dir = "/srv/icmp-rl-survey/speedtrap/multilevel-traceroutes/"

    n = 0
    fingerprinting = {}
    for multilevel_file in os.listdir(multilevel_traceroutes_dir):
        n += 1
        print(n, len(fingerprinting), multilevel_file)

        g = load_graph(multilevel_traceroutes_dir + multilevel_file)

        fingerprinting_g = g.vertex_properties["fingerprinting"]
        ip_address = g.vertex_properties["ip_address"]
        for v in g.vertices():
            if ip_address[v] in ips:
                fingerprinting[ip_address[v]] = list(fingerprinting_g[v])
        if len(fingerprinting) >= len(ips) - 1:
            break

    with open(
        "resources/results/survey/fingerprinting_" + ip_version + ".json", "w"
    ) as f:
        json.dump(fingerprinting, f)


def update_fingerprinting_with_pings(ip_version):
    ping_path = "resources/results/survey/validation/pings/"
    regex_ttl = re.compile("ttl=([0-9]{1,3})")
    with open(
        "resources/results/survey/fingerprinting_" + ip_version + ".json", "r"
    ) as f:
        fingerprinting = json.load(f)

    for ip, fingerprint in fingerprinting.items():
        if len(fingerprint) < 2:
            print(ip)
            continue
        ping_file = ping_path + ip + ".ping"
        if os.path.exists(ping_file):
            with open(ping_file) as p:
                for line in p:
                    m = regex_ttl.search(line)
                    if m is not None:
                        ttl = int(m.group(0).split("ttl=")[1])
                        fingerprint[0] = ttl
                        break
    with open(
        "resources/results/survey/fingerprinting_" + ip_version + ".json", "w"
    ) as f:
        json.dump(fingerprinting, f)


if __name__ == "__main__":

    ping_dir = "resources/results/survey/validation/pings/"
    ips_v4 = "resources/survey/ips4"

    ips_v6 = "resources/survey/ips6"

    i = 0
    ips = []

    with open(ips_v6) as f:
        for ip in f:
            ip_address = ip.strip("\n")
            ips.append(ip_address)
            i += 1
            if i == 10530:
                break
    # extract_fingerprinting(set(ips))

    # for ip in ips:
    #     if os.path.exists(ping_dir + ip + ".ping"):
    #         continue
    #     print (ip)
    #     execute_ping_command(ip, ping_dir + ip)

    update_fingerprinting_with_pings("6")
