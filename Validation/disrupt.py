from subprocess import PIPE, Popen
from threading import Thread
from Files.utils import ipv4_regex
import re
import time


def traceroute(destination, disrupt_dir, is_simultaneous_midar):
    traceroute_cmd = (
        "traceroute -n "
        + destination
        + " > "
        + disrupt_dir
        + destination
        + ".traceroute"
    )
    if is_simultaneous_midar:
        traceroute_cmd += "_midar"
    traceroute_process = Popen(traceroute_cmd, stdout=PIPE, stderr=PIPE, shell=True)
    print(traceroute_cmd)
    traceroute_process.communicate()


def midar(candidate, disrupt_dir):
    midar_output_prefix = candidate
    midar_candidate_file = disrupt_dir + candidate
    with open(midar_candidate_file, "w") as f:
        f.write(candidate + "\n")

    midar_cmd = (
        "cd resources/results/survey/midar/disrupt; "
        "/root/midar-0.7.1/midar/midar-cor --method=udp --rounds 5 "
        "--out " + midar_output_prefix + " " + candidate
    )
    print(midar_cmd)
    midar_process = Popen(midar_cmd, stdout=PIPE, stderr=PIPE, shell=True)
    midar_process.communicate()


def test_midar_disrupt(candidate, disrupt_dir):
    """
    First run classic traceroute
    Simultaneously launch midar and traceroute
    :return:
    """

    traceroute(candidate, disrupt_dir, False)
    midar_thread = Thread(target=midar, args=(candidate, disrupt_dir))
    traceroute_thread = Thread(target=traceroute, args=(candidate, disrupt_dir, True))

    midar_thread.start()
    time.sleep(2)
    traceroute_thread.start()
    midar_thread.join()
    traceroute_thread.join()


def midar_disrupt(disrupt_dir, candidates):

    for i in range(0, len(candidates)):
        print(i, candidates[i])
        test_midar_disrupt(candidates[i], disrupt_dir)


hop_re = re.compile("[0-9]{1,2}\s")


def extract_hop(traceroute, ip):
    with open(traceroute) as f1:
        # Extract hop at which the destination is found if found
        for line in f1:
            if ip in line and not line.startswith("traceroute"):
                hop_t1 = re.search(hop_re, line).group(0)
                return int(hop_t1)
    return None


def compare_traceroute(traceroute1, traceroute2):
    destination = re.search(ipv4_regex, traceroute1).group(0)
    hop_t1 = extract_hop(traceroute1, destination)
    hop_t2 = extract_hop(traceroute2, destination)

    if hop_t1 is not None or hop_t2 is not None:
        if hop_t1 != hop_t2:
            print(destination)
            return 1
    return 0


if __name__ == "__main__":
    disrupt_dir = "resources/results/survey/midar/disrupt/"
    candidates_file = "resources/survey/ips4"
    candidates = []

    with open(candidates_file) as f:
        for candidate in f:
            if len(candidates) < 10460:
                candidates.append(candidate.strip())

    # midar_disrupt(disrupt_dir, candidates)
    n_disrupt = 0
    for candidate in candidates:
        t1 = disrupt_dir + candidate + ".traceroute"
        t2 = disrupt_dir + candidate + ".traceroute_midar"
        n_disrupt += compare_traceroute(t1, t2)

    print("Number of traceroutes disrupted by MIDAR: " + str(n_disrupt))
