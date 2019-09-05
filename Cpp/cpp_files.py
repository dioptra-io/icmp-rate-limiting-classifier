import re
import os
from Files.utils import build_candidate_line, build_witness_line

ipv4_regex = re.compile(
    "(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)"
)
ipv6_regex = re.compile(
    "((([0-9A-Fa-f]{1,4}:){7}([0-9A-Fa-f]{1,4}|:))|(([0-9A-Fa-f]{1,4}:){6}(:[0-9A-Fa-f]{1,4}|((25[0-5]|2[0-4]d|1dd|[1-9]?d)(.(25[0-5]|2[0-4]d|1dd|[1-9]?d)){3})|:))|(([0-9A-Fa-f]{1,4}:){5}(((:[0-9A-Fa-f]{1,4}){1,2})|:((25[0-5]|2[0-4]d|1dd|[1-9]?d)(.(25[0-5]|2[0-4]d|1dd|[1-9]?d)){3})|:))|(([0-9A-Fa-f]{1,4}:){4}(((:[0-9A-Fa-f]{1,4}){1,3})|((:[0-9A-Fa-f]{1,4})?:((25[0-5]|2[0-4]d|1dd|[1-9]?d)(.(25[0-5]|2[0-4]d|1dd|[1-9]?d)){3}))|:))|(([0-9A-Fa-f]{1,4}:){3}(((:[0-9A-Fa-f]{1,4}){1,4})|((:[0-9A-Fa-f]{1,4}){0,2}:((25[0-5]|2[0-4]d|1dd|[1-9]?d)(.(25[0-5]|2[0-4]d|1dd|[1-9]?d)){3}))|:))|(([0-9A-Fa-f]{1,4}:){2}(((:[0-9A-Fa-f]{1,4}){1,5})|((:[0-9A-Fa-f]{1,4}){0,3}:((25[0-5]|2[0-4]d|1dd|[1-9]?d)(.(25[0-5]|2[0-4]d|1dd|[1-9]?d)){3}))|:))|(([0-9A-Fa-f]{1,4}:){1}(((:[0-9A-Fa-f]{1,4}){1,6})|((:[0-9A-Fa-f]{1,4}){0,4}:((25[0-5]|2[0-4]d|1dd|[1-9]?d)(.(25[0-5]|2[0-4]d|1dd|[1-9]?d)){3}))|:))|(:(((:[0-9A-Fa-f]{1,4}){1,7})|((:[0-9A-Fa-f]{1,4}){0,5}:((25[0-5]|2[0-4]d|1dd|[1-9]?d)(.(25[0-5]|2[0-4]d|1dd|[1-9]?d)){3}))|:)))(%.+)?s*(\/([0-9]|[1-9][0-9]|1[0-1][0-9]|12[0-8]))?"
)


def write_cpp_candidates_file(ip_version, candidates, witnesses, cpp_targets_file):
    af_family = ip_version
    probing_type = "DIRECT"
    probing_protocol = "icmp"

    with open(cpp_targets_file, "w") as cpp_targets_file_fp:
        for candidate in candidates:
            real_target = candidate

            cpp_targets_file_fp.write(
                build_candidate_line(
                    "1",
                    af_family,
                    probing_type,
                    probing_protocol,
                    real_target,
                    real_target,
                )
                + "\n"
            )
        for witness in witnesses:
            cpp_targets_file_fp.write(
                build_witness_line(
                    "1", af_family, probing_type, probing_protocol, witness, witness
                )
                + "\n"
            )


def write_cpp_witness_file(ip_version, witnesses, cpp_witness_file):
    af_family = ip_version
    probing_type = "DIRECT"
    probing_protocol = "icmp"
    with open(cpp_witness_file, "w") as cpp_witness_file_fp:
        for witness in witnesses:
            cpp_witness_file_fp.write(
                build_candidate_line(
                    "1", af_family, probing_type, probing_protocol, witness, witness
                )
                + "\n"
            )


def is_witness_already_probed(ip_witness, individual_pcap_dir):
    for file in os.listdir(individual_pcap_dir):
        if ip_witness in file:
            return True
    return False


def is_group_already_probed(ip, group_pcap_dir):
    for file in os.listdir(group_pcap_dir):
        split = file.split("_")
        if len(split) > 4:
            ip_file = split[4]
            if ip == ip_file:
                return True
    return False


if __name__ == "__main__":
    candidates_file = "resources/internet2/ips6"
    cpp_targets_file = "/root/ICMPRateLimiting/resources/cpp_targets_file6"
    candidates = []
    with open(candidates_file) as candidates_file_fp:
        for line in candidates_file_fp:
            line = line.strip("\n")
            if re.match(ipv6_regex, line):
                candidates.append(line)
    write_cpp_candidates_file("6", candidates, [], cpp_targets_file)
