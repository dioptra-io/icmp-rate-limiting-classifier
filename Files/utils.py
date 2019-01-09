import re
import os

ipv4_regex = re.compile("(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)")
ipv6_regex = re.compile("((([0-9A-Fa-f]{1,4}:){7}([0-9A-Fa-f]{1,4}|:))|(([0-9A-Fa-f]{1,4}:){6}(:[0-9A-Fa-f]{1,4}|((25[0-5]|2[0-4]d|1dd|[1-9]?d)(.(25[0-5]|2[0-4]d|1dd|[1-9]?d)){3})|:))|(([0-9A-Fa-f]{1,4}:){5}(((:[0-9A-Fa-f]{1,4}){1,2})|:((25[0-5]|2[0-4]d|1dd|[1-9]?d)(.(25[0-5]|2[0-4]d|1dd|[1-9]?d)){3})|:))|(([0-9A-Fa-f]{1,4}:){4}(((:[0-9A-Fa-f]{1,4}){1,3})|((:[0-9A-Fa-f]{1,4})?:((25[0-5]|2[0-4]d|1dd|[1-9]?d)(.(25[0-5]|2[0-4]d|1dd|[1-9]?d)){3}))|:))|(([0-9A-Fa-f]{1,4}:){3}(((:[0-9A-Fa-f]{1,4}){1,4})|((:[0-9A-Fa-f]{1,4}){0,2}:((25[0-5]|2[0-4]d|1dd|[1-9]?d)(.(25[0-5]|2[0-4]d|1dd|[1-9]?d)){3}))|:))|(([0-9A-Fa-f]{1,4}:){2}(((:[0-9A-Fa-f]{1,4}){1,5})|((:[0-9A-Fa-f]{1,4}){0,3}:((25[0-5]|2[0-4]d|1dd|[1-9]?d)(.(25[0-5]|2[0-4]d|1dd|[1-9]?d)){3}))|:))|(([0-9A-Fa-f]{1,4}:){1}(((:[0-9A-Fa-f]{1,4}){1,6})|((:[0-9A-Fa-f]{1,4}){0,4}:((25[0-5]|2[0-4]d|1dd|[1-9]?d)(.(25[0-5]|2[0-4]d|1dd|[1-9]?d)){3}))|:))|(:(((:[0-9A-Fa-f]{1,4}){1,7})|((:[0-9A-Fa-f]{1,4}){0,5}:((25[0-5]|2[0-4]d|1dd|[1-9]?d)(.(25[0-5]|2[0-4]d|1dd|[1-9]?d)){3}))|:)))(%.+)?s*(\/([0-9]|[1-9][0-9]|1[0-1][0-9]|12[0-8]))?")

def build_candidate_line(group, af_family, probing_type, probing_protocol, real_target, probing_target):
    return group + ", " + af_family + ", " + probing_type + ", " + probing_protocol + ", " + "CANDIDATE, " + real_target + ", " + probing_target


def build_witness_line(group, af_family, probing_type, probing_protocol, real_target, probing_target):
    return group + ", " + af_family + ", " + probing_type + ", " + probing_protocol + ", " + "WITNESS, " + real_target + ", " + probing_target


def extract_candidates(cw_file):
    candidates = []
    witnesses = []
    ip_index = 5
    for line in cw_file:
        line = line.replace(" ", "")
        fields = line.split(",")
        if "CANDIDATE" in fields:
            candidates.append(fields[ip_index])
        elif "WITNESS" in fields:
            witnesses.append(fields[ip_index])

    return candidates, witnesses


def extract_ip_witness(version, traceroute_output, dst_ip):

    previous_lines = []
    # Skip first line which only gives infos.
    for line in traceroute_output[1:]:
        if dst_ip in line:
            # Extract previous ip
            for previous_line in reversed(previous_lines):
                if version == "4":
                    match = re.findall(ipv4_regex, previous_line)
                elif version == "6":
                    match = re.findall(ipv6_regex, previous_line)
                if len(match) > 0:
                    if version == "4":
                        return  match[0]
                    elif version == "6":
                        return match[0][0]
        else:
            previous_lines.append(line)
    return ""
