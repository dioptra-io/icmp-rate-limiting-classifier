import re
import os

ipv4_regex = re.compile("(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)")
ipv6_regex = re.compile("((([0-9A-Fa-f]{1,4}:){7}([0-9A-Fa-f]{1,4}|:))|(([0-9A-Fa-f]{1,4}:){6}(:[0-9A-Fa-f]{1,4}|((25[0-5]|2[0-4]d|1dd|[1-9]?d)(.(25[0-5]|2[0-4]d|1dd|[1-9]?d)){3})|:))|(([0-9A-Fa-f]{1,4}:){5}(((:[0-9A-Fa-f]{1,4}){1,2})|:((25[0-5]|2[0-4]d|1dd|[1-9]?d)(.(25[0-5]|2[0-4]d|1dd|[1-9]?d)){3})|:))|(([0-9A-Fa-f]{1,4}:){4}(((:[0-9A-Fa-f]{1,4}){1,3})|((:[0-9A-Fa-f]{1,4})?:((25[0-5]|2[0-4]d|1dd|[1-9]?d)(.(25[0-5]|2[0-4]d|1dd|[1-9]?d)){3}))|:))|(([0-9A-Fa-f]{1,4}:){3}(((:[0-9A-Fa-f]{1,4}){1,4})|((:[0-9A-Fa-f]{1,4}){0,2}:((25[0-5]|2[0-4]d|1dd|[1-9]?d)(.(25[0-5]|2[0-4]d|1dd|[1-9]?d)){3}))|:))|(([0-9A-Fa-f]{1,4}:){2}(((:[0-9A-Fa-f]{1,4}){1,5})|((:[0-9A-Fa-f]{1,4}){0,3}:((25[0-5]|2[0-4]d|1dd|[1-9]?d)(.(25[0-5]|2[0-4]d|1dd|[1-9]?d)){3}))|:))|(([0-9A-Fa-f]{1,4}:){1}(((:[0-9A-Fa-f]{1,4}){1,6})|((:[0-9A-Fa-f]{1,4}){0,4}:((25[0-5]|2[0-4]d|1dd|[1-9]?d)(.(25[0-5]|2[0-4]d|1dd|[1-9]?d)){3}))|:))|(:(((:[0-9A-Fa-f]{1,4}){1,7})|((:[0-9A-Fa-f]{1,4}){0,5}:((25[0-5]|2[0-4]d|1dd|[1-9]?d)(.(25[0-5]|2[0-4]d|1dd|[1-9]?d)){3}))|:)))(%.+)?s*(\/([0-9]|[1-9][0-9]|1[0-1][0-9]|12[0-8]))?")

hop_in_traceroute_regex = re.compile("\s*[0-9]{1,2}\s")

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
            # Extract the hop at which the alias has been found.
            hop = line[0:2]
            hop = hop.strip()
            hop = int(hop)
            # Extract previous ip
            for previous_line in reversed(previous_lines):
                if version == "4":
                    match = re.findall(ipv4_regex, previous_line)
                elif version == "6":
                    match = re.findall(ipv6_regex, previous_line)
                if len(match) > 0:
                    if version == "4":
                        return  match[0], hop
                    elif version == "6":
                        return match[0][0], hop
        else:
            previous_lines.append(line)
    return "" , -1


def is_unresponsive(out):
    is_witness_unresponsive = False
    for line in out:
        if "0 received" in line:
            is_witness_unresponsive = True
            break
    return is_witness_unresponsive

if __name__ == "__main__":
    '''
    Tansforrm an IP list to a candidate file 
    '''

    ip_file = "resources/survey/ips"
    output_file = "/root/ICMPRateLimiting/resources/survey_ips"

    group = "1"
    af_family = "4"
    probing_type = "DIRECT"
    probing_protocol = "icmp"


    with open(ip_file) as ip_file_fp:
        with open(output_file, "w") as output_file_fp:
            for line in ip_file_fp:
                ip = line.strip("\n")
                candidate_line = build_candidate_line(group, af_family, probing_type, probing_protocol, ip, ip)
                output_file_fp.write(candidate_line + "\n")
