from Files.utils import *
from subprocess import Popen, PIPE

def execute_icmp_rate_limiting_command(binary_path,
                                       cpp_targets_file,
                                       cpp_options,
                                       output_file,
                                       is_witness,
                                       only_analyse):
    analyse_only_option = ""
    if only_analyse:
        analyse_only_option = " -a "
    if is_witness:
        icmp_cmd = binary_path + " -I -t " + cpp_targets_file + " -o " + output_file + \
        " -T " + cpp_options.target_loss_rate_interval + \
        " -i " + cpp_options.pcap_dir_individual + \
        " -g " + cpp_options.pcap_dir_groups + \
        " -x " + cpp_options.pcap_prefix + \
        analyse_only_option
    else:
        icmp_cmd = binary_path + " -Gu -m " + str(10) + " -t " + cpp_targets_file + " -o " + output_file + \
                  " -T " + cpp_options.target_loss_rate_interval + \
                  " -i " + cpp_options.pcap_dir_individual + \
                  " -g " + cpp_options.pcap_dir_groups + \
                  " -x " + cpp_options.pcap_prefix + \
        analyse_only_option
    print(icmp_cmd)

    icmp_process = Popen(icmp_cmd
            ,
            stdout=PIPE, stderr = PIPE, shell=True)

    out, err = icmp_process.communicate()

    out = [line.decode("utf-8") for line in out.splitlines()]

    return out, err



def find_witness(ip_version, candidates):
    if ip_version == "4":
        traceroute = "traceroute"
        ping = "ping"
    elif ip_version == "6":
        traceroute = "traceroute6"
        ping = "ping6"
    # Execute a traceroute until a candidate is responsive and has a witness.
    for candidate in candidates:
        print("Starting traceroute to " + str(candidate))
        traceroute_cmd = "sudo " + traceroute + " --icmp -n " + candidate
        traceroute_process = Popen(
            traceroute_cmd,
            stdout=PIPE, stderr=PIPE, shell=True)

        out, err  = traceroute_process.communicate()
        out = [line.decode("utf-8") for line in out.splitlines()]
        ip_witness = extract_ip_witness(ip_version, traceroute_output=out, dst_ip= candidate)

        # Check if the witness is pingable.
        if ip_witness != "":
            ping_cmd = "sudo " + ping + " -w3 " + ip_witness
            ping_process = Popen(
                ping_cmd,
                stdout=PIPE, stderr=PIPE, shell=True)
            out, err = ping_process.communicate()
            out = [line.decode("utf-8") for line in out.splitlines()]
            is_witness_unresponsive = False
            for line in out:
                if "0 received" in line:
                    is_witness_unresponsive = True
                    break
            if is_witness_unresponsive:
                return None, None

            print("Found a responsive candidate and a witness for candidate " + str(candidate))
            return candidate, ip_witness
    return None, None