from Files.utils import *
from subprocess import Popen, PIPE
from threading import Thread


def execute_ping_and_icmp_rate_limiting_command(high_rate_candidate,
                                       binary_path,
                                       cpp_targets_file,
                                       cpp_options,
                                       output_file,
                                       is_individual,
                                       only_analyse):

    icmp_rate_limiting_thread = Thread(target = execute_icmp_rate_limiting_command, args = (binary_path,
                                       cpp_targets_file,
                                       cpp_options,
                                       output_file,
                                       is_individual,
                                       only_analyse, ))


    ping_thread = Thread(target=execute_ping_command, args = (high_rate_candidate,output_file, ))

    icmp_rate_limiting_thread.start()
    ping_thread.start()
    ping_thread.join()
    icmp_rate_limiting_thread.join()

def execute_ping_command(high_rate_candidate, output_file):
    ping_cmd = "ping -w6 " + high_rate_candidate + " > " + output_file + ".ping"
    ping_process = Popen(ping_cmd
          ,
          stdout=PIPE, stderr=PIPE, shell=True)
    out, err = ping_process.communicate()

def execute_icmp_rate_limiting_command(
                                       binary_path,
                                       cpp_targets_file,
                                       cpp_options,
                                       output_file,
                                       is_individual,
                                       only_analyse):

    # Launch a simultaneous normal ping to the candidate for ethical disruptions.


    analyse_only_option = ""
    if only_analyse:
        analyse_only_option = " -a "
    if is_individual:
        icmp_cmd = binary_path + " -I -t " + cpp_targets_file + " -o " + output_file + \
        " -T " + cpp_options.target_loss_rate_interval + \
        " -i " + cpp_options.pcap_dir_individual + \
        " -g " + cpp_options.pcap_dir_groups + \
        " -x " + cpp_options.pcap_prefix + \
        " -r " + str(cpp_options.low_rate_dpr) + \
        " --start-probing-rate=" + str(cpp_options.starting_probing_rate) + \
        " -e " + str(cpp_options.exponential_ratio) + \
        analyse_only_option
    else:
        # Here we assume the triggering rate has already been found, so no need to specify a target window.
        icmp_cmd = binary_path + " -Gu -m " + str(cpp_options.measurement_time) + " -t " + cpp_targets_file + " -o " + output_file + \
                  " -T [0.00,0.99]" +\
                  " -i " + cpp_options.pcap_dir_individual + \
                  " -g " + cpp_options.pcap_dir_groups + \
                  " -x " + cpp_options.pcap_prefix + \
                  " -r " + str(cpp_options.low_rate_dpr) + \
                  " --individual-result-file=" + cpp_options.individual_result_file + \
                  " --start-probing-rate=" + str(cpp_options.starting_probing_rate) + \
                  " -e " + str(cpp_options.exponential_ratio) + \
                   analyse_only_option
    print(icmp_cmd)

    icmp_process = Popen(icmp_cmd
            ,
            stdout=PIPE, stderr = PIPE, shell=True)

    out, err = icmp_process.communicate()




def find_witness(ip_version, candidates):
    witness_by_candidate = {}
    hop_by_candidate = {}
    if ip_version == "4":
        traceroute = "traceroute"
        ping = "ping"
    elif ip_version == "6":
        traceroute = "traceroute6"
        ping = "ping6"
    # Execute a traceroute until a candidate is responsive and has a witness.
    for candidate in candidates:
        traceroute_file = "resources/traceroutes/" + candidate + ".traceroute"
        if os.path.isfile(traceroute_file):
            with open(traceroute_file) as traceroute_fp:
                out = [line for line in traceroute_fp]
                ip_witness, hop = extract_ip_witness(ip_version, traceroute_output=out, dst_ip=candidate)
        else:
            print("Starting traceroute to " + str(candidate))
            traceroute_cmd = "sudo " + traceroute + " --icmp -n " + candidate
            traceroute_process = Popen(
                traceroute_cmd,
                stdout=PIPE, stderr=PIPE, shell=True)

            out, err  = traceroute_process.communicate()
            out = [line.decode("utf-8") for line in out.splitlines()]
            with open(traceroute_file, "w") as traceroute_fp:
                for line in out:
                    traceroute_fp.write(line + "\n")
            ip_witness, hop = extract_ip_witness(ip_version, traceroute_output=out, dst_ip= candidate)

        # Check if the witness is pingable.
        if hop != -1:
            hop_by_candidate[candidate] = hop

        if ip_witness != "":
            ping_file = "resources/pings/" + ip_witness + ".ping"
            if os.path.isfile(ping_file):
                with open(ping_file) as ping_fp:
                    out = [line for line in ping_fp]
                    is_witness_unresponsive = is_unresponsive(out)
            else:
                ping_cmd = "sudo " + ping + " -w3 " + ip_witness
                ping_process = Popen(
                    ping_cmd,
                    stdout=PIPE, stderr=PIPE, shell=True)
                out, err = ping_process.communicate()
                out = [line.decode("utf-8") for line in out.splitlines()]
                with open(ping_file, "w") as ping_fp:
                    for line in out:
                        ping_fp.write(line + "\n")
                is_witness_unresponsive = is_unresponsive(out)
            if is_witness_unresponsive:
                continue
            print("Found a responsive candidate and a witness for candidate " + str(candidate))
            witness_by_candidate[candidate] = ip_witness
    return witness_by_candidate, hop_by_candidate