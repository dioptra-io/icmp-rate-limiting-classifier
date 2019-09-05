import os


def monitor():
    ips = set()
    ips_l = []
    pcap_dir = "/home/kevin/ICMPRateLimiting/resources/pcap/individual/"

    for pcap_file in os.listdir(pcap_dir):
        if pcap_file.startswith("dum"):
            continue
        pcap_file_split = pcap_file.split("_")
        ip = pcap_file_split[3]
        ips_l.append(ip)
        ips.add(ip)

    print(len(ips))
    print(len(set([ip for ip in ips_l if ips_l.count(ip) > 1])))


def concat_files(f1, f2):
    with open(f1, "a") as f1_fp:
        with open(f2, "r") as f2_fp:
            for l in f2_fp:
                f1_fp.write(l)


def write_individual_files():
    results_dir = "/srv/icmp-rl-survey/midar/survey/internet2/results/"
    for result_file in os.listdir(results_dir):
        if os.path.isdir(results_dir + result_file):
            continue
        node = result_file.split("_")[0]
        individual_file_results = (
            "resources/results/internet2/individual/" + node + "_individual_results"
        )
        if node in result_file:
            concat_files(individual_file_results, results_dir + result_file)


def remove_doubles():
    results_dir = "resources/results/internet2/individual/"
    for result_file in os.listdir(results_dir):
        ips = []
        with open(results_dir + result_file) as result_file_fp:
            print(result_file)
            for l in result_file_fp:
                l = l.strip()
                split = l.split(",")
                ip_address = split[0]
                if ip_address in ips:
                    print(ip_address)
                else:
                    ips.append(ip_address)


if __name__ == "__main__":
    # write_individual_files()
    remove_doubles()
