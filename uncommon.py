import os
from Validation.midar import extract_routers

from ICMPMain import main
from subprocess import Popen, PIPE


def rerun_lovebirds():
    # This tool evaluates the uncommon IPs found between Lovebird and Midar
    routers_dir = "resources/results/survey/uncommon/v4/midar-not-lovebirds-disagreement/"
    uncommon_routers = extract_routers(routers_dir)

    router_id = 0
    for router_name, routers in sorted(uncommon_routers.items()):
        router_id = int(router_name.split("_")[4].split("midar")[1])
        # if not "midar_not_lovebirds" in router_name:
        # if not "lovebirds_not_midar" in router_name:
        #     continue
        try:
            main(routers_dir + router_name, router_id)
        except:
            continue

def rerun_midar():
    midar_dir = "/root/midar-0.7.1/midar/"
    midar_bin = midar_dir + "midar-cor --rounds 20 --method icmp "

    results_dir = "/root/midar-0.7.1/internet2-rerun/"

    routers_dir = "resources/results/internet2/uncommon/v4/comparable-lovebirds-not-midar/"
    uncommon_routers = extract_routers(routers_dir)
    for router_name, routers in sorted(uncommon_routers.items()):
        if os.path.exists(results_dir + router_name):
            continue
        with open(results_dir + router_name, "w") as f:
            for ip in routers:
                f.write(ip + "\n")
        p = Popen("cd " + results_dir + ";" + midar_bin + "--out " + router_name + " " + results_dir + router_name,
                  shell=True,
                  stdout=PIPE,
                  stderr=PIPE)

        out, err = p.communicate()
        print (out)
        print (err)



if __name__ == "__main__":
    rerun_midar()




    #

