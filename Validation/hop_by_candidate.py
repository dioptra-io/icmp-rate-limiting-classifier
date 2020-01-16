import re
import os
import json

dic = {}
trouve = False


def hop_by_candidate():
    traceroute_dir = "resources/traceroutes/"

    with open("resources/survey/ips6") as f:

        ips6 = [line.strip() for line in f]

        for ip in ips6:
            trouve = False
            for file_name in os.listdir(traceroute_dir):
                if file_name.split(".")[0] == ip:
                    with open(traceroute_dir + file_name, "r") as f1:
                        lines = f1.readlines()
                        for line in lines:
                            if re.search("traceroute", line):
                                continue
                            if re.search(ip, line):
                                hop = line.split()[0]
                                dic[ip] = hop
                                trouve = True
                                break
                            else:
                                continue
                    if trouve:
                        break
                    else:
                        continue
                else:
                    continue

    with open("hop_by_candidate6_random.json", "w") as f1:
        json.dump(dic, f1, indent=4)
