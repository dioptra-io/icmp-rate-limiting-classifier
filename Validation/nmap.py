import subprocess
import os
import re
import json


if __name__ == "__main__":

    dic = {}

    with open("/home/kevin/hop_by_candidate4.json", "r") as json_f:
        ips = json.loads(json_f.read())

    if False:
        for ip in ips:
            nmap_ofile = "/srv/nmap/results/" + ip + ".nmap"
            print(ip)
            if os.path.exists(nmap_ofile):
                continue

            nmap = subprocess.Popen(
                "sudo nmap  -O --osscan-guess " + ip,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                shell=True,
            )
            try:
                out, err = nmap.communicate(timeout=15)
                out = [line.decode("utf-8") for line in out.splitlines()]
                with open(nmap_ofile, "w") as f:
                    for line in out:
                        f.write(line + "\n")
            except subprocess.TimeoutExpired:
                continue

    for result in os.listdir("/srv/nmap/results"):
        ip = result.split("_")[0]

        with open("/srv/nmap/results/" + result, "r") as f:
            for line in f.readlines():
                if re.search("Aggressive OS guesses:", line):
                    os = line.split(":")[1].strip()
                    dic[ip] = os
                    break

    with open("resources/results/survey/ip_os_nmap.json", "w") as f:
        json.dump(dic, f, indent=4)
