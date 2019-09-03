import os
def get_alias_pairs(routers):
    alias_pairs = set()
    for router_name, router in routers.items():
        for i in range(0, len(router)):
            for j in range(i+1, len(router)):
                alias_pairs.add(frozenset({router[i], router[j]}))
    return alias_pairs

def count_ips(routers_path):
    ips = set()
    ips_l = []
    routers = {}
    i = 0
    for router_f in os.listdir(routers_path):
        # if router_f.startswith("ple41"):
        router = []
        with open(routers_path + router_f) as f:
            for ip in f:
                ip = ip.strip("\n")
                ips_l.append(ip)
                ips.add(ip)
                router.append(ip)
        routers[i] = router
        i += 1
    print("Distincts IPs: " + str(len(ips)))
    print("IPs: " + str(len(ips_l)))


    alias_pairs = get_alias_pairs(routers)
    print("Unique alias pairs", len(alias_pairs))



# def count_pairs(routers_path):
#     routers = []
#
#     for router_f in os.listdir(routers_path):
#         if router_f.startswith("ple41"):
#             router = set()
#             with open(routers_path + router_f) as f:
#                 for ip in f:
#                     ip = ip.strip("\n")
#                     ips.add(ip)
#
if __name__ == "__main__":

    routers_path = "/home/kevin/mda-lite-v6-survey/resources/midar/batch2/routers/"
    #
    # count_ips(routers_path)

    routers_path = "/home/kevin/mda-lite-v6-survey/resources/speedtrap/routers/"

    count_ips(routers_path)