from Validation.midar import extract_routers


def find_corresponding_router(ip, ground_truth_routers):
    for router_name, router in ground_truth_routers.items():
        for ip_router in router:
            if ip_router == ip:
                return router_name
    return None

def combin(n, k):
    """Number of combinations C(n,k)"""
    if k > n//2:
        k = n-k
    x = 1
    y = 1
    i = n-k+1
    while i <= n:
        x = (x*i)//y
        y += 1
        i += 1
    return x




def evaluate(routers, ground_truth_routers):


    for router_name, router in routers.items():
        for i in range(0, len(router)):
            router_i = find_corresponding_router(router[i], ground_truth_routers)
            for j in range(i+1, len(router)):
                router_j = find_corresponding_router(router[j], ground_truth_routers)
                if router_i != router_j:
                    print (router[i], router[j])


    for file_name, ips in ground_truth_routers.items():
        ips = set(ips)
        print (file_name)
        for rl_router in routers:
            if len(ips.intersection(rl_router)) > 0:
                print ("Common: " + str(ips.intersection(rl_router)))
                print ("FP: " + str(rl_router - ips))
                # print ("FN: " + str(ips - rl_router))

    """Stats"""
    res_gt = 0

    for router, ips in sorted(ground_truth_routers.items()):
        res_gt += combin(len(ips), 2)

    res = 0

    for router in routers:
        res += combin(len(router), 2)

    print ("Ground Truth pairs: ")
    print (res_gt)
    print ("Rate Limiting pairs :")
    print (res)

    print ("RATIO :")
    print (float(res)/res_gt)


if __name__ == "__main__":
    routers = extract_routers("resources/results/internet2/aliases/")

    ground_truth_routers = extract_routers("resources/internet2/routers/v4/")

    evaluate(routers, ground_truth_routers)


