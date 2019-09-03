from Algorithms.algorithms import transitive_closure


def extract_speedtrap_routers(speedtrap_result_file):
    '''
    Speedtrap outputs results by pair. Rebuilds then the transitive closure
    :param speedtrap_result_file:
    :return:
    '''

    routers = []
    with open(speedtrap_result_file) as speedtrap_fp:
        for line in speedtrap_fp:
            # Split the line in 2 members of the pair.
            split = line.split(" ")
            if len(split) == 2:
                ip1 = split[0]
                ip2 = split[1]
                routers.append({ip1, ip2})

    tc_routers = transitive_closure(routers)

    final_routers = {}
    for i in range(0, len(tc_routers)):
        final_routers[i] = tc_routers[i]

    return final_routers