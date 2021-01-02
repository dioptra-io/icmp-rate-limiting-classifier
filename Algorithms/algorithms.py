def rotate(l, n):
    return l[n:] + l[:n]


def connected(routers):
    # Apply transitive closure
    old_len_tc = len(routers)
    new_len = 0
    # End condition
    while old_len_tc != new_len:
        old_len_tc = new_len
        tc_routers = []
        # Now apply transitive closure
        merged = []
        for i in range(0, len(routers)):
            if routers[i] in merged:
                continue
            for j in range(i + 1, len(routers)):
                if len(routers[i].intersection(routers[j])) > 0:
                    routers[i].update(routers[j])
                    if routers[j] not in merged:
                        merged.append(routers[j])
            tc_routers.append(routers[i])

        routers = tc_routers
        new_len = len(tc_routers)
    final_routers = tc_routers
    return final_routers
