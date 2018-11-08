import pandas as pd
import copy
from Validation.midar import compute_midar_routers
def extract_router_set(points):
    labels = points[[points.index, "label"]]

    print labels.to_string()


def extract_candidates(df):
    pairs = []
    for row in df.itertuples(index=True):
        index = row[0]
        split_index = index.split("_")
        candidate1 = split_index[1]
        candidate2 = split_index[2]
        witness = split_index[3]


        pairs.append({candidate1, candidate2})

    return pairs



if __name__ == "__main__":

    midar_path = "resources/internet2/midar/v4/"
    labeled_df_file = "resources/labeled_test_set"

    df = pd.read_csv(labeled_df_file, index_col=0)

    labeled_df = df[["label"]]

    print labeled_df.to_string()

    # Extract P, U and N labels

    p_labeled_df = labeled_df[labeled_df["label"] == 1]
    u_labeled_df = labeled_df[labeled_df["label"] == 2]
    n_labeled_df = labeled_df[labeled_df["label"] == 0]

    # Build routers from positives labels.
    routers = compute_midar_routers(midar_path)



    # Hack here
    pair_routers = copy.deepcopy(routers)
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
            for j in range(i+1, len(routers)):
                if len(routers[i].intersection(routers[j])) > 0:
                    routers[i].update(routers[j])
                    if routers[j] not in merged:
                        merged.append(routers[j])
            tc_routers.append(routers[i])

        routers = tc_routers
        new_len = len(tc_routers)
    final_routers = tc_routers

    print "Number of routers found " + str(len(tc_routers))

    list_routers = [list(router) for router in final_routers]
    distinct_ips = set()
    for router in routers:
        for ip in router:
            distinct_ips.add(ip)
    print len(distinct_ips)

    already_found_router = []

    for router in list_routers:
        import os
        # Find to which router this ip belongs to
        internet2_router_conf_path = "resources/internet2/routers/v4/"
        internet2_midar_routers_path = "resources/internet2/midar/v4/routers/"
        found_router_conf = False
        corresponding_router = ""

        for router_conf_file in os.listdir(internet2_router_conf_path):

            with open(internet2_router_conf_path + router_conf_file, "r") as f:
                for line in f.readlines():
                    line = line.strip()
                    if router[0] == line:
                        found_router_conf = True
                        corresponding_router = router_conf_file
                        break
            if found_router_conf:
                break
        if found_router_conf:
            mode = "w"
            if corresponding_router in already_found_router:
                mode = "a"
            already_found_router.append(corresponding_router)
            with open(internet2_midar_routers_path + corresponding_router, mode) as f:
                for ip in router:
                    f.write(ip + "\n")



    # See if some have been discarded by "N" labels
    discarded_pairs = extract_candidates(n_labeled_df)
    disagreements = []
    for router in tc_routers:
        for pair in discarded_pairs:
            if len(router.intersection(pair)) == len(pair):
                # Means that the pair has been discarded by another measurement.
                disagreements.append(pair)


    print "Number of false positives: " +str(len(disagreements))

    # Find the files that are responsible of false positives before transitive closure
    for pair in disagreements:
        if pair in pair_routers:
            print pair









