import pandas as pd
import copy

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


    labeled_df_file = "resources/labeled_test_set"

    df = pd.read_csv(labeled_df_file, index_col=0)

    labeled_df = df[["label"]]

    print labeled_df.to_string()

    # Extract P, U and N labels

    p_labeled_df = labeled_df[labeled_df["label"] == "P"]
    u_labeled_df = labeled_df[labeled_df["label"] == "U"]
    n_labeled_df = labeled_df[labeled_df["label"] == "N"]

    # Build routers from positives labels.
    routers = extract_candidates(p_labeled_df)
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









