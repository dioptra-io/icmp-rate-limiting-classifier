import pandas as pd
import json

if __name__ == "__main__":
    nodes4 = [
        "ple41.planet-lab.eu",
        "cse-yellow.cse.chalmers.se",
        "ple1.cesnet.cz",
        "planetlab1.cs.vu.nl",
    ]

    nodes6 = ["ple41.planet-lab.eu", "ple1.cesnet.cz"]

    unresponsives = None
    for node in nodes6:
        result_individual_file = (
            "resources/results/internet2/individual/" + node + "_individual_results6"
        )
        columns = [
            "ip_address",
            "probing_type",
            "probing_rate",
            "changing_behaviour",
            "loss_rate",
            "transition_0_0",
            "transition_0_1",
            "transition_1_0",
            "transition_1_1",
        ]
        df_individual = pd.read_csv(
            result_individual_file,
            names=columns,
            skipinitialspace=True,
            # encoding="utf-8",
            engine="python",
            index_col=False,
        )

        df_unresponsive = df_individual[df_individual["loss_rate"] == 1]

        unresponsive_addresses = list(df_unresponsive["ip_address"])
        if unresponsives is None:
            unresponsives = set(unresponsive_addresses)
        else:
            unresponsives = unresponsives.intersection(set(unresponsive_addresses))

    print("total IP addresses: " + str(df_individual.shape[0]))
    print("Unresponsive addresse: " + str(len(unresponsives)))
    print("Ratios: " + str(float(len(unresponsives) / df_individual.shape[0])))
    # with open("resources/results/internet2/unresponsive/" + node +"_internet2_unresponsive6.json", "w") as fp:
    #     json.dump(unresponsive_addresses, fp)
