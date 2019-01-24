import pandas as pd
import json

if __name__ == "__main__":
    node = "ple41.planet-lab.eu"

    result_individual_file = "resources/results/internet2/individual/" + node + "_internet2_individual6"
    columns = ["ip_address", "probing_type", "probing_rate", "changing_behaviour", "loss_rate",
               "transition_0_0", "transition_0_1", "transition_1_0", "transition_1_1"]
    df_individual = pd.read_csv(result_individual_file,
                               names=columns,
                               skipinitialspace=True,
                               # encoding="utf-8",
                               engine="python",
                               index_col=False)

    df_unresponsive = df_individual[df_individual["loss_rate"] == 1]

    unresponsive_addresses = list(df_unresponsive["ip_address"])



    with open("resources/results/internet2/unresponsive/" + node +"_internet2_unresponsive6.json", "w") as fp:
        json.dump(unresponsive_addresses, fp)
