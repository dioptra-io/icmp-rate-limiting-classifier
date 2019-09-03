import os
import json
from Validation.evaluation import extract_routers_by_node

def evaluate_rounds(rounds, rounds_dir):
    for round_file in os.listdir(rounds_dir):
        if os.stat(rounds_dir + round_file).st_size == 0:
            continue
        split = round_file.split("_")
        prefix = ""
        for i in range(0, len(split) - 1):
            prefix += split[i]
            prefix += "_"
        iteration = int(split[-1])
        if iteration not in rounds:
            rounds[iteration] = []

        i = 1
        next_iteration_file = prefix + str(iteration + i)
        is_no_alias = False
        while os.path.exists(rounds_dir + next_iteration_file):
            if os.stat(rounds_dir + next_iteration_file).st_size == 0:
                is_no_alias = True
                break
            i += 1
            next_iteration_file = prefix + str(iteration + i)
        if is_no_alias:
            continue
        predicted_aliases = 0
        with open(rounds_dir + round_file) as f:
            for line in f:
                predicted_aliases += 1
        if iteration > 3:
            iteration = 3
        # if iteration == 3:
        #     continue
        rounds[iteration].append(predicted_aliases)

if __name__ == "__main__":

    '''
    This script evaluates the number of rounds necessary on Limited Ltd. algorithm.
    '''

    rounds_dir_v4 = "resources/results/survey/aliases/v4/ple41.planet-lab.eu/"
    rounds_dir_v6 = "resources/results/survey/aliases/v6/ple2.planet-lab.eu/"


    rounds = {}
    evaluate_rounds(rounds, rounds_dir_v4)
    # evaluate_rounds(rounds, rounds_dir_v6)

    print(rounds)
    node = "ple41.planet-lab.eu"
    midar_routers = extract_routers_by_node("/home/kevin/mda-lite-v6-survey/resources/midar/batch2/routers/")[node]
    rounds["midar"] = []
    for router_id, router in midar_routers.items():
        rounds["midar"].append(len(router))
    with open("/home/kevin/icmp-rate-limiting-paper/resources/multiple_rounds.json", "w") as f:
        json.dump(rounds, f)


