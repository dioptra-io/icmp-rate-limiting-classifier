"""Perform the rate limiting algorithm on a list of IP addresses."""
import argparse
import configparser
import json
import pandas as pd
import re

from joblib import load

from Algorithms.algorithms import transitive_closure
from Classification.classifier_options import ClassifierOptions
from Cpp.cpp_files import ipv4_regex, ipv6_regex
from Cpp.cpp_options import generate_cpp_options
from Data.preprocess import extract_individual
from Probing.probing_algorithm import (
    execute_individual,
    find_witness_phase,
    simultaneous_phase,
)


def get_candidates(targets_file, max_candidates=None):
    """Get candidate from the target file."""
    candidates = []
    for line in open(targets_file, "r"):
        if max_candidates is not None and len(candidates) >= max_candidates:
            break

        line = line.strip("\n")
        if re.match(ipv4_regex, line) or re.match(ipv6_regex, line):
            if line not in candidates:
                candidates.append(line)
    return candidates


def cluster_by_triggering_rate(df_individual, candidates):
    """Cluster the results by triggering rates."""
    clusters = []
    for i in range(7, 16):
        cluster_triggering_rate = 2 ** i
        probing_rate_column = df_individual["probing_rate"]
        cluster = list(
            df_individual[
                (2 ** i <= probing_rate_column) & (probing_rate_column < 2 ** (i + 1))
            ]["ip_address"]
        )

        clusters.append(
            (
                sorted(list(set(cluster).intersection(candidates))),
                cluster_triggering_rate,
            )
        )
    return clusters


def main(config, candidates):
    """The algorithm is O(n)."""
    #
    # ---- Options ----
    #

    # Binary options
    cpp_options = generate_cpp_options(config)

    # Classifier options
    classifier = load(filename=config["CLASSIFIER"]["ClassifierPath"])
    classifier_options = ClassifierOptions()

    #
    # ---- Individual phase ----
    #

    # Batch candidates into 1 candidate because of RAM issues.
    do_individual = True
    if do_individual:
        execute_individual(candidates, config, cpp_options, classifier_options)
    else:
        # Take the intersection of the candidates list and the individual file.
        individual_ips = set(
            extract_individual(
                config["BINARY_OPTIONS"]["IndividualResultFile"],
                classifier_options.global_raw_columns,
            )["ip_address"]
        )

        candidates = list(set(candidates).intersection(individual_ips))
        print(
            "Only "
            + str(len(candidates))
            + " found in the individual file. Will use the algorithm on these."
        )

    #
    # ---- Witness phase ----
    #

    # Be careful, this function remove candidates with no witness.
    do_witness = True
    if do_witness:
        witness_by_candidate, hop_by_candidate = find_witness_phase(
            candidates, config, cpp_options, classifier_options
        )

        if len(witness_by_candidate) == 0 or len(hop_by_candidate) == 0:
            return

        df_individual_witness = extract_individual(
            config["BINARY_OPTIONS"]["WitnessResultFile"],
            classifier_options.global_raw_columns,
        )
    df_individual = extract_individual(
        config["BINARY_OPTIONS"]["IndividualResultFile"],
        classifier_options.global_raw_columns,
    )
    # Remove unresponsive addresses
    df_individual = df_individual.apply(pd.to_numeric, errors="ignore")
    df_individual = df_individual[df_individual["loss_rate"].apply(pd.to_numeric) < 1]

    #
    # ---- Alias resolution phase ----
    #

    aliases = []
    unresponsive_candidates = []

    use_cluster = True
    if use_cluster:
        clusters = cluster_by_triggering_rate(df_individual, candidates)
    else:
        clusters = [(candidates, 0)]

    for cluster, cluster_triggering_rate in clusters:
        print(
            "Performing alias resolution on cluster with triggering rate "
            + str(cluster_triggering_rate)
        )

        iteration = 0
        while len(cluster) > 1:
            print(
                "Performing iteration "
                + str(iteration)
                + " on candidates: "
                + str(len(cluster))
            )
            aliases_found, cluster, new_unresponsive_candidates = simultaneous_phase(
                cluster,
                config,
                cpp_options,
                classifier,
                classifier_options,
                iteration,
                cluster_triggering_rate,
                witness_by_candidate,
                hop_by_candidate,
                df_individual,
                df_individual_witness,
            )
            unresponsive_candidates.extend(new_unresponsive_candidates)
            aliases.extend(aliases_found)
            iteration += 1
            if not use_cluster:
                break

    if len(aliases) > 0:
        final_aliases = transitive_closure(aliases)
        with open(config["OUTPUT"]["AliasFile"], "w") as fp:
            serializable_aliases = [list(router) for router in final_aliases]
            json.dump(serializable_aliases, fp)
        with open(config["OUTPUT"]["UnresponsiveFile"], "w") as fp:
            json.dump(unresponsive_candidates, fp)
        print("Aliases : " + str(final_aliases))
        print("Unresponsive: " + str(unresponsive_candidates))


if __name__ == "__main__":
    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("targets", help="Path of targets file.")
    parser.add_argument(
        "--configuration",
        help="Path of configuration file.",
        default="configuration/default.ini",
    )
    args = parser.parse_args()

    # Configuration
    config = configparser.ConfigParser(
        interpolation=configparser.ExtendedInterpolation()
    )
    config.read(args.configuration)

    # Candidates
    candidates = get_candidates(args.targets)

    # Main execution
    main(config, candidates)
