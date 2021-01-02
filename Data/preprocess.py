from sklearn.preprocessing import MinMaxScaler
import numpy as np
import copy
import pandas as pd
import re
from Files.utils import ipv6_regex, ipv4_regex

minimum_probing_rate = 512


def minmax_scale(features):
    scaler = MinMaxScaler()
    scaler.fit(features.astype(float))
    return scaler.transform(features)


def build_missing_candidates(
    n_candidates,
    n_candidates_max,
    n_witnesses,
    n_witnesses_max,
    default_columns,
    skip_fields,
    probing_type_suffix,
    probing_type_rates,
):

    new_entry = build_missing_candidates_impl(
        n_candidates,
        n_candidates_max,
        default_columns,
        skip_fields,
        probing_type_suffix,
        probing_type_rates,
        "_c",
    )
    new_entry.update(
        build_missing_candidates_impl(
            n_witnesses,
            n_witnesses_max,
            default_columns,
            skip_fields,
            probing_type_suffix,
            probing_type_rates,
            "_w",
        )
    )
    return new_entry


def build_missing_candidates_impl(
    n_min,
    n_max,
    default_columns,
    skip_fields,
    probing_type_suffix,
    probing_type_rates,
    interface_type_suffix,
):
    new_entry = {}
    for i in range(n_min, n_max):
        for probing_type, probing_rates in probing_type_rates.iteritems():
            for probing_rate in probing_rates:
                for column in default_columns:
                    if column in skip_fields:
                        continue
                    # if default_value_feature.has_key(column):
                    #     value = default_value_feature[column]
                    # else:
                    value = 0.0
                    new_entry[
                        "".join(
                            [
                                column,
                                "_",
                                probing_type_suffix[probing_type],
                                interface_type_suffix,
                                str(i),
                                "_",
                                str(probing_rate),
                            ]
                        )
                    ] = value
        new_entry["".join(["label", interface_type_suffix, str(i)])] = 0
    return new_entry


def build_new_row(
    df_result,
    candidates,
    witnesses,
    skip_fields,
    probing_type_suffix,
    probing_type_rates,
    is_lr_classifier,
):

    ips = copy.deepcopy(candidates)
    ips.extend(witnesses)

    new_entry = {}

    columns = list(df_result.columns)
    probing_rate_column_index = columns.index("probing_rate")
    ip_address_column_index = columns.index("ip_address")
    probing_type_column_index = columns.index("probing_type")
    for row in df_result.itertuples():
        probing_rate = row[probing_rate_column_index + 1]

        ip_address = row[ip_address_column_index + 1]
        probing_type = row[probing_type_column_index + 1]

        if probing_rate not in probing_type_rates[probing_type]:
            continue

        for i in range(0, len(columns)):

            # Skip some fields
            skip = False
            for skip_field in skip_fields:
                if skip_field in columns[i]:
                    skip = True
                    break
            if skip:
                continue

            if ip_address in candidates:
                additional_suffix = "c"
                key = "".join(
                    [
                        columns[i],
                        "_",
                        probing_type_suffix[probing_type],
                        "_",
                        additional_suffix,
                        str(candidates.index(ip_address)),
                    ]
                )
                if not is_lr_classifier:
                    key += "_" + str(probing_rate)
                # if probing_rate == minimum_probing_rate:
                #     if not key + "_min" in new_entry:
                #         key += "_min"

                new_entry[key] = row[i + 1]

            elif ip_address in witnesses:
                additional_suffix = "w"
                key = "".join(
                    [
                        columns[i],
                        "_",
                        probing_type_suffix[probing_type],
                        "_",
                        additional_suffix,
                        str(witnesses.index(ip_address)),
                    ]
                )

                if not is_lr_classifier:
                    key += "_" + str(probing_rate)
                # if probing_rate == minimum_probing_rate:
                #     if not key + "_min" in new_entry:
                #         key += "_min"
                new_entry[key] = row[i + 1]

    return new_entry


def find_index(e, l):
    for i in range(0, len(l)):
        if l[i] == e:
            return i
    return None


def parse_correlation(df, rates_dpr, candidates, witnesses):

    correlations = {}

    high_rate_candidate_ip = candidates[0]

    for rate in rates_dpr:
        df_filter_dpr_rate_high_rate_candidate = df[
            (df["probing_rate"] == rate)
            & (df["ip_address"] == high_rate_candidate_ip)
            & (df["probing_type"] == "GROUPDPR")
        ].filter(regex="correlation.*", axis=1)
        correlations_row = df_filter_dpr_rate_high_rate_candidate.iloc[0]

        for i in range(1, len(candidates)):
            correlation_i = correlations_row["correlation_c" + str(i)]
            correlation_split = correlation_i.split(": ")
            ip_correlation = correlation_split[0].strip()
            correlation = correlation_split[1].strip()
            correlations[ip_correlation] = float(correlation)
        for i in range(0, len(witnesses)):
            correlation_i = correlations_row["correlation_w" + str(i)]
            correlation_split = correlation_i.split(": ")
            ip_correlation = correlation_split[0].strip()
            correlation = correlation_split[1].strip()
            correlations[ip_correlation] = float(correlation)
    return correlations


def get_pairwise_correlation_row(correlations, candidates, witnesses):
    row = {}
    for i in range(1, len(candidates)):
        if candidates[i] in correlations:
            row["correlation_c" + str(i)] = correlations[candidates[i]]
        else:
            row["correlation_c" + str(i)] = 0
    for i in range(0, len(witnesses)):
        if witnesses[i] in correlations:
            row["correlation_w" + str(i)] = correlations[witnesses[i]]
        else:
            row["correlation_w" + str(i)] = 0
    return row


def remove_uninteresting_rate(
    new_entry, rates, probing_type, default_full_responsive_value
):

    for rate in rates:
        filter_rate_entry = [
            (key, value)
            for key, value in new_entry.iteritems()
            if key.endswith(str(rate))
            and key.startswith("".join(["changing_behaviour_", probing_type]))
        ]
        is_fully_responsive = True
        for key, value in filter_rate_entry:
            if value != default_full_responsive_value:
                is_fully_responsive = False
                break
        if is_fully_responsive:
            for key, value in filter_rate_entry:
                del new_entry[key]


def is_broken_witness(df_result, probing_type, rates, witnesses, candidates):
    broken_witness = False
    for rate in rates:
        df_witness = df_result[
            (df_result["probing_type"] == probing_type)
            & (df_result["ip_address"].isin(witnesses))
            & (df_result["probing_rate"] == rate)
        ]
        df_candidates = df_result[
            (df_result["probing_type"] == probing_type)
            & (df_result["ip_address"].isin(candidates))
            & (df_result["probing_rate"] == rate)
        ]

        if len(df_candidates) != 0 and len(df_witness) != 0:
            lr_candidates = max(df_candidates["loss_rate"])
            lr_witness = max(df_witness["loss_rate"])

            if lr_witness != 1 and lr_witness > lr_candidates:
                broken_witness = True
                break
    return broken_witness


def sort_ips_by(df_result, candidates, witnesses, filter):
    # First candidate is always the first. Trick here is to set the LR > 1 to be sure candidates[0] is first.
    candidates_loss_rate_order = [(candidates[0], 1.1)]
    witnesses_loss_rate_order = []

    df_sort_candidates = df_result[filter]
    for row in df_sort_candidates.itertuples():
        ip_address = row.ip_address
        loss_rate = row.loss_rate

        if ip_address == candidates[0]:
            continue

        if ip_address in candidates:
            candidates_loss_rate_order.append((ip_address, loss_rate))
        if ip_address in witnesses:
            witnesses_loss_rate_order.append((ip_address, loss_rate))

    # Reorder the candidates by the loss rate
    candidates_loss_rate_order.sort(key=lambda x: x[1], reverse=True)

    candidates_sorted = [x[0] for x in candidates_loss_rate_order]
    witnesses_sorted = [x[0] for x in witnesses_loss_rate_order]

    return candidates_sorted, witnesses_sorted


def extract_feature_labels_columns(df_computed_result, is_pairwise):
    # Select features
    feature_columns = []
    for column in df_computed_result.columns:
        if not column.startswith("label") and not column.startswith("ip_address"):
            if "transition" in column and not "dpr" in column:
                continue
            if "changing_behaviour" in column:
                continue
            if "probing_rate" in column and not "ind" in column:
                continue
            if "spr" in column:
                continue
            feature_columns.append(column)

    labels_column = []
    for column in df_computed_result.columns:
        if is_pairwise:
            if column == ("label_pairwise"):
                labels_column.append(column)
        else:
            if column.startswith("label"):
                labels_column.append(column)

    return feature_columns, labels_column


def build_labeled_df(is_pairwise, df_computed_result):
    for column in df_computed_result.columns:
        # if column.startswith("changing_behaviour") or column.startswith("probing_rate"):
        if column.startswith("probing_rate"):
            df_computed_result[column] = minmax_scale(
                np.array(df_computed_result[column]).reshape(-1, 1)
            )
        if column.startswith("label"):
            df_computed_result[column] = df_computed_result[column].apply(np.int64)

    labeled_df = df_computed_result
    labeled_df = labeled_df.reset_index()

    feature_columns, labels_column = extract_feature_labels_columns(
        df_computed_result, is_pairwise
    )

    labeled_df = labeled_df.dropna(subset=feature_columns)

    # Now train a classifier on the labeled data.
    #
    # Shuffle the data
    labeled_df = labeled_df.sort_index(axis=1)
    return labeled_df, feature_columns, labels_column


def parse_labels_and_features(dataset, label_column, features_columns):
    """Extracts labels and features.

    This is a good place to scale or transform the features if needed.

    Args:
      dataset: A Pandas `Dataframe`, containing the label on the first column and
        monochrome pixel values on the remaining columns, in row major order.
    Returns:
      A `tuple` `(labels, features)`:
        labels: A Pandas `Series`.
        features: A Pandas `DataFrame`.
    """
    labels = dataset[label_column]

    # DataFrame.loc index ranges are inclusive at both ends.
    features = dataset[features_columns]
    features = features.reindex(sorted(features.columns), axis=1)
    return labels, features


def extract_individual(cpp_output_individual_file, raw_columns):
    df_raw = pd.read_csv(
        cpp_output_individual_file,
        names=raw_columns,
        skipinitialspace=True,
        # encoding="utf-8",
        engine="python",
        index_col=False,
    )
    df_individual = df_raw[df_raw["probing_type"] == "INDIVIDUAL"]
    return df_individual


def build_classifier_entry_from_csv(
    targets_file,
    global_raw_columns,
    skip_fields,
    probing_type_suffix,
    cpp_output_file,
    df_individual,
    df_witness_individual,
    witness_by_candidate,
):

    unresponsive_candidates = []

    candidates = []
    witnesses = []
    ip_index = 5
    df_computed_result = None
    try:
        with open(targets_file) as cw_file:
            for line in cw_file:
                line = line.replace(" ", "")
                fields = line.split(",")
                ip_address = fields[ip_index]
                if "CANDIDATE" in fields:
                    candidates.append(ip_address)
                elif "WITNESS" in fields:
                    witnesses.append(ip_address)
    except IOError as e:
        return None

    raw_columns = copy.deepcopy(global_raw_columns)

    # # Add correlation columns
    for i in range(1, len(candidates)):
        raw_columns.append("correlation_c" + str(i))

    for i in range(0, len(witnesses)):
        raw_columns.append("correlation_w" + str(i))

    # print "Routers with more than 2 interfaces: " + str(multiple_interfaces_router)
    # 1 point is represented by different dimensions:
    df_raw = pd.read_csv(
        cpp_output_file,
        names=raw_columns,
        skipinitialspace=True,
        # encoding="utf-8",
        engine="python",
        index_col=False,
    )

    # group_spr_probing_rate = df_raw[df_raw["probing_type"] == "GROUPSPR"]["probing_rate"].iloc[0]
    group_dpr_probing_rate = df_raw[df_raw["probing_type"] == "GROUPDPR"][
        "probing_rate"
    ].iloc[0]

    correlations = parse_correlation(
        df_raw, [group_dpr_probing_rate], candidates, witnesses
    )
    # Remove correlations columns
    df_raw = df_raw[df_raw.columns.drop(list(df_raw.filter(regex="correlation.*")))]

    if df_individual is not None and df_witness_individual is not None:
        df_raw = pd.concat(
            [df_individual, df_witness_individual, df_raw],
            sort=False,
            ignore_index=True,
        )

    # usecols=[x for x in range(0, 9)])

    ##################################### Split the df in pairwise elements #############################

    rows_to_adds = []
    keys = None

    df_list_pairwise = []
    pairwise_candidates_list = []
    for i in range(1, len(candidates)):
        if candidates[i] not in witness_by_candidate:
            unresponsive_candidates.append(candidates[i])
            continue
        df_pairwise = df_raw[
            df_raw["ip_address"].isin(
                [candidates[0], candidates[i], witness_by_candidate[candidates[i]]]
            )
        ]
        df_list_pairwise.append(df_pairwise)
        pairwise_candidates_list.append([candidates[0], candidates[i]])

    if not df_list_pairwise:
        return unresponsive_candidates, None, None, None

    for i in range(0, len(df_list_pairwise)):
        # print(i)
        # if i == 5:
        #     break
        new_entry = {}
        pairwise_candidates = pairwise_candidates_list[i]
        df_result = df_list_pairwise[i]

        for k in range(0, len(pairwise_candidates)):
            new_entry["ip_address_c" + str(k)] = pairwise_candidates[k]

        # Skip the measurement if all the loss rates are 1.
        not_usable = False
        for candidate in pairwise_candidates:
            df_not_usable = df_result[
                (df_result["loss_rate"] == 1)
                & (df_result["ip_address"] == candidate)
                & (df_result["probing_type"] == "GROUPDPR")
            ]
            if len(df_not_usable) > 0:
                not_usable |= df_not_usable["loss_rate"].iloc[0] == 1
        if not_usable:
            unresponsive_candidates.append(pairwise_candidates[1])
            continue

        ind_probing_rate = df_result[df_result["probing_type"] == "INDIVIDUAL"][
            "probing_rate"
        ]
        probing_type_rates = {
            "INDIVIDUAL": list(ind_probing_rate),
            # "GROUPSPR": [group_spr_probing_rate],
            "GROUPDPR": [group_dpr_probing_rate],
        }

        witnesses = [witness_by_candidate[pairwise_candidates[1]]]
        new_row = build_new_row(
            df_result,
            pairwise_candidates,
            witnesses,
            skip_fields,
            probing_type_suffix,
            probing_type_rates,
            is_lr_classifier=True,
        )

        correlation_row = get_pairwise_correlation_row(
            correlations, pairwise_candidates, witnesses
        )

        new_row.update(correlation_row)

        new_entry["measurement_id"] = targets_file
        new_entry.update(new_row)

        # In case we are missing a witness individual features, artificially create it.

        if keys is None:
            # Total number of figures
            df_computed_result = pd.DataFrame(columns=new_entry.keys())
            keys = list(new_entry.keys())
        if len(new_entry) != df_computed_result.shape[1]:
            print("Bad csv " + str(pairwise_candidates[1]))
            continue
        rows_to_adds.append(new_entry)
        # df_computed_result.loc[len(df_computed_result)] = new_entry
    df_computed_result = pd.DataFrame(rows_to_adds, columns=keys)

    df_computed_result.set_index("measurement_id", inplace=True)

    labeled_df, features_columns, labels_column = build_labeled_df(
        df_computed_result=df_computed_result, is_pairwise=True
    )
    return unresponsive_candidates, labeled_df, features_columns, labels_column


if __name__ == "__main__":

    from Classification.classifier_options import ClassifierOptions

    # DEBUG stuff
    ip_version = "4"

    icmp_install_dir = "/root/ICMPRateLimiting/"
    classifier_options = ClassifierOptions()

    cpp_individual_file = (
        icmp_install_dir + "resources/results/survey_individual" + ip_version
    )
    cpp_individual_file_witness = (
        icmp_install_dir + "resources/results/survey_individual_witness" + ip_version
    )

    from joblib import load

    classifier_file_name = "resources/random_forest_classifier4" + ".joblib"
    classifier = load(filename=classifier_file_name)

    # Recreate a context

    df_individual = extract_individual(
        cpp_individual_file, classifier_options.global_raw_columns
    )
    df_individual_witness = extract_individual(
        cpp_individual_file_witness, classifier_options.global_raw_columns
    )

    targets_file = "/root/ICMPRateLimiting/resources/cpp_targets_file_cluster2048_0_0"
    witness_by_candidate_file = "resources/witness_by_candidate" + ip_version + ".json"
    import json

    with open(witness_by_candidate_file) as witness_by_candidate_fp:
        witness_by_candidate = json.load(witness_by_candidate_fp)

    # Remove unresponsive addresses
    df_individual = df_individual[df_individual["loss_rate"] < 1]

    from Cpp.launcher import find_witness

    candidates = list(df_individual["ip_address"])
    witness_by_candidate, hop_by_candidate = find_witness(ip_version, candidates)

    with open("resources/hop_by_candidate4.json", "w") as fp:
        json.dump(hop_by_candidate, fp)

    cpp_output_file = "/root/ICMPRateLimiting/test_cluster2048_0_0"

    unresponsive_candidates, labeled_df, features_columns, labels_column = build_classifier_entry_from_csv(
        targets_file,
        classifier_options.global_raw_columns,
        classifier_options.skip_fields,
        classifier_options.probing_type_suffix,
        cpp_output_file,
        df_individual,
        df_individual_witness,
        witness_by_candidate,
    )

    labels, features_data = parse_labels_and_features(
        labeled_df, labels_column, features_columns
    )
    # import pandas as pd
    # features_data = pd.read_csv("resources/features_data.csv", index_col=0)
    # features_data = features_data.reindex(sorted(features_data.columns), axis=1)
    print(" done")
    # Classify
    print("Predicting probabilities...", end="", flush=True)
    predictions = []
    try:
        probabilities = classifier.predict_proba(features_data)
    except ValueError as e:
        predictions = [0 for x in range(0, len(labeled_df))]

    print(predictions)
