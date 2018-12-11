from sklearn.preprocessing import MinMaxScaler
import copy

def minmax_scale(features):
    scaler = MinMaxScaler()
    scaler.fit(features)
    return scaler.transform(features)



def build_missing_candidates(n_candidates,
                             n_candidates_max,
                             n_witnesses,
                             n_witnesses_max,
                             default_columns,
                             skip_fields,
                             probing_type_suffix,
                             probing_type_rates):

    new_entry = build_missing_candidates_impl(n_candidates, n_candidates_max, default_columns, skip_fields, probing_type_suffix, probing_type_rates, "_c")
    new_entry.update(build_missing_candidates_impl(n_witnesses, n_witnesses_max, default_columns, skip_fields, probing_type_suffix, probing_type_rates, "_w"))
    return new_entry


def build_missing_candidates_impl(n_min, n_max, default_columns, skip_fields, probing_type_suffix, probing_type_rates, interface_type_suffix):
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
                    new_entry["".join([column,
                                       "_",
                                       probing_type_suffix[probing_type],
                                       interface_type_suffix,
                                       str(i),
                                       "_",
                                       str(probing_rate)])] = value
        new_entry["".join(["label", interface_type_suffix,  str(i)])] = 0
    return new_entry


def build_new_row(df_result, candidates, witnesses, skip_fields, probing_type_suffix, probing_type_rates, is_lr_classifier):

    ips = copy.deepcopy(candidates)
    ips.extend(witnesses)

    new_entry = {}

    for row in df_result.itertuples():
        probing_rate = row.probing_rate

        ip_address = row.ip_address
        probing_type = row.probing_type

        if probing_rate not in probing_type_rates[probing_type]:
            continue

        for i in range(0, len(row._fields)):

            # Skip some fields
            skip = False
            for skip_field in skip_fields:
                if skip_field in row._fields[i]:
                    skip = True
                    break
            if skip:
                continue


            if ip_address in candidates:
                additional_suffix = "c"
                key = "".join([row._fields[i],
                               "_",
                               probing_type_suffix[probing_type],
                               "_", additional_suffix,
                               str(candidates.index(row.ip_address))
                               ])
                if not is_lr_classifier:
                    key += "_" + str(probing_rate)

                new_entry[key] = row[i]

            elif ip_address in witnesses:
                additional_suffix = "w"
                key = "".join([row._fields[i],
                               "_",
                               probing_type_suffix[probing_type],
                               "_", additional_suffix,
                               str(witnesses.index(row.ip_address))
                              ])

                if not is_lr_classifier:
                    key += "_" + str(probing_rate)

                new_entry[key] = row[i]

    return new_entry



def find_index(e, l):
    for i in range(0, len(l)):
        if l[i] == e:
            return i
    return None


def parse_correlation(df, rates_dpr, candidates, witnesses):


    row = {}
    high_rate_candidate_ip = candidates[0]

    for rate in rates_dpr:
        df_filter_dpr_rate_high_rate_candidate = df[(df["probing_rate"]== rate) & \
                                                     (df["ip_address"] == high_rate_candidate_ip) & \
                                                      (df["probing_type"] ==  "GROUPDPR")]

        for field in df_filter_dpr_rate_high_rate_candidate.keys():
            if field.startswith("correlation"):
                correlation_field = df_filter_dpr_rate_high_rate_candidate[field].iloc[0]

                # In case candidates correlation are not in the same order
                correlation_split = correlation_field.split(":")
                ip_correlation = correlation_split[0].strip()
                correlation = correlation_split[1].strip()



                ip_corr_index = find_index(ip_correlation, candidates)
                if ip_corr_index is not None:
                    row["correlation_c" + str(ip_corr_index)] = float(correlation)
                else:
                    ip_corr_index = find_index(ip_correlation, witnesses)
                    if ip_corr_index is not None:
                        row["correlation_w" + str(ip_corr_index)] = float(correlation)

    return row

def remove_uninteresting_rate(new_entry, rates, probing_type,default_full_responsive_value):

    for rate in rates:
        filter_rate_entry = [(key, value) for key, value in new_entry.iteritems() \
                             if key.endswith(str(rate)) \
                             and key.startswith("".join(["changing_behaviour_", probing_type]))]
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
            (df_result["probing_type"] == probing_type) & (df_result["ip_address"].isin(witnesses)) & (
            df_result["probing_rate"] == rate)]
        df_candidates = df_result[
            (df_result["probing_type"] == probing_type) & (df_result["ip_address"].isin(candidates)) & (
            df_result["probing_rate"] == rate)]

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