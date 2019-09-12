class CppOptions(object):
    def __init__(self):
        self.analyse_only = False
        self.probe_only = False
        self.group_only = False
        self.individual_only = False
        self.is_custom_probing_rates = False
        self.custom_probing_rates = []
        self.use_individual_for_analyse = False
        self.use_group_for_analyse = False
        self.first_only = False
        self.pcap_dir_individual = ""
        self.pcap_dir_groups = ""
        self.output_file = ""
        self.target_loss_rate_interval = ""
        self.measurement_time = 0
        self.low_rate_dpr = 0
        self.pcap_prefix = ""
        self.individual_result_file = ""
        self.starting_probing_rate = 512
        self.exponential_ratio = 1.05

    def to_string(self):
        options_str = ""
        if self.analyse_only:
            options_str += " -a "
        if self.probe_only:
            options_str += " -p "
        if self.group_only:
            options_str += " -G "
        if self.individual_only:
            options_str += " -I "
        if self.use_group_for_analyse:
            options_str += " -U "
        if self.use_individual_for_analyse:
            options_str += " -u "
        if self.pcap_dir_individual != "":
            options_str += " -i " + self.pcap_dir_individual
        if self.pcap_dir_groups != "":
            options_str += " -g " + self.pcap_dir_groups
        if self.pcap_prefix != "":
            options_str += " -x " + self.pcap_prefix
        if self.output_file != "":
            options_str += " -o " + self.output_file
        if self.target_loss_rate_interval != "":
            options_str += " -T " + self.target_loss_rate_interval
        if self.measurement_time != 0:
            options_str += " -m " + str(self.measurement_time)
        if self.low_rate_dpr != 0:
            options_str += " -r " + str(self.low_rate_dpr)

        return options_str


def generate_cpp_options(config):
    """Generate a Cpp option instance."""
    # TODO Directly pass the config in the CppOptions class
    cpp_options = CppOptions()
    cpp_options.pcap_dir_individual = config["BINARY_OPTIONS"]["PCAPDirIndividual"]
    cpp_options.pcap_dir_groups = config["BINARY_OPTIONS"]["PCAPDirIndividual"]
    cpp_options.pcap_prefix = config["BINARY_OPTIONS"]["PCAPPrefix"]
    cpp_options.low_rate_dpr = config["BINARY_OPTIONS"]["LowRateDPR"]
    cpp_options.measurement_time = config["BINARY_OPTIONS"]["MeasurementTime"]
    cpp_options.output_file = config["BINARY_OPTIONS"]["OutputFile"]
    cpp_options.target_loss_rate_interval = config["BINARY_OPTIONS"][
        "TargetLossRateInterval"
    ]
    cpp_options.exponential_ratio = config["BINARY_OPTIONS"]["ExponentialRatio"]
    cpp_options.individual_result_file = config["BINARY_OPTIONS"][
        "IndividualResultFile"
    ]
    return cpp_options
