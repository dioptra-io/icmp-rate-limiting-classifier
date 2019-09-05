class CppOptions:
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
