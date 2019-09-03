

class ClassifierOptions(object):

    def __init__(self):

        self._probing_type_suffix = {"INDIVIDUAL": "ind", "GROUPSPR": "spr", "GROUPDPR": "dpr"}
        self._global_raw_columns = ["ip_address", "probing_type", "probing_rate", "changing_behaviour", "loss_rate",
                          "transition_0_0", "transition_0_1", "transition_1_0", "transition_1_1"]
        self._skip_fields = ["Index", "probing_type", "ip_address", "correlation"]
        self._probability_threshold = 0.6

    @property
    def probing_type_suffix(self):
        return self._probing_type_suffix

    @property
    def global_raw_columns(self):
        return self._global_raw_columns

    @property
    def skip_fields(self):
        return self._skip_fields

    @property
    def probability_threshold(self):
        return self._probability_threshold

    @probability_threshold.setter
    def probability_threshold(self, new_probability_threshold):
        self._probability_threshold = new_probability_threshold
