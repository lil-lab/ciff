from setup_agreement.abstract_validate_setup import AbstractSetupValidator

HOUSE_CONFIG_KEYS = ["num_actions",
                     "action_names",
                     "vocab_file",
                     "use_manipulation",
                     "num_manipulation_row",
                     "num_manipulation_col",
                     "stop_action",
                     "image_height",
                     "image_width"]


class HouseSetupValidator(AbstractSetupValidator):

    def validate_environment_specific(self, config):
        for config_key in HOUSE_CONFIG_KEYS:
            if config_key not in config:
                raise KeyError("missing key %r from nav-drone config file"
                               % config)
