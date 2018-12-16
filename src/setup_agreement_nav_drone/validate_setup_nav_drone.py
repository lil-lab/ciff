from setup_agreement.abstract_validate_setup import AbstractSetupValidator

NAV_DRONE_CONFIG_KEYS = ["port",
                         "hostname",
                         "action_names",
                         "vocab_file",
                         "stop_action",
                         "resources_dir",
                         "use_pointer_model"]


class NavDroneSetupValidator(AbstractSetupValidator):

    def validate_environment_specific(self, config):
        for config_key in NAV_DRONE_CONFIG_KEYS:
            if config_key not in config:
                raise KeyError("missing key %r from nav-drone config file"
                               % config)
