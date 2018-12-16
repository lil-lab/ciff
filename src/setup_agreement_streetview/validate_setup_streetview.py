from setup_agreement.abstract_validate_setup import AbstractSetupValidator

STREETVIEW_CONFIG_KEYS = ["num_actions",
                          "action_names",
                          "vocab_file",
                          "node_file",
                          "link_file",
                          "image_feature_folder",
                          "stop_action",
                          "image_height",
                          "image_width"]


class StreetViewSetupValidator(AbstractSetupValidator):

    def validate_environment_specific(self, config):
        for config_key in STREETVIEW_CONFIG_KEYS:
            if config_key not in config:
                raise KeyError("missing key %r from street-view config file"
                               % config)
