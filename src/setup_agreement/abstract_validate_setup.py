REQUIRED_CONFIG_KEYS = ["image_height",
                        "image_width",
                        "vocab_size",
                        "num_actions",
                        "use_paragraphs"]
REQUIRED_CONSTANTS_KEYS = ["image_emb_dim",
                           "max_num_images",
                           "word_emb_dim",
                           "lstm_emb_dim",
                           "action_emb_dim",
                           "learning_rate",
                           "entropy_coefficient",
                           "max_epochs",
                           "max_extra_horizon",
                           "max_extra_horizon_auto_segmented"]


class AbstractSetupValidator:
    def validate(self, config, constants):
        AbstractSetupValidator.validate_core(config, constants)
        self.validate_environment_specific(config)

    @staticmethod
    def validate_core(config, constants):
        for config_key in REQUIRED_CONFIG_KEYS:
            if config_key not in config:
                raise KeyError("missing key %r from config file"
                               % config_key)
        for constants_key in REQUIRED_CONSTANTS_KEYS:
            if constants_key not in constants:
                raise KeyError("missing key %r from config file"
                               % constants_key)

    def validate_environment_specific(self, config):
        raise NotImplementedError()
