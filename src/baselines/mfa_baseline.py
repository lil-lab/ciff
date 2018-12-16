from baselines.abstract_baseline import AbstractBaseline


class MFABaseline(AbstractBaseline):
    def __init__(self, server, action_space, meta_data_util, config, constants,
                 action_name):
        AbstractBaseline.__init__(self, server, action_space, meta_data_util,
                                  config, constants)
        self.baseline_name = "mfa_baseline"
        self.action_name = action_name

    def get_next_action(self, data_point, num_actions):
        return self.action_space.get_action_index(self.action_name)
