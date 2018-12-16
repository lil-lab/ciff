from baselines.abstract_baseline import AbstractBaseline


class StopBaseline(AbstractBaseline):
    def __init__(self, server, action_space, meta_data_util, config, constants):
        AbstractBaseline.__init__(self, server, action_space, meta_data_util,
                                  config, constants)
        self.baseline_name = "stop_baseline"

    def get_next_action(self, data_point, num_actions):
        return self.action_space.get_stop_action_index()