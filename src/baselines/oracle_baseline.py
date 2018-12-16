from baselines.abstract_baseline import AbstractBaseline


class OracleBaseline(AbstractBaseline):
    def __init__(self, server, action_space, meta_data_util, config, constants):
        AbstractBaseline.__init__(self, server, action_space, meta_data_util,
                                  config, constants)
        self.baseline_name = "oracle_baseline"

    def get_next_action(self, data_point, num_actions):
        gold_trajectory = data_point.get_trajectory()
        if num_actions < len(gold_trajectory):
            return gold_trajectory[num_actions]
        else:
            return self.action_space.get_stop_action_index()
