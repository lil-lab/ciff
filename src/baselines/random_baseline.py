import random

from baselines.abstract_baseline import AbstractBaseline

class RandomBaseline(AbstractBaseline):
    def __init__(self, server, action_space, meta_data_util, config, constants):
        AbstractBaseline.__init__(self, server, action_space, meta_data_util,
                                  config, constants)
        self.baseline_name = "random_baseline"

    def get_next_action(self, data_point, num_actions):
        # stop_action = self.action_space.get_stop_action_index()
        # actions = [i for i in range(self.action_space.num_actions())
        #            if i != stop_action]
        actions = list(range(self.action_space.num_actions()))

        return random.choice(actions)
