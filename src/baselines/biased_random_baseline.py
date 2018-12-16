import random

from baselines.abstract_baseline import AbstractBaseline

FREQS = {
    "Forward": 37948,
    "TurnRight": 13089,
    "TurnLeft": 13938,
}
ACTION_SAMPLE = [k for k, v in FREQS.iteritems() for _ in xrange(v)]

class BiasedRandomBaseline(AbstractBaseline):
    def __init__(self, server, action_space, meta_data_util, config, constants):
        AbstractBaseline.__init__(self, server, action_space, meta_data_util,
                                  config, constants)
        self.baseline_name = "random_baseline"
        self.action_sample = [self.action_space.get_action_index(a)
                              for a in ACTION_SAMPLE]

    def get_next_action(self, data_point, num_actions):
        return random.choice(self.action_sample)
