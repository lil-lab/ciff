from dataset_agreement.abstract_action_space import AbstractActionSpace


class ActionSpace(AbstractActionSpace):

    def __init__(self, act_names, stop_name):
        self.act_names = act_names
        self.num_act = len(act_names)
        self.stop_idx = act_names.index(stop_name)
        AbstractActionSpace.__init__(self)

    def num_actions(self):
        return self.num_act

    def get_action_name(self, act_idx):
        return self.act_names[act_idx]

    def get_action_index(self, act_name):
        return self.act_names.index(act_name)

    def get_stop_action_index(self):
        return self.stop_idx
