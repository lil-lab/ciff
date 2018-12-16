from dataset_agreement.abstract_action_space import AbstractActionSpace


class ActionSpace(AbstractActionSpace):

    def __init__(self, action_names, stop_act_name):
        AbstractActionSpace.__init__(self)
        self.action_names = action_names
        self.num_act = len(action_names)
        self.stop_act_idx = action_names.index(stop_act_name)

    def get_stop_action_index(self):
        return self.stop_act_idx

    def num_actions(self):
        return self.num_act

    def get_action_name(self, act_idx):
        return self.action_names[act_idx]

    def get_action_index(self, act_name):
        return self.action_names.index(act_name)
