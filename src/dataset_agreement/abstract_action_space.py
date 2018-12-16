class AbstractActionSpace:

    def __init__(self):
        pass

    def num_actions(self):
        raise NotImplementedError()

    def get_action_name(self, act_idx):
        raise NotImplementedError()

    def get_action_index(self, act_name):
        raise NotImplementedError()

    def get_stop_action_index(self):
        raise NotImplementedError()