from dataset_agreement.abstract_action_space import AbstractActionSpace


class ActionSpace(AbstractActionSpace):

    def __init__(self, act_names, stop_name, use_manipulation, num_manipulation_row, num_manipulation_col):

        self.use_manipulation = use_manipulation
        self.num_manipulation_row = num_manipulation_row
        self.num_manipulation_col = num_manipulation_col

        if self.use_manipulation:
            self.act_names = list(act_names)
            for row in range(0, self.num_manipulation_row):
                for col in range(0, self.num_manipulation_col):
                    self.act_names.append("interact %r %r" % (row, col))
        else:
            self.act_names = list(act_names)

        self.num_act = len(self.act_names)
        self.stop_idx = self.act_names.index(stop_name)

        AbstractActionSpace.__init__(self)

    def num_actions(self):
        return self.num_act

    def get_action_name(self, act_idx):
        return self.act_names[act_idx]

    def get_action_index(self, act_name):
        return self.act_names.index(act_name)

    def get_stop_action_index(self):
        return self.stop_idx
