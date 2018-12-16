from dataset_agreement.abstract_action_space import AbstractActionSpace


class ActionSpace(AbstractActionSpace):

    def __init__(self, config):
        self.num_blocks = config["num_blocks"]
        self.num_directions = config["num_directions"]
        self.num_actions_ = self.num_blocks * self.num_directions + 1  # 1 for stopping
        AbstractActionSpace.__init__(self)

    def num_actions(self):
        return self.num_actions_

    def get_action_name(self, act_idx):
        if act_idx == self.num_actions_ - 1:
            return "Stop"

        # block_id major format
        block_id = int(act_idx / self.num_directions)
        direction_id = act_idx % self.num_directions

        if direction_id == 0:
            direction_id_str = "north"
        elif direction_id == 1:
            direction_id_str = "south"
        elif direction_id == 2:
            direction_id_str = "east"
        elif direction_id == 3:
            direction_id_str = "west"
        else:
            raise AssertionError("Found direction id of %s, expected one in 0, 1, 2, 3. " % direction_id)

        return str(block_id) + " " + direction_id_str

    def get_action_index(self, act_name):

        if act_name == "Stop":
            return self.num_actions_ - 1

        words = act_name.split(" ")
        block_id = int(words[0])
        direction_id = int(words[1])

        act_idx = block_id * self.num_directions + direction_id
        return act_idx

    def get_stop_action_index(self):
        return self.num_actions_ - 1
