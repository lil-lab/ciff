import logging

# from agent_observed_state import AgentObservedState
from utils.debug_nav_drone_instruction import instruction_to_string


class _GetchUnix:

    def __init__(self):
        pass

    def __call__(self):
        import sys, tty, termios
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(sys.stdin.fileno())
            ch = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        return ch


class KeyboardAgent:
    """ A class that allows users to control the agent and provides visual display
    TODO: Currently only supports nav_drone. Extend it to other types of agent.
    """

    def __init__(self, server, action_space, meta_data_util, config, constants):
        self.server = server
        self.action_space = action_space
        self.meta_data_util = meta_data_util
        self.config = config
        self.constants = constants

    def get_an_action(self):

        for i in range(0, 5):
            action = _GetchUnix()()
            action = action.lower()
            if action == "w":
                return 0   # forward
            if action == "d":
                return 1   # right
            if action == "a":
                return 2   # left
            if action == "p":
                return 3   # stop
            if action == "m":
                print("Thanks for playing")
                return -1
            print("Wrong key. w: Forward, d: right, a: Left, p: Stop and m: quit")
            continue

    def show_instruction(self, data_point, show_discourse=True):

        if show_discourse:
            paragraph_instruction = data_point.get_paragraph_instruction()
            start_index, end_index = data_point.get_instruction_indices()

            previous_instruction_string = instruction_to_string(paragraph_instruction[:start_index], self.config)
            instruction_string = instruction_to_string(paragraph_instruction[start_index:end_index], self.config)
            future_instruction_string = instruction_to_string(paragraph_instruction[end_index:], self.config)
            return previous_instruction_string + " \n /** " + instruction_string + " **/\n " + future_instruction_string
        else:
            instruction_string = instruction_to_string(data_point.get_instruction(), self.config)
            return instruction_string

    def test(self, test_dataset, show_discourse=True):

        self.server.clear_metadata()
        action_counts = [0] * self.action_space.num_actions()
        dataset_size = len(test_dataset)

        print("Hi user. There are 25 examples in this task")
        if show_discourse:
            print("For each example, follow the highlighted instruction in between /** **/. "
                  "Rest of it is for your clarification.")
        else:
            print("Follow the instruction for each example.")
        print("Key controls are---  w: Forward, d: right, a: Left, p: Stop and m: quit")

        metadata = {"feedback": ""}
        for data_point_ix, data_point in enumerate(test_dataset):
            image, metadata = self.server.reset_receive_feedback(data_point)
            print("\n\n\n\n\n\n\n")

            num_actions = 0
            max_num_actions = 500  # len(data_point.get_trajectory())
            max_num_actions += self.constants["max_extra_horizon"]

            instruction_string = self.show_instruction(data_point, show_discourse)
            logging.info("Instruction no. %r is %r", data_point_ix + 1, str(instruction_string))
            print("Instruction " + str(data_point_ix + 1) + " is: " + str(instruction_string))

            while True:

                # Use test policy to get the action
                action = self.get_an_action()
                if action == -1:  # player wants to quit.
                    return
                action_counts[action] += 1

                if action == self.action_space.get_stop_action_index() or num_actions >= max_num_actions:
                    # Send the action and get feedback
                    image, reward, metadata = self.server.halt_and_receive_feedback()

                    # Update the scores based on meta_data
                    self.meta_data_util.log_results(metadata)

                    print("Pressed stopped or ran out of actions. Done "
                          + str(data_point_ix + 1) + " out of " + str(dataset_size))
                    break
                else:
                    # Send the action and get feedback
                    image, reward, metadata = self.server.send_action_receive_feedback(action)

                    num_actions += 1

        logging.info("Human accuracy on dataset of size %r", len(test_dataset))
        self.meta_data_util.log_results(metadata)
        logging.info("Testing data action counts %r", action_counts)
