import logging
import numpy as np

from agents.agent_observed_state import AgentObservedState


class AbstractBaseline(object):
    def __init__(self, server, action_space, meta_data_util, config, constants):
        self.server = server
        self.action_space = action_space
        self.meta_data_util = meta_data_util
        self.config = config
        self.constants = constants
        self.baseline_name = "abstract_baseline"

    def test_baseline(self, test_dataset):

        self.server.clear_metadata()

        metadata = {"feedback": ""}
        num_actions_list = []
        task_completion_accuracy = 0
        for data_point in test_dataset:
            image, metadata = self.server.reset_receive_feedback(data_point)
            state = AgentObservedState(instruction=data_point.instruction,
                                       config=self.config,
                                       constants=self.constants,
                                       start_image=image,
                                       previous_action=None)

            num_actions = 0
            # max_num_actions = len(data_point.get_trajectory())
            # max_num_actions += self.constants["max_extra_horizon"]
            num_segments = len(data_point.get_instruction_oracle_segmented())
            max_num_actions = self.constants["horizon"] * num_segments

            while True:

                action = self.get_next_action(data_point, num_actions)

                if action == self.action_space.get_stop_action_index() or num_actions >= max_num_actions:
                    # Send the action and get feedback
                    image, reward, metadata = self.server.halt_and_receive_feedback()
                    num_actions_list.append(num_actions)
                    self.meta_data_util.log_results(metadata)

                    if metadata["stop_dist_error"] < 5.0:
                        task_completion_accuracy += 1
                    break

                else:
                    # Send the action and get feedback
                    image, reward, metadata = self.server.send_action_receive_feedback(action)

                    # Update the agent state
                    state = state.update(image, action)
                    num_actions += 1

                    # self._save_agent_state(state, num_actions)

        self.meta_data_util.log_results(metadata)
        task_completion_accuracy /= float(max(len(test_dataset), 1))
        task_completion_accuracy *= 100.0
        mean_num_actions = float(np.array(num_actions_list).mean())
        logging.info("Task completion accuracy %r", task_completion_accuracy)
        logging.info("Done testing baseline %r, mean num actions is %f",
                     self.baseline_name, mean_num_actions)

    def get_next_action(self, data_point, num_actions):
        raise NotImplementedError()