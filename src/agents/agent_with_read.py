import logging
import scipy.misc
import numpy as np
import time
import torch

from agents.agent_observed_state import AgentObservedState
from utils.debug_nav_drone_instruction import instruction_to_string


class ReadPointerAgent:

    READ_MODE, ACT_MODE = range(2)

    def __init__(self, server, model, test_policy, action_space, meta_data_util,
                 config, constants):
        self.server = server
        self.model = model
        self.test_policy = test_policy
        self.action_space = action_space
        self.meta_data_util = meta_data_util
        self.config = config
        self.constants = constants

    @staticmethod
    def _save_agent_state(state, id_):
        image_memory = state.image_memory
        for ix, img in enumerate(image_memory):
            img = np.swapaxes(img, 0, 1)
            img = np.swapaxes(img, 1, 2)
            scipy.misc.imsave("./images/state_" + str(id_) + "_image_" + str(ix + 1) + ".png", img)

    @staticmethod
    def _save_image(img, id_):
        img = np.swapaxes(img, 0, 1)
        img = np.swapaxes(img, 1, 2)
        scipy.misc.imsave("./images/image_" + str(id_) + ".png", img)

    def test(self, test_dataset, tensorboard=None):

        self.server.clear_metadata()
        action_counts = dict()
        action_counts[ReadPointerAgent.READ_MODE] = [0] * 2
        action_counts[ReadPointerAgent.ACT_MODE] = [0] * self.action_space.num_actions()

        metadata = {"feedback": ""}

        for data_point in test_dataset:
            image, metadata = self.server.reset_receive_feedback(data_point)
            state = AgentObservedState(instruction=data_point.instruction,
                                       config=self.config,
                                       constants=self.constants,
                                       start_image=image,
                                       previous_action=None)

            num_actions = 0
            max_num_actions = len(data_point.get_trajectory())
            max_num_actions += self.constants["max_extra_horizon"]
            # self._save_agent_state(state, num_actions)
            mode = ReadPointerAgent.READ_MODE
            last_action_was_stop = False

            instruction = instruction_to_string(
                data_point.get_instruction(), self.config)
            print("TEST INSTRUCTION: %r" % instruction)
            print("")

            while True:
                # Generate probabilities over actions given the state and the mode
                probabilities = list(torch.exp(self.model.get_probs(state, mode).data))

                # Use test policy to get the action
                action = self.test_policy(probabilities)
                action_counts[mode][action] += 1
                # logging.info("Taking action-num=%d horizon=%d action=%s from %s",
                #              num_actions, max_num_actions, str(action), str(probabilities))

                if mode == ReadPointerAgent.READ_MODE:
                    # Take read action in this mode
                    if not state.are_tokens_left_to_be_read():
                        # force halt
                        action = 1
                    elif num_actions >= max_num_actions or last_action_was_stop:
                        # force read
                        action = 0

                    if action == 0:  # 0 is read action
                        state = state.update_on_read()
                        last_action_was_stop = False
                    elif action == 1:  # 1 is halt action
                        mode = ReadPointerAgent.ACT_MODE
                        last_action_was_stop = True
                    else:
                        raise AssertionError("Read mode only supports two actions: read(0) and halt(1). "
                                             + "Found " + str(action))
                elif mode == ReadPointerAgent.ACT_MODE:
                    # Take physical actions in this mode
                    if action == self.action_space.get_stop_action_index() or num_actions >= max_num_actions:
                        # If there are still tokens left to be read then read the token else halt
                        if state.are_tokens_left_to_be_read():
                            mode = ReadPointerAgent.READ_MODE
                            state = state.update_on_act_halt()
                            last_action_was_stop = True
                        else:
                            # Send the action and get feedback
                            image, reward, metadata = self.server.halt_and_receive_feedback()
                            if tensorboard is not None:
                                tensorboard.log_test_error(metadata["error"])

                            # Update the scores based on meta_data
                            # self.meta_data_util.log_results(metadata)
                            break
                    else:
                        # Send the action and get feedback
                        image, reward, metadata = self.server.send_action_receive_feedback(action)

                        # Update the agent state
                        state = state.update(image, action)
                        num_actions += 1
                        last_action_was_stop = False

                        # self._save_agent_state(state, num_actions)
                else:
                    raise AssertionError("Mode should be either read or act. Unhandled mode: " + str(mode))

        self.meta_data_util.log_results(metadata)
        logging.info("Testing data action counts %r", action_counts)

    def test_forced_reading(self, test_dataset, tensorboard=None):

        self.server.clear_metadata()
        action_counts = dict()
        action_counts[ReadPointerAgent.READ_MODE] = [0] * 2
        action_counts[ReadPointerAgent.ACT_MODE] = [0] * self.action_space.num_actions()

        metadata = {"feedback": ""}

        for data_point in test_dataset:
            image, metadata = self.server.reset_receive_feedback(data_point)
            oracle_segments = data_point.get_instruction_oracle_segmented()
            pose = int(metadata["y_angle"] / 15.0)
            state = AgentObservedState(instruction=data_point.instruction,
                                       config=self.config,
                                       constants=self.constants,
                                       start_image=image,
                                       previous_action=None,
                                       pose=pose)

            num_actions = 0
            max_num_actions = len(data_point.get_trajectory())
            max_num_actions += self.constants["max_extra_horizon"]
            current_segment_ix = 0
            mode = ReadPointerAgent.READ_MODE

            per_segment_budget = int(max_num_actions / len(oracle_segments))
            num_segment_actions = 0

            while True:

                if mode == ReadPointerAgent.READ_MODE:

                    num_segment_size = len(oracle_segments[current_segment_ix])
                    current_segment_ix += 1
                    for i in range(0, num_segment_size):
                        state = state.update_on_read()
                    mode = ReadPointerAgent.ACT_MODE

                elif mode == ReadPointerAgent.ACT_MODE:
                    # Take physical actions in this mode

                    # Generate probabilities over actions given the state and the mode
                    probabilities = list(torch.exp(self.model.get_probs(state, mode).data))

                    # Use test policy to get the action
                    action = self.test_policy(probabilities)
                    action_counts[mode][action] += 1

                    if action == self.action_space.get_stop_action_index() \
                            or num_actions >= max_num_actions or num_segment_actions > per_segment_budget:
                        # If there are still tokens left to be read then read the token else halt
                        self.server.flag_act_halt()
                        if state.are_tokens_left_to_be_read():
                            mode = ReadPointerAgent.READ_MODE
                            state = state.update_on_act_halt()
                            num_segment_actions = 0
                        else:
                            # Send the action and get feedback
                            image, reward, metadata = self.server.halt_and_receive_feedback()
                            if tensorboard is not None:
                                tensorboard.log_all_test_errors(
                                    metadata["edit_dist_error"],
                                    metadata["closest_dist_error"],
                                    metadata["stop_dist_error"])

                            # Update the scores based on meta_data
                            self.meta_data_util.log_results(metadata)
                            break
                    else:
                        # Send the action and get feedback
                        image, reward, metadata = self.server.send_action_receive_feedback(action)

                        # Update the agent state
                        pose = int(metadata["y_angle"] / 15.0)
                        state = state.update(image, action, pose=pose)
                        num_actions += 1
                        num_segment_actions += 1
                else:
                    raise AssertionError("Mode should be either read or act. Unhandled mode: " + str(mode))

        self.meta_data_util.log_results(metadata)
        logging.info("Testing data action counts %r", action_counts)
