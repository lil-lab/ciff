import logging
import time

import numpy as np
import torch
import nltk
import scipy.misc
import matplotlib.pyplot as plt

from agents.agent_observed_state import AgentObservedState
from utils.debug_nav_drone_instruction import instruction_to_string
from models.model.abstract_model import AbstractModel
from models.incremental_model.abstract_incremental_model import AbstractIncrementalModel


class TmpHouseAgent:

    def __init__(self, server, model, test_policy, action_space, meta_data_util,
                 config, constants):
        self.server = server
        self.model = model
        self.test_policy = test_policy
        self.action_space = action_space
        self.meta_data_util = meta_data_util
        self.config = config
        self.constants = constants

    def show_goal_location(self, image, metadata, goal_type="gold", size=1):

        new_image = image.swapaxes(0, 1).swapaxes(1, 2)
        row, col, row_real, col_real = self.get_goal(metadata, goal_type)

        # Take the values
        kernel = np.zeros(shape=(32, 32 * size), dtype=np.float32)
        if row is not None and col is not None:
            for i in range(-3, 3):
                for j in range(-3, 3):
                    if 0 <= i + row < 32 and 0 <= j + col < 32 * size:
                        kernel[i + row][j + col] = 1.0

        kernel = scipy.misc.imresize(kernel, (128, 128 * size))

        plt.clf()
        plt.imshow(new_image)
        plt.imshow(kernel, alpha=0.5)
        plt.show(block=False)

    def debug_manual_control(self, data_point, vocab):

        self.server.clear_metadata()
        task_completion_accuracy = 0

        image, metadata = self.server.reset_receive_feedback(data_point)
        state = AgentObservedState(instruction=data_point.instruction,
                                   config=self.config,
                                   constants=self.constants,
                                   start_image=image,
                                   previous_action=None,
                                   data_point=data_point)
        num_actions = 0
        print("Instruction is ", " ".join([vocab[index] for index in data_point.instruction]))
        plt.ion()

        while True:
            # Show the goal location
            self.show_goal_location(image, metadata)

            incorrect_action = True
            action_string = None
            while incorrect_action:
                action_string = input("Take the action. 0: Forward, 1: Left, 2: Right, 3: Stop, 4: Interact\n")
                if action_string in ['0', '1', '2', '3', '4']:
                    incorrect_action = False
                if action_string == '4':
                    interact_values = input("Enter the row and column in format: row col")
                    row, col = interact_values.split()
                    row, col = int(row), int(col)
                    action_string = 4 + row * 32 + col

            action = int(action_string)
            action_name = self.action_space.get_action_name(action)

            if action == self.action_space.get_stop_action_index():
                # Send the action and get feedback
                image, reward, metadata = self.server.halt_and_receive_feedback()

                print("Metadata is ", metadata)
                if metadata["navigation-error"] <= 1.0:
                    task_completion_accuracy += 1
                break
            else:
                # Send the action and get feedback
                image, reward, metadata = self.server.send_action_receive_feedback(action)
                # Update the agent state
                state = state.update(
                    image, action, data_point=data_point)
                num_actions += 1

            print("Metadata is ", metadata)
            print("Took action %r, Got reward %r" % (action_name, reward))

    def test_human_performance(self, dataset, vocab, logger):

        self.server.clear_metadata()

        for data_point in dataset:

            task_completion_accuracy = 0

            image, metadata = self.server.reset_receive_feedback(data_point)
            state = AgentObservedState(instruction=data_point.instruction,
                                       config=self.config,
                                       constants=self.constants,
                                       start_image=image,
                                       previous_action=None,
                                       data_point=data_point)
            num_actions = 0
            print("Instruction is ", " ".join([vocab[index] for index in data_point.instruction]))

            while True:

                incorrect_action = True
                action_string = None
                while incorrect_action:
                    action_string = input("Take the action. 0: Forward, 1: Left, 2: Right, 3: Stop, 4: Interact\n")
                    if action_string in ['0', '1', '2', '3', '4']:
                        incorrect_action = False
                    if action_string == '4':
                        interact_values = input("Enter the row and column in format: row col")
                        row, col = interact_values.split()
                        row, col = int(row), int(col)
                        action_string = 4 + row * 32 + col

                action = int(action_string)

                if action == self.action_space.get_stop_action_index():
                    # Send the action and get feedback
                    image, reward, metadata = self.server.halt_and_receive_feedback()

                    if metadata["navigation-error"] <= 1.0:
                        task_completion_accuracy += 1
                        logger.log("Completed the task")
                    logger.log("Meta data is %r " % metadata)
                    break
                else:
                    # Send the action and get feedback
                    image, reward, metadata = self.server.send_action_receive_feedback(action)
                    # Update the agent state
                    state = state.update(
                        image, action, data_point=data_point)
                    num_actions += 1

    def debug_tracking(self, data_point, vocab):

        self.server.clear_metadata()
        task_completion_accuracy = 0

        image, metadata = self.server.reset_receive_feedback(data_point)
        state = AgentObservedState(instruction=data_point.instruction,
                                   config=self.config,
                                   constants=self.constants,
                                   start_image=image,
                                   previous_action=None,
                                   data_point=data_point)
        num_actions = 0
        print("Instruction is ", " ".join([vocab[index] for index in data_point.instruction]))
        plt.ion()

        # Get the panoramic image
        panorama, _ = self.server.explore()

        # Show the goal location
        self.show_goal_location(panorama, metadata, size=6)

        tracking_values = input("Enter the region, row and column for tracking.")
        region_ix, row, col = [int(w) for w in tracking_values.split()]
        if region_ix == 0:
            camera_ix = 3
        elif region_ix == 1:
            camera_ix = 4
        elif region_ix == 2:
            camera_ix = 5
        elif region_ix == 3:
            camera_ix = 0
        elif region_ix == 4:
            camera_ix = 1
        elif region_ix == 5:
            camera_ix = 2
        else:
            raise AssertionError("Region ix should be in {0, 1, 2, 3, 4, 5}")
        
        row_value = row/32.0
        col_value = col/32.0
        self.server.set_tracking(camera_ix, row_value, col_value)
        input("Tracking done. Enter to continue")

        while True:

            # Show the goal location
            self.show_goal_location(image, metadata, goal_type="inferred", size=1)

            incorrect_action = True
            action_string = None
            while incorrect_action:
                action_string = input("Take the action. 0: Forward, 1: Left, 2: Right, 3: Stop, 4: Interact\n")
                if action_string in ['0', '1', '2', '3', '4']:
                    incorrect_action = False
                if action_string == '4':
                    interact_values = input("Enter the row and column in format: row col")
                    row, col = interact_values.split()
                    row, col = int(row), int(col)
                    action_string = 4 + row * 32 + col

            action = int(action_string)
            action_name = self.action_space.get_action_name(action)

            if action == self.action_space.get_stop_action_index():
                # Send the action and get feedback
                image, reward, metadata = self.server.halt_and_receive_feedback()

                print("Metadata is ", metadata)
                if metadata["navigation-error"] <= 1.0:
                    task_completion_accuracy += 1
                break
            else:
                # Send the action and get feedback
                image, reward, metadata = self.server.send_action_receive_feedback(action)
                # Update the agent state
                state = state.update(
                    image, action, data_point=data_point)
                num_actions += 1

            print("Metadata is ", metadata)
            print("Took action %r, Got reward %r" % (action_name, reward))

    def get_goal(self, metadata, goal_type="gold"):

        if goal_type == "gold":

            if metadata["goal-screen"] is None:
                return None, None, None, None

            left, bottom, depth = metadata["goal-screen"]

        elif goal_type == "inferred":

            print("Meta Data is ", metadata["tracking"])
            if metadata["tracking"] is None:
                return None, None, None, None

            left, bottom, depth = metadata["tracking"]

        else:
            raise AssertionError("Unknown goal type %r. Supported goal types are gold and inferred.", goal_type)

        if 0.01 < left < self.config["image_width"] and \
                                0.01 < bottom < self.config["image_height"] and depth > 0.01:

            scaled_left = left / float(self.config["image_width"])
            scaled_top = 1.0 - bottom / float(self.config["image_height"])

            row_real = self.config["num_manipulation_row"] * scaled_top
            col_real = self.config["num_manipulation_col"] * scaled_left
            row, col = round(row_real), round(col_real)

            if row < 0:
                row = 0
            elif row >= self.config["num_manipulation_row"]:
                row = self.config["num_manipulation_row"] - 1
            if col < 0:
                col = 0
            elif col >= self.config["num_manipulation_col"]:
                col = self.config["num_manipulation_col"] - 1

            return row, col, row_real, col_real
        else:
            return None, None, None, None

    def test(self, test_dataset, vocab, tensorboard=None, logger=None,
             pushover_logger=None):

        self.server.clear_metadata()
        action_counts = [0] * self.action_space.num_actions()
        task_completion_accuracy = 0

        metadata = {"feedback": ""}
        for data_point_ix, data_point in enumerate(test_dataset):
            image, metadata = self.server.reset_receive_feedback(data_point)
            state = AgentObservedState(instruction=data_point.instruction,
                                       config=self.config,
                                       constants=self.constants,
                                       start_image=image,
                                       previous_action=None,
                                       data_point=data_point)
            state.goal = self.get_goal(metadata)
            # state.start_read_pointer, state.end_read_pointer = data_point.get_instruction_indices()
            num_actions = 0
            max_num_actions = self.constants["horizon"]
            model_state = None

            while True:

                # Generate probabilities over actions
                if isinstance(self.model, AbstractModel):
                    probabilities = list(torch.exp(self.model.get_probs(state).data))
                elif isinstance(self.model, AbstractIncrementalModel):
                    log_probabilities, model_state, _, _ = self.model.get_probs(state, model_state, volatile=True)
                    probabilities = list(torch.exp(log_probabilities.data))[0]
                else:
                    log_probabilities, model_state = self.model.get_probs(state, model_state)
                    probabilities = list(torch.exp(log_probabilities.data))
                    # raise AssertionError("Unhandled Model type.")

                # Use test policy to get the action
                action = self.test_policy(probabilities)
                action_counts[action] += 1

                if action == self.action_space.get_stop_action_index() or num_actions >= max_num_actions:
                    # Send the action and get feedback
                    image, reward, metadata = self.server.halt_and_receive_feedback()
                    # if tensorboard is not None:
                    #     tensorboard.log_all_test_errors(
                    #         metadata["edit_dist_error"],
                    #         metadata["closest_dist_error"],
                    #         metadata["stop_dist_error"])

                    # self.log("Testing: Taking stop action and got reward %r " % reward, logger)

                    if metadata["navigation-error"] <= 1.0:
                        task_completion_accuracy += 1

                    # Update the scores based on meta_data
                    # self.meta_data_util.log_results(metadata, logger)
                    # self.log("Overall test results: %r " % metadata, logger)
                    break
                else:
                    # Send the action and get feedback
                    image, reward, metadata = self.server.send_action_receive_feedback(action)
                    # self.log("Testing: Taking action %r and got reward %r " % (action, reward), logger)
                    # time.sleep(0.5)
                    # Update the agent state
                    state = state.update(image, action, data_point=data_point)
                    state.goal = self.get_goal(metadata)
                    num_actions += 1

        task_completion_accuracy = (task_completion_accuracy * 100.0)/float(max(len(test_dataset), 1))
        self.log("Overall test results:", logger)
        self.log("Testing: Final Metadata: %r" % metadata, logger)
        self.log("Testing: Action Distribution: %r" % action_counts, logger)
        self.log("Testing: Task Completion Accuracy: %r " % task_completion_accuracy, logger)
        # self.meta_data_util.log_results(metadata, logger)
        self.log("Testing data action counts %r" % action_counts, logger)
        if pushover_logger is not None:
            pushover_feedback = str(metadata["feedback"])
            pushover_logger.log(pushover_feedback)

    def test_auto_segmented(self, test_dataset, tensorboard=None,
                            segmenting_type="auto"):
        assert segmenting_type in ("auto", "oracle")
        self.server.clear_metadata()
        action_counts = [0] * self.action_space.num_actions()

        metadata = ""

        for data_point in test_dataset:
            if segmenting_type == "auto":
                segmented_instruction = data_point.get_instruction_auto_segmented()
            else:
                segmented_instruction = data_point.get_instruction_oracle_segmented()
            num_segments = len(segmented_instruction)
            gold_num_actions = len(data_point.get_trajectory())
            horizon = gold_num_actions // num_segments
            horizon += self.constants["max_extra_horizon_auto_segmented"]

            image, metadata = self.server.reset_receive_feedback(data_point)

            instruction = instruction_to_string(
                data_point.get_instruction(), self.config)
            print ("TEST INSTRUCTION: %r" % instruction)
            print ("")

            for instruction_i, instruction in enumerate(segmented_instruction):

                state = AgentObservedState(instruction=instruction,
                                           config=self.config,
                                           constants=self.constants,
                                           start_image=image,
                                           previous_action=None,
                                           prev_instruction=data_point.get_prev_instruction(),
                                           next_instruction=data_point.get_next_instruction)

                num_actions = 0
                # self._save_agent_state(state, num_actions)

                while True:

                    # Generate probabilities over actions
                    probabilities = list(torch.exp(self.model.get_probs(state).data))
                    # print "test probs:", probabilities

                    # Use test policy to get the action
                    action = self.test_policy(probabilities)
                    action_counts[action] += 1

                    # logging.info("Taking action-num=%d horizon=%d action=%s from %s",
                    #              num_actions, max_num_actions, str(action), str(probabilities))

                    if action == self.action_space.get_stop_action_index() or num_actions >= horizon:
                        break

                    else:
                        # Send the action and get feedback
                        image, reward, metadata = self.server.send_action_receive_feedback(action)

                        # Update the agent state
                        state = state.update(image, action)
                        num_actions += 1

            _,  _, metadata = self.server.halt_and_receive_feedback()
            if tensorboard is not None:
                tensorboard.log_test_error(metadata["error"])

        self.meta_data_util.log_results(metadata)
        logging.info("Testing data action counts %r", action_counts)

    def log(self, message, logger=None):
        if logger is not None:
            logger.log(message)
        else:
            logging.info(message)
