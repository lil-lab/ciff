import json
import logging
import textwrap
import traceback
import numpy as np
import sys

import time
import torch
import scipy.misc
import matplotlib.pyplot as plt
import utils.generic_policy as gp

from agents.agent_observed_state import AgentObservedState
from models.model.abstract_model import AbstractModel
from models.incremental_model.abstract_incremental_model import AbstractIncrementalModel
from utils.launch_unity import launch_k_unity_builds
from utils.tensorboard import Tensorboard


class HouseDecoupledPredictorNavigatorAgent:

    def __init__(self, server, goal_prediction_model, navigation_model, action_type_model,
                 test_policy, action_space, meta_data_util, config, constants):
        self.server = server
        self.goal_prediction_model = goal_prediction_model
        self.navigation_model = navigation_model
        self.action_type_model = action_type_model
        self.test_policy = test_policy
        self.action_space = action_space
        self.meta_data_util = meta_data_util
        self.config = config
        self.constants = constants

    def show_goal_location(self, image, metadata):

        new_image = image.swapaxes(0, 1).swapaxes(1, 2)
        row, col, row_real, col_real = self.get_goal(metadata)

        # Take the values
        kernel = np.zeros(shape=(32, 32), dtype=np.float32)
        if row is not None and col is not None:
            for i in range(-3, 3):
                for j in range(-3, 3):
                    if 0 <= i + row < 32 and 0 <= j + col < 32:
                        kernel[i + row][j + col] = 1.0

        kernel = scipy.misc.imresize(kernel, (128, 128))

        plt.clf()
        plt.imshow(new_image)
        plt.imshow(kernel, alpha=0.5)
        plt.show(block=False)

    @staticmethod
    def save_panorama_heat_maps(fig_id, panorama, region_ix, row, col, instruction_string):

        new_image = panorama.swapaxes(0, 1).swapaxes(1, 2)

        # Take the values
        kernel = np.zeros(shape=(32, 32 * 6), dtype=np.float32)
        if row is not None and col is not None:
            pad = region_ix * 32
            for i in range(-5, 5):
                for j in range(-5, 5):
                    if 0 <= i + row < 32 and 0 <= j + col + pad < 32 * 6:
                        kernel[i + row][j + col + pad] = 1.0

        kernel = scipy.misc.imresize(kernel, (128, 128 * 6))

        plt.clf()
        title = instruction_string + ". (Region=%d, Row=%d, Col=%d)" % (region_ix, row, col)
        plt.title("\n".join(textwrap.wrap(title, 80)))
        plt.imshow(new_image)
        plt.imshow(kernel, alpha=0.5)
        plt.savefig("./heat_maps/figure_%d.png" % fig_id)

    @staticmethod
    def save_large_panorama_heat_maps(fig_id, panorama, attention_prob, instruction_string, scale=1):

        new_image = panorama.swapaxes(0, 1).swapaxes(1, 2)

        # Take the values
        kernel = attention_prob[:-1].view(32, 192).cpu().data.numpy()
        kernel = scipy.misc.imresize(kernel, (128 * scale, 128 * scale * 6))
        print("Image shape %r and kernel shape %r" % (new_image.shape, kernel.shape))
        plt.clf()
        title = instruction_string
        plt.title("\n".join(textwrap.wrap(title, 80)))
        plt.imshow(new_image)
        plt.imshow(kernel, alpha=0.5)
        plt.savefig("./heat_maps/figure_%d.png" % fig_id)

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
                if metadata["navigation-error"] < 5.0:
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

    def get_goal(self, metadata, goal_type):

        if goal_type == "gold":

            if metadata["goal-screen"] is None:
                return None, None, None, None

            left, bottom, depth = metadata["goal-screen"]

        elif goal_type == "inferred":

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

    def _explore_and_set_tracking(self, data_point, data_point_ix, instruction_string):

        # Get the panoramic image
        panorama, _ = self.server.explore()

        ###########################################
        # original_large_panorama = panorama.copy()
        # panorama = scipy.misc.imresize(panorama.swapaxes(0, 1).swapaxes(1, 2), (128, 128*6, 3)).swapaxes(1, 2).swapaxes(0, 1)
        ###########################################

        # Get the panorama and predict the goal location
        state = AgentObservedState(instruction=data_point.instruction,
                                   config=self.config,
                                   constants=self.constants,
                                   start_image=panorama,
                                   previous_action=None,
                                   pose=None,
                                   position_orientation=None,
                                   data_point=data_point)
        volatile = self.goal_prediction_model.get_attention_prob(state, model_state=None)
        inferred_ix = int(torch.max(volatile["attention_logits"], 0)[1].data.cpu().numpy()[0])

        ##########################################
        # self.save_large_panorama_heat_maps(data_point_ix, original_large_panorama,
        #                                    volatile["attention_probs"], instruction_string, scale=5)
        ##########################################

        if inferred_ix == 6 * self.config["num_manipulation_row"] * self.config["num_manipulation_col"]:
            print("Predicting Out-of-sight")
            return None

        assert 0 <= inferred_ix < 6 * self.config["num_manipulation_row"] * self.config["num_manipulation_col"]

        row = int(inferred_ix / (6 * self.config["num_manipulation_col"]))
        col = inferred_ix % (6 * self.config["num_manipulation_col"])
        region_ix = int(col / self.config["num_manipulation_col"])

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
            raise AssertionError("region ix should be in {0, 1, 2, 3, 4, 5}. Found ", region_ix)

        col = col % self.config["num_manipulation_col"]

        # Set tracking
        row_value = min(1.0, (row + 0.5) / float(self.config["num_manipulation_row"]))
        col_value = min(1.0, (col + 0.5) / float(self.config["num_manipulation_col"]))

        message = self.server.set_tracking(camera_ix, row_value, col_value)

        # self.save_panorama_heat_maps(data_point_ix, panorama, region_ix, row, col, instruction_string)
        return message.decode("utf-8")

    def test_goal_distance(self, house_id, test_dataset, vocab, goal_type="inferred",
                         tensorboard=None, logger=None, pushover_logger=None):
        """ Perform a single step testing i.e. the goal prediction module is called only once. """

        self.server.clear_metadata()
        num_cases = 0
        task_completion_accuracy = 0
        mean_goal_distance = 0

        metadata = {"feedback": ""}

        save_json = []

        for data_point_ix, data_point in enumerate(test_dataset):

            instruction_string = " ".join([vocab[token_id] for token_id in data_point.instruction])
            self.log("Instruction is %r " % instruction_string, logger)

            # Call the navigation model
            image, metadata = self.server.reset_receive_feedback(data_point)

            # Get the panorama and set tracking
            message = self._explore_and_set_tracking(data_point, data_point_ix, instruction_string)
            next_goal_dist = message.split("#")[2].lower()
            self.log("Next goal distance %r" % next_goal_dist, logger)

            # Send the action and get feedback
            image, reward, metadata = self.server.halt_and_receive_feedback()

            if next_goal_dist != "none":

                save_json_ = dict()
                save_json_["Instruction"] = instruction_string
                next_goal_dist = float(next_goal_dist)
                save_json_["GoalPredictionError"] = next_goal_dist
                num_cases += 1
                mean_goal_distance += next_goal_dist

                if next_goal_dist <= 1.0:
                    save_json_["GoalTaskCompletion"] = "Success"
                    task_completion_accuracy += 1.0
                else:
                    save_json_["GoalTaskCompletion"] = "Failure"

                save_json_["StopDist"] = float(metadata["distance-to-next-goal"])
                save_json_["TrajLength"] = len(data_point.trajectory)
                save_json.append(save_json_)

            with open('chai_goal_prediction_error_house%d.txt' % house_id, 'w') as fout:
                json.dump(save_json, fout)

        mean_goal_distance = mean_goal_distance / float(max(num_cases, 1))
        task_completion_accuracy = (task_completion_accuracy * 100.0)/float(max(num_cases, 1))

        self.log("Overall test results:", logger)
        self.log("Testing: Num cases computed %d" % num_cases, logger)
        self.log("Testing: Mean Goal Distance Error: %r " % mean_goal_distance, logger)
        self.log("Testing: Mean Task Completion Accuracy: %r " % task_completion_accuracy, logger)

        if pushover_logger is not None:
            pushover_feedback = str(metadata["feedback"])
            pushover_logger.log(pushover_feedback)

    def test_single_step(self, test_dataset, vocab, goal_type="gold",
                         tensorboard=None, logger=None, pushover_logger=None):
        """ Perform a single step testing i.e. the goal prediction module is called only once. """

        self.server.clear_metadata()
        action_counts = [0] * self.action_space.num_actions()
        task_completion_accuracy = 0

        metadata = {"feedback": ""}

        for data_point_ix, data_point in enumerate(test_dataset):

            instruction_string = " ".join([vocab[token_id] for token_id in data_point.instruction])
            self.log("Instruction is %r " % instruction_string, logger)

            # Call the navigation model
            image, metadata = self.server.reset_receive_feedback(data_point)

            if goal_type == "inferred":
                # Get the panorama and set tracking
                self._explore_and_set_tracking(data_point, data_point_ix, instruction_string)

            state = AgentObservedState(instruction=data_point.instruction,
                                       config=self.config,
                                       constants=self.constants,
                                       start_image=image,
                                       previous_action=None,
                                       data_point=data_point)
            state.goal = self.get_goal(metadata, goal_type)
            num_actions = 0
            max_num_actions = self.constants["horizon"]
            model_state = None

            while True:

                # Generate probabilities over actions
                if isinstance(self.navigation_model, AbstractModel):
                    probabilities = list(torch.exp(self.navigation_model.get_probs(state).data))
                elif isinstance(self.navigation_model, AbstractIncrementalModel):
                    log_probabilities, model_state, _, _ = self.navigation_model.get_probs(state, model_state, volatile=True)
                    probabilities = list(torch.exp(log_probabilities.data))[0]
                else:
                    log_probabilities, model_state = self.navigation_model.get_probs(state, model_state)
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
                    self.log("Overall test results: %r " % metadata, logger)

                    #############################################
                    # Take a dummy manipulation action
                    # row, col, row_real, col_real = state.goal
                    # if row is not None and col is not None:
                    #     act_name = "interact %r %r" % (row, col)
                    #     interact_action = self.action_space.get_action_index(act_name)
                    #     image, reward, metadata = self.server.send_action_receive_feedback(interact_action)
                    #############################################

                    break
                else:
                    # Send the action and get feedback
                    image, reward, metadata = self.server.send_action_receive_feedback(action)
                    # self.log("Testing: Taking action %r and got reward %r " % (action, reward), logger)
                    # time.sleep(0.5)
                    # Update the agent state
                    state = state.update(image, action, data_point=data_point)
                    state.goal = self.get_goal(metadata, goal_type)
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

    def test_multi_step(self, test_dataset, vocab, num_outer_loop_steps, num_inner_loop_steps, goal_type="gold",
                        tensorboard=None, logger=None, pushover_logger=None):
        """ Perform a single step testing i.e. the goal prediction module is called only once. """

        self.server.clear_metadata()
        action_counts = [0] * self.action_space.num_actions()
        task_completion_accuracy = 0

        metadata = {"feedback": ""}

        for data_point_ix, data_point in enumerate(test_dataset):

            instruction_string = " ".join([vocab[token_id] for token_id in data_point.instruction])
            self.log("Instruction is %r " % instruction_string, logger)

            # Call the navigation model
            image, metadata = self.server.reset_receive_feedback(data_point)

            state = AgentObservedState(instruction=data_point.instruction,
                                       config=self.config,
                                       constants=self.constants,
                                       start_image=image,
                                       previous_action=None,
                                       data_point=data_point)
            num_actions = 0
            max_num_actions = self.constants["horizon"]
            model_state = None

            for outer_loop_iter in range(0, num_outer_loop_steps):

                if goal_type == "inferred":
                    # Get the panorama and set tracking
                    self._explore_and_set_tracking(data_point, data_point_ix, instruction_string)

                state.goal = self.get_goal(metadata, goal_type)

                for inner_loop_iter in range(0, num_inner_loop_steps):

                    # Generate probabilities over actions
                    if isinstance(self.navigation_model, AbstractModel):
                        probabilities = list(torch.exp(self.navigation_model.get_probs(state).data))
                    elif isinstance(self.navigation_model, AbstractIncrementalModel):
                        log_probabilities, model_state, _, _ = self.navigation_model.get_probs(state, model_state, volatile=True)
                        probabilities = list(torch.exp(log_probabilities.data))[0]
                    else:
                        log_probabilities, model_state = self.navigation_model.get_probs(state, model_state)
                        probabilities = list(torch.exp(log_probabilities.data))

                    # Use test policy to get the action
                    action = self.test_policy(probabilities)
                    action_counts[action] += 1

                    #############################################
                    # Take a dummy manipulation action
                    # row, col, row_real, col_real = state.goal
                    # if row is not None and col is not None:
                    #     act_name = "interact %r %r" % (row, col)
                    #     interact_action = self.action_space.get_action_index(act_name)
                    #     image, reward, metadata = self.server.send_action_receive_feedback(interact_action)
                    #############################################

                    if action == self.action_space.get_stop_action_index() or num_actions >= max_num_actions:
                        break
                    else:
                        # Send the action and get feedback
                        image, reward, metadata = self.server.send_action_receive_feedback(action)

                        # Update the agent state
                        state = state.update(image, action, data_point=data_point)
                        state.goal = self.get_goal(metadata, goal_type)
                        num_actions += 1

                if num_actions >= max_num_actions:
                    break

            # Send the action and get feedback
            image, reward, metadata = self.server.halt_and_receive_feedback()

            if metadata["navigation-error"] <= 1.0:
                task_completion_accuracy += 1

            # Update the scores based on meta_data
            # self.meta_data_util.log_results(metadata, logger)
            self.log("Overall test results: %r " % metadata, logger)

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

    def test_multi_step_action_types(self, test_dataset, vocab, goal_type=None,
                                     tensorboard=None, logger=None, pushover_logger=None):
        """ Perform a single step testing i.e. the goal prediction module is called only once. """

        self.server.clear_metadata()
        action_counts = [0] * self.action_space.num_actions()
        task_completion_accuracy = 0

        metadata = {"feedback": ""}
        text_embedding_model = self.goal_prediction_model.text_module

        for data_point_ix, data_point in enumerate(test_dataset):

            instruction_string = " ".join([vocab[token_id] for token_id in data_point.instruction])
            self.log("Instruction is %r " % instruction_string, logger)

            # Call the action type model to determine the number of steps
            token_indices = self.action_type_model.decoding_from_indices_to_indices(data_point.instruction,
                                                                                    text_embedding_model)

            print("Token indices ", token_indices)
            assert len(token_indices) <= 5

            # Call the navigation model
            image, metadata = self.server.reset_receive_feedback(data_point)

            state = AgentObservedState(instruction=data_point.instruction,
                                       config=self.config,
                                       constants=self.constants,
                                       start_image=image,
                                       previous_action=None,
                                       data_point=data_point)
            num_actions = 0
            max_num_actions = self.constants["horizon"]
            num_inner_loop_steps = int(max_num_actions/max(1, len(token_indices)))
            model_state = None

            for outer_loop_iter in range(0, len(token_indices)):

                if goal_type == "inferred":
                    # Get the panorama and set tracking
                    self._explore_and_set_tracking(data_point, data_point_ix, instruction_string)

                state.goal = self.get_goal(metadata, goal_type)

                for inner_loop_iter in range(0, num_inner_loop_steps):

                    # Generate probabilities over actions
                    if isinstance(self.navigation_model, AbstractModel):
                        probabilities = list(torch.exp(self.navigation_model.get_probs(state).data))
                    elif isinstance(self.navigation_model, AbstractIncrementalModel):
                        log_probabilities, model_state, _, _ = self.navigation_model.get_probs(state, model_state, volatile=True)
                        probabilities = list(torch.exp(log_probabilities.data))[0]
                    else:
                        log_probabilities, model_state = self.navigation_model.get_probs(state, model_state)
                        probabilities = list(torch.exp(log_probabilities.data))

                    # Use test policy to get the action
                    action = self.test_policy(probabilities)
                    action_counts[action] += 1

                    if token_indices[outer_loop_iter] == 1:
                        print("Performing interaction")
                        row, col, row_real, col_real = state.goal
                        if row is not None and col is not None:
                            act_name = "interact %r %r" % (row, col)
                            interact_action = self.action_space.get_action_index(act_name)
                            image, reward, metadata = self.server.send_action_receive_feedback(interact_action)

                    if action == self.action_space.get_stop_action_index() or num_actions >= max_num_actions:
                        break
                    else:
                        # Send the action and get feedback
                        image, reward, metadata = self.server.send_action_receive_feedback(action)

                        # Update the agent state
                        state = state.update(image, action, data_point=data_point)
                        state.goal = self.get_goal(metadata, goal_type)
                        num_actions += 1

                if num_actions >= max_num_actions:
                    break

            # Send the action and get feedback
            image, reward, metadata = self.server.halt_and_receive_feedback()

            if metadata["navigation-error"] <= 1.0:
                task_completion_accuracy += 1

            # Update the scores based on meta_data
            # self.meta_data_util.log_results(metadata, logger)
            self.log("Overall test results: %r " % metadata, logger)

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
        raise NotImplementedError()

    @staticmethod
    def do_test(house_id, goal_prediction_model, navigation_model, action_type_model, config, action_space,
                meta_data_util, constants, test_dataset, experiment_name, rank, server,
                logger, vocab, goal_type, use_pushover=False):
        try:
            HouseDecoupledPredictorNavigatorAgent.do_test_(house_id, goal_prediction_model, navigation_model,
                                                           action_type_model, config, action_space, meta_data_util,
                                                           constants, test_dataset, experiment_name, rank, server,
                                                           logger, vocab, goal_type, use_pushover)
        except Exception:
            exc_info = sys.exc_info()
            traceback.print_exception(*exc_info)

    @staticmethod
    def do_test_(house_id, goal_prediction_model, navigation_model, action_type_model, config,
                 action_space, meta_data_util, constants, test_dataset,
                 experiment_name, rank, server, logger, vocab, goal_type, use_pushover=False):

        logger.log("In Testing...")
        launch_k_unity_builds([config["port"]], "./house_" + str(house_id) + "_elmer.x86_64",
                              arg_str="--config ./AssetsHouse/config" + str(house_id) + ".json",
                              cwd="./simulators/house/")
        logger.log("Launched Builds.")
        server.initialize_server()
        logger.log("Server Initialized.")

        # Test policy
        test_policy = gp.get_argmax_action

        if rank == 0:  # client 0 creates a tensorboard server
            tensorboard = Tensorboard(experiment_name)
            logger.log('Created Tensorboard Server.')
        else:
            tensorboard = None

        if use_pushover:
            pushover_logger = None
        else:
            pushover_logger = None

        # Create the Agent
        tmp_agent = HouseDecoupledPredictorNavigatorAgent(server=server,
                                                          goal_prediction_model=goal_prediction_model,
                                                          navigation_model=navigation_model,
                                                          action_type_model=action_type_model,
                                                          test_policy=test_policy,
                                                          action_space=action_space,
                                                          meta_data_util=meta_data_util,
                                                          config=config,
                                                          constants=constants)
        logger.log("Created Agent.")
        tune_dataset_size = len(test_dataset)

        if tune_dataset_size > 0:
            # Test on tuning data
            # tmp_agent.test_single_step(test_dataset, vocab, goal_type=goal_type, tensorboard=tensorboard,
            #                            logger=logger, pushover_logger=pushover_logger)
            # tmp_agent.test_multi_step(test_dataset, vocab, num_outer_loop_steps=10, num_inner_loop_steps=4,
            #                           goal_type=goal_type, tensorboard=tensorboard, logger=logger,
            #                           pushover_logger=pushover_logger)
            # tmp_agent.test_multi_step_action_types(test_dataset, vocab, goal_type=goal_type, tensorboard=tensorboard,
            #                                        logger=logger, pushover_logger=pushover_logger)
            tmp_agent.test_goal_distance(house_id, test_dataset, vocab, goal_type=goal_type, tensorboard=tensorboard,
                                                   logger=logger, pushover_logger=pushover_logger)

    def log(self, message, logger=None):
        if logger is not None:
            logger.log(message)
        else:
            logging.info(message)
