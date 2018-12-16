import json
import logging

import time
import torch
import scipy.misc
import os
import math
import random
import numpy as np
import torch.nn.functional as F

from agents.agent_observed_state import AgentObservedState
from learning.auxiliary_objective.goal_prediction import GoalPrediction
from utils.debug_nav_drone_instruction import instruction_to_string
from models.model.abstract_model import AbstractModel
from models.incremental_model.abstract_incremental_model import AbstractIncrementalModel
from utils.geometry import get_turn_angle

NO_BUCKETS = 48
BUCKET_WIDTH = 360.0/(1.0*NO_BUCKETS)


class Agent:

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
    def save_image_and_metadata(image, state, data_point_ix, img_ctr):
        # Save the image
        scipy.misc.imsave("./synthetic_v2_10k_images/test_images/example_"
                          + str(data_point_ix) + "/image_" + str(img_ctr) + ".png",
                          image.swapaxes(0, 1).swapaxes(1, 2))

        # Save the visible information
        x_pos, z_pos, y_angle = state.get_position_orientation()
        landmark_pos_dict = state.get_landmark_pos_dict()
        symbolic_image = Agent.get_visible_landmark_r_theta(x_pos, z_pos, y_angle, landmark_pos_dict)
        with open("./synthetic_v2_10k_images/test_images/example_" + str(data_point_ix)
                           + "/data_" + str(img_ctr) + ".json", 'w') as fp:
            json.dump(symbolic_image, fp)

    def save_start_360_images(self, dataset, folder_name, save_format="numpy"):

        start = time.time()
        if os.path.exists(folder_name):
            raise AssertionError("Folder Exists. Refusing to proceed to avoid deleting existing content.")

        os.mkdir(folder_name)

        self.server.clear_metadata()

        dataset_size = len(dataset)
        for data_point_ix, data_point in enumerate(dataset):

            if (data_point_ix + 1) % 100 == 0:
                time_taken = time.time() - start
                processing_rate = (data_point_ix + 1) / float(time_taken)
                expected_completion_time = ((dataset_size - data_point_ix) * processing_rate)/60.0  # in minutes
                print("Processed %r out of %r, expected completion time %r min " % (data_point_ix + 1, dataset_size,
                                                                                    expected_completion_time))

            self.server.reset_receive_feedback(data_point)

            # Take the explore action which returns the 360 degree view
            image, _, _ = self.server.explore()

            os.mkdir(folder_name + "/example_" + str(data_point_ix))

            if save_format == "numpy":
                image = image.astype(np.float32, copy=False)
                np.save(folder_name + "/example_" + str(data_point_ix) + "/image_numpy.npy", image)
            elif save_format == "png":
                for img_ctr in range(0, 6):

                    image_slice = image[img_ctr * 3: (img_ctr + 1) * 3, :, :]

                    # Save the image
                    scipy.misc.imsave(folder_name + "/example_" + str(data_point_ix) +
                                      "/image_" + str(img_ctr) + ".png",
                                      image_slice.swapaxes(0, 1).swapaxes(1, 2))
            else:
                raise AssertionError("Unhandled save format ", save_format)

            self.server.halt_and_receive_feedback()

        logging.info("Processed. Total time in seconds: %r ", (time.time() - start))

    @staticmethod
    def get_all_landmark_r_theta(x_pos, z_pos, y_angle, landmark_pos_dict):
        landmark_r_theta_dict = {}
        for landmark, (landmark_x, landmark_z) in landmark_pos_dict.items():
            # get angle between drone's current orientation and landmark
            landmark_angle = 90.0 - np.arctan2(landmark_z - z_pos, landmark_x - x_pos) * 180.0 / math.pi
            angle_diff = landmark_angle - y_angle
            while angle_diff > 180.0:
                angle_diff -= 360.0
            while angle_diff < -180.0:
                angle_diff += 360.0
            angle_discrete = int((angle_diff + 180.0) / BUCKET_WIDTH)

            # get discretized radius
            radius = ((landmark_x - x_pos) ** 2 + (landmark_z - z_pos) ** 2) ** 0.5
            radius_discrete = int(radius / 5.0)

            landmark_r_theta_dict[landmark] = (radius_discrete, angle_discrete)
        return landmark_r_theta_dict

    @staticmethod
    def get_visible_landmark_r_theta(x_pos, z_pos, y_angle, landmark_pos_dict):
        landmark_r_theta_dict = {}
        for landmark, (landmark_x, landmark_z) in landmark_pos_dict.items():
            # get angle between drone's current orientation and landmark
            landmark_angle = 90.0 - np.arctan2(landmark_z - z_pos, landmark_x - x_pos) * 180.0 / math.pi
            angle_diff = landmark_angle - y_angle
            while angle_diff > 180.0:
                angle_diff -= 360.0
            while angle_diff < -180.0:
                angle_diff += 360.0
            if abs(angle_diff) <= 30.0:
                angle_discrete = int((angle_diff + 30.0) / BUCKET_WIDTH)
            else:
                angle_discrete = -1

            # get discretized radius
            radius = ((landmark_x - x_pos) ** 2 + (landmark_z - z_pos) ** 2) ** 0.5
            radius_discrete = int(radius / 5.0)

            landmark_r_theta_dict[landmark] = (radius_discrete, angle_discrete)
        return landmark_r_theta_dict

    def test_save_oracle_images(self, test_dataset, max_traj_len=None):

        self.server.clear_metadata()
        action_counts = [0] * self.action_space.num_actions()

        metadata = {"feedback": ""}
        for data_point_ix, data_point in enumerate(test_dataset):
            print("Resetting ", data_point.get_instruction())
            image, metadata = self.server.reset_receive_feedback(data_point)
            pose = int(metadata["y_angle"] / 15.0)
            position_orientation = (metadata["x_pos"], metadata["z_pos"],
                                    metadata["y_angle"])
            state = AgentObservedState(instruction=data_point.instruction,
                                       config=self.config,
                                       constants=self.constants,
                                       start_image=image,
                                       previous_action=None,
                                       pose=pose,
                                       position_orientation=position_orientation,
                                       data_point=data_point)

            num_actions = 0
            trajectory = data_point.get_trajectory()
            os.mkdir("./synthetic_v2_10k_images/test_images/example_" + str(data_point_ix))
            Agent.save_image_and_metadata(image, state, data_point_ix, num_actions)

            while True:

                # Use test policy to get the action
                if num_actions == len(trajectory) or (max_traj_len is not None and num_actions >= max_traj_len):
                    action = self.action_space.get_stop_action_index()
                else:
                    action = trajectory[num_actions]

                if action == self.action_space.get_stop_action_index():
                    # Send the action and get feedback
                    image, reward, metadata = self.server.halt_and_receive_feedback()

                    # Update the scores based on meta_data
                    self.meta_data_util.log_results(metadata)
                    break
                else:
                    # Send the action and get feedback
                    image, reward, metadata = self.server.send_action_receive_feedback(action)

                    # Update the agent state
                    pose = int(metadata["y_angle"] / 15.0)
                    position_orientation = (metadata["x_pos"],
                                            metadata["z_pos"],
                                            metadata["y_angle"])
                    state = state.update(
                        image, action, pose=pose,
                        position_orientation=position_orientation,
                        data_point=data_point)
                    num_actions += 1
                    Agent.save_image_and_metadata(image, state, data_point_ix, num_actions)

        logging.info("Overall test result: ")
        self.meta_data_util.log_results(metadata)
        logging.info("Testing data action counts %r", action_counts)

    @staticmethod
    def goal_prediction_accuracy(goal, volatile):
        return GoalPrediction.get_loss_and_prob(volatile, goal, 32, 32)

    @staticmethod
    def _l2_distance(pt1, pt2):
        x1, z1 = pt1
        x2, z2 = pt2
        return math.sqrt((x1 - z1) * (x1 - z1) + (x2 - z2) * (x2 - z2))

    @staticmethod
    def get_angle_distance(metadata, data_point):

        agent_pos = metadata["x_pos"], metadata["z_pos"]
        start_pose = metadata["y_angle"]
        dest_list = data_point.get_destination_list()
        pt_g = (dest_list[0][0], dest_list[0][1])

        # Compute the distance
        l2_distance = Agent._l2_distance(agent_pos, pt_g)

        # Compute the angle [-180, 180]
        turn_angle = get_turn_angle(agent_pos, start_pose, pt_g)

        return l2_distance, turn_angle

    @staticmethod
    def bennett_metric(datapoint1, datapoint2, metadata1, metadata2):
        """ Compute the Bennett metric given by:
            dist = e_12 + e_21 - e_11 - e22
            bennett_metric = 1/2 + dist/(4 * dist_12)
            larger the bennett metric the better it is. Values are bounded in [0, 1]
        """

        dest_1 = metadata1["x_pos"], metadata1["z_pos"]
        dest_2 = metadata2["x_pos"], metadata2["z_pos"]
        dest_list1 = datapoint1.get_destination_list()
        dest_list2 = datapoint2.get_destination_list()

        assert len(dest_list1) == 1 and len(dest_list2) == 1, "Should be a segment level task."

        pt_g1 = (dest_list1[0][0], dest_list1[0][1])
        pt_g2 = (dest_list2[0][0], dest_list2[0][1])

        dist_12 = Agent._l2_distance(pt_g1, pt_g2)
        e11 = Agent._l2_distance(dest_1, pt_g1)
        e22 = Agent._l2_distance(dest_2, pt_g2)
        e12 = Agent._l2_distance(dest_1, pt_g2)
        e21 = Agent._l2_distance(dest_2, pt_g1)

        metric_val = 0.5 + (e12 + e21 - e11 - e22) / max((4.0 * dist_12), 0.00001)
        return metric_val

    def test_goal_prediction(self, test_dataset, tensorboard=None, logger=None, pushover_logger=None):

        self.server.clear_metadata()
        action_counts = [0] * self.action_space.num_actions()
        task_completion_accuracy = 0

        sum_loss, count, sum_prob, goal_prob_count = 0, 0, 0, 0

        metadata = {"feedback": ""}
        for data_point_ix, data_point in enumerate(test_dataset):
            print("Datapoint index ", data_point_ix)
            image, metadata = self.server.reset_receive_feedback(data_point)
            pose = int(metadata["y_angle"] / 15.0)
            position_orientation = (metadata["x_pos"], metadata["z_pos"],
                                    metadata["y_angle"])
            state = AgentObservedState(instruction=data_point.instruction,
                                       config=self.config,
                                       constants=self.constants,
                                       start_image=image,
                                       previous_action=None,
                                       pose=pose,
                                       position_orientation=position_orientation,
                                       data_point=data_point,
                                       prev_instruction=data_point.get_prev_instruction(),
                                       next_instruction=data_point.get_next_instruction())

            ##################################
            state.goal = GoalPrediction.get_goal_location(metadata, data_point, 8, 8)
            print ("Instruction is ", instruction_to_string(data_point.instruction, self.config))
            ##################################

            # state.start_read_pointer, state.end_read_pointer = data_point.get_instruction_indices()
            num_actions = 0
            max_num_actions = self.constants["horizon"]
            model_state = None
            trajectory = data_point.get_trajectory()[0:1]
            trajectory_len = len(trajectory)

            while True:

                if num_actions == trajectory_len:
                    action = self.action_space.get_stop_action_index()
                else:
                    action = trajectory[num_actions]

                # Generate probabilities over actions
                if isinstance(self.model, AbstractModel):
                    raise NotImplementedError()
                elif isinstance(self.model, AbstractIncrementalModel):
                    log_probabilities, model_state, _, volatile = self.model.get_probs(state, model_state, volatile=True)
                    probabilities = list(torch.exp(log_probabilities.data))[0]
                    # Compute goal prediction accuracy
                    goal_loss, prob, _ = self.goal_prediction_accuracy(state.goal, volatile)
                    sum_loss += goal_loss
                    count += 1
                    if prob is not None:
                        sum_prob += prob
                        goal_prob_count += 1
                else:
                    raise NotImplementedError()
                    # log_probabilities, model_state = self.model.get_probs(state, model_state)
                    # probabilities = list(torch.exp(log_probabilities.data))

                action_counts[action] += 1

                if action == self.action_space.get_stop_action_index() or num_actions >= max_num_actions:
                    # Send the action and get feedback
                    image, reward, metadata = self.server.halt_and_receive_feedback()
                    if tensorboard is not None:
                        tensorboard.log_all_test_errors(
                            metadata["edit_dist_error"],
                            metadata["closest_dist_error"],
                            metadata["stop_dist_error"])

                    if metadata["stop_dist_error"] < 5.0:
                        task_completion_accuracy += 1

                    # Update the scores based on meta_data
                    self.meta_data_util.log_results(metadata)
                    break
                else:
                    # Send the action and get feedback
                    image, reward, metadata = self.server.send_action_receive_feedback(action)
                    # Update the agent state
                    pose = int(metadata["y_angle"] / 15.0)
                    position_orientation = (metadata["x_pos"],
                                            metadata["z_pos"],
                                            metadata["y_angle"])
                    state = state.update(
                        image, action, pose=pose,
                        position_orientation=position_orientation,
                        data_point=data_point)
                    ##################################
                    state.goal = GoalPrediction.get_goal_location(metadata, data_point, 8, 8)
                    ##################################
                    num_actions += 1

        print("Finished testing. Now logging.")
        task_completion_accuracy = (task_completion_accuracy * 100.0) / float(max(len(test_dataset), 1))
        self.log("Overall test results:", logger)
        self.log("Testing: Task completion accuracy is: %r" % task_completion_accuracy, logger)
        self.log("Testing: Final Metadata: %r" % metadata, logger)
        self.log("Testing: Action Distribution: %r" % action_counts, logger)
        self.log("Goal Count %r, Mean Goal Loss %r" % (count, sum_loss / float(count)), logger)
        self.log("Goal Prob Count %r, Mean Goal Prob %r" % (goal_prob_count, sum_prob / float(goal_prob_count)), logger)

        self.meta_data_util.log_results(metadata, logger)
        self.log("Testing data action counts %r" % action_counts, logger)
        if pushover_logger is not None:
            pushover_feedback = str(metadata["feedback"]) + " --- " + "task_completion_accuracy=%r" % task_completion_accuracy
            pushover_logger.log(pushover_feedback)

    def _test(self, data_point, tensorboard=None):

        image, metadata = self.server.reset_receive_feedback(data_point)
        pose = int(metadata["y_angle"] / 15.0)
        position_orientation = (metadata["x_pos"], metadata["z_pos"],
                                metadata["y_angle"])
        state = AgentObservedState(instruction=data_point.instruction,
                                   config=self.config,
                                   constants=self.constants,
                                   start_image=image,
                                   previous_action=None,
                                   pose=pose,
                                   position_orientation=position_orientation,
                                   data_point=data_point,
                                   prev_instruction=data_point.get_prev_instruction(),
                                   next_instruction=data_point.get_next_instruction())

        ##################################
        state.goal = GoalPrediction.get_goal_location(metadata, data_point, 32, 32)
        ##################################

        # state.start_read_pointer, state.end_read_pointer = data_point.get_instruction_indices()
        num_actions = 0
        max_num_actions = self.constants["horizon"]
        model_state = None
        actions = []

        ###################################
        # distance, angle = self.get_angle_distance(metadata, data_point)
        ###################################

        while True:

            # Generate probabilities over actions
            if isinstance(self.model, AbstractModel):
                probabilities = list(torch.exp(self.model.get_probs(state).data))
            elif isinstance(self.model, AbstractIncrementalModel):
                log_probabilities, model_state, _, _ = self.model.get_probs(state, model_state, volatile=True)
                probabilities = list(torch.exp(log_probabilities.data))[0]
            else:
                raise AssertionError("Unhandled Model type.")

            # Use test policy to get the action
            action = self.test_policy(probabilities)
            actions.append(action)

            if action == self.action_space.get_stop_action_index() or num_actions >= max_num_actions:
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
                position_orientation = (metadata["x_pos"],
                                        metadata["z_pos"],
                                        metadata["y_angle"])
                state = state.update(
                    image, action, pose=pose,
                    position_orientation=position_orientation,
                    data_point=data_point)
                ##################################
                state.goal = GoalPrediction.get_goal_location(metadata, data_point, 32, 32)
                ##################################
                num_actions += 1

        # logging.info("Error, Start-Distance, Turn-Angle,  %r %r %r", metadata["stop_dist_error"], distance, angle)
        return metadata, actions

    def test_minimal_linguistic_pairs(self, test_dataset, tensorboard=None, logger=None, pushover_logger=None):

        self.server.clear_metadata()
        action_counts = [0] * self.action_space.num_actions()

        # For minimal linguistic pairs we compute two success metric:
        task_completion_accuracy = 0
        task_bennett_metric = 0

        # Accuracy for individual examples
        single_completion_accuracy = 0
        single_distance_regression = 0

        metadata = {"feedback": ""}
        data_point_ix = 0
        for data_point_ix, tup_data_point in enumerate(test_dataset):

            # The two minimal linguistic pairs
            data_point1 = tup_data_point[0]
            data_point2 = tup_data_point[1]

            metadata1, _ = self._test(data_point1, tensorboard)
            metadata2, _ = self._test(data_point2, tensorboard)
            metadata = metadata2  # Last metadata

            if metadata1["stop_dist_error"] < 5.0:
                single_completion_accuracy += 1
            if metadata2["stop_dist_error"] < 5.0:
                single_completion_accuracy += 1

            if metadata1["stop_dist_error"] < 5.0 and metadata2["stop_dist_error"] < 5.0:
                task_completion_accuracy += 1
            task_bennett_metric += self.bennett_metric(data_point1, data_point2, metadata1, metadata2)

            single_distance_regression += metadata1["stop_dist_error"] + metadata2["stop_dist_error"]

            if (data_point_ix + 1) % 100 == 0:  # print intermediate results
                self.log("Results after %r " % (data_point_ix + 1), logger)
                temp_task_completion_accuracy = (task_completion_accuracy * 100.0) / float(data_point_ix + 1)
                temp_task_bennett_metric = (task_bennett_metric * 100.0) / float(data_point_ix + 1)
                temp_single_completion_accuracy = (single_completion_accuracy * 100.0) / (2.0 * float(data_point_ix + 1))
                temp_single_distance_regression = single_distance_regression / (2.0 * float(data_point_ix + 1))
                self.log("Testing: Minimal Linguistic Pair Task completion accuracy is: %r"
                         % temp_task_completion_accuracy, logger)
                self.log("Testing: Minimal Linguistic Pair Bennett Metric is: %r" % temp_task_bennett_metric, logger)
                self.log("Testing: Single completion accuracy is: %r" % temp_single_completion_accuracy, logger)
                self.log("Testing: Single distance regression accuracy is: %r" % temp_single_distance_regression, logger)

        dataset_size = data_point_ix + 1
        task_completion_accuracy = (task_completion_accuracy * 100.0) / float(max(dataset_size, 1))
        task_bennett_metric = (task_bennett_metric * 100.0) / float(max(dataset_size, 1))
        single_completion_accuracy = (single_completion_accuracy * 100.0) / float(max(2 * dataset_size, 1))
        single_distance_regression = single_distance_regression / float(max(2 * dataset_size, 1))

        self.log("Overall test results:", logger)
        self.log("Testing: Minimal Linguistic Pair Task completion accuracy is: %r" % task_completion_accuracy, logger)
        self.log("Testing: Minimal Linguistic Pair Bennett Metric is: %r" % task_bennett_metric, logger)
        self.log("Testing: Single completion accuracy is: %r" % single_completion_accuracy, logger)
        self.log("Testing: Single distance regression accuracy is: %r" % single_distance_regression, logger)
        self.log("Testing: Final Metadata: %r" % metadata, logger)
        self.log("Testing data action counts %r" % action_counts, logger)
        self.meta_data_util.log_results(metadata, logger)
        if pushover_logger is not None:
            pushover_feedback = str(metadata["feedback"]) +\
                                " --- " + "task_completion_accuracy=%r" % task_completion_accuracy
            pushover_logger.log(pushover_feedback)

    def test(self, test_dataset, tensorboard=None, logger=None,
             pushover_logger=None):

        self.server.clear_metadata()
        action_counts = [0] * self.action_space.num_actions()
        task_completion_accuracy = 0

        metadata = {"feedback": ""}
        for data_point_ix, data_point in enumerate(test_dataset):
            metadata, actions_taken = self._test(data_point, tensorboard)

            if metadata["stop_dist_error"] < 5.0:
                task_completion_accuracy += 1

            for action in actions_taken:
                action_counts[action] += 1

        task_completion_accuracy = (task_completion_accuracy * 100.0) / float(max(len(test_dataset), 1))
        self.log("Overall test results:", logger)
        self.log("Testing: Task completion accuracy is: %r" % task_completion_accuracy, logger)
        self.log("Testing: Final Metadata: %r" % metadata, logger)
        self.log("Testing: Action Distribution: %r" % action_counts, logger)
        self.log("Testing data action counts %r" % action_counts, logger)
        self.meta_data_util.log_results(metadata, logger)
        if pushover_logger is not None:
            pushover_feedback = str(metadata["feedback"]) +\
                                " --- " + "task_completion_accuracy=%r" % task_completion_accuracy
            pushover_logger.log(pushover_feedback)

    def test_auto_segmented(self, test_dataset, segmenting_type="oracle",
                            tensorboard=None, logger=None, pushover_logger=None):

        assert segmenting_type in ("auto", "oracle")
        self.server.clear_metadata()
        action_counts = [0] * self.action_space.num_actions()

        self.log("Performing testing on paragraphs with segmenting type %r" % segmenting_type, logger)
        metadata = {"feedback": ""}

        task_completion_accuracy = 0

        for data_point in test_dataset:
            if segmenting_type == "auto":
                segmented_instruction = data_point.get_instruction_auto_segmented()
            else:
                segmented_instruction = data_point.get_instruction_oracle_segmented()

            max_num_actions = self.constants["horizon"]
            image, metadata = self.server.reset_receive_feedback(data_point)

            for instruction_i, instruction in enumerate(segmented_instruction):

                pose = int(metadata["y_angle"] / 15.0)
                position_orientation = (metadata["x_pos"], metadata["z_pos"],
                                        metadata["y_angle"])
                state = AgentObservedState(instruction=instruction,
                                           config=self.config,
                                           constants=self.constants,
                                           start_image=image,
                                           previous_action=None,
                                           pose=pose,
                                           position_orientation=position_orientation,
                                           data_point=data_point,
                                           prev_instruction=data_point.get_prev_instruction(),
                                           next_instruction=data_point.get_next_instruction())

                # Reset the actions taken and model state
                num_actions = 0
                model_state = None

                while True:

                    # Generate probabilities over actions
                    if isinstance(self.model, AbstractModel):
                        probabilities = list(torch.exp(self.model.get_probs(state).data))
                    elif isinstance(self.model, AbstractIncrementalModel):
                        log_probabilities, model_state, _, _ = self.model.get_probs(state, model_state, volatile=True)
                        probabilities = list(torch.exp(log_probabilities.data))[0]
                    else:
                        raise AssertionError("Unhandled Model type.")

                    # Use test policy to get the action
                    action = self.test_policy(probabilities)
                    action_counts[action] += 1

                    if action == self.action_space.get_stop_action_index() or num_actions >= max_num_actions:
                        # Compute the l2 distance

                        intermediate_goal = data_point.get_destination_list()[instruction_i]
                        agent_position = metadata["x_pos"], metadata["z_pos"]
                        distance = self._l2_distance(agent_position, intermediate_goal)
                        # logging.info("Agent: Position %r got Distance %r " % (instruction_i + 1, distance))
                        # self.log("Agent: Position %r got Distance %r " % (instruction_i + 1, distance), logger)
                        break

                    else:
                        # Send the action and get feedback
                        image, reward, metadata = self.server.send_action_receive_feedback(action)

                        # Update the agent state
                        pose = int(metadata["y_angle"] / 15.0)
                        position_orientation = (metadata["x_pos"],
                                                metadata["z_pos"],
                                                metadata["y_angle"])
                        state = state.update(
                            image, action, pose=pose,
                            position_orientation=position_orientation,
                            data_point=data_point)
                        num_actions += 1

            image, reward, metadata = self.server.halt_and_receive_feedback()
            if tensorboard is not None:
                tensorboard.log_all_test_errors(
                    metadata["edit_dist_error"],
                    metadata["closest_dist_error"],
                    metadata["stop_dist_error"])

            # Update the scores based on meta_data
            self.meta_data_util.log_results(metadata)

            if metadata["stop_dist_error"] < 5.0:
                task_completion_accuracy += 1

        logging.info("Testing data action counts %r", action_counts)
        task_completion_accuracy = (task_completion_accuracy * 100.0) / float(max(len(test_dataset), 1))
        self.log("Overall test results:", logger)
        self.log("Testing: Task completion accuracy is: %r" % task_completion_accuracy, logger)
        self.log("Testing: Final Metadata: %r" % metadata, logger)
        self.log("Testing: Action Distribution: %r" % action_counts, logger)
        self.log("Testing data action counts %r" % action_counts, logger)
        self.meta_data_util.log_results(metadata, logger)
        if pushover_logger is not None:
            pushover_feedback = str(metadata["feedback"]) + \
                                " --- " + "task_completion_accuracy=%r" % task_completion_accuracy
            pushover_logger.log(pushover_feedback)

    @staticmethod
    def log(message, logger=None):
        if logger is not None:
            logger.log(message)
        else:
            logging.info(message)
