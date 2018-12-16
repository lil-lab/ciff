import json
import logging
import torch
import scipy.misc
import os
import math
import numpy as np

from agent_observed_state import AgentObservedState
from utils.debug_nav_drone_instruction import instruction_to_string
from models.model.abstract_model import AbstractModel
from models.incremental_model.abstract_incremental_model import AbstractIncrementalModel

from utils.nav_drone_symbolic_instructions import BUCKET_WIDTH
from utils.nav_drone_symbolic_instructions import NO_BUCKETS


class TwoStageAgent:

    def __init__(self, server, model, symbolic_text_model, angle_prediction_model, test_policy, action_space, meta_data_util,
                 config, constants):
        self.server = server
        self.symbolic_text_model = symbolic_text_model  # stage 1 model for converting text to symbolic form
        self.angle_prediction_model = angle_prediction_model
        self.model = model  # stage 2 model
        self.test_policy = test_policy
        self.action_space = action_space
        self.meta_data_util = meta_data_util
        self.config = config
        self.constants = constants

    @staticmethod
    def save_image_and_metadata(image, state, data_point_ix, img_ctr):
        # Save the image
        scipy.misc.imsave("./oracle_images/train_images/example_"
                          + str(data_point_ix) + "/image_" + str(img_ctr) + ".png",
                          image.swapaxes(0, 1).swapaxes(1, 2))

        # Save the visible information
        x_pos, z_pos, y_angle = state.get_position_orientation()
        landmark_pos_dict = state.get_landmark_pos_dict()
        symbolic_image = TwoStageAgent.get_visible_landmark_r_theta(x_pos, z_pos, y_angle, landmark_pos_dict)
        with open("./oracle_images/train_images/example_" + str(data_point_ix)
                           + "/data_" + str(img_ctr) + ".json", 'w') as fp:
            json.dump(symbolic_image, fp)

    @staticmethod
    def get_visible_landmark_r_theta(x_pos, z_pos, y_angle, landmark_pos_dict):
        landmark_r_theta_dict = {}
        for landmark, (landmark_x, landmark_z) in landmark_pos_dict.iteritems():
            # get angle between drone's current orientation and landmark
            landmark_angle = 90.0 - np.arctan2(landmark_z - z_pos, landmark_x - x_pos) * 180.0 / math.pi
            angle_diff = landmark_angle - y_angle
            while angle_diff > 180.0:
                angle_diff -= 360.0
            while angle_diff < -180.0:
                angle_diff += 360.0
            if abs(angle_diff) <= 30.0:
                angle_discrete = int((angle_diff + 30.0) / 7.5)
            else:
                angle_discrete = -1

            # get discretized radius
            radius = ((landmark_x - x_pos) ** 2 + (landmark_z - z_pos) ** 2) ** 0.5
            radius_discrete = int(radius / 5.0)

            landmark_r_theta_dict[landmark] = (radius_discrete, angle_discrete)
        return landmark_r_theta_dict

    def test_save_oracle_images(self, test_dataset):

        self.server.clear_metadata()
        action_counts = [0] * self.action_space.num_actions()

        metadata = {"feedback": ""}
        for data_point_ix, data_point in enumerate(test_dataset):
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
            max_num_actions = self.constants["horizon"]
            trajectory = data_point.get_trajectory()
            os.mkdir("./oracle_images/train_images/example_" + str(data_point_ix))
            TwoStageAgent.save_image_and_metadata(image, state, data_point_ix, num_actions)

            while True:

                # Use test policy to get the action
                if num_actions == len(trajectory):
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
                    TwoStageAgent.save_image_and_metadata(image, state, data_point_ix, num_actions)

        logging.info("Overall test result: ")
        self.meta_data_util.log_results(metadata)
        logging.info("Testing data action counts %r", action_counts)

    def infer_symbolic_text(self, symbolic_text_prob):

        num_categories = len(symbolic_text_prob)
        categories = []
        for i in range(0, num_categories):
            probs = list(torch.exp(symbolic_text_prob[i].data[0]))
            category = self.test_policy(probs)
            categories.append(category)
        return tuple(categories)

    @staticmethod
    def symbolic_text_accuracy(symbolic_text_logs):

        landmark_accuracy = 0
        theta_1_accuracy = 0
        theta_2_accuracy = 0
        theta_1_regression_accuracy = 0
        theta_2_regression_accuracy = 0
        r_accuracy = 0
        accuracy = 0

        for tup in symbolic_text_logs:
            gold_symbolic_text, predicted_symbolic_text = tup
            gold_landmark, gold_theta_1, gold_theta_2, gold_r = gold_symbolic_text
            landmark, theta_1, theta_2, r = predicted_symbolic_text

            if gold_landmark == landmark:
                landmark_accuracy += 1

            if gold_theta_1 == theta_1:
                theta_1_accuracy += 1

            if gold_theta_2 == theta_2:
                theta_2_accuracy += 1

            theta_1_regression_accuracy += min((gold_theta_1 - theta_1) % NO_BUCKETS,
                                               NO_BUCKETS - (gold_theta_1 - theta_1) % NO_BUCKETS)
            theta_2_regression_accuracy += min((gold_theta_2 - theta_2) % NO_BUCKETS,
                                               NO_BUCKETS - (gold_theta_2 - theta_2) % NO_BUCKETS)

            if gold_r == r:
                r_accuracy += 1

            if gold_landmark == landmark and gold_theta_1 == theta_1 and gold_theta_2 == theta_2 and gold_r == r:
                accuracy += 1

        dataset_size = len(symbolic_text_logs)
        landmark_accuracy = (landmark_accuracy * 100) / float(max(1, dataset_size))
        theta_1_accuracy = (theta_1_accuracy * 100) / float(max(1, dataset_size))
        theta_2_accuracy = (theta_2_accuracy * 100) / float(max(1, dataset_size))
        theta_1_regression_accuracy = (BUCKET_WIDTH * theta_1_regression_accuracy) / float(max(1, dataset_size))
        theta_2_regression_accuracy = (BUCKET_WIDTH * theta_2_regression_accuracy) / float(max(1, dataset_size))
        r_accuracy = (r_accuracy * 100) / float(max(1, dataset_size))
        accuracy = (accuracy * 100) / float(max(1, dataset_size))
        logging.info(
            "Test accuracy on dataset of size %r is landmark %r, theta1 %r %r angle, theta2 %r %r angle, "
            "r %r, total acc %r",
            dataset_size, landmark_accuracy, theta_1_accuracy, theta_1_regression_accuracy,
            theta_2_accuracy, theta_2_regression_accuracy, r_accuracy, accuracy)

    def test(self, test_dataset, test_images, tensorboard=None):

        self.server.clear_metadata()
        action_counts = [0] * self.action_space.num_actions()

        metadata = {"feedback": ""}
        num_task_completed = 0
        symbolic_text_logs = []
        for data_point_ix, data_point in enumerate(test_dataset):
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
            # symbolic_text_probs = self.symbolic_text_model.get_symbolic_text_batch([state])
            # inferred_symbolic_text = self.infer_symbolic_text(symbolic_text_probs)
            gold_symbolic_text = state.get_symbolic_instruction()
            # Compute the angle for all landmark
            log_prob_theta = self.angle_prediction_model.get_probs([[test_images[data_point_ix][0]]])
            prob_theta = list(torch.exp(log_prob_theta.data)[0])
            inferred_theta = self.test_policy(prob_theta[gold_symbolic_text[0]])
            inferred_theta_correct = (inferred_theta - 6) % 12
            inferred_symbolic_text = (gold_symbolic_text[0],
                                      inferred_theta_correct, # inferred_symbolic_text[1],
                                      gold_symbolic_text[2],
                                      gold_symbolic_text[3])
            symbolic_text_logs.append((gold_symbolic_text, inferred_symbolic_text))
            # state.start_read_pointer, state.end_read_pointer = data_point.get_instruction_indices()

            num_actions = 0
            max_num_actions = self.constants["horizon"]
            model_state = None

            while True:

                # Generate probabilities over actions
                if isinstance(self.model, AbstractModel):
                    probabilities = list(torch.exp(self.model.get_probs(state).data))
                elif isinstance(self.model, AbstractIncrementalModel):
                    log_probabilities, model_state, _, _ = self.model.get_probs_symbolic_text(
                        state, inferred_symbolic_text, model_state, mode=None, volatile=False)
                    # log_probabilities, model_state, _, _ = self.model.get_probs(state, model_state, volatile=True)
                    probabilities = list(torch.exp(log_probabilities.data))[0]
                else:
                    raise AssertionError("Unhandled Model type.")

                # Use test policy to get the action
                action = self.test_policy(probabilities)
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
                        num_task_completed += 1

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

        logging.info("Symbolic Text Log accuracy: ")
        self.symbolic_text_accuracy(symbolic_text_logs)
        logging.info("Overall test result: ")
        self.meta_data_util.log_results(metadata)
        task_completion_accuracy = (num_task_completed * 100.0)/float(max(1, len(test_dataset)))
        logging.info("Task completion accuracy: %r", task_completion_accuracy)
        logging.info("Testing data action counts %r", action_counts)

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
            print("TEST INSTRUCTION: %r" % instruction)
            print("")

            for instruction_i, instruction in enumerate(segmented_instruction):

                state = AgentObservedState(instruction=instruction,
                                           config=self.config,
                                           constants=self.constants,
                                           start_image=image,
                                           previous_action=None)

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
