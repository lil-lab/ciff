import json
import logging
import torch
import scipy.misc
import math
import numpy as np
import matplotlib.pyplot as plt

from agents.agent_observed_state import AgentObservedState
from learning.auxiliary_objective.goal_prediction import GoalPrediction
from learning.single_client.goal_prediction_single_360_image_supervised_from_disk import \
    GoalPredictionSingle360ImageSupervisedLearningFromDisk
from utils.camera_mapping import get_inverse_object_position
from utils.debug_nav_drone_instruction import instruction_to_string
from models.model.abstract_model import AbstractModel
from models.incremental_model.abstract_incremental_model import AbstractIncrementalModel
from utils.geometry import get_turn_angle, current_pose_from_metadata, current_pos_from_metadata, get_distance

NO_BUCKETS = 48
BUCKET_WIDTH = 360.0/(1.0*NO_BUCKETS)


class PredictorPlannerAgent:

    def __init__(self, server, predictor_model, model, test_policy, action_space, meta_data_util,
                 config, constants):
        self.server = server
        self.predictor_model = predictor_model
        self.model = model
        self.test_policy = test_policy
        self.action_space = action_space
        self.meta_data_util = meta_data_util
        self.config = config
        self.constants = constants
        self.global_id = 0

    @staticmethod
    def goal_prediction_accuracy(goal, volatile):
        return GoalPrediction.get_loss_and_prob(volatile, goal, 32, 32)

    @staticmethod
    def _l2_distance(pt1, pt2):
        x1, z1 = pt1
        x2, z2 = pt2
        return math.sqrt((x1 - x2) ** 2 + (z1 - z2) ** 2)

    @staticmethod
    def get_angle_distance(metadata, data_point):

        agent_pos = metadata["x_pos"], metadata["z_pos"]
        start_pose = metadata["y_angle"]
        dest_list = data_point.get_destination_list()
        pt_g = (dest_list[0][0], dest_list[0][1])

        # Compute the distance
        l2_distance = PredictorPlannerAgent._l2_distance(agent_pos, pt_g)

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

        dist_12 = PredictorPlannerAgent._l2_distance(pt_g1, pt_g2)
        e11 = PredictorPlannerAgent._l2_distance(dest_1, pt_g1)
        e22 = PredictorPlannerAgent._l2_distance(dest_2, pt_g2)
        e12 = PredictorPlannerAgent._l2_distance(dest_1, pt_g2)
        e21 = PredictorPlannerAgent._l2_distance(dest_2, pt_g1)

        metric_val = 0.5 + (e12 + e21 - e11 - e22) / max((4.0 * dist_12), 0.00001)
        return metric_val

    def get_exploration_image(self):
        # Predict the goal by performing an exploration image and then finding the next suitable place to visit
        exploration_image, _, _ = self.server.explore()
        image_slices = []
        for img_ctr in range(0, 6):
            image_slice = exploration_image[img_ctr * 3: (img_ctr + 1) * 3, :, :]  # 3 x height x width
            # Scale the intensity of the image as done by scipy.misc.imsave
            image_slice = scipy.misc.bytescale(image_slice.swapaxes(0, 1).swapaxes(1, 2))
            image_slices.append(image_slice)

        # Reorder and horizontally stitch the images
        reordered_images = [image_slices[3], image_slices[4], image_slices[5],
                            image_slices[0], image_slices[1], image_slices[2]]
        exploration_image = np.hstack(reordered_images).swapaxes(1, 2).swapaxes(0, 1)  # 3 x height x (width*6)

        return exploration_image

    def get_3d_location(self, exploration_image, data_point, panaroma=True):

        state = AgentObservedState(instruction=data_point.instruction,
                                   config=self.config,
                                   constants=self.constants,
                                   start_image=exploration_image,
                                   previous_action=None,
                                   pose=None,
                                   position_orientation=data_point.get_start_pos(),
                                   data_point=data_point)

        volatile = self.predictor_model.get_attention_prob(state, model_state=None)
        inferred_ix = int(torch.max(volatile["attention_logits"], 0)[1].data.cpu().numpy()[0])
        # Max pointed about that when inferred ix above is the last value then calculations are buggy. He is right.

        predicted_row = int(inferred_ix / float(192))
        predicted_col = inferred_ix % 192
        screen_pos = (predicted_row, predicted_col)

        if panaroma:
            # Index of the 6 image where the goal is
            region_index = int(predicted_col / 32)
            predicted_col = predicted_col % 32  # Column within that image where the goal is
            pos = data_point.get_start_pos()
            new_pos_angle = GoalPredictionSingle360ImageSupervisedLearningFromDisk.\
                get_new_pos_angle_from_region_index(region_index, pos)
            metadata = {"x_pos": pos[0], "z_pos": pos[1], "y_angle": new_pos_angle}
        else:
            pos = data_point.get_start_pos()
            metadata = {"x_pos": pos[0], "z_pos": pos[1], "y_angle": pos[2]}

        row, col = predicted_row + 0.5, predicted_col + 0.5

        start_pos = current_pos_from_metadata(metadata)
        start_pose = current_pose_from_metadata(metadata)

        goal_pos = data_point.get_destination_list()[-1]
        height_drone = 2.5
        x_gen, z_gen = get_inverse_object_position(row, col, height_drone, 30, 32, 32,
                                                   (start_pos[0], start_pos[1], start_pose))
        predicted_goal_pos = (x_gen, z_gen)
        x_goal, z_goal = goal_pos

        x_diff = x_gen - x_goal
        z_diff = z_gen - z_goal

        dist = math.sqrt(x_diff * x_diff + z_diff * z_diff)

        return predicted_goal_pos, dist, screen_pos, volatile["attention_probs"]

    def save_attention_prob(self, image, attention_prob, instruction, scene_name):
        self.global_id += 1

        image_flipped = image.swapaxes(0, 1).swapaxes(1, 2)
        image_flipped = scipy.misc.imresize(image_flipped, (128, 128 * 6))
        attention_prob = attention_prob[:-1].view(32, 192).cpu().data.numpy()
        resized_kernel = scipy.misc.imresize(attention_prob, (128, 128 * 6))

        plt.title(instruction)
        plt.imshow(image_flipped)
        plt.imshow(resized_kernel, cmap='jet', alpha=0.5)
        plt.savefig("./dev_attention_prob/" + str(scene_name) + ".png")
        plt.clf()

    def get_3d_location_for_paragraphs(self, exploration_image, instruction, start_pos, goal_pos, panaroma=True):

        state = AgentObservedState(instruction=instruction,
                                   config=self.config,
                                   constants=self.constants,
                                   start_image=exploration_image,
                                   previous_action=None,
                                   pose=None,
                                   position_orientation=start_pos,
                                   data_point=None)

        volatile = self.predictor_model.get_attention_prob(state, model_state=None)
        inferred_ix = int(torch.max(volatile["attention_logits"], 0)[1].data.cpu().numpy()[0])

        ########################################
        # inst_string = instruction_to_string(instruction, self.config)
        # self.save_attention_prob(exploration_image, volatile["attention_probs"][:-1].view(32, 192), inst_string)
        ########################################

        predicted_row = int(inferred_ix / float(192))
        predicted_col = inferred_ix % 192

        if panaroma:
            # Index of the 6 image where the goal is
            region_index = int(predicted_col / 32)
            predicted_col = predicted_col % 32  # Column within that image where the goal is
            pos = start_pos
            new_pos_angle = GoalPredictionSingle360ImageSupervisedLearningFromDisk.\
                get_new_pos_angle_from_region_index(region_index, pos)
            metadata = {"x_pos": pos[0], "z_pos": pos[1], "y_angle": new_pos_angle}
        else:
            pos = start_pos
            metadata = {"x_pos": pos[0], "z_pos": pos[1], "y_angle": pos[2]}

        row, col = predicted_row + 0.5, predicted_col + 0.5

        start_pos = current_pos_from_metadata(metadata)
        start_pose = current_pose_from_metadata(metadata)

        height_drone = 2.5
        x_gen, z_gen = get_inverse_object_position(row, col, height_drone, 30, 32, 32,
                                                   (start_pos[0], start_pos[1], start_pose))
        predicted_goal_pos = (x_gen, z_gen)
        x_goal, z_goal = goal_pos

        x_diff = x_gen - x_goal
        z_diff = z_gen - z_goal

        dist = math.sqrt(x_diff * x_diff + z_diff * z_diff)

        return predicted_goal_pos, dist

    @staticmethod
    def get_goal_location(current_bot_location, current_bot_pose, goal_3d_location, height, width, max_angle=30):

        # compute angle which is in -180 to 180 degree
        angle_phi = get_turn_angle(current_bot_location, current_bot_pose, goal_3d_location)

        if angle_phi < -max_angle or angle_phi > max_angle:
            return None, None, None, None
        distance = get_distance(current_bot_location, goal_3d_location)  # return distance
        tan_phi = math.tan(math.radians(angle_phi))
        cos_phi = math.cos(math.radians(angle_phi))
        tan_theta = math.tan(math.radians(30.0))  # camera width is 30.0
        height_drone = 2.5

        height_half = int(height / 2)
        width_half = int(width / 2)
        try:
            row_real = height_half + (height_half * height_drone) / (distance * cos_phi * tan_theta)
            col_real = width_half + (width_half * tan_phi) / tan_theta
            row = int(round(row_real))  # int(row_real)
            col = int(round(col_real))  # int(col_real)
        except:
            print("Found exception. Cosphi is %r, tan theta is %r, and distance is %r" % (cos_phi, tan_theta, distance))
            return None, None, None, None

        if row < 0:
            row = 0
        elif row >= height:
            row = height - 1

        if col < 0:
            col = 0
        elif col >= width:
            col = width - 1

        return row, col, row_real, col_real

    def _test(self, data_point_ix, data_point, test_image, tensorboard=None, debug=False):

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

        # Test image
        if test_image is None:
            test_image_example = self.get_exploration_image()
        else:
            test_image_example = test_image[data_point_ix][0]

        # Predict the goal
        predicted_goal, predictor_error, predicted_pixel, attention_prob = self.get_3d_location(
            test_image_example, data_point, panaroma=True)
        current_bot_location = metadata["x_pos"], metadata["z_pos"]
        current_bot_pose = metadata["y_angle"]
        state.goal = PredictorPlannerAgent.get_goal_location(
            current_bot_location, current_bot_pose, predicted_goal, 32, 32)
        print("Predicted Error ", predictor_error)

        num_actions = 0
        max_num_actions = self.constants["horizon"]
        model_state = None
        actions = []
        info = dict()

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

                if debug:
                    # Dictionary to contain key results
                    info["instruction_string"] = instruction_to_string(data_point.instruction, self.config)
                    info["datapoint_id"] = data_point.get_scene_name()
                    info["stop_dist_error"] = metadata["stop_dist_error"]
                    info["closest_dist_error"] = metadata["closest_dist_error"]
                    info["edit_dist_error"] = metadata["edit_dist_error"]
                    info["num_actions_taken"] = num_actions
                    info["predicted_goal"] = predicted_goal
                    info["predicted_error"] = predictor_error
                    info["gold_goal"] = data_point.get_destination_list()[-1]
                    info["final_location"] = (metadata["x_pos"], metadata["z_pos"])
                    info["predicted_screen_pixels"] = predicted_pixel

                    self.save_attention_prob(test_image_example, attention_prob, info["instruction_string"],
                                             info["datapoint_id"])
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

                # Set the goal based on the current position and angle
                current_bot_location = metadata["x_pos"], metadata["z_pos"]
                current_bot_pose = metadata["y_angle"]
                state.goal = PredictorPlannerAgent.get_goal_location(
                    current_bot_location, current_bot_pose, predicted_goal, 32, 32)
                num_actions += 1

        # logging.info("Error, Start-Distance, Turn-Angle,  %r %r %r", metadata["stop_dist_error"], distance, angle)
        return metadata, actions, predictor_error, info

    def test_minimal_linguistic_pairs(self, test_dataset, tensorboard=None, logger=None, pushover_logger=None):

        self.server.clear_metadata()
        action_counts = [0] * self.action_space.num_actions()

        # For minimal linguistic pairs we compute two success metric:
        task_completion_accuracy = 0
        task_bennett_metric = 0

        # Accuracy for individual examples
        single_completion_accuracy = 0
        single_distance_regression = 0
        single_predictor_error = 0

        metadata = {"feedback": ""}
        data_point_ix = 0
        for data_point_ix, tup_data_point in enumerate(test_dataset):

            # The two minimal linguistic pairs
            data_point1 = tup_data_point[0]
            data_point2 = tup_data_point[1]

            metadata1, _, predictor_error1 = self._test(data_point_ix, data_point1, test_image=None,
                                                        test_goal_location=None, tensorboard=tensorboard)
            metadata2, _, predictor_error2 = self._test(data_point_ix, data_point2,
                                      test_image=None, test_goal_location=None, tensorboard=tensorboard)
            metadata = metadata2  # Last metadata

            if metadata1["stop_dist_error"] < 5.0:
                single_completion_accuracy += 1
            if metadata2["stop_dist_error"] < 5.0:
                single_completion_accuracy += 1

            if metadata1["stop_dist_error"] < 5.0 and metadata2["stop_dist_error"] < 5.0:
                task_completion_accuracy += 1
            task_bennett_metric += self.bennett_metric(data_point1, data_point2, metadata1, metadata2)

            single_predictor_error += predictor_error1 + predictor_error2
            single_distance_regression += metadata1["stop_dist_error"] + metadata2["stop_dist_error"]

            if (data_point_ix + 1) % 100 == 0:  # print intermediate results
                self.log("Results after %r " % (data_point_ix + 1), logger)
                temp_task_completion_accuracy = (task_completion_accuracy * 100.0) / float(data_point_ix + 1)
                temp_task_bennett_metric = (task_bennett_metric * 100.0) / float(data_point_ix + 1)
                temp_single_completion_accuracy = (single_completion_accuracy * 100.0) / (2.0 * float(data_point_ix + 1))
                temp_single_distance_regression = single_distance_regression / (2.0 * float(data_point_ix + 1))
                self.log("Testing: Minimal Linguistic Pair Task completion accuracy is: %r"
                         % temp_task_completion_accuracy, logger)
                self.log("Predictor error %r " % (single_predictor_error / float(data_point_ix + 1)), logger)
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
        self.log("Predictor error %r " % (single_predictor_error / float(max(dataset_size, 1))), logger)
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
        mean_predictor_error = 0

        test_image, _ = GoalPredictionSingle360ImageSupervisedLearningFromDisk.parse(
            "./6000_dev_split_start_images", test_dataset, self.model, self.config, format_type="png")

        assert len(test_dataset) == len(test_image)

        metadata = {"feedback": ""}
        info_list = []
        for data_point_ix, data_point in enumerate(test_dataset):
            metadata, actions_taken, predicted_error, info = self._test(
                data_point_ix, data_point, test_image, tensorboard, debug=False)

            if metadata["stop_dist_error"] < 5.0:
                task_completion_accuracy += 1

            for action in actions_taken:
                action_counts[action] += 1

            mean_predictor_error += predicted_error

            info_list.append(info)

        task_completion_accuracy = (task_completion_accuracy * 100.0) / float(max(len(test_dataset), 1))
        mean_predictor_error = mean_predictor_error / float(max(len(test_dataset), 1))

        with open("test_results_planner_with_image.json", "w") as debug_f:
            json.dump(info_list, debug_f)

        self.log("Overall test results:", logger)
        self.log("Testing: Task completion accuracy is: %r" % task_completion_accuracy, logger)
        self.log("Testing: Predicted Error is: %r" % mean_predictor_error, logger)
        self.log("Testing: Final Metadata: %r" % metadata, logger)
        self.log("Testing: Action Distribution: %r" % action_counts, logger)
        self.log("Testing data action counts %r" % action_counts, logger)
        self.meta_data_util.log_results(metadata, logger)
        if pushover_logger is not None:
            pushover_feedback = str(metadata["feedback"]) +\
                                " --- " + "task_completion_accuracy=%r" % task_completion_accuracy
            pushover_logger.log(pushover_feedback)

    def test_auto_segmented(self, test_dataset, logger=None, tensorboard=None, segmenting_type="oracle"):

        assert segmenting_type in ("auto", "oracle")
        self.server.clear_metadata()
        action_counts = [0] * self.action_space.num_actions()

        self.log("Performing testing on paragraphs with segmenting type %r" % segmenting_type, logger)
        metadata = {"feedback": ""}

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

                # Predict the goal by performing an exploration image and then finding the next suitable place to visit
                exploration_image, _, _ = self.server.explore()
                image_slices = []
                for img_ctr in range(0, 6):
                    image_slice = exploration_image[img_ctr * 3: (img_ctr + 1) * 3, :, :]  # 3 x height x width
                    # Scale the intensity of the image as done by scipy.misc.imsave
                    image_slice = scipy.misc.bytescale(image_slice.swapaxes(0, 1).swapaxes(1, 2))
                    image_slices.append(image_slice)

                # Reorder and horizontally stitch the images
                reordered_images = [image_slices[3], image_slices[4], image_slices[5],
                                    image_slices[0], image_slices[1], image_slices[2]]
                exploration_image = np.hstack(reordered_images).swapaxes(1, 2).swapaxes(0, 1)  # 3 x height x (width*6)

                start_pos = (metadata["x_pos"], metadata["z_pos"], metadata["y_angle"])
                goal_pos = data_point.get_destination_list()[instruction_i]
                predicted_goal, predictor_error = self.get_3d_location_for_paragraphs(
                    exploration_image, instruction, start_pos, goal_pos, panaroma=True)
                current_bot_location = metadata["x_pos"], metadata["z_pos"]
                current_bot_pose = metadata["y_angle"]
                state.goal = PredictorPlannerAgent.get_goal_location(
                    current_bot_location, current_bot_pose, predicted_goal, 32, 32)
                print("Predicted Error ", predictor_error)

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

                        intermediate_goal = data_point.get_destination_list()[instruction_i]
                        agent_position = metadata["x_pos"], metadata["z_pos"]
                        distance = self._l2_distance(agent_position, intermediate_goal)
                        self.log("Instruction is %r " % instruction, logger)
                        self.log("Predicted Goal is %r, Goal Reached is %r and Real goal is %r " %
                                 (predicted_goal, agent_position, intermediate_goal), logger)
                        self.log("Agent: Position %r got Distance %r " % (instruction_i + 1, distance), logger)
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

                        # Set the goal based on the current position and angle
                        current_bot_location = metadata["x_pos"], metadata["z_pos"]
                        current_bot_pose = metadata["y_angle"]
                        state.goal = PredictorPlannerAgent.get_goal_location(
                            current_bot_location, current_bot_pose, predicted_goal, 32, 32)

                        num_actions += 1

            image, reward, metadata = self.server.halt_and_receive_feedback()
            if tensorboard is not None:
                tensorboard.log_all_test_errors(
                    metadata["edit_dist_error"],
                    metadata["closest_dist_error"],
                    metadata["stop_dist_error"])

            # Update the scores based on meta_data
            self.meta_data_util.log_results(metadata)

        self.meta_data_util.log_results(metadata)
        logging.info("Testing data action counts %r", action_counts)

    @staticmethod
    def log(message, logger=None):
        if logger is not None:
            logger.log(message)
        else:
            logging.info(message)
