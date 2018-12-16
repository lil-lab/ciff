import json

import torch
import time
import os
import math
import random
import logging
import torch.optim as optim
import numpy as np
import scipy.misc
import matplotlib.pyplot as plt

from agents.agent_observed_state import AgentObservedState
from agents.replay_memory_item import ReplayMemoryItem
from learning.auxiliary_objective.goal_prediction import GoalPrediction
from learning.auxiliary_objective.object_pixel_identification import ObjectPixelIdentification
from learning.auxiliary_objective.symbolic_language_prediction import SymbolicLanguagePrediction
from learning.single_client.abstract_learning import AbstractLearning
from learning.auxiliary_objective.action_prediction import ActionPrediction
from learning.auxiliary_objective.temporal_autoencoder import TemporalAutoEncoder
from utils.camera_mapping import get_inverse_object_position
from utils.cuda import cuda_var
from models.incremental_model.incremental_model_recurrent_implicit_factorization_resnet import \
    IncrementalModelRecurrentImplicitFactorizationResnet
from utils.debug_nav_drone_instruction import instruction_to_string
from utils.geometry import current_pos_from_metadata, current_pose_from_metadata, get_turn_angle_from_metadata_datapoint


class GoalPredictionSingle360ImageSupervisedLearningFromDisk(AbstractLearning):
    """ Perform goal prediction on single images (as opposed to doing it for sequence)
    stored on disk and hence does not need client or server. """

    CLOCKWISE, BACKSTICH = range(2)
    image_stich = CLOCKWISE

    MODE, MEAN, REALMEAN, MEANAROUNDMODE = range(4)

    def __init__(self, model, action_space, meta_data_util, config, constants, tensorboard):
        self.max_epoch = constants["max_epochs"]
        self.model = model
        self.action_space = action_space
        self.meta_data_util = meta_data_util
        self.config = config
        self.constants = constants
        self.tensorboard = tensorboard
        self.entropy = None
        self.cross_entropy = None
        self.epoch = 0
        self.global_id = 0
        self.entropy_coef = constants["entropy_coefficient"]
        self.final_num_channels, self.final_height, self.final_width = model.image_module.get_final_dimension()

        self.ignore_none = True
        self.inference_procedure = GoalPredictionSingle360ImageSupervisedLearningFromDisk.MODE

        self.vocab = {}
        vocab_path = config["vocab_file"]
        word_index = 0
        with open(vocab_path) as f:
            for line in f.readlines():
                token = line.strip()
                self.vocab[token] = word_index
                word_index += 1

        # Auxiliary Objectives
        if self.config["do_action_prediction"]:
            self.action_prediction_loss_calculator = ActionPrediction(self.model)
            self.action_prediction_loss = None
        if self.config["do_temporal_autoencoding"]:
            self.temporal_autoencoder_loss_calculator = TemporalAutoEncoder(self.model)
            self.temporal_autoencoder_loss = None
        if self.config["do_object_detection"]:
            self.object_detection_loss_calculator = ObjectPixelIdentification(
                self.model, num_objects=67, camera_angle=60, image_height=self.final_height,
                image_width=self.final_width, object_height=0)  # -2.5)
            self.object_detection_loss = None
        if self.config["do_symbolic_language_prediction"]:
            self.symbolic_language_prediction_loss_calculator = SymbolicLanguagePrediction(self.model)
            self.symbolic_language_prediction_loss = None
        if self.config["do_goal_prediction"]:
            self.goal_prediction_calculator = GoalPrediction(self.model, self.final_height, self.final_width)
            self.goal_prediction_loss = None

        self.cross_entropy_loss = None
        self.dist_loss = None

        self.optimizer = optim.Adam(model.get_parameters(),
                                    lr=constants["learning_rate"])
        AbstractLearning.__init__(self, self.model, self.calc_loss, self.optimizer,
                                  self.config, self.constants, self.tensorboard)

        logging.info("Created Single Image goal predictor with ignore_none %r", self.ignore_none)

    def calc_loss(self, batch_replay_items):

        # Only compute the goal prediction loss
        # loss = None
        # for replay_item in batch_replay_items:
        #     self.goal_prediction_loss, self.goal_prob, meta = self.goal_prediction_calculator.calc_loss(
        #         [replay_item])
        #     if loss is None:
        #         loss = self.goal_prediction_loss
        #     else:
        #         loss += self.goal_prediction_loss
        #
        # loss = loss / float(len(batch_replay_items))
        self.goal_prediction_loss, self.goal_prob, meta = self.goal_prediction_calculator.calc_loss(batch_replay_items)
        loss = self.goal_prediction_loss

        if self.config["do_object_detection"]:
            self.object_detection_loss = self.object_detection_loss_calculator.calc_loss(batch_replay_items)
            if self.object_detection_loss is not None:
                self.object_detection_loss = self.constants["object_detection_coeff"] * self.object_detection_loss
                loss = loss + self.object_detection_loss
        else:
            self.object_detection_loss = None

        self.cross_entropy_loss = meta["cross_entropy"]
        self.dist_loss = meta["dist_loss"]

        return loss

    @staticmethod
    def get_new_pos_angle(data_point):

        pos = data_point.get_start_pos()
        metadata = {"x_pos": pos[0], "z_pos": pos[1], "y_angle": pos[2]}
        turn_angle = get_turn_angle_from_metadata_datapoint(metadata, data_point)

        assert 180.0 >= turn_angle >= -180.0
        if 30.0 >= turn_angle > -30.0:
            ix = 3  # ix = 0
            mean_turn_angle = 0
        elif 90.0 >= turn_angle > 30.0:
            ix = 4  # ix = 1
            mean_turn_angle = 60
        elif 150.0 >= turn_angle > 90.0:
            ix = 5  # ix = 2
            mean_turn_angle = 120
        elif -30 >= turn_angle > -90.0:
            ix = 2  # ix = 5
            mean_turn_angle = -60
        elif -90.0 >= turn_angle > -150.0:
            ix = 1  # ix = 4
            mean_turn_angle = -120
        else:
            ix = 0  # ix = 3
            mean_turn_angle = 180

        new_pos_angle = pos[2] + mean_turn_angle
        while new_pos_angle < -180:
            new_pos_angle += 360.0
        while new_pos_angle > 180:
            new_pos_angle -= 360.0

        return new_pos_angle, ix

    @staticmethod
    def get_new_pos_angle_from_region_index(region_index, start_pos):

        if region_index == 3:
            mean_turn_angle = 0
        elif region_index == 4:
            mean_turn_angle = 60
        elif region_index == 5:
            mean_turn_angle = 120
        elif region_index == 0:
            mean_turn_angle = 180
        elif region_index == 1:
            mean_turn_angle = -120
        elif region_index == 2:
            mean_turn_angle = -60
        else:
            raise AssertionError("Region index should be in 0 to 6.")

        new_pos_angle = start_pos[2] + mean_turn_angle
        while new_pos_angle < -180:
            new_pos_angle += 360.0
        while new_pos_angle > 180:
            new_pos_angle -= 360.0

        return new_pos_angle

    @staticmethod
    def parse(folder_name, dataset, model, config, format_type="numpy"):

        start = time.time()
        num_channel, height, width = model.image_module.get_final_dimension()

        # Read images
        num_examples = len(os.listdir(folder_name))

        if format_type == "numpy":

            image_dataset = []

            # Read panaroma images
            for i in range(0, num_examples):
                example_folder_name = folder_name + "/example_" + str(i)
                image_np = np.load(example_folder_name + "/image_numpy.npy")

                slices = []
                for i in range(0, 6):
                    slices.append(image_np[i*3:(i + 1)*3, :, :].swapaxes(0, 1).swapaxes(1, 2))  # height x width x 3

                images = [slices[3], slices[4], slices[5], slices[0], slices[1], slices[2]]
                images = np.hstack(images)
                images = images.swapaxes(1, 2).swapaxes(0, 1)
                image_dataset.append([images])

        elif format_type == "png":
            image_dataset = []

            # Read panaroma images
            for i in range(0, num_examples):
                example_folder_name = folder_name + "/example_" + str(i)
                images = []

                # image_order = range(0, 6)  # clockwise
                image_order = [3, 4, 5, 0, 1, 2]

                for ix in image_order:  # panaroma consists of 6 images stitched together
                    img = scipy.misc.imread(example_folder_name + "/image_" + str(ix) + ".png")
                    images.append(img)
                images = np.hstack(images)
                images = images.swapaxes(1, 2).swapaxes(0, 1)
                image_dataset.append([images])
        else:
            raise AssertionError("")

        # Read the goal state. The data for the single image can be
        # directly computed and does not need to be saved.
        goal_dataset = []
        for i in range(0, num_examples):

            data_point = dataset[i]
            new_pos_angle, ix = GoalPredictionSingle360ImageSupervisedLearningFromDisk.get_new_pos_angle(data_point)

            # Modify the pos to turn towards the image so we can compute the goal location relative to a single image.
            pos = data_point.get_start_pos()
            new_pos = (pos[0], pos[1], new_pos_angle)
            original_start_pos = pos
            data_point.start_pos = new_pos
            metadata = {"x_pos": new_pos[0], "z_pos": new_pos[1], "y_angle": new_pos[2]}
            new_turn_angle = get_turn_angle_from_metadata_datapoint(metadata, data_point)
            assert 30.0 >= new_turn_angle >= -30.0, "Found turn angle of " + str(new_turn_angle)
            goal_location = [GoalPrediction.get_goal_location(metadata, data_point, height=32, width=32)]
            _, _, row, col = goal_location[0]
            data_point.start_pos = original_start_pos

            # print("Drone's original angle is %r, New Pos Angle is %r ", original_start_pos[2], new_pos_angle)
            # print("Index is %r, Goal is %r " % (ix, goal_location[0]))
            if row is not None and col is not None:
                row = row
                col = col + ix * 32.0
                row_new, col_new, row1, col1 = [int(round(row)), int(round(col)), row, col]
                if row_new >= 32:
                    row_new = 31
                elif row_new < 0:
                    row_new = 0
                if col_new >= 192:
                    col_new = 191
                elif col_new < 0:
                    col_new = 0

                goal = [row_new, col_new, row1, col1]
                goal_location = [goal]

                # image = image_dataset[i][0]
                # goal_prob = GoalPrediction.generate_gold_prob(goal, 32, 32 * 6)
                # goal_prob = goal_prob[:-1].view(32, 32 * 6)
                # image_flipped = image[:, :, :].swapaxes(0, 1).swapaxes(1, 2)
                # image_flipped = scipy.misc.imresize(image_flipped, (128 * 5, 128 * 6 * 5))
                # goal_map = goal_prob.cpu().data.numpy()
                # if np.sum(goal_map) > 0.01:
                #     for k in range(0, 32):
                #         for j in range(0, 32 * 6):
                #             if goal_map[k][j] < 0.01:
                #                 goal_map[k][j] = 0.0
                #         goal_map = scipy.misc.imresize(goal_map, (128 * 5, 128 * 6 * 5))
                # else:
                #     goal_map = None
                #
                # plt.imshow(image_flipped)
                # # if goal_map is not None:
                # #     plt.imshow(goal_map, cmap='jet', alpha=0.5)
                #
                # plt.title(instruction_to_string(data_point.instruction, config))
                # plt.savefig("./paper_figures/goal_" + str(i) + "_1.png")
                # plt.clf()

            # start_pos = current_pos_from_metadata(metadata)
            # start_pose = current_pose_from_metadata(metadata)
            # if row is not None and col is not None:
            #     goal_pos = data_point.get_destination_list()[-1]
            #     x_goal, z_goal = goal_pos
            #     height_drone = 2.5
            #     x_gen, z_gen = get_inverse_object_position(row, col, height_drone, 30, height, width,
            #                                                (start_pos[0], start_pos[1], start_pose))
            #     x_diff = x_gen - x_goal
            #     z_diff = z_gen - z_goal
            #     dist = math.sqrt(x_diff * x_diff + z_diff * z_diff)
            #     assert dist < 0.5, "forward computation of goal should match inverse computation"
            # else:
            #     print("Warning: None found! ")

            goal_dataset.append(goal_location)

        assert len(image_dataset) == len(dataset) and len(goal_dataset) == len(dataset)

        end = time.time()
        logging.info("Parsed dataset of size %r in time % seconds", len(image_dataset), (end - start))

        return image_dataset, goal_dataset

    @staticmethod
    def parse_oracle_turn(folder_name, dataset, model):

        start = time.time()

        num_channel, height, width = model.image_module.get_final_dimension()

        # Read images
        image_dataset = []
        num_examples = len(os.listdir(folder_name))
        for i in range(0, num_examples):
            data_point = dataset[i]

            ################################################
            pos = data_point.get_start_pos()
            metadata = {"x_pos": pos[0], "z_pos": pos[1], "y_angle": pos[2]}

            turn_angle = get_turn_angle_from_metadata_datapoint(metadata, data_point)

            assert 180.0 >= turn_angle >= -180.0

            if 30.0 >= turn_angle > -30.0:
                ix = 0
                mean_turn_angle = 0
            elif 90.0 >= turn_angle > 30.0:
                ix = 1
                mean_turn_angle = 60
            elif 150.0 >= turn_angle > 90.0:
                ix = 2
                mean_turn_angle = 120
            elif -30 >= turn_angle > -90.0:
                ix = 5
                mean_turn_angle = -60
            elif -90.0 >= turn_angle > -150.0:
                ix = 4
                mean_turn_angle = -120
            else:
                ix = 3
                mean_turn_angle = 180

            print("Pose is %r, Turn Angle is %r and Mean Turn Angle is %r " % (pos[2], turn_angle, mean_turn_angle))
            new_pos_angle = pos[2] + mean_turn_angle

            while new_pos_angle < -180:
                new_pos_angle += 360.0
            while new_pos_angle > 180:
                new_pos_angle -= 360.0

            # Modify the pos to turn towards the image
            new_pos = (pos[0], pos[1], new_pos_angle)
            data_point.start_pos = new_pos

            pos = data_point.get_start_pos()
            metadata = {"x_pos": pos[0], "z_pos": pos[1], "y_angle": pos[2]}
            new_turn_angle = get_turn_angle_from_metadata_datapoint(metadata, data_point)
            assert 30.0 >= new_turn_angle >= -30.0, "Found turn angle of " + str(new_turn_angle)
            ################################################

            example_folder_name = folder_name + "/example_" + str(i)
            img = scipy.misc.imread(example_folder_name + "/image_" + str(ix) + ".png").swapaxes(1, 2).swapaxes(0, 1)
            images = [img]
            image_dataset.append(images)

        assert len(image_dataset) == len(dataset)

        # Read the goal state. The data for the single image can be
        # directly computed and does not need to be saved.
        goal_dataset = []
        for i in range(0, num_examples):
            data_point = dataset[i]

            pos = data_point.get_start_pos()
            metadata = {"x_pos": pos[0], "z_pos": pos[1], "y_angle": pos[2]}

            goal_location = [GoalPrediction.get_goal_location(metadata, data_point, height, width)]
            _, _, row, col = goal_location[0]

            start_pos = current_pos_from_metadata(metadata)
            start_pose = current_pose_from_metadata(metadata)

            if row is not None and col is not None:
                goal_pos = data_point.get_destination_list()[-1]
                x_goal, z_goal = goal_pos
                height_drone = 2.5
                x_gen, z_gen = get_inverse_object_position(row, col, height_drone, 30, height, width,
                                                           (start_pos[0], start_pos[1], start_pose))
                x_diff = x_gen - x_goal
                z_diff = z_gen - z_goal
                dist = math.sqrt(x_diff * x_diff + z_diff * z_diff)
                assert dist < 0.5, "forward computation of goal should match inverse computation"
            else:
                print("Warning: None found! ")

            goal_dataset.append(goal_location)

        end = time.time()
        logging.info("Parsed dataset of size %r in time % seconds", len(image_dataset), (end - start))
        return image_dataset, goal_dataset

    def convert_to_id(self, instruction):
        tk_seq = instruction.split()
        token_ids = []
        for tk in tk_seq:
            if tk in self.vocab:
                token_ids.append(self.vocab[tk])
            else:
                print("Out of vocabulary word. Ignoring ", tk)
        return token_ids

    def is_close_enough(self, inferred_ix, row, col):
        predicted_row = int(inferred_ix / float(self.final_width))
        predicted_col = inferred_ix % self.final_width

        row_diff = row - predicted_row
        col_diff = col - predicted_col

        dist = math.sqrt(row_diff * row_diff + col_diff * col_diff)

        max_dim = float(max(self.final_height, self.final_width))
        if dist < 0.1 * max_dim:
            return True
        else:
            return False

    def compute_distance_in_real_world(self, inferred_ix, row_col, data_point, panaroma=True):

        if row_col is None:
            predicted_row = int(inferred_ix / float(self.final_width))
            predicted_col = inferred_ix % self.final_width
        else:
            predicted_row, predicted_col = row_col

        if panaroma:
            region_index = int(predicted_col / 32)
            predicted_col = predicted_col % 32
            pos = data_point.get_start_pos()
            new_pos_angle = GoalPredictionSingle360ImageSupervisedLearningFromDisk.\
                get_new_pos_angle_from_region_index(region_index, pos)
            metadata = {"x_pos": pos[0], "z_pos": pos[1], "y_angle": new_pos_angle}
        else:
            pos = data_point.get_start_pos()
            metadata = {"x_pos": pos[0], "z_pos": pos[1], "y_angle": pos[2]}

        if row_col is None:
            row, col = predicted_row + 0.5, predicted_col + 0.5
        else:
            row, col = predicted_row, predicted_col

        start_pos = current_pos_from_metadata(metadata)
        start_pose = current_pose_from_metadata(metadata)

        goal_pos = data_point.get_destination_list()[-1]
        height_drone = 2.5
        x_gen, z_gen = get_inverse_object_position(row, col, height_drone, 30, 32, 32,
                                                   (start_pos[0], start_pos[1], start_pose))
        x_goal, z_goal = goal_pos

        x_diff = x_gen - x_goal
        z_diff = z_gen - z_goal

        dist = math.sqrt(x_diff * x_diff + z_diff * z_diff)
        return (x_gen, z_gen), dist

    def get_inferred_value(self, volatile):

        if self.inference_procedure == GoalPredictionSingle360ImageSupervisedLearningFromDisk.MODE:
            # Mode setting
            inferred_ix = int(torch.max(volatile["attention_logits"], 0)[1].data.cpu().numpy()[0])

            return inferred_ix, None

        elif self.inference_procedure == GoalPredictionSingle360ImageSupervisedLearningFromDisk.MEAN:
            prob_values = volatile["attention_probs"][:-1].view(32, 192).data.cpu().numpy()
            expected_row = 0
            expected_col = 0
            for row in range(0, 32):
                for col in range(0, 192):
                    expected_row = expected_row + row * prob_values[row, col]
                    expected_col = expected_col + col * prob_values[row, col]

            mode_ix = int(torch.max(volatile["attention_logits"], 0)[1].data.cpu().numpy()[0])
            row_ = int(mode_ix/192)
            col_ = mode_ix % 192
            inferred_ix = expected_row * 192.0 + expected_col
            print("Expected Row is %r Mode Row is %r and and Expected Col is %r, Mode Col is %r "
                  % (expected_row, row_, expected_col, col_))
            if inferred_ix > 32 * 192:
                inferred_ix = 32 * 192

            return inferred_ix, None

        elif self.inference_procedure == GoalPredictionSingle360ImageSupervisedLearningFromDisk.MEANAROUNDMODE:

            mode_ix = int(torch.max(volatile["attention_logits"], 0)[1].data.cpu().numpy()[0])
            mode_row = int(mode_ix / 192)
            mode_col = mode_ix % 192

            expected_row = 0
            expected_col = 0
            prob_values = volatile["attention_probs"][:-1].view(32, 192).data.cpu().numpy()
            z = 0.0
            for i in range(0, 1):
                for j in range(-1, 2):
                    row = mode_row + i
                    col = mode_col + j
                    if row < 0 or row >= 32 or col < 0 or col >= 192:
                        continue
                    expected_row = expected_row + row * prob_values[row, col]
                    expected_col = expected_col + col * prob_values[row, col]
                    z = z + prob_values[row, col]

            # print("Prob Values is %r, Mode Row is %r, Mode Col is %r, Expected Row is %r, Expected Col is %r, Z is %r"
            #       % (prob_values[mode_row, mode_col], mode_row, mode_col, expected_row, expected_col, z))
            inferred_ix = (expected_row * 192.0 + expected_col)/z
            if inferred_ix > 32 * 192:
                inferred_ix = 32 * 192

            print("Predicted Inferred ix is %r, Was %r " % (inferred_ix, mode_ix))

            return inferred_ix, (expected_row + 0.5, expected_col)

        else:
            raise AssertionError("Not handled")

        return inferred_ix, None

    def save_attention_prob(self, image, attention_prob, instruction, goal_prob=None):
        self.global_id += 1

        image_flipped = image.swapaxes(0, 1).swapaxes(1, 2)
        image_flipped = scipy.misc.imresize(image_flipped, (128, 128 * 6))
        attention_prob = attention_prob.cpu().data.numpy()
        resized_kernel = scipy.misc.imresize(attention_prob, (128*5, 128*5 * 6))
        if goal_prob is not None:
            goal_location = goal_prob.cpu().data.numpy()
            if np.sum(goal_location) > 0.01:
                for i in range(0, 32):
                    for j in range(0, 192):
                            if goal_location[i][j] < 0.01:
                                goal_location[i][j] = 0.0
                goal_location = scipy.misc.imresize(goal_location, (128, 128 * 6))
        else:
            goal_location = None

        plt.title(instruction)
        plt.imshow(resized_kernel, cmap='jet', alpha=0.5)
        plt.savefig("./final_figures/image_" + str(self.global_id) + ".png")
        plt.clf()

        # f, axarr = plt.subplots(1, 2)
        # if instruction is not None:
        #     f.suptitle(instruction)
        # axarr[0].set_title("Predicted Attention")
        # # axarr[0].imshow(image_flipped)
        # axarr[0].imshow(resized_kernel, cmap='jet', alpha=0.5)
        # axarr[1].set_title("Gold Attention (Goal)")
        # axarr[1].imshow(image_flipped)
        # if goal_location is not None:
        #     axarr[1].imshow(goal_location, cmap='jet', alpha=0.5)
        # plt.savefig("./attention_prob/image_" + str(self.global_id) + ".png")
        # plt.clf()

    def show_image(self, goal, predicted_goal, start_pos, instruction):
        self.global_id += 1

        # image_flipped = image.swapaxes(0, 1).swapaxes(1, 2)
        # image_flipped = scipy.misc.imresize(image_flipped, (128, 128 * 6))
        goal_map = np.zeros((50, 50))
        predicted_goal_map = np.zeros((50, 50))

        x_1, y_1 = goal
        x_2, y_2 = predicted_goal
        x_3, y_3, _ = start_pos

        x_1 = min(x_1, 274.99)
        y_1 = min(y_1, 274.99)
        x_2 = min(x_2, 274.99)
        y_2 = min(y_2, 274.99)
        x_3 = min(x_3, 274.99)
        y_3 = min(y_3, 274.99)

        print(" %r %r %r %r " % (x_1, y_1, x_2, y_2))
        assert 225.0 <= x_1 <= 275.0
        assert 225.0 <= x_2 <= 275.0
        assert 225.0 <= x_3 <= 275.0
        assert 225.0 <= y_1 <= 275.0
        assert 225.0 <= y_2 <= 275.0
        assert 225.0 <= y_3 <= 275.0

        i1, j1 = int((x_1 - 225.0)), int((y_1 - 225.0))
        i2, j2 = int((x_2 - 225.0)), int((y_2 - 225.0))
        i3, j3 = int((x_3 - 225.0)), int((y_3 - 225.0))

        goal_map[i1, j1] = 1.0
        goal_map[i3, j3] = 0.75
        predicted_goal_map[i2, j2] = 1.0
        predicted_goal_map[i3, j3] = 0.75

        f, axarr = plt.subplots(1, 2)
        if instruction is not None:
            f.suptitle(instruction)
        axarr[0].set_title("Predicted Goal")
        # axarr[0].imshow(image_flipped)
        axarr[0].imshow(predicted_goal_map, cmap='jet', alpha=0.5)
        axarr[1].set_title("Gold Goal")
        # axarr[1].imshow(image_flipped)
        axarr[1].imshow(goal_map, cmap='jet', alpha=0.5)
        plt.savefig("./attention_prob/image_" + str(self.global_id) + "_maps.png")
        plt.clf()

    def interactive_shell(self, train_dataset, train_images):

        traj_len = len(train_dataset)
        keep = False
        image_id = 1
        while True:

            # Sample a random dataset
            if not keep:
                ix = random.randint(0, traj_len - 1)
            data_point = train_dataset[ix]
            image = train_images[ix][0]

            # Show the image in pyplot
            plt.imshow(image.swapaxes(0, 1).swapaxes(1, 2))
            plt.ion()
            plt.show()

            # Get the instruction
            print("Enter the instruction below (q or quit to quit)\n")
            print("Sample instruction is ", instruction_to_string(data_point.instruction, self.config))
            while True:
                instruction = input()
                if instruction == "q" or instruction == "quit":
                    break
                elif len(instruction) == 0:
                    print("Enter a non-empty instruction (q or quit to quit)")
                else:
                    break

            instruction_id = self.convert_to_id(instruction)
            state = AgentObservedState(instruction=instruction_id,
                                       config=self.config,
                                       constants=self.constants,
                                       start_image=image,
                                       previous_action=None,
                                       pose=None,
                                       position_orientation=None,
                                       data_point=data_point)

            # Show the attention mask
            _, _, _, volatile = self.model.get_attention_prob(state, model_state=None)

            attention_prob = volatile["attention_probs"][:-1].view(self.final_height, self.final_width)
            attention_prob = attention_prob.cpu().data.numpy()
            resized_kernel = scipy.misc.imresize(attention_prob,
                                                 (self.config["image_height"], self.config["image_width"]))
            plt.clf()
            plt.title(instruction)
            plt.imshow(image.swapaxes(0, 1).swapaxes(1, 2))
            plt.imshow(resized_kernel, cmap="jet", alpha=0.5)

            print("Enter s to save, k to keep working on this environment, sk to do both. Other key to simply continue")
            key_ = input()
            if key_ == "s":
                plt.savefig("interactive_image_" + str(image_id) + ".png")
                image_id += 1

            if key_ == "k":
                keep = True
            else:
                keep = False

            if key_ == "sk":
                plt.savefig("image_" + str(image_id) + ".png")
                image_id += 1
                keep = True

            plt.clf()

    def test(self, tune_dataset, tune_image, tune_goal_location, tensorboard):

        total_validation_loss = 0
        total_validation_prob = 0
        total_validation_exact_accuracy = 0
        total_goal_distance = 0
        num_items = 0

        # Next metric measures when the goal is visible and prediction is within 10\% radius
        total_epsilon_accuracy = 0
        num_visible_items = 0

        # Next metric measures distance in real world and only when goal is visible
        total_real_world_distance = 0

        correct = 0
        count_correct = 0

        for data_point_ix, data_point in enumerate(tune_dataset):
            tune_image_example = tune_image[data_point_ix]
            goal_location = tune_goal_location[data_point_ix]
            image = tune_image_example[0]

            model_state = None
            state = AgentObservedState(instruction=data_point.instruction,
                                       config=self.config,
                                       constants=self.constants,
                                       start_image=image,
                                       previous_action=None,
                                       pose=None,
                                       position_orientation=data_point.get_start_pos(),
                                       data_point=data_point)

            num_items_ = 0
            sum_loss = 0
            sum_prob = 0
            sum_acc = 0
            sum_dist = 0
            sum_real_world_distance = 0

            goal = goal_location[0]
            state.goal = goal
            volatile = self.model.get_attention_prob(state, model_state)
            row, col, _, _ = goal

            if not self.ignore_none or row is not None:
                if row is None or col is None:
                    gold_ix = self.final_height * self.final_width
                else:
                    gold_ix = row * self.final_width + col
                loss, prob, meta = GoalPrediction.get_loss_and_prob(
                    volatile, goal, self.final_height, self.final_width)
                num_items_ += 1
                sum_loss = sum_loss + float(loss.data.cpu().numpy()[0])
                sum_prob = sum_prob + float(prob.data.cpu().numpy()[0])

                inferred_ix, row_col = self.get_inferred_value(volatile)
                # Center pixel prediction
                # inferred_ix, row_col = 20 * 192 + 32 * 3 + 16, None

                if gold_ix == inferred_ix:
                    sum_acc = sum_acc + 1.0
                if row is not None and col is not None:
                    sum_dist = sum_dist + abs(row - int(round(inferred_ix/self.final_width)))\
                               + abs(col - int(inferred_ix % self.final_height))
                    num_visible_items += 1
                    if self.is_close_enough(inferred_ix, row, col):
                        total_epsilon_accuracy += 1
                    predicted_goal, real_world_distance = self.compute_distance_in_real_world(inferred_ix, row_col, data_point)
                    sum_real_world_distance += real_world_distance

                    count_correct += 1.0
                    if real_world_distance <= 5.0:
                        correct += 1.0

                    # # Save the map
                    # instruction_string = instruction_to_string(data_point.instruction, self.config)
                    # goal_x, goal_y = data_point.get_destination_list()[-1]
                    # goal_x, goal_y = round(goal_x, 2), round(goal_y, 2)
                    # predicted_goal_x, predicted_goal_y = predicted_goal
                    # predicted_goal_x, predicted_goal_y = round(predicted_goal_x, 2), round(predicted_goal_y, 2)
                    # instruction_string = instruction_string + \
                    #                      "\n (Error: " + str(round(sum_real_world_distance, 2)) + ")" + \
                    #                      "\n %r %r %r %r \n" % (goal_x, goal_y, predicted_goal_x, predicted_goal_y)
                    # self.show_image(data_point.get_destination_list()[-1], predicted_goal, data_point.get_start_pos(),
                    #                 instruction_string)
                    #
                    # # Save the generated image
                    # goal_prob = GoalPrediction.generate_gold_prob(goal, 32, 192)
                    # predicted_goal = (int(inferred_ix/192), inferred_ix % 192, int(inferred_ix/192), inferred_ix % 192)
                    # predicted_goal_prob = GoalPrediction.generate_gold_prob(predicted_goal, 32, 192)
                    # self.save_attention_prob(image, volatile["attention_probs"][:-1].view(32, 192),
                    #                          instruction_string, goal_prob[:-1].view(32, 192))
                    # self.save_attention_prob(image, predicted_goal_prob[:-1].view(32, 192),
                    #                          instruction_string, goal_prob[:-1].view(32, 192))

            total_validation_loss += sum_loss
            total_validation_prob += sum_prob
            total_goal_distance += sum_dist
            total_validation_exact_accuracy += sum_acc
            total_real_world_distance += sum_real_world_distance
            num_items += num_items_

        mean_total_goal_distance = total_goal_distance / float(max(num_items, 1))
        mean_total_validation_loss = total_validation_loss / float(max(num_items, 1))
        mean_total_validation_prob = total_validation_prob / float(max(num_items, 1))
        mean_total_validation_accuracy = (total_validation_exact_accuracy * 100.0) / float(max(num_items, 1))
        mean_total_epsilon_accuracy = (total_epsilon_accuracy * 100.0) / float(max(num_visible_items, 1))
        mean_real_world_distance = total_real_world_distance / float(max(num_visible_items, 1))

        logging.info("Mean Test result: L1 Distance is %r, Loss %r, Prob %r, Acc is %r, Epsilon Accuracy is %r"
                     % (mean_total_goal_distance, mean_total_validation_loss, mean_total_validation_prob,
                        mean_total_validation_accuracy, mean_total_epsilon_accuracy))
        logging.info("Num visible items %r, Num Exact Match items is %r, Num epsilon match %r, Num Items is %r "
                     % (num_visible_items, total_validation_exact_accuracy, total_epsilon_accuracy, num_items))
        logging.info("Num visible items %r, Mean Real World Distance %r "
                     % (num_visible_items, mean_real_world_distance))
        logging.info("Num counts %r, Task Completion Accuracy %r "
                     % (count_correct, (correct * 100.0)/float(max(1, count_correct))))


        return mean_real_world_distance

    def do_train(self, train_dataset, train_images, train_goal_location,
                 tune_dataset, tune_images, tune_goal_location, experiment_name, save_best_model=False):
        """ Perform training """

        dataset_size = len(train_dataset)
        tensorboard = self.tensorboard

        # Test on tuning data with initialized model
        mean_real_world_distance = self.test(tune_dataset, tune_images, tune_goal_location, tensorboard=tensorboard)
        best_real_world_distance = mean_real_world_distance

        for epoch in range(1, self.max_epoch + 1):

            logging.info("Starting epoch %d", epoch)

            batch_replay_items = []
            best_real_world_distance = min(best_real_world_distance, mean_real_world_distance)

            for data_point_ix, data_point in enumerate(train_dataset):

                if (data_point_ix + 1) % 100 == 0:
                    logging.info("Done %d out of %d", data_point_ix, dataset_size)

                train_images_example = train_images[data_point_ix]
                goal_location = train_goal_location[data_point_ix]
                image = train_images_example[0]

                model_state = None
                state = AgentObservedState(instruction=data_point.instruction,
                                           config=self.config,
                                           constants=self.constants,
                                           start_image=image,
                                           previous_action=None,
                                           pose=None,
                                           position_orientation=data_point.get_start_pos(),
                                           data_point=data_point)

                # Generate attention probabilities
                volatile = self.model.get_attention_prob(state, model_state)
                goal = goal_location[0]

                # Store it in the replay memory list
                if not self.ignore_none or goal[0] is not None:
                    replay_item = ReplayMemoryItem(state, None, 0, volatile=volatile, goal=goal)
                    batch_replay_items.append(replay_item)

                # Perform update
                if len(batch_replay_items) > 0:
                    loss_val = self.do_update(batch_replay_items)
                    batch_replay_items = []
                    if tensorboard is not None:
                        tensorboard.log_scalar("Loss", loss_val)
                        if self.goal_prediction_loss is not None:
                            goal_prediction_loss = float(self.goal_prediction_loss.data[0])
                            tensorboard.log_scalar("goal_prediction_loss", goal_prediction_loss)
                        if self.goal_prob is not None:
                            goal_prob = float(self.goal_prob.data[0])
                            tensorboard.log_scalar("goal_prob", goal_prob)
                        if self.object_detection_loss is not None:
                            object_detection_loss = float(self.object_detection_loss.data[0])
                            tensorboard.log_scalar("object_detection_loss", object_detection_loss)
                        if self.cross_entropy_loss is not None:
                            cross_entropy_loss = float(self.cross_entropy_loss.data[0])
                            tensorboard.log_scalar("Cross_entropy_loss", cross_entropy_loss)
                        if self.dist_loss is not None:
                            dist_loss = float(self.dist_loss.data[0])
                            tensorboard.log_scalar("Dist_loss", dist_loss)

            mean_real_world_distance = self.test(tune_dataset, tune_images, tune_goal_location, tensorboard=tensorboard)

            # Save the model
            if save_best_model:
                if mean_real_world_distance < best_real_world_distance:
                    self.model.save_model(experiment_name + "/goal_prediction_single_supervised_epoch_" + str(epoch))
            else:
                self.model.save_model(experiment_name + "/goal_prediction_single_supervised_epoch_" + str(epoch))
