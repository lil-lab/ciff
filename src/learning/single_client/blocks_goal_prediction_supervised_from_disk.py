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
from learning.asynchronous.tmp_blocks_asynchronous_contextual_bandit_learning import TmpAsynchronousContextualBandit
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


class BlockGoalPredictionSupervisedLearningFromDisk(AbstractLearning):
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
        self.inference_procedure = BlockGoalPredictionSupervisedLearningFromDisk.MODE

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
    def parse(folder_name, dataset, vocab, debug=False):

        start = time.time()

        with open(folder_name + "/dataset_goal.json") as f:

            data = json.load(f)

            for datapoint in data:

                # Read dataset information
                i = int(datapoint["id"])
                screen_width = datapoint["screenWidth"]
                screen_height = datapoint["screenHeight"]
                instr_str = datapoint["instruction"]
                gold_block_id = int(datapoint["goldBlockId"])

                start_loc_json = datapoint["startLocation"]
                start_loc_str = [start_loc_json["x"], start_loc_json["y"], start_loc_json["z"]]
                start_loc = [float(w) for w in start_loc_str]

                goal_loc_json = datapoint["goalLocation"]
                goal_loc_str = [goal_loc_json["x"], goal_loc_json["y"], goal_loc_json["z"]]
                goal_loc = [float(w) for w in goal_loc_str]

                goal_pixel_json = datapoint["goalPixel"]
                goal_pixel_str = [goal_pixel_json["x"], goal_pixel_json["y"], goal_pixel_json["z"]]
                goal_pixel = [float(w) for w in goal_pixel_str]

                # Read the image
                image = np.load(folder_name + "/example_%s/image.npy" % i)
                image = image.swapaxes(0, 1).swapaxes(1, 2)
                image = np.rot90(image, k=2)
                image = np.fliplr(image)

                # Read the goal information
                lines = open(folder_name + "/example_%s/instruction.txt" % i).readlines()
                assert len(lines) == 2
                instruction = lines[0]

                pixel = ((screen_height - goal_pixel[1])/float(screen_height), (goal_pixel[0])/float(screen_width))

                if debug:
                    # save the image for debugging
                    pixel_row_real = int(128 * pixel[0])   # the additional slack is for boundary
                    pixel_col_real = int(128 * pixel[1])

                    if pixel_row_real < 0 or pixel_row_real >= 128 or pixel_col_real < 0 or pixel_col_real >= 128:
                        raise AssertionError("failed")

                    goal = np.zeros((128, 128))
                    for i1 in range(-3, 3):
                        for j1 in range(-3, 3):
                            if pixel_row_real + i1 < 0 or pixel_row_real + i1 >= 128:
                                continue
                            if pixel_col_real + j1 < 0 or pixel_col_real + j1 >= 128:
                                continue
                            goal[pixel_row_real + i1][pixel_col_real + j1] = 1.0

                    f, axarr = plt.subplots(1, 2)
                    if instruction is not None:
                        f.suptitle(instruction)
                    axarr[0].imshow(image)
                    axarr[0].imshow(goal, cmap='jet', alpha=0.5)
                    plt.savefig("./goal_block/image_" + str(i) + ".png")
                    plt.clf()

                instruction_indices = TmpAsynchronousContextualBandit.convert_text_to_indices(instruction, vocab)

                python_datapoint = dataset[i - 1]
                python_datapoint.set_instruction(instruction_indices, instruction)
                python_datapoint.set_start_image(image.swapaxes(1, 2).swapaxes(0, 1))
                python_datapoint.set_block_id(gold_block_id)
                python_datapoint.set_start_location(start_loc)
                python_datapoint.set_goal_location(goal_loc)

                scaled_pixel_row, scaled_pixel_col = int(pixel[0] * 32), int(pixel[1] * 32)
                if scaled_pixel_row < 0:
                    scaled_pixel_row = 0
                elif scaled_pixel_row >= 32:
                    scaled_pixel_row = 31

                if scaled_pixel_col < 0:
                    scaled_pixel_col = 0
                elif scaled_pixel_col >= 32:
                    scaled_pixel_col = 31

                python_datapoint.set_goal_pixel((scaled_pixel_row, scaled_pixel_col))

                predicted_x = 1.25 - 2.5 * (scaled_pixel_col / 32.0)  # ranges between -1.25 to 1.25
                predicted_z = 2.5 * (scaled_pixel_row / 32.0) - 1.25  # ranges between -1.25 to 1.25

                print("Pixel is %r and Goal is %r and Prediction is %r " % ((scaled_pixel_row, scaled_pixel_col),
                                                                            (goal_loc[0], goal_loc[2]),
                                                                            (predicted_x, predicted_z)))

        end = time.time()
        logging.info("Parsed dataset of size %r in time % seconds", len(dataset), (end - start))

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

    def compute_distance_in_real_world(self, inferred_ix, data_point):

        # find predicted pixel
        row = int(inferred_ix/32) + 0.5
        col = inferred_ix % 32 + 0.5

        # convert to real world location
        predicted_x = 1.25 - 2.5 * (col / 32.0)  # ranges between -1.25 to 1.25
        predicted_z = 2.5 * (row / 32.0) - 1.25  # ranges between -1.25 to 1.25

        # gold location
        gold_x, _, gold_z = data_point.goal_location

        # print("Predicted Goal %r and Goal is %r " % ((row, col), data_point.goal_pixel))
        # print("Predicted Goal Location %r and Goal Location %r" % ((predicted_x, predicted_z), (gold_x, gold_z)))

        l2_distance = math.sqrt(
            (predicted_x - gold_x) * (predicted_x - gold_x) + (predicted_z - gold_z) * (predicted_z - gold_z))

        block_size = 0.1524
        bisk_distance = l2_distance/block_size
        return bisk_distance


    def get_inferred_value(self, volatile):

        # Mode setting
        inferred_ix = int(torch.max(volatile["attention_logits"], 0)[1].data.cpu().numpy()[0])
        return inferred_ix, None

    def save_attention_prob(self, image, attention_prob, instruction, goal_prob=None):

        image_flipped = image.swapaxes(0, 1).swapaxes(1, 2)
        image_flipped = scipy.misc.imresize(image_flipped, (128, 128))
        attention_prob = attention_prob.cpu().data.numpy()
        resized_kernel = scipy.misc.imresize(attention_prob, (128, 128))
        if goal_prob is not None:
            goal_location = goal_prob.cpu().data.numpy()
            if np.sum(goal_location) > 0.01:
                for i in range(0, 32):
                    for j in range(0, 32):
                            if goal_location[i][j] < 0.01:
                                goal_location[i][j] = 0.0
                goal_location = scipy.misc.imresize(goal_location, (128, 128))
        else:
            goal_location = None

        f, axarr = plt.subplots(1, 2)
        if instruction is not None:
            f.suptitle(instruction)
        axarr[0].set_title("Predicted Attention")
        axarr[0].imshow(image_flipped)
        axarr[0].imshow(resized_kernel, cmap='jet', alpha=0.5)
        axarr[1].set_title("Gold Attention (Goal)")
        axarr[1].imshow(image_flipped)
        if goal_location is not None:
            axarr[1].imshow(goal_location, cmap='jet', alpha=0.5)
        plt.savefig("./attention_prob/image_" + str(self.global_id) + ".png")
        plt.clf()

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

    def test(self, tune_dataset, tensorboard):

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

        for data_point_ix, data_point in enumerate(tune_dataset):

            model_state = None
            state = AgentObservedState(instruction=data_point.instruction,
                                       config=self.config,
                                       constants=self.constants,
                                       start_image=data_point.start_image,
                                       previous_action=None,
                                       pose=None,
                                       position_orientation=None,
                                       data_point=data_point)

            num_items_ = 0
            sum_loss = 0
            sum_prob = 0
            sum_acc = 0
            sum_dist = 0
            sum_real_world_distance = 0

            row, col = data_point.goal_pixel
            goal = row, col, row, col
            state.goal = goal
            volatile = self.model.get_attention_prob(state, model_state)

            if not self.ignore_none or row is not None:
                gold_ix = row * self.final_width + col
                loss, prob, meta = GoalPrediction.get_loss_and_prob(
                    volatile, goal, self.final_height, self.final_width)
                num_items_ += 1
                sum_loss = sum_loss + float(loss.data.cpu().numpy()[0])
                sum_prob = sum_prob + float(prob.data.cpu().numpy()[0])

                inferred_ix, row_col = self.get_inferred_value(volatile)

                if gold_ix == inferred_ix:
                    sum_acc = sum_acc + 1.0
                if row is not None and col is not None:
                    sum_dist = sum_dist + abs(row - int(round(inferred_ix/self.final_width)))\
                               + abs(col - int(inferred_ix % self.final_height))
                    num_visible_items += 1
                    if self.is_close_enough(inferred_ix, row, col):
                        total_epsilon_accuracy += 1
                    real_world_distance = self.compute_distance_in_real_world(inferred_ix, data_point)
                    sum_real_world_distance += real_world_distance

                    # Save the map
                    instruction_string = instruction_to_string(data_point.instruction, self.config)
                    # goal_x, goal_y = data_point.goal_location
                    # goal_x, goal_y = round(goal_x, 2), round(goal_y, 2)
                    # predicted_goal_x, predicted_goal_y = predicted_goal
                    # predicted_goal_x, predicted_goal_y = round(predicted_goal_x, 2), round(predicted_goal_y, 2)
                    # instruction_string = instruction_string + \
                    #                      "\n (Error: " + str(round(sum_real_world_distance, 2)) + ")" + \
                    #                      "\n %r %r %r %r \n" % (goal_x, goal_y, predicted_goal_x, predicted_goal_y)
                    # self.show_image(data_point.get_destination_list()[-1], predicted_goal, data_point.get_start_pos(),
                    #                 instruction_string)

                    # Save the generated image
                    self.global_id += 1
                    if self.global_id % 25 == 0:
                        goal_prob = GoalPrediction.generate_gold_prob(goal, 32, 32)
                        predicted_goal = (int(inferred_ix/32), inferred_ix % 32, int(inferred_ix/32), inferred_ix % 32)
                        predicted_goal_prob = GoalPrediction.generate_gold_prob(predicted_goal, 32, 32)
                        self.save_attention_prob(data_point.start_image, volatile["attention_probs"][:-1].view(32, 32),
                                                 data_point.instruction_string, goal_prob[:-1].view(32, 32))
                        self.save_attention_prob(data_point.start_image, predicted_goal_prob[:-1].view(32, 32),
                                                 data_point.instruction_string, goal_prob[:-1].view(32, 32))

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

        return mean_real_world_distance

    def do_train(self, train_dataset, tune_dataset, experiment_name, save_best_model=False):
        """ Perform training """

        dataset_size = len(train_dataset)
        tensorboard = self.tensorboard

        # Test on tuning data with initialized model
        mean_real_world_distance = self.test(tune_dataset, tensorboard=tensorboard)
        best_real_world_distance = mean_real_world_distance

        for epoch in range(1, self.max_epoch + 1):

            logging.info("Starting epoch %d", epoch)

            batch_replay_items = []
            best_real_world_distance = min(best_real_world_distance, mean_real_world_distance)

            for data_point_ix, data_point in enumerate(train_dataset):

                if (data_point_ix + 1) % 100 == 0:
                    logging.info("Done %d out of %d", data_point_ix, dataset_size)

                model_state = None
                state = AgentObservedState(instruction=data_point.instruction,
                                           config=self.config,
                                           constants=self.constants,
                                           start_image=data_point.start_image,
                                           previous_action=None,
                                           pose=None,
                                           position_orientation=None,
                                           data_point=data_point)

                # Generate attention probabilities
                volatile = self.model.get_attention_prob(state, model_state)
                row, col = data_point.goal_pixel
                goal = row, col, row, col

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

            mean_real_world_distance = self.test(tune_dataset, tensorboard=tensorboard)

            # Save the model
            if save_best_model:
                if mean_real_world_distance < best_real_world_distance:
                    self.model.save_model(experiment_name + "/goal_prediction_single_supervised_epoch_" + str(epoch))
            else:
                self.model.save_model(experiment_name + "/goal_prediction_single_supervised_epoch_" + str(epoch))
