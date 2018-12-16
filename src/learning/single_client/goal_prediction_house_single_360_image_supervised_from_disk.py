import json
import torch
import nltk
import time
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


class GoalPredictionHouseSingle360ImageSupervisedLearningFromDisk(AbstractLearning):
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

        self.ignore_none = False
        self.inference_procedure = GoalPredictionHouseSingle360ImageSupervisedLearningFromDisk.MODE

        # Auxiliary Objectives
        if self.config["do_action_prediction"]:
            self.action_prediction_loss_calculator = ActionPrediction(self.model)
            self.action_prediction_loss = None
        if self.config["do_temporal_autoencoding"]:
            self.temporal_autoencoder_loss_calculator = TemporalAutoEncoder(self.model)
            self.temporal_autoencoder_loss = None
        if self.config["do_object_detection"]:
            self.object_detection_loss_calculator = None
            # self.object_detection_loss_calculator = ObjectPixelIdentification(
            #     self.model, num_objects=67, camera_angle=60, image_height=self.final_height,
            #     image_width=self.final_width, object_height=0)  # -2.5)
            self.object_detection_loss = None
        if self.config["do_symbolic_language_prediction"]:
            self.symbolic_language_prediction_loss_calculator = SymbolicLanguagePrediction(self.model)
            self.symbolic_language_prediction_loss = None
        if self.config["do_goal_prediction"]:
            self.goal_prediction_calculator = GoalPrediction(self.model, self.final_height, self.final_width)
            self.goal_prediction_loss = None

        self.cross_entropy_loss = None
        self.dist_loss = None
        self.object_detection_loss = None

        self.optimizer = optim.Adam(model.get_parameters(),
                                    lr=constants["learning_rate"])
        AbstractLearning.__init__(self, self.model, self.calc_loss, self.optimizer,
                                  self.config, self.constants, self.tensorboard)

        logging.info("Created Single Image goal predictor with ignore_none %r", self.ignore_none)

    def calc_loss(self, batch_replay_items):

        self.goal_prediction_loss, self.goal_prob, meta = self.goal_prediction_calculator.calc_loss(batch_replay_items)
        loss = self.goal_prediction_loss

        # if self.config["do_object_detection"]:
        #     self.object_detection_loss = self.object_detection_loss_calculator.calc_loss(batch_replay_items)
        #     if self.object_detection_loss is not None:
        #         self.object_detection_loss = self.constants["object_detection_coeff"] * self.object_detection_loss
        #         loss = loss + self.object_detection_loss
        # else:
        #     self.object_detection_loss = None

        self.cross_entropy_loss = meta["cross_entropy"]
        self.dist_loss = meta["dist_loss"]

        return loss

    @staticmethod
    def parse(house_id, vocab, size):

        start = time.time()
        lines = open("./house_house%r_goal_prediction_data.json" % house_id).readlines()

        dataset = []
        for line in lines:

            jobj = json.loads(line)

            # Read the instruction
            instruction_string = jobj["instruction"].lower()
            instruction = GoalPredictionHouseSingle360ImageSupervisedLearningFromDisk.convert_to_id(
                instruction_string, vocab)

            # Read the image
            image_file_name = jobj["imageFileName"].replace("goal_images", "goal_images_%r" % size)
            images = []

            # image_order = range(0, 6)  # clockwise
            image_order = [3, 4, 5, 0, 1, 2]

            for ix in image_order:  # panaroma consists of 6 images stitched together
                img = scipy.misc.imread(image_file_name + "/image_" + str(ix + 1) + ".png")
                images.append(img)
            images = np.hstack(images)
            panoramic_image = images.swapaxes(1, 2).swapaxes(0, 1)
            panoramic_image = scipy.misc.imresize(panoramic_image, (size, size * 6)).swapaxes(1, 2).swapaxes(0, 1)

            # Read and calculate the goal pixel in the panaromic image
            screen_point_left = jobj["allPixelsFromLeft"]
            screen_point_bottom = jobj["allPixelsFromBottom"]
            screen_point_depth = jobj["allPixelsZ"]

            valid_regions = []
            for ix in image_order:

                # There is maximum one region in which the goal is visible.
                # Check if the goal is visible in this region.
                left, bottom, depth = screen_point_left[ix], screen_point_bottom[ix], screen_point_depth[ix]

                if 0.01 < left < size and 0.01 < bottom < size and depth > 0.01:
                    valid_regions.append(ix)

            if len(valid_regions) == 0:
                # Goal is not visible in any region
                goal_pixel = None, None, None, None
            elif len(valid_regions) == 1:
                valid_region = valid_regions[0]

                scaled_left = screen_point_left[valid_region]/float(size)
                scaled_top = 1.0 - screen_point_bottom[valid_region]/float(size)

                row_real = 32 * scaled_top
                col_real = 32 * scaled_left

                # Using a panaroma will change the column position based on which image has the goal.
                if valid_region == 0:
                    column_padding = 32 * 3
                elif valid_region == 1:
                    column_padding = 32 * 4
                elif valid_region == 2:
                    column_padding = 32 * 5
                elif valid_region == 3:
                    column_padding = 0
                elif valid_region == 4:
                    column_padding = 32 * 1
                elif valid_region == 5:
                    column_padding = 32 * 2
                else:
                    raise AssertionError("Valid region must be in {0, 1, 2, 3, 4, 5}")

                col_real += column_padding

                row, col = round(row_real), round(col_real)

                if row < 0:
                    row = 0
                elif row >= 32:
                    row = 31
                if col < 0:
                    col = 0
                elif col >= 32 * 6:
                    col = 32 * 6 - 1

                goal_pixel = row, col, row_real, col_real
            else:
                raise AssertionError("Goal cannot be visible in two region of a panaroma")

            data_point = GoalPredictionDataPoint(jobj["id"], instruction, instruction_string,
                                                 [panoramic_image], [goal_pixel], jobj["goalDefinition"])
            dataset.append(data_point)

        end = time.time()
        logging.info("Parsed dataset of size %r in time % seconds", len(dataset), (end - start))

        return dataset

    def save_datapoint(self, datapoint):

        self.global_id += 1
        image_flipped = datapoint.image[0].swapaxes(0, 1).swapaxes(1, 2)
        image_flipped = scipy.misc.imresize(image_flipped, (128, 128 * 6))
        attention_prob = np.zeros((32, 32 * 6))
        row, col, _, _ = datapoint.goal_pixel[0]

        if row is not None and col is not None:
            for i in range(-3, 3):
                for j in range(-3, 3):
                    if 0 <= row + i < 32 and 0 <= col + j < 192:
                        attention_prob[row + i][col + j] = 1.0
        attention_prob = scipy.misc.imresize(attention_prob, (128, 128 * 6))

        f, axarr = plt.subplots(1, 1)
        f.suptitle(datapoint.instruction_string)
        axarr.set_title(datapoint.goal_definition)
        axarr.imshow(image_flipped)
        axarr.imshow(attention_prob, cmap='jet', alpha=0.5)
        plt.savefig("./house_panorama/image_" + str(self.global_id) + ".png")
        plt.clf()

    @staticmethod
    def convert_to_id(instruction, vocab):
        tk_seq = nltk.tokenize.word_tokenize(instruction)
        token_ids = []
        for tk in tk_seq:
            if tk in vocab:
                token_ids.append(vocab[tk])
            else:
                token_ids.append(vocab["$UNK$"])
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

    def get_inferred_value(self, volatile):

        if self.inference_procedure == GoalPredictionHouseSingle360ImageSupervisedLearningFromDisk.MODE:
            # Mode setting
            inferred_ix = int(torch.max(volatile["attention_logits"], 0)[1].data.cpu().numpy()[0])

            return inferred_ix, None

        elif self.inference_procedure == GoalPredictionHouseSingle360ImageSupervisedLearningFromDisk.MEAN:
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

        elif self.inference_procedure == GoalPredictionHouseSingle360ImageSupervisedLearningFromDisk.MEANAROUNDMODE:

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

    def save_attention_prob(self, image, attention_prob, instruction, goal_prob=None):
        self.global_id += 1

        image_flipped = image.swapaxes(0, 1).swapaxes(1, 2)
        image_flipped = scipy.misc.imresize(image_flipped, (128, 128 * 6))
        attention_prob = attention_prob.cpu().data.numpy()
        resized_kernel = scipy.misc.imresize(attention_prob, (128, 128 * 6))
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
        total_validation_visible_exact_accuracy = 0
        total_goal_distance = 0
        num_items = 0

        # Next metric measures when the goal is visible and prediction is within 10\% radius
        total_epsilon_accuracy = 0
        num_visible_items = 0

        for data_point_ix, data_point in enumerate(tune_dataset):
            tune_image_example = data_point.image
            goal_location = data_point.goal_pixel
            image = tune_image_example[0]

            model_state = None
            state = AgentObservedState(instruction=data_point.instruction,
                                       config=self.config,
                                       constants=self.constants,
                                       start_image=image,
                                       previous_action=None,
                                       pose=None,
                                       position_orientation=None,
                                       data_point=data_point)

            num_items_ = 0
            sum_loss = 0
            sum_prob = 0
            sum_acc = 0
            sum_visible_exact = 0
            sum_dist = 0

            goal = goal_location[0]
            state.goal = goal
            volatile = self.model.get_attention_prob(state, model_state)
            row, col, _, _ = goal

            logging.info("Instruction is %s " % data_point.instruction_string)

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

                if gold_ix == inferred_ix:
                    sum_acc = sum_acc + 1.0
                    logging.info("Exact Match")
                else:
                    logging.info("Did Not Match Exactly")
                if row is not None and col is not None:
                    sum_dist = sum_dist + abs(row - int(round(inferred_ix/self.final_width)))\
                               + abs(col - int(inferred_ix % self.final_height))
                    num_visible_items += 1
                    if self.is_close_enough(inferred_ix, row, col):
                        total_epsilon_accuracy += 1

                    if gold_ix == inferred_ix:
                        sum_visible_exact += 1

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
            total_validation_visible_exact_accuracy += sum_visible_exact
            num_items += num_items_

        # Metric over the entire data
        mean_total_validation_accuracy = (total_validation_exact_accuracy * 100.0) / float(max(num_items, 1))
        mean_total_validation_loss = total_validation_loss / float(max(num_items, 1))
        mean_total_validation_prob = total_validation_prob / float(max(num_items, 1))

        # Metric over examples which are visible
        mean_total_goal_distance = total_goal_distance / float(max(num_visible_items, 1))
        mean_total_epsilon_accuracy = (total_epsilon_accuracy * 100.0) / float(max(num_visible_items, 1))
        visible_accuracy = (total_validation_visible_exact_accuracy * 100.0) / float(max(num_visible_items, 1))

        logging.info("Mean Test result (All Data): Num items %r, Exact Match %r, Loss %r, Prob %r, "
                     % (num_items, mean_total_validation_accuracy, mean_total_validation_loss, mean_total_validation_prob))
        logging.info("Num visible items %r, Visible Exact Match Accuracy %r, L1 distance %r, Epsilon Accuracy %r"
                     % (num_visible_items, visible_accuracy, mean_total_goal_distance, mean_total_epsilon_accuracy))

        return mean_total_validation_accuracy

    def do_train(self, train_dataset, tune_dataset, experiment_name, save_best_model=False):
        """ Perform training """

        dataset_size = len(train_dataset)
        tensorboard = self.tensorboard

        # Test on tuning data with initialized model
        mean_distance = self.test(tune_dataset, tensorboard=tensorboard)
        best_distance = mean_distance

        for epoch in range(1, self.max_epoch + 1):

            logging.info("Starting epoch %d", epoch)

            batch_replay_items = []
            best_distance = min(best_distance, mean_distance)

            for data_point_ix, data_point in enumerate(train_dataset):

                if (data_point_ix + 1) % 100 == 0:
                    logging.info("Done %d out of %d", data_point_ix, dataset_size)

                model_state = None
                state = AgentObservedState(instruction=data_point.instruction,
                                           config=self.config,
                                           constants=self.constants,
                                           start_image=data_point.image[0],
                                           previous_action=None,
                                           pose=None,
                                           position_orientation=None,
                                           data_point=data_point)

                # Generate attention probabilities
                goal = data_point.goal_pixel[0]
                # state.goal = goal
                volatile = self.model.get_attention_prob(state, model_state)

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

            mean_distance = self.test(tune_dataset, tensorboard=tensorboard)

            # Save the model
            if save_best_model:
                if mean_distance < best_distance:
                    self.model.save_model(experiment_name + "/goal_prediction_single_supervised_epoch_" + str(epoch))
            else:
                self.model.save_model(experiment_name + "/goal_prediction_single_supervised_epoch_" + str(epoch))


class GoalPredictionDataPoint:

    def __init__(self, task_id, instruction, instruction_string, image, goal_pixel, goal_definition):
        self.task_id = task_id
        self.instruction = instruction
        self.instruction_string = instruction_string
        self.image = image
        self.goal_pixel = goal_pixel
        self.goal_definition = goal_definition
