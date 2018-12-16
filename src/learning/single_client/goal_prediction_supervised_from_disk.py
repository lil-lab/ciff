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
from utils.geometry import current_pos_from_metadata, current_pose_from_metadata


class GoalPredictionSupervisedLearningFromDisk(AbstractLearning):
    """ Perform goal prediction on oracle trajectories using images stored on disk
    and hence does not need client or server. """

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
        self.entropy_coef = constants["entropy_coefficient"]
        self.final_height, self.final_width = 32, 32

        self.ignore_none = True
        self.only_first = True

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

        logging.info("Created Goal predictor with ignore_none %r and only_first %r", self.ignore_none, self.only_first)

    def calc_loss(self, batch_replay_items):

        # Only compute the goal prediction loss
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
    def parse(folder_name, dataset):

        start = time.time()

        image_dataset = []
        num_examples = len(os.listdir(folder_name))
        for i in range(0, num_examples):
            example_folder_name = folder_name + "/example_" + str(i)
            image_names = [file for file in os.listdir(example_folder_name) if file.endswith('.png')]
            num_actions = len(image_names)
            images = []
            for j in range(0, num_actions):
                img = scipy.misc.imread(example_folder_name + "/image_" + str(j) + ".png").swapaxes(1, 2).swapaxes(0, 1)
                images.append(img)
            image_dataset.append(images)

        # goal_dataset = []
        # num_examples = len(os.listdir(folder_name))
        # for i in range(0, num_examples):
        #     example_folder_name = folder_name + "/example_" + str(i)
        #     lines = open(example_folder_name + "/goal.txt").readlines()
        #     goals = []
        #     for line in lines:
        #         words = line.strip().split()
        #         assert len(words) == 4
        #         if words[0] == "None" or words[1] == "None":
        #             row, col, row_real, col_real = None, None, None, None
        #         else:
        #             row, col, row_real, col_real = int(words[0]), int(words[1]), float(words[2]), float(words[3])
        #             assert 0 <= row < 8 and 0 <= col < 8
        #         goals.append((row, col, row_real, col_real))
        #     goal_dataset.append(goals)
        #
        # assert len(image_dataset) == len(goal_dataset)
        assert len(image_dataset) == len(dataset)

        ####################################
        #  Hack for synythetic data
        goal_dataset = []
        for i in range(0, num_examples):
            data_point = dataset[i]
            pos = data_point.get_start_pos()
            metadata = {"x_pos": pos[0], "z_pos": pos[1], "y_angle": pos[2]}

            goal_location = [GoalPrediction.get_goal_location(metadata, data_point, 32, 32)]
            _, _, row, col = goal_location[0]

            start_pos = current_pos_from_metadata(metadata)
            start_pose = current_pose_from_metadata(metadata)

            if row is not None and col is not None:
                goal_pos = data_point.get_destination_list()[-1]
                height_drone = 2.5
                x_gen, y_gen = get_inverse_object_position(row, col, height_drone, 30, 32, 32,
                                                           (start_pos[0], start_pos[1], start_pose))

            goal_dataset.append(goal_location)
            data_point.trajectory = [0]
            # if len(image_dataset[i]) >= 2:
            #     data_point.trajectory = [0]  # dummy action added
        #####################################

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
            _, _, _, \
            volatile = self.model.get_attention_prob(state, model_state=None)

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
            trajectory = data_point.get_trajectory()
            if self.only_first:
                trajectory = trajectory[0:1]
            traj_len = len(trajectory)
            num_items_ = 0

            sum_loss = 0
            sum_prob = 0
            sum_acc = 0
            sum_dist = 0

            for action_ix, action in enumerate(trajectory):
                state.goal = goal_location[action_ix]
                volatile = self.model.get_attention_prob(state, model_state)
                goal = goal_location[action_ix]
                row, col, _, _ = goal

                if not self.ignore_none or row is not None:
                    if row is None:
                        gold_ix = self.final_height * self.final_width
                    else:
                        gold_ix = row * self.final_width + col
                    loss, prob, meta = GoalPrediction.get_loss_and_prob(
                        volatile, goal, self.final_height, self.final_width)
                    num_items_ += 1
                    sum_loss = sum_loss + float(loss.data.cpu().numpy()[0])
                    sum_prob = sum_prob + float(prob.data.cpu().numpy()[0])

                    inferred_ix = int(torch.max(volatile["attention_logits"], 0)[1].data.cpu().numpy()[0])
                    if gold_ix == inferred_ix:
                        sum_acc = sum_acc + 1.0
                    if row is not None:
                        sum_dist = sum_dist + abs(row - int(round(inferred_ix/self.final_width)))\
                                   + abs(col - int(inferred_ix % self.final_height))

                    if row is not None:
                        num_visible_items += 1
                        if self.is_close_enough(inferred_ix, row, col):
                            total_epsilon_accuracy += 1

                if not self.only_first:
                    image = tune_image_example[action_ix + 1]
                    state = state.update(image, action, pose=None,  position_orientation=None, data_point=data_point)

            if not self.only_first:
                state.goal = goal_location[traj_len]
                volatile = self.model.get_attention_prob(state, model_state)
                goal = goal_location[traj_len]
                row, col, _, _ = goal

                if not self.ignore_none or row is not None:
                    if row is None:
                        gold_ix = self.final_height * self.final_width
                    else:
                        gold_ix = row * self.final_width + col

                    loss, prob, _ = GoalPrediction.get_loss_and_prob(volatile, goal,
                                                                     self.final_height, self.final_width)
                    num_items_ += 1
                    sum_loss = sum_loss + float(loss.data.cpu().numpy()[0])
                    sum_prob = sum_prob + float(prob.data.cpu().numpy()[0])
                    inferred_ix = int(torch.max(volatile["attention_logits"], 0)[1].data.cpu().numpy()[0])
                    if gold_ix == inferred_ix:
                        sum_acc = sum_acc + 1.0
                    if row is not None:
                        sum_dist = sum_dist + abs(row - int(round(inferred_ix/self.final_width))) \
                                   + abs(col - int(inferred_ix % self.final_width))

                    if row is not None:
                        num_visible_items += 1
                        if self.is_close_enough(inferred_ix, row, col):
                            total_epsilon_accuracy += 1

            total_validation_loss += sum_loss
            total_validation_prob += sum_prob
            total_goal_distance += sum_dist
            total_validation_exact_accuracy += sum_acc
            num_items += num_items_

        mean_total_goal_distance = total_goal_distance / float(max(num_items, 1))
        mean_total_validation_loss = total_validation_loss / float(max(num_items, 1))
        mean_total_validation_prob = total_validation_prob / float(max(num_items, 1))
        mean_total_validation_accuracy = (total_validation_exact_accuracy * 100.0) / float(max(num_items, 1))
        mean_total_epsilon_accuracy = (total_epsilon_accuracy * 100.0) / float(max(num_visible_items, 1))

        logging.info("Mean Test result: L1 Distance is %r, Loss %r, Prob %r, Acc is %r, Epsilon Accuracy is %r"
                     % (mean_total_goal_distance, mean_total_validation_loss, mean_total_validation_prob,
                        mean_total_validation_accuracy, mean_total_epsilon_accuracy))
        logging.info("Num visible items %r, Num Exact Match items is %r, Num epsilon match %r, Num Items is %r "
                     % (num_visible_items, total_validation_exact_accuracy, total_epsilon_accuracy, num_items))

    def do_train(self, train_dataset, train_images, train_goal_location,
                 tune_dataset, tune_images, tune_goal_location, experiment_name):
        """ Perform training """

        dataset_size = len(train_dataset)
        tensorboard = self.tensorboard

        for epoch in range(1, self.max_epoch + 1):

            logging.info("Starting epoch %d", epoch)

            # Test on tuning data
            self.test(tune_dataset, tune_images, tune_goal_location, tensorboard=tensorboard)

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

                trajectory = data_point.get_trajectory()
                traj_len = len(trajectory)
                if self.only_first:
                    trajectory = trajectory[0:1]
                batch_replay_items = []

                for action_ix, action in enumerate(trajectory):

                    # Sample action using the policy
                    # Generate probabilities over actions
                    volatile = self.model.get_attention_prob(state, model_state)
                    goal = goal_location[action_ix]

                    # Store it in the replay memory list
                    if not self.ignore_none or goal[0] is not None:
                        replay_item = ReplayMemoryItem(state, action, 0, volatile=volatile, goal=goal)
                        batch_replay_items.append(replay_item)

                    if not self.only_first:
                        # Send the action and get feedback
                        image = train_images_example[action_ix + 1]

                        # Update the agent state
                        state = state.update(
                            image, action, pose=None,
                            position_orientation=None,
                            data_point=data_point)

                # Store it in the replay memory list
                if not self.only_first:
                    goal = goal_location[traj_len]
                    if not self.ignore_none or goal[0] is not None:
                        volatile = self.model.get_attention_prob(state, model_state)
                        replay_item = ReplayMemoryItem(
                            state, self.action_space.get_stop_action_index(), 0, volatile=volatile, goal=goal)
                        batch_replay_items.append(replay_item)

                # Perform update
                if len(batch_replay_items) > 0:
                    loss_val = self.do_update(batch_replay_items)
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

            # Save the model
            self.model.save_model(experiment_name + "/goal_prediction_supervised_epoch_" + str(epoch))
