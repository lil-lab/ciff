import os

import time
import torch
import json
import logging
import scipy.misc
import torch.optim as optim
import utils.generic_policy as gp

from agents.agent_observed_state import AgentObservedState
from agents.replay_memory_item import ReplayMemoryItem
from abstract_learning import AbstractLearning
from utils.nav_drone_landmarks import get_name_of_landmark
from utils.nav_drone_landmarks import get_all_landmark_names


class SupervisedLearningDetectSymbolicEnvironment(AbstractLearning):
    """ Perform supervised learning to train a simple classifier on resnet features """

    def __init__(self, model, action_space, meta_data_util, config, constants, tensorboard):
        self.max_epoch = 100
        self.model = model
        self.action_space = action_space
        self.meta_data_util = meta_data_util
        self.config = config
        self.constants = constants
        self.tensorboard = tensorboard
        self.entropy = None
        self.cross_entropy = None
        self.entropy_coef = constants["entropy_coefficient"]
        self.optimizer = optim.Adam(model.get_parameters(),
                                    lr=constants["learning_rate"])

        self.confusion_num_count = []
        self.confusion_denom_count = []

        for i in range(0, 63):
            self.confusion_num_count.append([0.0] * 63)
            self.confusion_denom_count.append([0.0] * 63)

        AbstractLearning.__init__(self, self.model, self.calc_loss,
                                  self.optimizer, self.config, self.constants)

    def calc_loss(self, batch_replay_items):

        images = []
        visible_objects = []
        for replay_item in batch_replay_items:
            image, visible_objects_ = replay_item
            visible_objects.append(visible_objects_)
            images.append([image])

        landmark_log_prob, distance_log_prob, theta_log_prob = self.model.get_probs(images)
        num_states = int(landmark_log_prob.size()[0])
        
        landmark_objective = None
        distance_objective = None
        theta_objective = None
        num_visible = 0.0
        for i in range(0, num_states):
            visible_objects_example = visible_objects[i]
            for landmark in range(0, 63):
                # See if the landmark is present and visible in the agent's field of view
                if landmark in visible_objects_example and visible_objects_example[landmark][1] != -1:
                    r, theta = visible_objects_example[landmark]
                    if landmark_objective is None:
                        landmark_objective = landmark_log_prob[i, landmark, 1]
                    else:
                        landmark_objective = landmark_objective + landmark_log_prob[i, landmark, 1]

                    if distance_objective is None:
                        distance_objective = distance_log_prob[i, landmark, r]
                    else:
                        distance_objective = distance_objective + distance_log_prob[i, landmark, r]

                    if theta_objective is None:
                        theta_objective = theta_log_prob[i, landmark, theta]
                    else:
                        theta_objective = theta_objective + theta_log_prob[i, landmark, theta]

                    num_visible += 1.0
                else:
                    if landmark_objective is None:
                        landmark_objective = landmark_log_prob[i, landmark, 0]
                    else:
                        landmark_objective = landmark_objective + landmark_log_prob[i, landmark, 0]

        landmark_objective = landmark_objective / (num_states * 63.0)
        loss = - landmark_objective
        if num_visible != 0:
            distance_objective = distance_objective / num_visible
            theta_objective = theta_objective / num_visible
            loss = loss - distance_objective - theta_objective

        return loss

    def update_confusion_matrix(self, gold_set, predicted_set):

        for i in range(0, 63):
            for j in range(0, 63):
                # If i is present and predicted but j is not present
                if i in gold_set and i in predicted_set and j not in gold_set:
                    self.confusion_denom_count[i][j] += 1
                    if j in predicted_set:
                        self.confusion_num_count[i][j] += 1

    def print_confusion_matrix(self):

        for i in range(0, 63):
            normalized_vals = []
            for j in range(0, 63):
                normalized_val = self.confusion_num_count[i][j]/max(self.confusion_denom_count[i][j], 1.0)
                if normalized_val > 0.1:
                    normalized_vals.append([(get_name_of_landmark(j), normalized_val)])
            logging.info("Row %r is %r", get_name_of_landmark(i), normalized_vals)

    @staticmethod
    def get_f1_score(entire_field_gold_set, predicted_set):

        visible_gold_set = dict()  # contains landmark which are visible in the agent's field of view
        for landmark in entire_field_gold_set:
            if entire_field_gold_set[landmark][1] != -1:
                visible_gold_set[landmark] = entire_field_gold_set[landmark]

        precision = 0
        recall = 0

        distance_precision = 0
        theta_regression = 0
        num_distance_theta_cases = 0

        for val in predicted_set:
            if val in visible_gold_set:
                precision += 1
                r_gold, theta_gold = visible_gold_set[val]
                r, theta = predicted_set[val]
                num_distance_theta_cases += 1
                if r == r_gold:
                    distance_precision += 1
                mod_angle_diff = (theta_gold - theta) % 48
                theta_regression += min(mod_angle_diff, 48 - mod_angle_diff)

        for val in visible_gold_set:
            if val in predicted_set:
                recall += 1

        mean_distance_precision = distance_precision / float(max(num_distance_theta_cases, 1))
        mean_theta_regression = theta_regression / float(max(num_distance_theta_cases, 1))

        if len(visible_gold_set) == 0 and len(predicted_set) == 0:
            return 1, 1, 1, mean_distance_precision, mean_theta_regression, num_distance_theta_cases
        if len(visible_gold_set) == 0 and len(predicted_set) != 0:
            return 0, 0, 1, mean_distance_precision, mean_theta_regression, num_distance_theta_cases
        if len(visible_gold_set) != 0 and len(predicted_set) == 0:
            return 0, 1, 0, mean_distance_precision, mean_theta_regression, num_distance_theta_cases

        recall /= float(max(len(visible_gold_set), 1))
        precision /= float(max(len(predicted_set), 1))

        if precision == 0 and recall == 0:
            f1_score = 0
        else:
            f1_score = (2 * precision * recall) / (precision + recall)

        return f1_score, precision, recall, mean_distance_precision, mean_theta_regression, num_distance_theta_cases

    def test(self, test_dataset):

        mean_f1_score, mean_precision, mean_recall, mean_distance_precision, mean_theta_regression = 0, 0, 0, 0, 0
        num_data_points = 0
        num_distance_theta_cases = 0

        for data_point_ix, data_point in enumerate(test_dataset):

            image, visible_objects = data_point

            # Compute probabilities over list of visible objects
            log_prob_landmark, log_prob_distance, log_prob_theta = self.model.get_probs([[image]])

            prob_landmark = list(torch.exp(log_prob_landmark.data)[0])
            prob_distance = list(torch.exp(log_prob_distance.data)[0])
            prob_theta = list(torch.exp(log_prob_theta.data)[0])

            predicted_set = dict()
            for i in range(0, 63):
                argmax_val = gp.get_argmax_action(prob_landmark[i])
                if argmax_val == 1:
                    # predicted the distance and angle
                    argmax_landmark_val = gp.get_argmax_action(prob_distance[i])
                    argmax_theta_val = gp.get_argmax_action(prob_theta[i])
                    predicted_set[i] = (argmax_landmark_val, argmax_theta_val)

            f1_score, precision, recall, distance_precision, theta_regression, num_distance_theta_cases_ = \
                SupervisedLearningDetectSymbolicEnvironment.get_f1_score(visible_objects, predicted_set)
            # self.update_confusion_matrix(visible_objects, predicted_set)
            mean_f1_score += f1_score
            mean_precision += precision
            mean_recall += recall
            num_data_points += 1
            mean_distance_precision += distance_precision
            mean_theta_regression += theta_regression
            num_distance_theta_cases += num_distance_theta_cases_

        mean_f1_score /= float(max(num_data_points, 1))
        mean_precision /= float(max(num_data_points, 1))
        mean_recall /= float(max(num_data_points, 1))
        mean_distance_precision /= float(max(num_distance_theta_cases, 1))
        mean_theta_regression /= float(max(num_distance_theta_cases, 1))

        logging.info(
            "Object detection accuracy on a dataset of size %r the mean f1 score is %r, precision %r, recall %r",
            num_data_points, mean_f1_score, mean_precision, mean_recall)
        logging.info(
            "Object location accuracy on %r cases the mean distance precision is %r and theta regression is %r angle",
            num_distance_theta_cases, mean_distance_precision, mean_theta_regression * 7.5)

    @staticmethod
    def parse(folder_name):

        start = time.time()
        dataset = []
        landmark_names = get_all_landmark_names()
        num_examples = len(os.listdir(folder_name))
        for i in range(0, num_examples):
            example_folder_name = folder_name + "/example_" + str(i)
            image_names = [file for file in os.listdir(example_folder_name) if file.endswith('.png')]
            num_actions = len(image_names)
            for j in range(0, num_actions):
                img = scipy.misc.imread(example_folder_name + "/image_" + str(j) + ".png").swapaxes(1, 2).swapaxes(0, 1)
                with open(example_folder_name + "/data_" + str(j) + ".json", 'r') as fp:
                    data = json.load(fp)
                    new_data = dict()
                    for key in data:
                        new_data[landmark_names.index(key)] = data[key]
                    dataset.append((img, new_data))

        end = time.time()
        logging.info("Parsed dataset of size %r in time % seconds", len(dataset), (end - start))
        return dataset

    def do_train(self, train_dataset, tune_dataset, experiment_name):
        """ Perform training """

        dataset_size = len(train_dataset)

        for epoch in range(1, self.max_epoch + 1):

            logging.info("Starting epoch %d", epoch)

            # Test on tuning data
            self.test(tune_dataset)
            return
            # self.print_confusion_matrix()

            batch_items = []

            for data_point_ix, data_point in enumerate(train_dataset):

                if (data_point_ix + 1) % 10000 == 0:
                    logging.info("Done %d out of %d", data_point_ix, dataset_size)

                image, landmark_visible = data_point
                batch_items.append((image, landmark_visible))

                # Perform update
                if len(batch_items) > 32:
                    loss_val = self.do_update(batch_items)
                    if self.tensorboard is not None:
                        self.tensorboard.log_scalar("loss", loss_val)
                    batch_items = []

            # Save the model
            self.model.save_model(experiment_name + "/symbolic_image_detection_resnet_epoch_" + str(epoch))
