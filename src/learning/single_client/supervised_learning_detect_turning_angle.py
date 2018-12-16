import os
import time
import torch
import json
import logging
import scipy.misc
import numpy as np
import torch.optim as optim
import utils.generic_policy as gp
import torch.nn.functional as F

from abstract_learning import AbstractLearning
from utils import nav_drone_symbolic_instructions
from utils.cuda import cuda_var
from utils.nav_drone_landmarks import get_name_of_landmark
from utils.nav_drone_landmarks import get_all_landmark_names


class SupervisedLearningDetectTurningAngle(AbstractLearning):
    """ Perform supervised learning to train a simple classifier on resnet features """

    def __init__(self, model, action_space, meta_data_util, config, constants, tensorboard):
        self.max_epoch = 1500
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
        self.num_landmark = 67
        for i in range(0, self.num_landmark):
            self.confusion_num_count.append([0.0] * self.num_landmark)
            self.confusion_denom_count.append([0.0] * self.num_landmark)

        AbstractLearning.__init__(self, self.model, self.calc_loss,
                                  self.optimizer, self.config, self.constants)

    def calc_loss(self, batch_replay_items):

        images = []
        visible_objects = []
        for replay_item in batch_replay_items:
            image, visible_objects_ = replay_item
            visible_objects.append(visible_objects_)
            images.append([image])

        theta_logits = self.model.get_probs(images)  # batch x 67 x 12
        num_states = int(theta_logits.size()[0])
        one_hot_vector = torch.zeros(theta_logits.size())

        for i in range(0, num_states):
            visible_objects_example = visible_objects[i]
            for landmark in range(0, self.num_landmark):

                # See if the landmark is present and visible in the agent's field of view
                if landmark in visible_objects_example and visible_objects_example[landmark][1] != -1:
                    r, theta = visible_objects_example[landmark]
                    one_hot_vector[i,landmark, theta] = 1.0

        loss = F.binary_cross_entropy_with_logits(theta_logits, cuda_var(one_hot_vector).float())

        return loss

    def calc_loss_v1(self, batch_replay_items):

        images = []
        visible_objects = []
        for replay_item in batch_replay_items:
            image, visible_objects_ = replay_item
            visible_objects.append(visible_objects_)
            images.append([image])

        theta_log_prob = self.model.get_probs(images)
        num_states = int(theta_log_prob.size()[0])

        landmark_objective = None
        distance_objective = None
        theta_objective = None
        num_visible = 0.0
        for i in range(0, num_states):
            visible_objects_example = visible_objects[i]
            for landmark in range(0, self.num_landmark):
                # See if the landmark is present and visible in the agent's field of view
                if landmark in visible_objects_example and visible_objects_example[landmark][1] != -1:
                    r, theta = visible_objects_example[landmark]
                    # if landmark_objective is None:
                    #     landmark_objective = landmark_log_prob[i, landmark, 1]
                    # else:
                    #     landmark_objective = landmark_objective + landmark_log_prob[i, landmark, 1]
                    #
                    # if distance_objective is None:
                    #     distance_objective = distance_log_prob[i, landmark, r]
                    # else:
                    #     distance_objective = distance_objective + distance_log_prob[i, landmark, r]

                    if theta_objective is None:
                        theta_objective = theta_log_prob[i, landmark, theta]
                    else:
                        theta_objective = theta_objective + theta_log_prob[i, landmark, theta]

                    num_visible += 1.0
                # else:
                #     if landmark_objective is None:
                #         landmark_objective = landmark_log_prob[i, landmark, 0]
                #     else:
                #         landmark_objective = landmark_objective + landmark_log_prob[i, landmark, 0]

        if num_visible != 0:
            theta_objective = theta_objective / num_visible
            loss = - theta_objective
        else:
            loss = 0

        return loss

    def calc_loss_old(self, batch_replay_items):

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

        for i in range(0, self.num_landmark):
            for j in range(0, self.num_landmark):
                # If i is present and predicted but j is not present
                if i in gold_set and i in predicted_set and j not in gold_set:
                    self.confusion_denom_count[i][j] += 1
                    if j in predicted_set:
                        self.confusion_num_count[i][j] += 1

    def print_confusion_matrix(self):

        for i in range(0, self.num_landmark):
            normalized_vals = []
            for j in range(0, self.num_landmark):
                normalized_val = self.confusion_num_count[i][j]/max(self.confusion_denom_count[i][j], 1.0)
                if normalized_val > 0.1:
                    normalized_vals.append([(get_name_of_landmark(j), normalized_val)])
            logging.info("Row %r is %r", get_name_of_landmark(i), normalized_vals)

    def test1(self, test_dataset, test_real_dataset):

        num_data_points = 0
        num_used_landmarks = 0
        theta_accuracy = 0
        neighbouring_accuracy = 0
        symbolic_landmark_accuracy = 0

        for data_point_ix, data_point in enumerate(test_dataset):

            image, visible_objects = data_point

            # Compute probabilities over list of visible objects
            log_prob_theta = self.model.get_probs([[image]])
            prob_theta = list(torch.exp(log_prob_theta.data)[0])

            data_point_real = test_real_dataset[data_point_ix]
            gold_landmark, gold_theta_1, gold_theta_2, gold_r = \
                nav_drone_symbolic_instructions.get_nav_drone_symbolic_instruction_segment(data_point_real)
            gold_theta_1_corrected = (gold_theta_1 + 6) %12
            if gold_theta_1_corrected == gp.get_argmax_action(prob_theta[gold_landmark]):
                symbolic_landmark_accuracy += 1

            for i in range(0, self.num_landmark):
                if i in visible_objects:
                    # predicted the distance and angle
                    argmax_theta_val = gp.get_argmax_action(prob_theta[i])
                    gold_angle = visible_objects[i][1]
                    if argmax_theta_val == gold_angle:
                        theta_accuracy += 1
                    angle_diff = min((argmax_theta_val - gold_angle) % 12,
                                     (gold_angle - argmax_theta_val) % 12)
                    if angle_diff <= 1:
                        neighbouring_accuracy += 1
                    num_used_landmarks += 1

            num_data_points += 1

        theta_accuracy /= float(max(num_used_landmarks, 1))
        neighbouring_accuracy /= float(max(num_used_landmarks, 1))
        symbolic_landmark_accuracy /= float(max(len(test_real_dataset), 1))

        logging.info(
            "Num datapoints %r, num visible landmarks %r and mean theta accuracy %r and neigbhouring accuracy %r",
            num_data_points, num_used_landmarks, theta_accuracy, neighbouring_accuracy)
        logging.info("Accuracy for landmark mentioned in the text is %r", symbolic_landmark_accuracy)
        return theta_accuracy

    def test(self, test_dataset):

        num_data_points = 0
        num_used_landmarks = 0
        theta_accuracy = 0
        neighbouring_accuracy = 0

        for data_point_ix, data_point in enumerate(test_dataset):

            image, visible_objects = data_point

            # Compute probabilities over list of visible objects
            log_prob_theta = self.model.get_probs([[image]])

            prob_theta = list(torch.exp(log_prob_theta.data)[0])
            for i in range(0, self.num_landmark):
                if i in visible_objects:
                    # predicted the distance and angle
                    argmax_theta_val = gp.get_argmax_action(prob_theta[i])
                    gold_angle = visible_objects[i][1]
                    if argmax_theta_val == gold_angle:
                        theta_accuracy += 1
                    angle_diff = min((argmax_theta_val - gold_angle) % 12,
                                     (gold_angle - argmax_theta_val) % 12)
                    if angle_diff <= 1:
                        neighbouring_accuracy += 1
                    num_used_landmarks += 1

            num_data_points += 1

        theta_accuracy /= float(max(num_used_landmarks, 1))
        neighbouring_accuracy /= float(max(num_used_landmarks, 1))

        logging.info(
            "Num datapoints %r, num visible landmarks %r and mean theta accuracy %r and neigbhouring accuracy %r",
            num_data_points, num_used_landmarks, theta_accuracy, neighbouring_accuracy)
        return theta_accuracy

    @staticmethod
    def parse(folder_name):

        start = time.time()
        dataset = []
        landmark_names = get_all_landmark_names()
        num_examples = len(os.listdir(folder_name))
        for i in range(0, num_examples):
            example_folder_name = folder_name + "/example_" + str(i)
            images = []
            for j in range(0, 6):
                img = scipy.misc.imread(example_folder_name + "/image_" + str(j) + ".png").swapaxes(1, 2).swapaxes(0, 1)
                images.append(img)

            with open(example_folder_name + "/data.json", 'r') as fp:
                data = json.load(fp)
                new_data = dict()
                for key in data:
                    new_data[landmark_names.index(key)] = SupervisedLearningDetectTurningAngle.fix_angle(data[key])
                image_concatenate = np.concatenate(images, 0)
                dataset.append((image_concatenate, new_data))

        end = time.time()
        logging.info("Parsed dataset of size %r in time % seconds", len(dataset), (end - start))
        return dataset

    @staticmethod
    def fix_angle(data_val):
        r, theta = data_val
        theta_fixed = (theta + 6) % 12
        return (r, theta_fixed)

    def do_train(self, train_dataset, tune_dataset, experiment_name):
        """ Perform training """

        dataset_size = len(train_dataset)

        for epoch in range(1, self.max_epoch + 1):

            logging.info("Starting epoch %d", epoch)

            # Test on tuning data
            self.test(tune_dataset)
            # self.print_confusion_matrix()

            batch_items = []

            for data_point_ix, data_point in enumerate(train_dataset):

                if (data_point_ix + 1) % 10000 == 0:
                    logging.info("Done %d out of %d", data_point_ix, dataset_size)

                image, landmarks = data_point
                batch_items.append((image, landmarks))

                # Perform update
                if len(batch_items) > 32:
                    loss_val = self.do_update(batch_items)
                    if self.tensorboard is not None:
                        self.tensorboard.log_scalar("loss", loss_val)
                    batch_items = []

            # Save the model
            self.model.save_model(experiment_name + "/symbolic_image_detection_resnet_epoch_" + str(epoch))
