import os
import time
import torch
import json
import logging
import math
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


class SupervisedLearningDetectLandmarks(AbstractLearning):
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

        theta_logits = self.model.get_probs(images)  # batch x 67
        num_states = int(theta_logits.size()[0])
        one_hot_vector = torch.zeros(theta_logits.size())

        for i in range(0, num_states):
            visible_objects_example = visible_objects[i]
            for landmark in range(0, self.num_landmark):

                # See if the landmark is present and visible in the agent's field of view
                if landmark in visible_objects_example and visible_objects_example[landmark][1] != -1:
                    one_hot_vector[i, landmark] = 1.0

        loss = F.binary_cross_entropy_with_logits(theta_logits, cuda_var(one_hot_vector).float())

        return loss

    def test(self, test_dataset):

        num_data_points = 0

        precision_num, precision_denom = 0, 0
        recall_num, recall_denom = 0, 0

        for data_point_ix, data_point in enumerate(test_dataset):

            image, visible_objects = data_point

            # Compute probabilities over list of visible objects
            logits = self.model.get_probs([[image]])  # batch x 67
            logits = list(logits[0])
            for i in range(0, self.num_landmark):

                prob = 1/(1 + math.exp(-logits[i].data.cpu().numpy()[0]))
                present = prob > 0.5
                if present and i in visible_objects:
                    precision_num += 1
                    recall_num += 1
                if present:
                    precision_denom += 1
                if i in visible_objects:
                    recall_denom += 1

            num_data_points += 1

        precision = precision_num/float(precision_denom)
        recall = recall_num/float(recall_denom)
        f1 = (2 * precision * recall)/float(precision + recall)

        logging.info(
            "Num datapoints %r, precision %r, recall %r, f1-score %r",
            num_data_points, precision, recall, f1)
        return f1

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
                    new_data[landmark_names.index(key)] = data[key]
                image_concatenate = np.concatenate(images, 0)
                dataset.append((image_concatenate, new_data))

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
