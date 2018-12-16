import torch
import time
import os
import logging
import torch.optim as optim
import torch.nn as nn
import numpy as np
import scipy.misc
import torch.nn.functional as F

from learning.auxiliary_objective.object_pixel_identification import ObjectPixelIdentification
from learning.single_client.abstract_learning import AbstractLearning
from models.model.abstract_model import AbstractModel
from models.module.chaplot_text_module import ChaplotTextModule
from models.module.pixel_identification_module import PixelIdentificationModule
from utils.cuda import cuda_var
from utils.geometry import get_turn_angle_from_metadata_datapoint
from utils.nav_drone_landmarks import get_all_landmark_names


class FinalModule(torch.nn.Module):

    def __init__(self, text_module):
        super(FinalModule, self).__init__()

        self.input_dims = (3, 128, 128)
        self.conv1 = nn.Conv2d(3, 128, kernel_size=8, stride=4, padding=3)
        self.conv2 = nn.Conv2d(128, 64, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 32, kernel_size=3, stride=2, padding=1)

        self.text_module = text_module
        self.layer1 = nn.Linear(448, 128)
        self.layer2 = nn.Linear(128, 6)

    def forward(self, images, instruction):

        x = images.view(6, 3, 128, 128)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(6, 32, -1)
        x = x.mean(2)

        image_embedding = x.view(1, -1)
        _, text_embedding = self.text_module(instruction)

        x = torch.cat([image_embedding, text_embedding], dim=1)
        x = F.relu(self.layer1(x))
        log_prob = F.log_softmax(self.layer2(x))
        return log_prob


class ImagePredictionModel(AbstractModel):

    def __init__(self, config, constants):

        AbstractModel.__init__(self, config, constants)
        self.none_action = config["num_actions"]

        num_channels, image_height, image_width = 3, 3, 3

        self.text_module = ChaplotTextModule(
            emb_dim=32,
            hidden_dim=256,
            vocab_size=config["vocab_size"],
            image_height=image_height, image_width=image_width)

        self.final_module = FinalModule(self.text_module)

        if False: # config["do_object_detection"]:
            self.landmark_names = get_all_landmark_names()
            self.object_detection_module = PixelIdentificationModule(
                num_channels=num_channels, num_objects=67)
        else:
            self.object_detection_module = None

        if torch.cuda.is_available():
            self.text_module.cuda()
            self.final_module.cuda()
            if self.object_detection_module is not None:
                self.object_detection_module.cuda()

    def get_log_prob(self, replay_item):

        image_batch, instruction = replay_item
        image_seqs = [image_batch]
        image_batch = cuda_var(torch.from_numpy(np.array(image_seqs)).float())

        instructions = [instruction]
        instructions_batch = cuda_var(torch.from_numpy(np.array(instructions)).long())

        return self.final_module(image_batch, instructions_batch)

    def get_state_dict(self):
        nested_state_dict = dict()
        nested_state_dict["text_module"] = self.text_module.state_dict()
        nested_state_dict["final_module"] = self.final_module.state_dict()
        if self.object_detection_module is not None:
            nested_state_dict["od_module"] = self.object_detection_module.state_dict()
        return nested_state_dict

    def load_from_state_dict(self, nested_state_dict):
        self.text_module.load_state_dict(nested_state_dict["text_module"])
        self.final_module.load_state_dict(nested_state_dict["final_module"])
        if self.object_detection_module is not None:
            self.object_detection_module.load_state_dict(nested_state_dict["od_module"])

    def load_saved_model(self, load_dir):
        if torch.cuda.is_available():
            torch_load = torch.load
        else:
            torch_load = lambda f_: torch.load(f_, map_location=lambda s_, l_: s_)
        text_module_path = os.path.join(load_dir, "text_module_state.bin")
        self.text_module.load_state_dict(torch_load(text_module_path))
        final_module_path = os.path.join(load_dir, "final_module_state.bin")
        self.final_module.load_state_dict(torch_load(final_module_path), strict=False)
        if self.object_detection_module is not None:
            auxiliary_object_detection_path = os.path.join(load_dir, "auxiliary_object_detection.bin")
            self.object_detection_module.load_state_dict(torch_load(auxiliary_object_detection_path))

    def save_model(self, save_dir):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # save state file for image nn
        text_module_path = os.path.join(save_dir, "text_module_state.bin")
        torch.save(self.text_module.state_dict(), text_module_path)
        final_module_path = os.path.join(save_dir, "final_module_state.bin")
        torch.save(self.final_module.state_dict(), final_module_path)
        # save the auxiliary models
        if self.object_detection_module is not None:
            auxiliary_object_detection_path = os.path.join(save_dir, "auxiliary_object_detection.bin")
            torch.save(self.object_detection_module.state_dict(), auxiliary_object_detection_path)

    def get_parameters(self):
        parameters = list(self.final_module.parameters())
        if self.object_detection_module is not None:
            parameters += list(self.object_detection_module.parameters())
        return parameters

    def get_named_parameters(self):
        named_parameters = list(self.final_module.named_parameters())
        if self.object_detection_module is not None:
            named_parameters += list(self.object_detection_module.named_parameters())
        return named_parameters


class ImagePredictionLearning(AbstractLearning):
    """ Perform goal prediction on single images (as opposed to doing it for sequence)
    stored on disk and hence does not need client or server. """

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

        self.vocab = {}
        vocab_path = config["vocab_file"]
        word_index = 0
        with open(vocab_path) as f:
            for line in f.readlines():
                token = line.strip()
                self.vocab[token] = word_index
                word_index += 1

        # Auxiliary Objectives
        if False: # self.config["do_object_detection"]:
            self.object_detection_loss_calculator = ObjectPixelIdentification(
                self.model, num_objects=67, camera_angle=60, image_height=self.final_height,
                image_width=self.final_width, object_height=0)  # -2.5)
            self.object_detection_loss = None

        self.cross_entropy_loss = None
        self.dist_loss = None

        self.optimizer = optim.Adam(model.get_parameters(),
                                    lr=constants["learning_rate"])
        AbstractLearning.__init__(self, self.model, self.calc_loss, self.optimizer,
                                  self.config, self.constants, self.tensorboard)

    def calc_loss(self, batch_replay_items):

        # Only compute the goal prediction loss

        loss = None
        for replay_item in batch_replay_items:
            image, instruction, gold_image_ix = replay_item
            log_prob = self.model.get_log_prob((image, instruction))
            loss_ = -log_prob[0, gold_image_ix]
            if loss is None:
                loss = loss_
            else:
                loss = loss - loss_

        loss = loss / len(batch_replay_items)

        if False:  # self.config["do_object_detection"]:
            self.object_detection_loss = self.object_detection_loss_calculator.calc_loss(batch_replay_items)
            if self.object_detection_loss is not None:
                self.object_detection_loss = self.constants["object_detection_coeff"] * self.object_detection_loss
                loss = loss + self.object_detection_loss
        else:
            self.object_detection_loss = None


        return loss

    @staticmethod
    def get_gold_image(data_point):

        pos = data_point.get_start_pos()
        metadata = {"x_pos": pos[0], "z_pos": pos[1], "y_angle": pos[2]}
        turn_angle = get_turn_angle_from_metadata_datapoint(metadata, data_point)

        assert 180.0 >= turn_angle >= -180.0
        if 30.0 >= turn_angle > -30.0:
            ix = 0
        elif 90.0 >= turn_angle > 30.0:
            ix = 1
        elif 150.0 >= turn_angle > 90.0:
            ix = 2
        elif -30 >= turn_angle > -90.0:
            ix = 5
        elif -90.0 >= turn_angle > -150.0:
            ix = 4
        else:
            ix = 3

        return ix

    @staticmethod
    def parse(folder_name, dataset, model):

        start = time.time()

        # Read images
        image_dataset = []
        num_examples = len(os.listdir(folder_name))

        # Read images
        for i in range(0, num_examples):
            example_folder_name = folder_name + "/example_" + str(i)
            images = []
            for ix in range(0, 6):  # panaroma consists of 6 images stitched together
                img = scipy.misc.imread(example_folder_name + "/image_" + str(ix) + ".png").swapaxes(1, 2).swapaxes(0, 1)
                images.append(img)
            image_dataset.append(images)

        # Read the goal state. The data for the single image can be
        # directly computed and does not need to be saved.
        image_index_dataset = []
        for data_point in dataset:
            ix = ImagePredictionLearning.get_gold_image(data_point)
            image_index_dataset.append(ix)

        assert len(image_dataset) == len(dataset) and len(image_index_dataset) == len(dataset)

        end = time.time()
        logging.info("Parsed dataset of size %r in time % seconds", len(image_dataset), (end - start))

        return image_dataset, image_index_dataset

    def test(self, tune_dataset, tune_image, tune_goal_location, tensorboard):

        total_validation_loss = 0
        total_validation_exact_accuracy = 0
        total_epsilon_accuracy = 0

        for data_point_ix, data_point in enumerate(tune_dataset):
            tune_image_example = tune_image[data_point_ix]
            tune_image_index = tune_goal_location[data_point_ix]

            log_prob = self.model.get_log_prob((tune_image_example, data_point.instruction))

            loss = -log_prob[0, tune_image_index]
            inferred_ix = int(torch.max(log_prob, 1)[1].data.cpu().numpy()[0])

            if tune_image_index == inferred_ix:
                total_validation_exact_accuracy += 1
            if min((tune_image_index - inferred_ix) % 6, (inferred_ix - tune_image_index) % 6) <= 1.0:
                total_epsilon_accuracy += 1

            total_validation_loss += loss

        num_items = len(tune_dataset)
        mean_total_validation_loss = total_validation_loss / float(max(num_items, 1))
        mean_total_validation_accuracy = (total_validation_exact_accuracy * 100.0) / float(max(num_items, 1))
        mean_total_epsilon_accuracy = (total_epsilon_accuracy * 100.0) / float(max(num_items, 1))

        logging.info("Mean Test result: Num items %r, Loss %r, Acc is %r, Epsilon Accuracy is %r"
                     % (num_items, mean_total_validation_loss,
                        mean_total_validation_accuracy, mean_total_epsilon_accuracy))

    def do_train(self, train_dataset, train_images, train_image_indices,
                 tune_dataset, tune_images, tune_goal_location, experiment_name):
        """ Perform training """

        dataset_size = len(train_dataset)
        tensorboard = self.tensorboard

        for epoch in range(1, self.max_epoch + 1):

            logging.info("Starting epoch %d", epoch)

            # Test on tuning data
            self.test(tune_dataset, tune_images, tune_goal_location, tensorboard=tensorboard)
            batch_replay_items = []

            for data_point_ix, data_point in enumerate(train_dataset):

                if (data_point_ix + 1) % 100 == 0:
                    logging.info("Done %d out of %d", data_point_ix, dataset_size)

                # Store it in the replay memory list
                replay_item = (train_images[data_point_ix], data_point.instruction, train_image_indices[data_point_ix])
                batch_replay_items.append(replay_item)

                # Perform update
                if len(batch_replay_items) > 0:
                    loss_val = self.do_update(batch_replay_items)
                    batch_replay_items = []
                    if tensorboard is not None:
                        tensorboard.log_scalar("Loss", loss_val)
                        if self.object_detection_loss is not None:
                            object_detection_loss = float(self.object_detection_loss.data[0])
                            tensorboard.log_scalar("object_detection_loss", object_detection_loss)

            # Save the model
            self.model.save_model(experiment_name + "/goal_prediction_single_supervised_epoch_" + str(epoch))
