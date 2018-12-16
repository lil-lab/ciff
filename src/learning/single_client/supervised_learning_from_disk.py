import torch
import time
import os
import json
import logging
import torch.optim as optim
import numpy as np
import scipy.misc

from agents.agent_observed_state import AgentObservedState
from agents.replay_memory_item import ReplayMemoryItem
from learning.auxiliary_objective.object_detection import ObjectDetection
from learning.auxiliary_objective.symbolic_language_prediction import SymbolicLanguagePrediction
from learning.single_client.abstract_learning import AbstractLearning
from learning.auxiliary_objective.action_prediction import ActionPrediction
from learning.auxiliary_objective.temporal_autoencoder import TemporalAutoEncoder
from utils import nav_drone_symbolic_instructions
from utils.cuda import cuda_var
from models.incremental_model.incremental_model_recurrent_implicit_factorization_resnet import \
    IncrementalModelRecurrentImplicitFactorizationResnet


class SupervisedLearningFromDisk(AbstractLearning):
    """ Perform maximum likelihood on oracle trajectories using images stored on disk
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
        self.entropy_coef = constants["entropy_coefficient"]

        # Auxiliary Objectives
        if self.config["do_action_prediction"]:
            self.action_prediction_loss_calculator = ActionPrediction(self.model)
            self.action_prediction_loss = None
        if self.config["do_temporal_autoencoding"]:
            self.temporal_autoencoder_loss_calculator = TemporalAutoEncoder(self.model)
            self.temporal_autoencoder_loss = None
        if self.config["do_object_detection"]:
            self.object_detection_loss_calculator = ObjectDetection(self.model)
            self.object_detection_loss = None
        if self.config["do_symbolic_language_prediction"]:
            self.symbolic_language_prediction_loss_calculator = SymbolicLanguagePrediction(self.model)
            self.symbolic_language_prediction_loss = None

        self.optimizer = optim.Adam(model.get_parameters(),
                                    lr=constants["learning_rate"])
        AbstractLearning.__init__(self, self.model, self.calc_loss,
                                  self.optimizer, self.config, self.constants, self.tensorboard)

    def calc_loss(self, batch_replay_items):

        agent_observation_state_ls = []
        action_batch = []
        log_probabilities = []
        factor_entropy = []
        for replay_item in batch_replay_items:
            agent_observation_state_ls.append(replay_item.get_agent_observed_state())
            action_batch.append(replay_item.get_action())
            log_probabilities.append(replay_item.get_log_prob())
            factor_entropy.append(replay_item.get_factor_entropy())

        log_probabilities = torch.cat(log_probabilities)
        action_batch = cuda_var(torch.from_numpy(np.array(action_batch)))

        num_states = int(action_batch.size()[0])
        model_log_prob_batch = log_probabilities
        # model_log_prob_batch = self.model.get_probs_batch(agent_observation_state_ls)
        chosen_log_probs = model_log_prob_batch.gather(1, action_batch.view(-1, 1))

        gold_distribution = cuda_var(torch.FloatTensor([0.6719, 0.1457, 0.1435, 0.0387]))
        model_prob_batch = torch.exp(model_log_prob_batch)
        mini_batch_action_distribution = torch.mean(model_prob_batch, 0)

        self.cross_entropy = -torch.sum(gold_distribution * torch.log(mini_batch_action_distribution))
        self.entropy = -torch.mean(torch.sum(model_log_prob_batch * model_prob_batch, 1))
        objective = torch.sum(chosen_log_probs) / num_states
        # Essentially we want the objective to increase and cross entropy to decrease
        loss = -objective # - self.entropy_coef * self.entropy
        # loss = -objective + self.entropy_coef * self.cross_entropy

        # Minimize the Factor Entropy if the model is implicit factorization model
        if isinstance(self.model, IncrementalModelRecurrentImplicitFactorizationResnet):
            self.mean_factor_entropy = torch.mean(torch.cat(factor_entropy))
            loss = loss + self.mean_factor_entropy
        else:
            self.mean_factor_entropy = None

        if self.config["do_action_prediction"]:
            self.action_prediction_loss = self.action_prediction_loss_calculator.calc_loss(batch_replay_items)
            if self.action_prediction_loss is not None:
                self.action_prediction_loss = self.constants["action_prediction_coeff"] * self.action_prediction_loss
                loss = loss + self.action_prediction_loss
        else:
            self.action_prediction_loss = None

        if self.config["do_temporal_autoencoding"]:
            self.temporal_autoencoder_loss = self.temporal_autoencoder_loss_calculator.calc_loss(batch_replay_items)
            if self.temporal_autoencoder_loss is not None:
                self.temporal_autoencoder_loss = \
                    self.constants["temporal_autoencoder_coeff"] * self.temporal_autoencoder_loss
                loss = loss + self.temporal_autoencoder_loss
        else:
            self.temporal_autoencoder_loss = None

        if self.config["do_object_detection"]:
            self.object_detection_loss = self.object_detection_loss_calculator.calc_loss(batch_replay_items)
            self.object_detection_loss = self.constants["object_detection_coeff"] * self.object_detection_loss
            loss = loss + self.object_detection_loss
        else:
            self.object_detection_loss = None

        if self.config["do_symbolic_language_prediction"]:
            self.symbolic_language_prediction_loss = \
                self.symbolic_language_prediction_loss_calculator.calc_loss(batch_replay_items)
            self.symbolic_language_prediction_loss = self.constants["symbolic_language_prediction_coeff"] * \
                                                     self.symbolic_language_prediction_loss
            loss = loss + self.symbolic_language_prediction_loss
        else:
            self.symbolic_language_prediction_loss = None

        return loss

    @staticmethod
    def parse(folder_name):

        start = time.time()
        dataset = []
        num_examples = len(os.listdir(folder_name))
        for i in range(0, num_examples):
            example_folder_name = folder_name + "/example_" + str(i)
            image_names = [file for file in os.listdir(example_folder_name) if file.endswith('.png')]
            num_actions = len(image_names)
            images = []
            for j in range(0, num_actions):
                img = scipy.misc.imread(example_folder_name + "/image_" + str(j) + ".png").swapaxes(1, 2).swapaxes(0, 1)
                images.append(img)
            dataset.append(images)
        end = time.time()
        logging.info("Parsed dataset of size %r in time % seconds", len(dataset), (end - start))
        return dataset

    def calc_log_prob(self, tune_dataset, tune_image, tensorboard):

        total_validation_log_probability = 0
        for data_point_ix, data_point in enumerate(tune_dataset):
            tune_image_example = tune_image[data_point_ix]
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
            trajectory = data_point.get_trajectory()

            validation_log_probability = 0

            for action_ix, action in enumerate(trajectory):
                log_probabilities, model_state, image_emb_seq = self.model.get_probs(state, model_state)
                validation_log_probability += float(log_probabilities.data[0][action])
                image = tune_image_example[action_ix + 1]
                state = state.update(image, action, pose=None,  position_orientation=None, data_point=data_point)

            log_probabilities, model_state, image_emb_seq = self.model.get_probs(state, model_state)
            validation_log_probability += float(log_probabilities.data[0][self.action_space.get_stop_action_index()])
            mean_validation_log_probability = validation_log_probability/float(len(trajectory) + 1)
            tensorboard.log_scalar("Validation_Log_Prob",  mean_validation_log_probability)
            total_validation_log_probability += mean_validation_log_probability
        total_validation_log_probability /= float(max(len(tune_dataset), 1))
        logging.info("Mean Validation Log Prob is %r", total_validation_log_probability)

    def do_train(self, train_dataset, train_images, tune_dataset, tune_images, experiment_name):
        """ Perform training """

        dataset_size = len(train_dataset)

        for epoch in range(1, self.max_epoch + 1):

            logging.info("Starting epoch %d", epoch)

            # Test on tuning data
            self.calc_log_prob(tune_dataset, tune_images, tensorboard=self.tensorboard)

            batch_replay_items = []
            episodes_in_batch = 0

            for data_point_ix, data_point in enumerate(train_dataset):

                if (data_point_ix + 1) % 100 == 0:
                    logging.info("Done %d out of %d", data_point_ix, dataset_size)

                train_images_example = train_images[data_point_ix]
                image = train_images_example[0]
                symbolic_form = nav_drone_symbolic_instructions.get_nav_drone_symbolic_instruction_segment(data_point)

                model_state = None
                state = AgentObservedState(instruction=data_point.instruction,
                                           config=self.config,
                                           constants=self.constants,
                                           start_image=image,
                                           previous_action=None,
                                           pose=None,
                                           position_orientation=None,
                                           data_point=data_point)

                trajectory = data_point.get_trajectory()
                for action_ix, action in enumerate(trajectory):

                    # Sample action using the policy
                    # Generate probabilities over actions
                    log_probabilities, model_state, image_emb_seq = self.model.get_probs(state, model_state)

                    # Send the action and get feedback
                    image = train_images_example[action_ix + 1]

                    # Store it in the replay memory list
                    replay_item = ReplayMemoryItem(
                        state, action, 0, log_prob=log_probabilities, symbolic_text=symbolic_form,
                        image_emb_seq=image_emb_seq, text_emb=model_state[0])
                    batch_replay_items.append(replay_item)

                    # Update the agent state
                    state = state.update(
                        image, action, pose=None,
                        position_orientation=None,
                        data_point=data_point)

                log_probabilities, model_state, image_emb_seq = self.model.get_probs(state, model_state)

                # Store it in the replay memory list
                replay_item = ReplayMemoryItem(
                    state, self.action_space.get_stop_action_index(), 0, log_prob=log_probabilities,
                    symbolic_text=symbolic_form, image_emb_seq=image_emb_seq, text_emb=model_state[0])
                batch_replay_items.append(replay_item)

                # Perform update
                episodes_in_batch += 1
                if episodes_in_batch == 1:
                    episodes_in_batch = 0
                    loss_val = self.do_update(batch_replay_items)
                    del batch_replay_items[:]  # in place list clear
                    self.tensorboard.log_scalar("loss", loss_val)
                    cross_entropy = float(self.cross_entropy.data[0])
                    self.tensorboard.log_scalar("cross_entropy", cross_entropy)
                    entropy = float(self.entropy.data[0])
                    self.tensorboard.log_scalar("entropy", entropy)
                    if self.action_prediction_loss is not None:
                        action_prediction_loss = float(self.action_prediction_loss.data[0])
                        self.tensorboard.log_action_prediction_loss(action_prediction_loss)
                    if self.temporal_autoencoder_loss is not None:
                        temporal_autoencoder_loss = float(self.temporal_autoencoder_loss.data[0])
                        self.tensorboard.log_temporal_autoencoder_loss(temporal_autoencoder_loss)
                    if self.object_detection_loss is not None:
                        object_detection_loss = float(self.object_detection_loss.data[0])
                        self.tensorboard.log_object_detection_loss(object_detection_loss)
                    if self.symbolic_language_prediction_loss is not None:
                        symbolic_language_prediction_loss = float(self.symbolic_language_prediction_loss.data[0])
                        self.tensorboard.log_scalar("sym_language_prediction_loss", symbolic_language_prediction_loss)
                    if self.mean_factor_entropy is not None:
                        mean_factor_entropy = float(self.mean_factor_entropy.data[0])
                        self.tensorboard.log_factor_entropy_loss(mean_factor_entropy)

            # Save the model
            self.model.save_model(experiment_name + "/contextual_bandit_resnet_epoch_" + str(epoch))
