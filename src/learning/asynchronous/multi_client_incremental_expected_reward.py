import sys
import traceback

import torch
import logging
import torch.optim as optim
import utils.generic_policy as gp
import numpy as np

from agents.agent import Agent
from agents.agent_observed_state import AgentObservedState
from agents.replay_memory_item import ReplayMemoryItem
from learning.auxiliary_objective.goal_prediction import GoalPrediction
from learning.auxiliary_objective.object_detection import ObjectDetection
from learning.auxiliary_objective.symbolic_language_prediction import SymbolicLanguagePrediction
from learning.asynchronous.abstract_learning import AbstractLearning
from learning.auxiliary_objective.action_prediction import ActionPrediction
from learning.auxiliary_objective.temporal_autoencoder import TemporalAutoEncoder
from utils.cuda import cuda_var
from models.incremental_model.incremental_model_recurrent_implicit_factorization_resnet import \
    IncrementalModelRecurrentImplicitFactorizationResnet
from utils.launch_unity import launch_k_unity_builds
from utils.pushover_logger import PushoverLogger
from utils.tensorboard import Tensorboard


class AsynchronousContextualBandit(AbstractLearning):
    """ Perform expected reward maximization """

    def __init__(self, shared_model, local_model, action_space, meta_data_util, config, constants, tensorboard):
        self.max_epoch = constants["max_epochs"]
        self.shared_model = shared_model
        self.local_model = local_model
        self.action_space = action_space
        self.meta_data_util = meta_data_util
        self.config = config
        self.constants = constants
        self.num_client = config["num_client"]
        self.tensorboard = tensorboard
        self.entropy = None
        self.cross_entropy = None
        self.entropy_coef = constants["entropy_coefficient"]

        # Auxiliary Objectives
        if self.config["do_action_prediction"]:
            self.action_prediction_loss_calculator = ActionPrediction(self.local_model)
            self.action_prediction_loss = None
        if self.config["do_temporal_autoencoding"]:
            self.temporal_autoencoder_loss_calculator = TemporalAutoEncoder(self.local_model)
            self.temporal_autoencoder_loss = None
        if self.config["do_object_detection"]:
            self.object_detection_loss_calculator = ObjectDetection(self.local_model, num_objects=67)
            self.object_detection_loss = None
        if self.config["do_symbolic_language_prediction"]:
            self.symbolic_language_prediction_loss_calculator = SymbolicLanguagePrediction(self.local_model)
            self.symbolic_language_prediction_loss = None
        if self.config["do_goal_prediction"]:
            self.goal_prediction_calculator = GoalPrediction(self.local_model)
            self.goal_prediction_loss = None

        self.optimizer = optim.Adam(shared_model.get_parameters(),
                                    lr=constants["learning_rate"])
        AbstractLearning.__init__(self, self.shared_model, self.local_model, self.calc_loss,
                                  self.optimizer, self.config, self.constants, self.tensorboard)

    def calc_loss(self, batch_replay_items):
        agent_observation_state_ls = []
        all_rewards = []
        log_probabilities = []
        factor_entropy = []
        for replay_item in batch_replay_items:
            agent_observation_state_ls.append(replay_item.get_agent_observed_state())
            log_probabilities.append(replay_item.get_log_prob())
            factor_entropy.append(replay_item.get_factor_entropy())
            all_rewards.append(replay_item.get_all_rewards())

        all_rewards = cuda_var(torch.from_numpy(np.array(all_rewards)).float())  # batch x action
        log_probabilities = torch.cat(log_probabilities)

        num_states = int(all_rewards.size()[0])
        model_log_prob_batch = log_probabilities
        model_prob_batch = torch.exp(model_log_prob_batch)
        reward_probs = all_rewards * model_prob_batch
        objective = torch.sum(reward_probs) / num_states

        gold_distribution = cuda_var(torch.FloatTensor([0.6719, 0.1457, 0.1435, 0.0387]))
        mini_batch_action_distribution = torch.mean(model_prob_batch, 0)
        self.cross_entropy = -torch.sum(gold_distribution * torch.log(mini_batch_action_distribution))
        self.entropy = -torch.mean(torch.sum(model_log_prob_batch * model_prob_batch, 1))

        # Essentially we want the objective to increase and cross entropy to decrease
        loss = -objective + self.entropy_coef * self.cross_entropy

        # Minimize the Factor Entropy if the model is implicit factorization model
        if isinstance(self.local_model, IncrementalModelRecurrentImplicitFactorizationResnet):
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

        return loss

    def get_all_rewards(self, metadata):
        rewards = []
        for i in range(0, self.config["num_actions"]):
            reward = metadata["reward_dict"][self.action_space.get_action_name(i)]
            rewards.append(reward)
        return rewards

    @staticmethod
    def do_train(shared_model, config, action_space, meta_data_util,
                 args, constants, train_dataset, tune_dataset, experiment,
                 experiment_name, rank, server, logger, model_type, use_pushover=False):
        try:
            AsynchronousContextualBandit.do_train_(shared_model, config, action_space, meta_data_util,
                                                   args, constants, train_dataset, tune_dataset, experiment,
                                                   experiment_name, rank, server, logger, model_type, use_pushover)
        except Exception:
            exc_info = sys.exc_info()
            traceback.print_exception(*exc_info)

    @staticmethod
    def do_train_(shared_model, config, action_space, meta_data_util, args, constants,
                  train_dataset, tune_dataset, experiment, experiment_name, rank, server,
                  logger, model_type, use_pushover=False):

        server.initialize_server()

        # Test policy
        test_policy = gp.get_argmax_action

        # torch.manual_seed(args.seed + rank)

        if rank == 0:  # client 0 creates a tensorboard server
            tensorboard = Tensorboard(experiment_name)
        else:
            tensorboard = None

        if use_pushover:
            pushover_logger = PushoverLogger(experiment_name)
        else:
            pushover_logger = None

        # Create a local model for rollouts
        local_model = model_type(args, config=config)
        if torch.cuda.is_available():
            local_model.cuda()
        local_model.train()

        # Create the Agent
        logger.log("STARTING AGENT")
        agent = Agent(server=server,
                      model=local_model,
                      test_policy=test_policy,
                      action_space=action_space,
                      meta_data_util=meta_data_util,
                      config=config,
                      constants=constants)
        logger.log("Created Agent...")

        action_counts = [0] * action_space.num_actions()
        max_epochs = constants["max_epochs"]
        dataset_size = len(train_dataset)
        tune_dataset_size = len(tune_dataset)

        # Create the learner to compute the loss
        learner = AsynchronousContextualBandit(shared_model, local_model, action_space, meta_data_util,
                                               config, constants, tensorboard)

        # Launch unity
        launch_k_unity_builds([config["port"]], "/home/dipendra/Downloads/NavDroneLinuxBuild/NavDroneLinuxBuild.x86_64")

        for epoch in range(1, max_epochs + 1):

            if tune_dataset_size > 0:
                # Test on tuning data
                agent.test(tune_dataset, tensorboard=tensorboard,
                           logger=logger, pushover_logger=pushover_logger)

            for data_point_ix, data_point in enumerate(train_dataset):

                # Sync with the shared model
                # local_model.load_state_dict(shared_model.state_dict())
                local_model.load_from_state_dict(shared_model.get_state_dict())

                if (data_point_ix + 1) % 100 == 0:
                    logging.info("Done %d out of %d", data_point_ix, dataset_size)
                    logging.info("Training data action counts %r", action_counts)

                num_actions = 0
                # max_num_actions = len(data_point.get_trajectory())
                # max_num_actions += self.constants["max_extra_horizon"]
                max_num_actions = constants["horizon"]

                image, metadata = agent.server.reset_receive_feedback(data_point)

                pose = int(metadata["y_angle"] / 15.0)
                position_orientation = (metadata["x_pos"], metadata["z_pos"],
                                        metadata["y_angle"])
                state = AgentObservedState(instruction=data_point.instruction,
                                           config=config,
                                           constants=constants,
                                           start_image=image,
                                           previous_action=None,
                                           pose=pose,
                                           position_orientation=position_orientation,
                                           data_point=data_point)

                model_state = None
                batch_replay_items = []
                total_reward = 0
                forced_stop = True

                while num_actions < max_num_actions:

                    # Sample action using the policy
                    log_probabilities, model_state, image_emb_seq, state_feature = \
                        local_model.get_probs(state, model_state)
                    probabilities = list(torch.exp(log_probabilities.data))[0]

                    # Sample action from the probability
                    action = gp.sample_action_from_prob(probabilities)
                    action_counts[action] += 1

                    if action == action_space.get_stop_action_index():
                        forced_stop = False
                        break

                    # Send the action and get feedback
                    image, reward, metadata = agent.server.send_action_receive_feedback(action)

                    # Store it in the replay memory list
                    rewards = learner.get_all_rewards(metadata)
                    replay_item = ReplayMemoryItem(state, action, reward,
                                                   log_prob=log_probabilities, all_rewards=rewards)
                    batch_replay_items.append(replay_item)

                    # Update the agent state
                    pose = int(metadata["y_angle"] / 15.0)
                    position_orientation = (metadata["x_pos"],
                                            metadata["z_pos"],
                                            metadata["y_angle"])
                    state = state.update(
                        image, action, pose=pose,
                        position_orientation=position_orientation,
                        data_point=data_point)

                    num_actions += 1
                    total_reward += reward

                # Send final STOP action and get feedback
                image, reward, metadata = agent.server.halt_and_receive_feedback()
                rewards = learner.get_all_rewards(metadata)
                total_reward += reward

                if tensorboard is not None:
                    tensorboard.log_all_train_errors(
                        metadata["edit_dist_error"], metadata["closest_dist_error"], metadata["stop_dist_error"])

                # Store it in the replay memory list
                if not forced_stop:
                    replay_item = ReplayMemoryItem(state, action_space.get_stop_action_index(),
                                                   reward, log_prob=log_probabilities, all_rewards=rewards)
                    batch_replay_items.append(replay_item)

                # Update the scores based on meta_data
                # self.meta_data_util.log_results(metadata)

                # Perform update
                if len(batch_replay_items) > 0:
                    loss_val = learner.do_update(batch_replay_items)
                    # self.action_prediction_loss_calculator.predict_action(batch_replay_items)
                    del batch_replay_items[:]  # in place list clear

                    if tensorboard is not None:
                        cross_entropy = float(learner.cross_entropy.data[0])
                        tensorboard.log(cross_entropy, loss_val, 0)
                        entropy = float(learner.entropy.data[0])
                        tensorboard.log_scalar("entropy", entropy)

                        ratio = float(learner.ratio.data[0])
                        tensorboard.log_scalar("Abs_objective_to_entropy_ratio", ratio)

                        if learner.action_prediction_loss is not None:
                            action_prediction_loss = float(learner.action_prediction_loss.data[0])
                            learner.tensorboard.log_action_prediction_loss(action_prediction_loss)
                        if learner.temporal_autoencoder_loss is not None:
                            temporal_autoencoder_loss = float(learner.temporal_autoencoder_loss.data[0])
                            tensorboard.log_temporal_autoencoder_loss(temporal_autoencoder_loss)
                        if learner.object_detection_loss is not None:
                            object_detection_loss = float(learner.object_detection_loss.data[0])
                            tensorboard.log_object_detection_loss(object_detection_loss)
                        if learner.symbolic_language_prediction_loss is not None:
                            symbolic_language_prediction_loss = float(learner.symbolic_language_prediction_loss.data[0])
                            tensorboard.log_scalar("sym_language_prediction_loss", symbolic_language_prediction_loss)
                        if learner.goal_prediction_loss is not None:
                            goal_prediction_loss = float(learner.goal_prediction_loss.data[0])
                            tensorboard.log_scalar("goal_prediction_loss", goal_prediction_loss)
                        if learner.mean_factor_entropy is not None:
                            mean_factor_entropy = float(learner.mean_factor_entropy.data[0])
                            tensorboard.log_factor_entropy_loss(mean_factor_entropy)

            # Save the model
            local_model.save_model(experiment + "/contextual_bandit_" + str(rank) + "_epoch_" + str(epoch))

            logging.info("Training data action counts %r", action_counts)
