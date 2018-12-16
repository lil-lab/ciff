import sys
import traceback

import time
import torch
import torch.optim as optim
import utils.generic_policy as gp
import numpy as np
import random

from agents.tmp_streetview_agent import Agent
from agents.agent_observed_state import AgentObservedState
from agents.replay_memory_item import ReplayMemoryItem
from learning.auxiliary_objective.goal_prediction import GoalPrediction
from learning.auxiliary_objective.object_detection import ObjectDetection
from learning.auxiliary_objective.object_pixel_identification import ObjectPixelIdentification
from learning.auxiliary_objective.symbolic_language_prediction import SymbolicLanguagePrediction
from learning.asynchronous.abstract_learning import AbstractLearning
from learning.auxiliary_objective.action_prediction import ActionPrediction
from learning.auxiliary_objective.temporal_autoencoder import TemporalAutoEncoder
from utils.cuda import cuda_var
from utils.launch_unity import launch_k_unity_builds
from utils.pushover_logger import PushoverLogger
from utils.tensorboard import Tensorboard


class TmpStreetViewAsynchronousSupervisedLearning(AbstractLearning):
    """ Temp file with modification for streetview corpus.
    Perform supervised learning (Kakade and Langford (circa 2006) & Misra, Langford and Artzi EMNLP 2017) """

    def __init__(self, shared_model, local_model, action_space, meta_data_util, config, constants, tensorboard):
        self.max_epoch = constants["max_epochs"]
        self.shared_model = shared_model
        self.local_model = local_model
        self.action_space = action_space
        self.meta_data_util = meta_data_util
        self.config = config
        self.constants = constants
        self.tensorboard = tensorboard
        self.entropy = None
        self.cross_entropy = None
        self.ratio = None
        self.epoch = 0
        self.entropy_coef = constants["entropy_coefficient"]

        self.optimizer = optim.Adam(shared_model.get_parameters(),
                                    lr=constants["learning_rate"])
        AbstractLearning.__init__(self, self.shared_model, self.local_model, self.calc_loss,
                                  self.optimizer, self.config, self.constants, self.tensorboard)

    def calc_loss(self, batch_replay_items):

        agent_observation_state_ls = []
        action_batch = []
        log_probabilities = []
        for replay_item in batch_replay_items:
            agent_observation_state_ls.append(replay_item.get_agent_observed_state())
            action_batch.append(replay_item.get_action())
            log_probabilities.append(replay_item.get_log_prob())

        log_probabilities = torch.cat(log_probabilities)
        action_batch = cuda_var(torch.from_numpy(np.array(action_batch)))

        num_states = int(action_batch.size()[0])
        model_log_prob_batch = log_probabilities
        # model_log_prob_batch = self.model.get_probs_batch(agent_observation_state_ls)
        chosen_log_probs = model_log_prob_batch.gather(1, action_batch.view(-1, 1)).view(-1)

        model_prob_batch = torch.exp(model_log_prob_batch)
        self.entropy = -torch.sum(torch.sum(model_log_prob_batch * model_prob_batch, 1))
        objective = torch.sum(chosen_log_probs)

        loss = -objective

        return loss

    @staticmethod
    def do_train(shared_model, config, action_space, meta_data_util,
                 constants, train_dataset, tune_dataset, experiment,
                 experiment_name, rank, server, logger, model_type, use_pushover=False):
        try:
            TmpStreetViewAsynchronousSupervisedLearning.do_train_(shared_model, config, action_space, meta_data_util,
                                                                  constants, train_dataset, tune_dataset, experiment,
                                                                  experiment_name, rank, server, logger, model_type,
                                                                  use_pushover)
        except Exception:
            exc_info = sys.exc_info()
            traceback.print_exception(*exc_info)

    @staticmethod
    def do_train_(shared_model, config, action_space, meta_data_util, constants,
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
        local_model = model_type(config, constants)
        # local_model.train()

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
        learner = TmpStreetViewAsynchronousSupervisedLearning(shared_model, local_model, action_space, meta_data_util,
                                                              config, constants, tensorboard)

        for epoch in range(1, max_epochs + 1):

            learner.epoch = epoch
            task_completion_accuracy = 0
            mean_stop_dist_error = 0

            time_taken = dict()
            time_taken["prob_time"] = 0.0
            time_taken["update_time"] = 0.0
            time_taken["server_time"] = 0.0
            time_taken["total_time"] = 0.0

            for data_point_ix, data_point in enumerate(train_dataset):

                start = time.time()

                # Sync with the shared model
                # local_model.load_state_dict(shared_model.state_dict())
                local_model.load_from_state_dict(shared_model.get_state_dict())

                if (data_point_ix + 1) % 100 == 0:
                    logger.log("Done %d out of %d" % (data_point_ix, dataset_size))
                    logger.log("Training data action counts %r" % action_counts)
                    logger.log("Total Time %f, Server Time %f, Update Time %f, Prob Time %f " %
                               (time_taken["total_time"], time_taken["server_time"],
                                time_taken["update_time"], time_taken["prob_time"]))

                num_actions = 0
                time_start = time.time()
                image, metadata = agent.server.reset_receive_feedback(data_point)
                time_taken["server_time"] += time.time() - time_start

                state = AgentObservedState(instruction=data_point.instruction,
                                           config=config,
                                           constants=constants,
                                           start_image=image,
                                           previous_action=None,
                                           data_point=data_point)
                # state.goal = GoalPrediction.get_goal_location(metadata, data_point,
                #                                               learner.image_height, learner.image_width)

                model_state = None
                batch_replay_items = []
                total_reward = 0

                trajectory = agent.server.get_trajectory_exact(data_point.trajectory)
                trajectory = trajectory[:min(len(trajectory), constants["horizon"])]

                for action in trajectory:

                    # Sample action using the policy
                    time_start = time.time()
                    log_probabilities, model_state, image_emb_seq, volatile = \
                        local_model.get_probs(state, model_state)
                    time_taken["prob_time"] += time.time() - time_start

                    # Sample action from the probability
                    action_counts[action] += 1

                    # Send the action and get feedback
                    time_start = time.time()
                    image, reward, metadata = agent.server.send_action_receive_feedback(action)
                    time_taken["server_time"] += time.time() - time_start

                    # Store it in the replay memory list
                    replay_item = ReplayMemoryItem(state, action, reward,
                                                   log_prob=log_probabilities, volatile=volatile, goal=None)
                    batch_replay_items.append(replay_item)

                    # Update the agent state
                    state = state.update(
                        image, action, data_point=data_point)
                    # state.goal = GoalPrediction.get_goal_location(metadata, data_point,
                    #                                               learner.image_height, learner.image_width)

                    num_actions += 1
                    total_reward += reward

                time_start = time.time()
                log_probabilities, model_state, image_emb_seq, volatile = \
                    local_model.get_probs(state, model_state)
                time_taken["prob_time"] += time.time() - time_start

                # Send final STOP action and get feedback
                time_start = time.time()
                image, reward, metadata = agent.server.halt_and_receive_feedback()
                time_taken["server_time"] += time.time() - time_start
                total_reward += reward

                if metadata["navigation_error"] <= 5.0:
                    task_completion_accuracy += 1
                mean_stop_dist_error += metadata["navigation_error"]

                if tensorboard is not None:
                    tensorboard.log_scalar("navigation_error", metadata["navigation_error"])

                # Store it in the replay memory list
                replay_item = ReplayMemoryItem(state, action_space.get_stop_action_index(),
                                               reward, log_prob=log_probabilities, volatile=volatile, goal=None)
                batch_replay_items.append(replay_item)

                # Update the scores based on meta_data
                # self.meta_data_util.log_results(metadata)

                # Perform update
                time_start = time.time()
                if len(batch_replay_items) > 0:  # 32:
                    loss_val = learner.do_update(batch_replay_items)
                    # self.action_prediction_loss_calculator.predict_action(batch_replay_items)
                    # del batch_replay_items[:]  # in place list clear

                    if tensorboard is not None:
                        cross_entropy = 0.0  # float(learner.cross_entropy.data[0])
                        tensorboard.log(cross_entropy, loss_val, 0)
                        entropy = float(learner.entropy.data[0])/float(num_actions + 1)
                        logger.log("Entropy %r, Total Reward %r, Loss %r, Num Actions %d, stop-error %r " %
                                   (entropy, total_reward, loss_val, num_actions + 1, metadata["navigation_error"]))
                        tensorboard.log_scalar("entropy", entropy)
                        tensorboard.log_scalar("total_reward", total_reward)

                time_taken["update_time"] += time.time() - time_start
                time_taken["total_time"] += time.time() - start

            # Save the model
            local_model.save_model(experiment + "/supervised_learning" + str(rank) + "_epoch_" + str(epoch))
            logger.log("Training data action counts %r" % action_counts)
            mean_stop_dist_error = mean_stop_dist_error / float(len(train_dataset))
            task_completion_accuracy = (task_completion_accuracy * 100.0)/float(len(train_dataset))
            logger.log("Training: Mean stop distance error %r" % mean_stop_dist_error)
            logger.log("Training: Task completion accuracy %r " % task_completion_accuracy)

            if tune_dataset_size > 0:

                # Test on tuning data
                agent.test(tune_dataset, tensorboard=tensorboard,
                           logger=logger, pushover_logger=pushover_logger)
