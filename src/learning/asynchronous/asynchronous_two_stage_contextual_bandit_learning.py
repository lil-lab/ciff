import sys
import traceback
import math
import torch
import torch.optim as optim
import utils.generic_policy as gp
import numpy as np
import random

from agents.agent import Agent
from agents.agent_observed_state import AgentObservedState
from agents.predicter_planner_agent import PredictorPlannerAgent
from agents.replay_memory_item import ReplayMemoryItem
from learning.auxiliary_objective.goal_prediction import GoalPrediction
from learning.auxiliary_objective.object_detection import ObjectDetection
from learning.auxiliary_objective.object_pixel_identification import ObjectPixelIdentification
from learning.auxiliary_objective.symbolic_language_prediction import SymbolicLanguagePrediction
from learning.asynchronous.abstract_learning import AbstractLearning
from learning.auxiliary_objective.action_prediction import ActionPrediction
from learning.auxiliary_objective.temporal_autoencoder import TemporalAutoEncoder
from learning.single_client.goal_prediction_single_360_image_supervised_from_disk import \
    GoalPredictionSingle360ImageSupervisedLearningFromDisk
from utils.camera_mapping import get_inverse_object_position
from utils.cuda import cuda_var
from utils.geometry import current_pos_from_metadata, current_pose_from_metadata
from utils.launch_unity import launch_k_unity_builds
from utils.pushover_logger import PushoverLogger
from utils.tensorboard import Tensorboard


class AsynchronousTwoStageContextualBandit(AbstractLearning):
    """ Perform Contextual Bandit learning (Kakade and Langford (circa 2006) & Misra, Langford and Artzi EMNLP 2017)
    on the two stage model. """

    def __init__(self, shared_navigator_model, local_navigator_model,
                 shared_predictor_model, local_predictor_model,
                 action_space, meta_data_util, config, constants, tensorboard):
        self.max_epoch = constants["max_epochs"]
        self.shared_navigator_model = shared_navigator_model
        self.local_navigator_model = local_navigator_model
        self.shared_predictor_model = shared_predictor_model
        self.local_predictor_model = local_predictor_model
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

        self.image_channels, self.image_height, self.image_width = shared_navigator_model.image_module.get_final_dimension()

        # Auxiliary Objectives
        if self.config["do_action_prediction"]:
            self.action_prediction_loss_calculator = ActionPrediction(self.local_navigator_model)
            self.action_prediction_loss = None
        if self.config["do_temporal_autoencoding"]:
            self.temporal_autoencoder_loss_calculator = TemporalAutoEncoder(self.local_navigator_model)
            self.temporal_autoencoder_loss = None
        if self.config["do_object_detection"]:
            self.object_detection_loss_calculator = ObjectPixelIdentification(
                self.local_navigator_model, num_objects=67, camera_angle=60, image_height=self.image_height,
                image_width=self.image_width, object_height=0)  # -2.5)
            self.object_detection_loss = None
        if self.config["do_symbolic_language_prediction"]:
            self.symbolic_language_prediction_loss_calculator = SymbolicLanguagePrediction(self.local_navigator_model)
            self.symbolic_language_prediction_loss = None
        if self.config["do_goal_prediction"]:
            self.goal_prediction_calculator = GoalPrediction(self.local_navigator_model, self.image_height, self.image_width)
            self.goal_prediction_loss = None

        parameters = self.shared_navigator_model.get_parameters()
        parameters.extend(self.shared_predictor_model.get_parameters())

        self.optimizer = optim.Adam(parameters,
                                    lr=constants["learning_rate"])
        AbstractLearning.__init__(self, self.shared_navigator_model, self.local_navigator_model, self.calc_loss,
                                  self.optimizer, self.config, self.constants, self.tensorboard)

    def calc_loss(self, batch_replay_items):

        agent_observation_state_ls = []
        immediate_rewards = []
        action_batch = []
        log_probabilities = []
        factor_entropy = []
        chosen_log_goal_prob = []
        for replay_item in batch_replay_items:
            agent_observation_state_ls.append(replay_item.get_agent_observed_state())
            action_batch.append(replay_item.get_action())
            immediate_rewards.append(replay_item.get_reward())
            log_probabilities.append(replay_item.get_log_prob())
            factor_entropy.append(replay_item.get_factor_entropy())
            chosen_log_goal_prob.append(replay_item.get_volatile_features()["goal_sample_prob"])

        log_probabilities = torch.cat(log_probabilities)
        action_batch = cuda_var(torch.from_numpy(np.array(action_batch)))
        immediate_rewards = cuda_var(torch.from_numpy(np.array(immediate_rewards)).float())

        num_states = int(action_batch.size()[0])
        model_log_prob_batch = log_probabilities
        # model_log_prob_batch = self.model.get_probs_batch(agent_observation_state_ls)
        chosen_log_action_probs = model_log_prob_batch.gather(1, action_batch.view(-1, 1))

        # Take the probability of goal generation into account
        chosen_log_goal_prob = torch.cat(chosen_log_goal_prob)

        chosen_log_probs = chosen_log_action_probs.view(-1) + chosen_log_goal_prob.view(-1)
        reward_log_probs = immediate_rewards * chosen_log_probs

        gold_distribution = cuda_var(torch.FloatTensor([0.6719, 0.1457, 0.1435, 0.0387]))
        model_prob_batch = torch.exp(model_log_prob_batch)
        mini_batch_action_distribution = torch.mean(model_prob_batch, 0)

        self.cross_entropy = -torch.sum(gold_distribution * torch.log(mini_batch_action_distribution))
        # self.entropy = -torch.mean(torch.sum(model_log_prob_batch * model_prob_batch, 1))
        self.entropy = -torch.sum(torch.sum(model_log_prob_batch * model_prob_batch, 1))
        objective = torch.sum(reward_log_probs) # / num_states
        # Essentially we want the objective to increase and cross entropy to decrease
        entropy_coef = max(0, self.entropy_coef - self.epoch * 0.01)
        loss = -objective - entropy_coef * self.entropy
        self.ratio = torch.abs(objective)/(entropy_coef * self.entropy)  # we want the ratio to be high

        # loss = -objective + self.entropy_coef * self.cross_entropy

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
            if self.object_detection_loss is not None:
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

        if self.config["do_goal_prediction"]:
            self.goal_prediction_loss, _, _ = self.goal_prediction_calculator.calc_loss(batch_replay_items)
            if self.goal_prediction_loss is not None:
                self.goal_prediction_loss = self.constants["goal_prediction_coeff"] * \
                                            self.goal_prediction_loss
                loss = loss + self.goal_prediction_loss  # * len(batch_replay_items)  # scale the loss
        else:
            self.goal_prediction_loss = None

        return loss

    def _sample_goal(self, exploration_image, data_point, panaroma=True):

        state = AgentObservedState(instruction=data_point.instruction,
                                   config=self.config,
                                   constants=self.constants,
                                   start_image=exploration_image,
                                   previous_action=None,
                                   pose=None,
                                   position_orientation=data_point.get_start_pos(),
                                   data_point=data_point)

        volatile = self.local_predictor_model.get_attention_prob(state, model_state=None)
        attention_prob = list(volatile["attention_probs"].view(-1)[:-1].data.cpu().numpy())
        sampled_ix = gp.sample_action_from_prob(attention_prob)
        sampled_prob = volatile["attention_probs"][sampled_ix]
        #################################################

        # Max pointed about that when inferred ix above is the last value then calculations are buggy. He is right.

        predicted_row = int(sampled_ix / float(192))
        predicted_col = sampled_ix % 192
        screen_pos = (predicted_row, predicted_col)

        if panaroma:
            # Index of the 6 image where the goal is
            region_index = int(predicted_col / 32)
            predicted_col = predicted_col % 32  # Column within that image where the goal is
            pos = data_point.get_start_pos()
            new_pos_angle = GoalPredictionSingle360ImageSupervisedLearningFromDisk.\
                get_new_pos_angle_from_region_index(region_index, pos)
            metadata = {"x_pos": pos[0], "z_pos": pos[1], "y_angle": new_pos_angle}
        else:
            pos = data_point.get_start_pos()
            metadata = {"x_pos": pos[0], "z_pos": pos[1], "y_angle": pos[2]}

        row, col = predicted_row + 0.5, predicted_col + 0.5

        start_pos = current_pos_from_metadata(metadata)
        start_pose = current_pose_from_metadata(metadata)

        goal_pos = data_point.get_destination_list()[-1]
        height_drone = 2.5
        x_gen, z_gen = get_inverse_object_position(row, col, height_drone, 30, 32, 32,
                                                   (start_pos[0], start_pos[1], start_pose))
        predicted_goal_pos = (x_gen, z_gen)
        x_goal, z_goal = goal_pos

        x_diff = x_gen - x_goal
        z_diff = z_gen - z_goal

        dist = math.sqrt(x_diff * x_diff + z_diff * z_diff)

        return predicted_goal_pos, dist, screen_pos, sampled_prob

    @staticmethod
    def do_train(shared_navigator_model, shared_predictor_model, config, action_space, meta_data_util, constants,
                 train_dataset, tune_dataset, experiment, experiment_name, rank, server,
                 logger, navigator_model_type, predictor_model_type, use_pushover=False):
        try:
            AsynchronousTwoStageContextualBandit.do_train_(shared_navigator_model, shared_predictor_model,
                                                           config, action_space, meta_data_util, constants,
                                                           train_dataset, tune_dataset, experiment, experiment_name, rank, server,
                                                           logger, navigator_model_type, predictor_model_type, use_pushover)
        except Exception:
            exc_info = sys.exc_info()
            traceback.print_exception(*exc_info)

    @staticmethod
    def do_train_(shared_navigator_model, shared_predictor_model, config, action_space, meta_data_util, constants,
                  train_dataset, tune_dataset, experiment, experiment_name, rank, server,
                  logger, navigator_model_type, predictor_model_type, use_pushover=False):

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
        local_predictor_model = predictor_model_type(config, constants, final_model_type="unet-positional-encoding",
                                                   final_dimension=(64, 32, 32 * 6))
        local_navigator_model = navigator_model_type(config, constants)
        # local_model.train()

        # Create the Agent
        logger.log("STARTING AGENT")
        agent = PredictorPlannerAgent(server=server,
                                      predictor_model=local_predictor_model,
                                      model=local_navigator_model,
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
        learner = AsynchronousTwoStageContextualBandit(shared_navigator_model, local_navigator_model,
                                                       shared_predictor_model, local_predictor_model,
                                                       action_space, meta_data_util,
                                                       config, constants, tensorboard)

        # Launch unity
        launch_k_unity_builds([config["port"]], "./simulators/NavDroneLinuxBuild.x86_64")

        for epoch in range(1, max_epochs + 1):

            learner.epoch = epoch
            task_completion_accuracy = 0
            mean_stop_dist_error = 0
            stop_dist_errors = []
            for data_point_ix, data_point in enumerate(train_dataset):

                # Sync with the shared model
                # local_model.load_state_dict(shared_model.state_dict())
                local_navigator_model.load_from_state_dict(shared_navigator_model.get_state_dict())
                local_predictor_model.load_from_state_dict(shared_predictor_model.get_state_dict())

                if (data_point_ix + 1) % 100 == 0:
                    logger.log("Done %d out of %d" %(data_point_ix, dataset_size))
                    logger.log("Training data action counts %r" % action_counts)

                num_actions = 0
                max_num_actions = constants["horizon"] + constants["max_extra_horizon"]

                image, metadata = agent.server.reset_receive_feedback(data_point)

                # Generate goal probability
                # Test image
                panorama = agent.get_exploration_image()

                # Sample a goal location and compute 3D mapping
                predicted_goal, predictor_error, predicted_pixel, sample_prob = learner._sample_goal(
                    panorama, data_point, panaroma=True)

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
                current_bot_location = metadata["x_pos"], metadata["z_pos"]
                current_bot_pose = metadata["y_angle"]
                state.goal = PredictorPlannerAgent.get_goal_location(
                    current_bot_location, current_bot_pose, predicted_goal, 32, 32)

                model_state = None
                batch_replay_items = []
                total_reward = 0
                forced_stop = True

                while num_actions < max_num_actions:

                    # Sample action using the policy
                    log_probabilities, model_state, image_emb_seq, volatile = \
                        local_navigator_model.get_probs(state, model_state)
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
                    volatile["goal_sample_prob"] = sample_prob
                    replay_item = ReplayMemoryItem(state, action, reward,
                                                   log_prob=log_probabilities, volatile=volatile)
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

                    current_bot_location = metadata["x_pos"], metadata["z_pos"]
                    current_bot_pose = metadata["y_angle"]
                    state.goal = PredictorPlannerAgent.get_goal_location(
                        current_bot_location, current_bot_pose, predicted_goal, 32, 32)

                    num_actions += 1
                    total_reward += reward

                # Send final STOP action and get feedback
                image, reward, metadata = agent.server.halt_and_receive_feedback()
                total_reward += reward

                if metadata["stop_dist_error"] < 5.0:
                    task_completion_accuracy += 1
                mean_stop_dist_error += metadata["stop_dist_error"]
                stop_dist_errors.append(metadata["stop_dist_error"])

                if tensorboard is not None:
                    tensorboard.log_all_train_errors(
                        metadata["edit_dist_error"], metadata["closest_dist_error"], metadata["stop_dist_error"])

                # Store it in the replay memory list
                if not forced_stop:
                    volatile["goal_sample_prob"] = sample_prob
                    replay_item = ReplayMemoryItem(state, action_space.get_stop_action_index(),
                                                   reward, log_prob=log_probabilities, volatile=volatile)
                    batch_replay_items.append(replay_item)

                # Update the scores based on meta_data
                # self.meta_data_util.log_results(metadata)

                # Perform update
                if len(batch_replay_items) > 0:  # 32:
                    loss_val = learner.do_update(batch_replay_items)
                    # self.action_prediction_loss_calculator.predict_action(batch_replay_items)
                    # del batch_replay_items[:]  # in place list clear

                    if tensorboard is not None:
                        tensorboard.log_scalar("gold_sample_prob", float(sample_prob.data[0]))
                        tensorboard.log_scalar("predicted_error", predictor_error)
                        cross_entropy = float(learner.cross_entropy.data[0])
                        tensorboard.log(cross_entropy, loss_val, 0)
                        entropy = float(learner.entropy.data[0])/float(num_actions + 1)
                        tensorboard.log_scalar("entropy", entropy)
                        tensorboard.log_scalar("total_reward", total_reward)

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

            # Save the model
            local_navigator_model.save_model(
                experiment + "/navigator_contextual_bandit_" + str(rank) + "_epoch_" + str(epoch))
            local_predictor_model.save_model(
                experiment + "/predictor_contextual_bandit_" + str(rank) + "_epoch_" + str(epoch))
            logger.log("Training data action counts %r" % action_counts)
            mean_stop_dist_error = mean_stop_dist_error / float(len(train_dataset))
            task_completion_accuracy = (task_completion_accuracy * 100.0)/float(len(train_dataset))
            logger.log("Training: Mean stop distance error %r" % mean_stop_dist_error)
            logger.log("Training: Task completion accuracy %r " % task_completion_accuracy)
            bins = range(0, 80, 3)  # range of distance
            histogram, _ = np.histogram(stop_dist_errors, bins)
            logger.log("Histogram of train errors %r " % histogram)

            if tune_dataset_size > 0:
                # Test on tuning data
                agent.test(tune_dataset, tensorboard=tensorboard,
                           logger=logger, pushover_logger=pushover_logger)
