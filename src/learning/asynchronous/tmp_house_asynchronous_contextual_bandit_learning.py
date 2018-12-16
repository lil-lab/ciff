import sys
import traceback
import torch
import torch.optim as optim
import utils.generic_policy as gp
import numpy as np
import nltk


from agents.agent_observed_state import AgentObservedState
from agents.replay_memory_item import ReplayMemoryItem
from agents.tmp_house_agent import TmpHouseAgent
from learning.auxiliary_objective.goal_prediction import GoalPrediction
from learning.auxiliary_objective.object_detection import ObjectDetection
from learning.auxiliary_objective.symbolic_language_prediction import SymbolicLanguagePrediction
from learning.asynchronous.abstract_learning import AbstractLearning
from learning.auxiliary_objective.action_prediction import ActionPrediction
from learning.auxiliary_objective.temporal_autoencoder import TemporalAutoEncoder
from utils.cuda import cuda_var
from utils.launch_unity import launch_k_unity_builds
# from utils.pushover_logger import PushoverLogger
from utils.tensorboard import Tensorboard


class TmpAsynchronousContextualBandit(AbstractLearning):
    """ Perform Contextual Bandit learning (Kakade and Langford (circa 2006) & Misra, Langford and Artzi EMNLP 2017) """

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
        self.entropy_coef = constants["entropy_coefficient"]
        self.logger = None

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
        immediate_rewards = []
        action_batch = []
        log_probabilities = []
        factor_entropy = []
        for replay_item in batch_replay_items:
            agent_observation_state_ls.append(replay_item.get_agent_observed_state())
            action_batch.append(replay_item.get_action())
            immediate_rewards.append(replay_item.get_reward())
            log_probabilities.append(replay_item.get_log_prob())
            factor_entropy.append(replay_item.get_factor_entropy())

        log_probabilities = torch.cat(log_probabilities)
        action_batch = cuda_var(torch.from_numpy(np.array(action_batch)))
        immediate_rewards = cuda_var(torch.from_numpy(np.array(immediate_rewards)).float())

        # self.logger.log("Learning from Log Probabilities is %r " % log_probabilities.data.cpu().numpy())
        # self.logger.log("Learning from Action Batch is %r " % action_batch.data.cpu().numpy())
        # self.logger.log("Learning from Immediate Rewards is %r " % immediate_rewards.data.cpu().numpy())

        # num_states = int(action_batch.size()[0])
        model_log_prob_batch = log_probabilities
        chosen_log_probs = model_log_prob_batch.gather(1, action_batch.view(-1, 1))
        reward_log_probs = immediate_rewards * chosen_log_probs.view(-1)

        # self.logger.log("Learning from Chosen Log Probs is %r " % chosen_log_probs.data.cpu().numpy())
        # self.logger.log("Learning from Reward Log Probs is %r " % reward_log_probs.data.cpu().numpy())

        model_prob_batch = torch.exp(model_log_prob_batch)
        # mini_batch_action_distribution = torch.mean(model_prob_batch, 0)
        # self.cross_entropy = -torch.sum(gold_distribution * torch.log(mini_batch_action_distribution))
        self.entropy = -torch.sum(torch.sum(model_log_prob_batch * model_prob_batch, 1))
        objective = torch.sum(reward_log_probs)

        # self.logger.log("Objective is %r and entropy is %r and entropy coef is %r " %
        #                 (objective, self.entropy, self.entropy_coef))

        # Essentially we want the objective to increase and cross entropy to decrease
        loss = -objective - self.entropy_coef * self.entropy
        self.ratio = torch.abs(objective)/(self.entropy_coef * self.entropy)  # we want the ratio to be high

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

        if self.config["do_goal_prediction"]:
            self.goal_prediction_loss = self.goal_prediction_calculator.calc_loss(batch_replay_items)
            self.goal_prediction_loss = self.constants["goal_prediction_coeff"] * self.goal_prediction_loss
            loss = loss + self.goal_prediction_loss
        else:
            self.goal_prediction_loss = None

        return loss

    @staticmethod
    def convert_text_to_indices(text, vocab, ignore_case=True):

        # Tokenize the text
        print ("instruction ", text)
        token_seq = nltk.word_tokenize(text)

        indices = []

        for token in token_seq:
            if ignore_case:
                ltoken = token.lower()
            else:
                ltoken = token
            if ltoken in vocab:
                indices.append(vocab[ltoken])
            else:
                indices.append(vocab["$UNK$"])

        return indices

    @staticmethod
    def convert_indices_to_text(indices, vocab):
        return " ".join([vocab[index] for index in indices])

    def get_goal(self, metadata):

        if metadata["goal-screen"] is None:
            return None, None, None, None

        left, bottom, depth = metadata["goal-screen"]

        if 0.01 < left < self.config["image_width"] and 0.01 < bottom < self.config["image_height"] and depth > 0.01:

            scaled_left = left / float(self.config["image_width"])
            scaled_top = 1.0 - bottom / float(self.config["image_height"])

            row_real = self.config["num_manipulation_row"] * scaled_top
            col_real = self.config["num_manipulation_col"] * scaled_left
            row, col = round(row_real), round(col_real)

            if row < 0:
                row = 0
            elif row >= self.config["num_manipulation_row"]:
                row = self.config["num_manipulation_row"] - 1
            if col < 0:
                col = 0
            elif col >= self.config["num_manipulation_col"]:
                col = self.config["num_manipulation_col"] - 1

            return row, col, row_real, col_real
        else:
            return None, None, None, None

    @staticmethod
    def do_train(house_id, shared_model, config, action_space, meta_data_util,
                 constants, train_dataset, tune_dataset, experiment,
                 experiment_name, rank, server, logger, model_type, vocab, use_pushover=False):
        try:
            TmpAsynchronousContextualBandit.do_train_(house_id, shared_model, config, action_space, meta_data_util,
                                                      constants, train_dataset, tune_dataset, experiment,
                                                      experiment_name, rank, server, logger, model_type,
                                                      vocab, use_pushover)
        except Exception:
            exc_info = sys.exc_info()
            traceback.print_exception(*exc_info)

    @staticmethod
    def do_train_(house_id, shared_model, config, action_space, meta_data_util, constants,
                  train_dataset, tune_dataset, experiment, experiment_name, rank, server,
                  logger, model_type, vocab, use_pushover=False):

        logger.log("In Training...")
        launch_k_unity_builds([config["port"]], "./house_" + str(house_id) + "_elmer.x86_64",
                              arg_str="--config ./AssetsHouse/config" + str(house_id) + ".json",
                              cwd="./simulators/house/")
        logger.log("Launched Builds.")
        server.initialize_server()
        logger.log("Server Initialized.")

        # Test policy
        test_policy = gp.get_argmax_action

        if rank == 0:  # client 0 creates a tensorboard server
            tensorboard = Tensorboard(experiment_name)
            logger.log('Created Tensorboard Server.')
        else:
            tensorboard = None

        if use_pushover:
            pushover_logger = None
        else:
            pushover_logger = None

        # Create a local model for rollouts
        local_model = model_type(config, constants)
        # local_model.train()

        # Create the Agent
        tmp_agent = TmpHouseAgent(server=server,
                                  model=local_model,
                                  test_policy=test_policy,
                                  action_space=action_space,
                                  meta_data_util=meta_data_util,
                                  config=config,
                                  constants=constants)
        logger.log("Created Agent.")

        action_counts = [0] * action_space.num_actions()
        max_epochs = 100000 # constants["max_epochs"]
        dataset_size = len(train_dataset)
        tune_dataset_size = len(tune_dataset)

        if tune_dataset_size > 0:
            # Test on tuning data
            tmp_agent.test(tune_dataset, vocab, tensorboard=tensorboard,
                           logger=logger, pushover_logger=pushover_logger)

        # Create the learner to compute the loss
        learner = TmpAsynchronousContextualBandit(shared_model, local_model, action_space, meta_data_util,
                                                  config, constants, tensorboard)
        # TODO change 2 --- unity launch moved up
        learner.logger = logger

        for epoch in range(1, max_epochs + 1):

            for data_point_ix, data_point in enumerate(train_dataset):

                # Sync with the shared model
                # local_model.load_state_dict(shared_model.state_dict())
                local_model.load_from_state_dict(shared_model.get_state_dict())

                if (data_point_ix + 1) % 100 == 0:
                    logger.log("Done %d out of %d" %(data_point_ix, dataset_size))
                    logger.log("Training data action counts %r" % action_counts)

                num_actions = 0
                max_num_actions = constants["horizon"]
                max_num_actions += constants["max_extra_horizon"]

                image, metadata = tmp_agent.server.reset_receive_feedback(data_point)
                instruction = data_point.get_instruction()
                # instruction_str = TmpAsynchronousContextualBandit.convert_indices_to_text(instruction, vocab)
                # print("Instruction str is ", instruction_str)

                # Pose and Orientation gone TODO change 3
                state = AgentObservedState(instruction=instruction,
                                           config=config,
                                           constants=constants,
                                           start_image=image,
                                           previous_action=None,
                                           data_point=data_point)
                state.goal = learner.get_goal(metadata)

                model_state = None
                batch_replay_items = []
                total_reward = 0
                forced_stop = True

                while num_actions < max_num_actions:

                    # logger.log("Training: Meta Data %r " % metadata)

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
                    image, reward, metadata = tmp_agent.server.send_action_receive_feedback(action)
                    # logger.log("Action is %r, Reward is %r Probability is %r " % (action, reward, probabilities))

                    # Store it in the replay memory list
                    replay_item = ReplayMemoryItem(state, action, reward, log_prob=log_probabilities)
                    batch_replay_items.append(replay_item)

                    # Update the agent state
                    # Pose and orientation gone, TODO change 4
                    state = state.update(image, action, data_point=data_point)
                    state.goal = learner.get_goal(metadata)

                    num_actions += 1
                    total_reward += reward

                # Send final STOP action and get feedback
                image, reward, metadata = tmp_agent.server.halt_and_receive_feedback()
                total_reward += reward

                # Store it in the replay memory list
                if not forced_stop:
                    # logger.log("Action is Stop, Reward is %r Probability is %r " % (reward, probabilities))
                    replay_item = ReplayMemoryItem(state, action_space.get_stop_action_index(),
                                                   reward, log_prob=log_probabilities)
                    batch_replay_items.append(replay_item)

                # Update the scores based on meta_data
                # self.meta_data_util.log_results(metadata)

                # Perform update
                if len(batch_replay_items) > 0:  # 32
                    loss_val = learner.do_update(batch_replay_items)

                    if tensorboard is not None:
                        # cross_entropy = float(learner.cross_entropy.data[0])
                        # tensorboard.log(cross_entropy, loss_val, 0)
                        tensorboard.log_scalar("loss", loss_val)
                        entropy = float(learner.entropy.data[0])/float(num_actions + 1)
                        tensorboard.log_scalar("entropy", entropy)
                        ratio = float(learner.ratio.data[0])
                        tensorboard.log_scalar("Abs_objective_to_entropy_ratio", ratio)
                        tensorboard.log_scalar("total_reward", total_reward)
                        tensorboard.log_scalar("mean navigation error", metadata['mean-navigation-error'])

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
            local_model.save_model(experiment + "/contextual_bandit_" + str(rank) + "_epoch_" + str(epoch))
            logger.log("Training data action counts %r" % action_counts)

            if tune_dataset_size > 0:
                # Test on tuning data
                tmp_agent.test(tune_dataset, vocab, tensorboard=tensorboard,
                               logger=logger, pushover_logger=pushover_logger)
