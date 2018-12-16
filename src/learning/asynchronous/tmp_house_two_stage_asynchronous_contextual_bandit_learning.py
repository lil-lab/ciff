import sys
import traceback
import torch
import torch.optim as optim
import utils.generic_policy as gp
import numpy as np
import nltk


from agents.agent_observed_state import AgentObservedState
from agents.house_decoupled_predictor_navigator_model import HouseDecoupledPredictorNavigatorAgent
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


class TmpTwoStageAsynchronousContextualBandit(AbstractLearning):
    """ Perform Contextual Bandit learning (Kakade and Langford (circa 2006) & Misra, Langford and Artzi EMNLP 2017) """

    def __init__(self, shared_navigator_model, local_navigator_model, shared_predictor_model, local_predictor_model,
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
        self.entropy_coef = constants["entropy_coefficient"]
        self.logger = None

        parameters = self.shared_navigator_model.get_parameters()
        parameters.extend(self.shared_predictor_model.get_parameters())

        self.optimizer = optim.Adam(parameters, lr=constants["learning_rate"])
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

        model_log_prob_batch = log_probabilities
        chosen_log_action_probs = model_log_prob_batch.gather(1, action_batch.view(-1, 1))

        # Take the probability of goal generation into account
        chosen_log_goal_prob = torch.cat(chosen_log_goal_prob)
        chosen_log_probs = chosen_log_action_probs.view(-1) + chosen_log_goal_prob.view(-1)

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

    def get_goal(self, metadata, goal_type):

        if goal_type == "gold":

            if metadata["goal-screen"] is None:
                return None, None, None, None

            left, bottom, depth = metadata["goal-screen"]

        elif goal_type == "inferred":

            if metadata["tracking"] is None:
                return None, None, None, None

            left, bottom, depth = metadata["tracking"]

        else:
            raise AssertionError("Unknown goal type %r. Supported goal types are gold and inferred.", goal_type)

        if 0.01 < left < self.config["image_width"] and \
                                0.01 < bottom < self.config["image_height"] and depth > 0.01:

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

    def _explore_and_set_tracking(self, server, data_point):

        # Get the panoramic image
        panorama, _ = server.explore()

        # Get the panorama and predict the goal location
        state = AgentObservedState(instruction=data_point.instruction,
                                   config=self.config,
                                   constants=self.constants,
                                   start_image=panorama,
                                   previous_action=None,
                                   pose=None,
                                   position_orientation=None,
                                   data_point=data_point)

        volatile = self.local_predictor_model.get_attention_prob(state, model_state=None)
        attention_prob = list(volatile["attention_probs"].view(-1)[:-1].data.cpu().numpy())
        inferred_ix = gp.sample_action_from_prob(attention_prob)
        sampled_prob = volatile["attention_probs"][inferred_ix]

        if inferred_ix == 6 * self.config["num_manipulation_row"] * self.config["num_manipulation_col"]:
            print("Predicting Out-of-sight")
            return

        assert 0 <= inferred_ix < 6 * self.config["num_manipulation_row"] * self.config["num_manipulation_col"]

        row = int(inferred_ix / (6 * self.config["num_manipulation_col"]))
        col = inferred_ix % (6 * self.config["num_manipulation_col"])
        region_ix = int(col / self.config["num_manipulation_col"])

        if region_ix == 0:
            camera_ix = 3
        elif region_ix == 1:
            camera_ix = 4
        elif region_ix == 2:
            camera_ix = 5
        elif region_ix == 3:
            camera_ix = 0
        elif region_ix == 4:
            camera_ix = 1
        elif region_ix == 5:
            camera_ix = 2
        else:
            raise AssertionError("region ix should be in {0, 1, 2, 3, 4, 5}. Found ", region_ix)

        col = col % self.config["num_manipulation_col"]

        # Set tracking
        row_value = min(1.0, (row + 0.5) / float(self.config["num_manipulation_row"]))
        col_value = min(1.0, (col + 0.5) / float(self.config["num_manipulation_col"]))

        server.set_tracking(camera_ix, row_value, col_value)

        return sampled_prob

    @staticmethod
    def do_train(house_id, shared_navigator_model, shared_predictor_model, config, action_space, meta_data_util,
                 constants, train_dataset, tune_dataset, experiment,
                 experiment_name, rank, server, logger, navigatormodel_type, predictor_model_type, vocab, use_pushover=False):
        try:
            TmpTwoStageAsynchronousContextualBandit.do_train_(house_id, shared_navigator_model, shared_predictor_model,
                                                              config, action_space, meta_data_util,
                                                              constants, train_dataset, tune_dataset, experiment,
                                                              experiment_name, rank, server, logger,
                                                              navigatormodel_type, predictor_model_type,
                                                              vocab, use_pushover)
        except Exception:
            exc_info = sys.exc_info()
            traceback.print_exception(*exc_info)

    @staticmethod
    def do_train_(house_id, shared_navigator_model, shared_predictor_model, config, action_space, meta_data_util,
                  constants, train_dataset, tune_dataset, experiment, experiment_name, rank, server,
                  logger, navigator_model_type, predictor_model_type, vocab, use_pushover=False):

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
        local_predictor_model = predictor_model_type(config, constants, final_model_type="unet-positional-encoding",
                                                     final_dimension=(64, 32, 32 * 6))
        local_navigator_model = navigator_model_type(config, constants)

        # Create the Agent
        tmp_agent = HouseDecoupledPredictorNavigatorAgent(server=server,
                                                          goal_prediction_model=local_predictor_model,
                                                          navigation_model=local_navigator_model,
                                                          action_type_model=None,
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

        # if tune_dataset_size > 0:
        #     # Test on tuning data
        #     tmp_agent.test_single_step(tune_dataset, vocab, goal_type="inferred", tensorboard=tensorboard,
        #                                logger=logger, pushover_logger=pushover_logger)

        # Create the learner to compute the loss
        learner = TmpTwoStageAsynchronousContextualBandit(shared_navigator_model, local_navigator_model,
                                                          shared_predictor_model, local_predictor_model,
                                                          action_space, meta_data_util,
                                                          config, constants, tensorboard)
        # TODO change 2 --- unity launch moved up
        learner.logger = logger

        for epoch in range(1, max_epochs + 1):

            for data_point_ix, data_point in enumerate(train_dataset):

                # Sync with the shared model
                # local_model.load_state_dict(shared_model.state_dict())
                local_navigator_model.load_from_state_dict(shared_navigator_model.get_state_dict())
                local_predictor_model.load_from_state_dict(shared_predictor_model.get_state_dict())

                if (data_point_ix + 1) % 100 == 0:
                    logger.log("Done %d out of %d" %(data_point_ix, dataset_size))
                    logger.log("Training data action counts %r" % action_counts)

                num_actions = 0
                max_num_actions = constants["horizon"]
                max_num_actions += constants["max_extra_horizon"]

                image, metadata = tmp_agent.server.reset_receive_feedback(data_point)
                instruction = data_point.get_instruction()
                # instruction_str = TmpTwoStageAsynchronousContextualBandit.convert_indices_to_text(instruction, vocab)
                # print("Instruction str is ", instruction_str)

                ################ SAMPLE A GOAL ###########################
                sample_prob = learner._explore_and_set_tracking(server, data_point)

                # Pose and Orientation gone TODO change 3
                state = AgentObservedState(instruction=instruction,
                                           config=config,
                                           constants=constants,
                                           start_image=image,
                                           previous_action=None,
                                           data_point=data_point)
                state.goal = learner.get_goal(metadata, "inferred")

                model_state = None
                batch_replay_items = []
                total_reward = 0
                forced_stop = True

                while num_actions < max_num_actions:

                    # logger.log("Training: Meta Data %r " % metadata)

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
                    image, reward, metadata = tmp_agent.server.send_action_receive_feedback(action)
                    # logger.log("Action is %r, Reward is %r Probability is %r " % (action, reward, probabilities))

                    # Store it in the replay memory list
                    volatile["goal_sample_prob"] = sample_prob
                    replay_item = ReplayMemoryItem(state, action, reward, log_prob=log_probabilities, volatile=volatile)
                    batch_replay_items.append(replay_item)

                    # Update the agent state
                    # Pose and orientation gone, TODO change 4
                    state = state.update(image, action, data_point=data_point)
                    state.goal = learner.get_goal(metadata, "inferred")

                    num_actions += 1
                    total_reward += reward

                # Send final STOP action and get feedback
                image, reward, metadata = tmp_agent.server.halt_and_receive_feedback()
                total_reward += reward

                # Store it in the replay memory list
                if not forced_stop:
                    volatile["goal_sample_prob"] = sample_prob
                    # logger.log("Action is Stop, Reward is %r Probability is %r " % (reward, probabilities))
                    replay_item = ReplayMemoryItem(state, action_space.get_stop_action_index(),
                                                   reward, log_prob=log_probabilities, volatile=volatile)
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

            # Save the model
            local_navigator_model.save_model(
                experiment + "/navigator_contextual_bandit_" + str(rank) + "_epoch_" + str(epoch))
            local_predictor_model.save_model(
                experiment + "/predictor_contextual_bandit_" + str(rank) + "_epoch_" + str(epoch))
            logger.log("Training data action counts %r" % action_counts)

            if tune_dataset_size > 0:
                # Test on tuning data
                tmp_agent.test_single_step(tune_dataset, vocab, goal_type="inferred", tensorboard=tensorboard,
                                           logger=logger, pushover_logger=pushover_logger)

