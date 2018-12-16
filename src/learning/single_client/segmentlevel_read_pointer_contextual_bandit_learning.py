import torch
import logging
import torch.optim as optim
from utils.debug_nav_drone_instruction import instruction_to_string
import utils.generic_policy as gp
import numpy as np
import math

from agents.agent_observed_state import AgentObservedState
from agents.replay_memory_item import ReplayMemoryItem
from abstract_learning import AbstractLearning
from utils.cuda import cuda_var


class SegmentLevelReadPointerContextualBandit(AbstractLearning):
    """ Perform Contextual Bandit learning (Kakade and Langford (circa 2006) & Misra, Langford and Artzi EMNLP 2017) """

    def __init__(self, model, action_space, meta_data_util, config, constants, tensorboard):
        self.max_epoch = 100  # constants["max_epochs"]
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
        AbstractLearning.__init__(self, self.model, self.calc_loss,
                                  self.optimizer, self.config, self.constants)

    def calc_loss(self, batch_replay_items):

        agent_observation_state_ls = []
        immediate_rewards = []
        action_batch = []
        for replay_item in batch_replay_items:
            agent_observation_state_ls.append(replay_item.get_agent_observed_state())
            action_batch.append(replay_item.get_action())
            immediate_rewards.append(replay_item.get_reward())

        action_batch = cuda_var(torch.from_numpy(np.array(action_batch)))
        immediate_rewards = cuda_var(torch.from_numpy(np.array(immediate_rewards)).float())

        num_states = int(action_batch.size()[0])
        model_prob_batch = self.model.get_probs_batch(agent_observation_state_ls)
        chosen_log_probs = model_prob_batch.gather(1, action_batch.view(-1, 1))
        reward_log_probs = immediate_rewards * chosen_log_probs.view(-1)

        gold_distribution = cuda_var(torch.FloatTensor([0.6719, 0.1457, 0.1435, 0.0387]))
        cross_entropy = -torch.mean(torch.sum(gold_distribution * model_prob_batch, 1))
        objective = torch.sum(reward_log_probs) / num_states
        loss = -(objective - self.entropy_coef * cross_entropy)
        self.cross_entropy = cross_entropy

        return loss

    def calc_loss_entropy(self, batch_replay_items):

        agent_observation_state_ls = []
        immediate_rewards = []
        action_batch = []
        for replay_item in batch_replay_items:
            agent_observation_state_ls.append(replay_item.get_agent_observed_state())
            action_batch.append(replay_item.get_action())
            immediate_rewards.append(replay_item.get_reward())

        action_batch = cuda_var(torch.from_numpy(np.array(action_batch)))
        immediate_rewards = cuda_var(torch.from_numpy(np.array(immediate_rewards)).float())

        num_states = int(action_batch.size()[0])
        model_prob_batch = self.model.get_probs_batch(agent_observation_state_ls)
        chosen_log_probs = model_prob_batch.gather(1, action_batch.view(-1, 1))
        reward_log_probs = immediate_rewards * chosen_log_probs.view(-1)

        entropy = -torch.mean(torch.sum(model_prob_batch * torch.exp(model_prob_batch), 1))
        objective = torch.sum(reward_log_probs) / num_states
        loss = -(objective + self.entropy_coef * entropy)
        self.entropy = entropy

        return loss

    def do_train(self, agent, train_dataset, tune_dataset, experiment_name):
        """ Perform training """

        dataset_size = len(train_dataset)

        for epoch in range(1, self.max_epoch + 1):

            logging.info("Starting epoch %d", epoch)
            action_counts = [0] * self.action_space.num_actions()

            # Test on tuning data
            agent.test(tune_dataset, tensorboard=self.tensorboard)

            batch_replay_items = []
            total_reward = 0
            episodes_in_batch = 0

            for data_point_ix, data_point in enumerate(train_dataset):

                if (data_point_ix + 1) % 100 == 0:
                    logging.info("Done %d out of %d", data_point_ix, dataset_size)
                    logging.info("Training data action counts %r", action_counts)

                # instruction = instruction_to_string(
                #     data_point.get_instruction(), self.config)
                # print "TRAIN INSTRUCTION: %r" % instruction
                # print ""

                instruction = data_point.get_paragraph_instruction()

                num_actions = 0
                max_num_actions = len(data_point.get_trajectory())
                max_num_actions += self.constants["max_extra_horizon"]

                image, metadata = agent.server.reset_receive_feedback(data_point)
                pose = int(metadata["y_angle"] / 15.0)
                position_orientation = (metadata["x_pos"], metadata["z_pos"],
                                        metadata["y_angle"])
                state = AgentObservedState(instruction=data_point.get_paragraph_instruction(),
                                           config=self.config,
                                           constants=self.constants,
                                           start_image=image,
                                           previous_action=None,
                                           pose=pose,
                                           position_orientation=position_orientation,
                                           data_point=data_point)
                state.start_read_pointer, state.end_read_pointer = data_point.get_instruction_indices()

                forced_stop = True

                while num_actions < max_num_actions:

                    # Sample action using the policy
                    # Generate probabilities over actions
                    probabilities = list(torch.exp(self.model.get_probs(state).data))

                    # Use test policy to get the action
                    action = gp.sample_action_from_prob(probabilities)
                    action_counts[action] += 1

                    if action == agent.action_space.get_stop_action_index():
                        forced_stop = False
                        break

                    # Send the action and get feedback
                    image, reward, metadata = agent.server.send_action_receive_feedback(action)

                    # Store it in the replay memory list
                    replay_item = ReplayMemoryItem(state, action, reward)
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
                total_reward += reward

                # Store it in the replay memory list
                if not forced_stop:
                    replay_item = ReplayMemoryItem(state, agent.action_space.get_stop_action_index(), reward)
                    batch_replay_items.append(replay_item)

                # Update the scores based on meta_data
                # self.meta_data_util.log_results(metadata)

                # Perform update
                episodes_in_batch += 1
                if episodes_in_batch == 1:
                    loss_val = self.do_update(batch_replay_items)
                    batch_replay_items = []
                    # entropy_val = float(self.entropy.data[0])
                    # self.tensorboard.log(entropy_val, loss_val, total_reward)
                    cross_entropy = float(self.cross_entropy.data[0])
                    self.tensorboard.log(cross_entropy, loss_val, total_reward)
                    total_reward = 0
                    episodes_in_batch = 0

                if self.tensorboard is not None:
                    self.tensorboard.log_all_train_errors(
                        metadata["edit_dist_error"], metadata["closest_dist_error"], metadata["stop_dist_error"])

            # Save the model
            self.model.save_model(experiment_name + "/contextual_bandit_resnet_epoch_" + str(epoch))

            logging.info("Training data action counts %r", action_counts)
