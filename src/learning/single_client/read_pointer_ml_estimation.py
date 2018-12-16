import time
import torch
import logging
import torch.optim as optim
from utils.debug_nav_drone_instruction import instruction_to_string
import utils.generic_policy as gp
import numpy as np
import utils.debug_nav_drone_instruction as debug

from agents.agent_observed_state import AgentObservedState
from agents.replay_memory_item import ReplayMemoryItem
from abstract_learning import AbstractLearning
from agents.agent_with_read import ReadPointerAgent
from reward_model.linguistic_prior import LinguisticPrior
from reward_model.alignment_reward import AlignmentReward
from utils.cuda import cuda_var


class ReadPointerMaximumLikelihoodEstimation(AbstractLearning):
    """ Perform Contextual Bandit learning (Kakade and Langford (circa 2006) & Misra, Langford and Artzi EMNLP 2017) """

    def __init__(self, model, action_space, meta_data_util, config, constants, tensorboard):
        self.max_epoch = 100  # constants["max_epochs"]
        self.model = model
        self.action_space = action_space
        self.meta_data_util = meta_data_util
        self.config = config
        self.constants = constants
        self.tensorboard = tensorboard
        self.entropy_coef = constants["entropy_coefficient"]
        self.optimizer = optim.Adam(model.get_parameters(),
                                    lr=constants["learning_rate"])
        self.entropy = None
        AbstractLearning.__init__(self, self.model, self.calc_loss,
                                  self.optimizer, self.config, self.constants)

    def calc_loss(self, batch_replay_items):

        agent_observation_state_ls = dict({ReadPointerAgent.READ_MODE: [], ReadPointerAgent.ACT_MODE: []})
        action_batch = dict({ReadPointerAgent.READ_MODE: [], ReadPointerAgent.ACT_MODE: []})
        for replay_item in batch_replay_items:
            agent_observation_state_ls[replay_item.get_mode()].append(replay_item.get_agent_observed_state())
            action_batch[replay_item.get_mode()].append(replay_item.get_action())

        if len(action_batch[ReadPointerAgent.READ_MODE]) == 0:
            loss_read, entropy_read = cuda_var(torch.FloatTensor([0.0])), cuda_var(torch.FloatTensor([0.0]))
        else:
            loss_read, entropy_read = self._calc_loss_each_mode(agent_observation_state_ls[ReadPointerAgent.READ_MODE],
                                                                action_batch[ReadPointerAgent.READ_MODE])

        if len(action_batch[ReadPointerAgent.ACT_MODE]) == 0:
            loss_act, entropy_act = cuda_var(torch.FloatTensor([0.0])), cuda_var(torch.FloatTensor([0.0]))
        else:
            loss_act, entropy_act = self._calc_loss_each_mode(agent_observation_state_ls[ReadPointerAgent.ACT_MODE],
                                                              action_batch[ReadPointerAgent.ACT_MODE])
        loss = loss_read + loss_act
        self.entropy = entropy_read + entropy_act

        return loss

    def _calc_loss_each_mode(self, agent_observation_state_ls, action_batch):

        action_batch = cuda_var(torch.from_numpy(np.array(action_batch)))

        num_states = int(action_batch.size()[0])
        model_prob_batch = self.model.get_probs_batch(agent_observation_state_ls)
        chosen_log_probs = model_prob_batch.gather(1, action_batch.view(-1, 1))

        entropy = -torch.mean(torch.sum(model_prob_batch * torch.exp(model_prob_batch), 1))
        objective = torch.sum(chosen_log_probs) / num_states
        loss = -(objective + self.entropy_coef * entropy)

        return loss, entropy

    def do_train_forced_reading(self, agent, train_dataset, tune_dataset, experiment_name):
        """ Perform training """

        assert isinstance(agent, ReadPointerAgent), "This learning algorithm works only with READPointerAgent"

        dataset_size = len(train_dataset)

        for epoch in range(1, self.max_epoch + 1):

            logging.info("Starting epoch %d", epoch)
            action_counts = dict()
            action_counts[ReadPointerAgent.READ_MODE] = [0] * 2
            action_counts[ReadPointerAgent.ACT_MODE] = [0] * self.action_space.num_actions()

            # Test on tuning data
            agent.test_forced_reading(tune_dataset, tensorboard=self.tensorboard)

            batch_replay_items = []
            total_reward = 0
            episodes_in_batch = 0

            for data_point_ix, data_point in enumerate(train_dataset):

                if (data_point_ix + 1) % 100 == 0:
                    logging.info("Done %d out of %d", data_point_ix, dataset_size)
                    logging.info("Training data action counts %r", action_counts)

                image, metadata = agent.server.reset_receive_feedback(data_point)
                pose = int(metadata["y_angle"] / 15.0)
                oracle_segments = data_point.get_instruction_oracle_segmented()
                state = AgentObservedState(instruction=data_point.instruction,
                                           config=self.config,
                                           constants=self.constants,
                                           start_image=image,
                                           previous_action=None,
                                           pose=pose)

                mode = ReadPointerAgent.READ_MODE
                current_segment_ix = 0

                trajectories = data_point.get_sub_trajectory_list()
                action_ix = 0

                while True:

                    if mode == ReadPointerAgent.READ_MODE:
                        # Find the number of tokens to read for the gold segment
                        num_segment_size = len(oracle_segments[current_segment_ix])
                        current_segment_ix += 1
                        for i in range(0, num_segment_size):
                            state = state.update_on_read()
                        mode = ReadPointerAgent.ACT_MODE

                    elif mode == ReadPointerAgent.ACT_MODE:

                        if action_ix == len(trajectories[current_segment_ix - 1]):
                            action = agent.action_space.get_stop_action_index()
                            action_ix = 0
                        else:
                            action = trajectories[current_segment_ix - 1][action_ix]
                            action_ix += 1

                        action_counts[mode][action] += 1

                        if action == agent.action_space.get_stop_action_index():
                            if state.are_tokens_left_to_be_read():
                                # Add to replay memory
                                replay_item = ReplayMemoryItem(state, agent.action_space.get_stop_action_index(),
                                                               1.0, mode)
                                batch_replay_items.append(replay_item)

                                mode = ReadPointerAgent.READ_MODE
                                agent.server.force_goal_update()
                                state = state.update_on_act_halt()
                            else:
                                break
                        else:
                            image, reward, metadata = agent.server.send_action_receive_feedback(action)

                            # Store it in the replay memory list
                            replay_item = ReplayMemoryItem(state, action, 1, mode=mode)
                            batch_replay_items.append(replay_item)

                            # Update the agent state
                            pose = int(metadata["y_angle"] / 15.0)
                            state = state.update(image, action, pose=pose)
                            total_reward += reward

                    else:
                        raise AssertionError("Mode should be either read or act. Unhandled mode: " + str(mode))

                assert mode == ReadPointerAgent.ACT_MODE, "Agent should end on Act Mode"

                # Send final STOP action and get feedback
                image, reward, metadata = agent.server.halt_and_receive_feedback()
                total_reward += reward

                # Store it in the replay memory list
                replay_item = ReplayMemoryItem(state, agent.action_space.get_stop_action_index(), 1, mode)
                batch_replay_items.append(replay_item)

                # Update the scores based on meta_data
                # self.meta_data_util.log_results(metadata)

                # Perform update
                episodes_in_batch += 1
                if episodes_in_batch == 1:
                    loss_val = self.do_update(batch_replay_items)
                    batch_replay_items = []
                    entropy_val = float(self.entropy.data[0])
                    self.tensorboard.log(entropy_val, loss_val, total_reward)
                    total_reward = 0
                    episodes_in_batch = 0

                self.tensorboard.log_train_error(metadata["error"])

            # Save the model
            self.model.save_model(
                experiment_name + "/ml_estimation_epoch_" + str(epoch))

            logging.info("Training data action counts %r", action_counts)
