import numpy as np
import torch
import torch.optim as optim

import utils.generic_policy as gp
from abstract_learning import AbstractLearning
from agents.agent_observed_state import AgentObservedState
from agents.replay_memory_item import ReplayMemoryItem
from utils.cuda import cuda_var
from utils.debug_nav_drone_instruction import instruction_to_string
from utils.tensorboard import Tensorboard


class ReinforceLearning(AbstractLearning):
    """ Perform Reinforce Learning (Williams 1992) """

    def __init__(self, model, action_space, meta_data_util, config, constants):
        self.max_epoch = constants["max_epochs"]
        self.model = model
        self.action_space = action_space
        self.meta_data_util = meta_data_util
        self.config = config
        self.constants = constants
        self.tensorboard = Tensorboard()
        self.entropy_coef = constants["entropy_coefficient"]
        self.optimizer = optim.Adam(model.get_parameters(),
                                    lr=constants["learning_rate"])
        AbstractLearning.__init__(self, self.model, self.calc_loss,
                                  self.optimizer, self.config, self.constants)

    def calc_loss(self, batch_replay_items):

        agent_observation_state_ls = []
        q_values = []
        action_batch = []
        for replay_item in batch_replay_items:
            agent_observation_state_ls.append(replay_item.get_agent_observed_state())
            action_batch.append(replay_item.get_action())
            q_values.append(replay_item.get_q_val())

        action_batch = cuda_var(torch.from_numpy(np.array(action_batch)))
        q_values = cuda_var(torch.from_numpy(np.array(q_values)))

        num_states = int(action_batch.size()[0])
        model_prob_batch = self.model.get_probs_batch(agent_observation_state_ls)
        chosen_log_probs = model_prob_batch.gather(1, action_batch.view(-1, 1))
        reward_log_probs = q_values * chosen_log_probs.view(-1)

        entropy = -torch.mean(torch.sum(model_prob_batch * torch.exp(model_prob_batch), 1))
        objective = torch.sum(reward_log_probs) / num_states
        loss = -(objective + self.entropy_coef * entropy)
        self.entropy = entropy

        return loss

    @staticmethod
    def _set_q_val(batch_replay_items):

        rewards = [replay_item.get_reward() for replay_item in batch_replay_items]

        for ix, replay_item in enumerate(batch_replay_items):
            q_val = sum(rewards[ix:])
            replay_item.set_q_val(q_val)

    def do_train(self, agent, train_dataset, tune_dataset, experiment_name):
        """ Perform training """

        for epoch in range(1, self.max_epoch + 1):

            # Test on tuning data
            agent.test(tune_dataset, tensorboard=self.tensorboard)

            for data_point in train_dataset:

                batch_replay_items = []
                num_actions = 0
                total_reward = 0
                max_num_actions = len(data_point.get_trajectory())
                max_num_actions += self.constants["max_extra_horizon"]

                image, metadata = agent.server.reset_receive_feedback(data_point)
                state = AgentObservedState(instruction=data_point.instruction,
                                           config=self.config,
                                           constants=self.constants,
                                           start_image=image,
                                           previous_action=None)

                forced_stop = True

                instruction = instruction_to_string(
                    data_point.get_instruction(), self.config)
                print "TRAIN INSTRUCTION: %r" % instruction
                print ""

                while num_actions < max_num_actions:

                    # Sample action using the policy
                    # Generate probabilities over actions
                    probabilities = list(torch.exp(self.model.get_probs(state).data))

                    # Use test policy to get the action
                    action = gp.sample_action_from_prob(probabilities)

                    if action == agent.action_space.get_stop_action_index():
                        forced_stop = False
                        break

                    # Send the action and get feedback
                    image, reward, metadata = agent.server.send_action_receive_feedback(action)
                    total_reward += reward

                    # Store it in the replay memory list
                    replay_item = ReplayMemoryItem(state, action, reward)
                    batch_replay_items.append(replay_item)

                    # Update the agent state
                    state = state.update(image, action)

                    num_actions += 1

                # Send final STOP action and get feedback
                image, reward, metadata = agent.server.halt_and_receive_feedback()
                total_reward += reward

                # Store it in the replay memory list
                if not forced_stop:
                    replay_item = ReplayMemoryItem(state, agent.action_space.get_stop_action_index(), reward)
                    batch_replay_items.append(replay_item)

                # Update the scores based on meta_data
                # self.meta_data_util.log_results(metadata)

                # Compute Q-values using sampled rollout
                ReinforceLearning._set_q_val(batch_replay_items)

                # Perform update
                loss_val = self.do_update(batch_replay_items)
                entropy_val = float(self.entropy.data[0])
                self.tensorboard.log(entropy_val, loss_val, total_reward)
                self.tensorboard.log_train_error(metadata["error"])

            # Save the model
            self.model.save_model(experiment_name + "/reinforce_epoch_" + str(epoch))
