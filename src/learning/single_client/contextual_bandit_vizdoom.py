import numpy as np
import torch
import torch.optim as optim
import logging

import utils.generic_policy as gp
from abstract_learning import AbstractLearning
from agents.agent_observed_state import AgentObservedState
from agents.replay_memory_item import ReplayMemoryItem
from utils.cuda import cuda_var
from utils.debug_nav_drone_instruction import instruction_to_string
from utils.tensorboard import Tensorboard


class ContextualBandit(AbstractLearning):
    """ Perform Contextual Bandit Learning """

    def __init__(self, model, config, constants, tensorboard):
        self.max_epoch = constants["max_epochs"]
        self.model = model
        self.config = config
        self.constants = constants
        self.tensorboard = tensorboard
        self.entropy_coef = constants["entropy_coefficient"]
        self.optimizer = optim.Adam(model.get_parameters(),
                                    lr=constants["learning_rate"])
        AbstractLearning.__init__(self, self.model, self.calc_loss,
                                  self.optimizer, self.config, self.constants, self.tensorboard)

    def calc_loss(self, batch_replay_items):

        log_probabilities = []
        rewards = []
        action_batch = []
        for replay_item in batch_replay_items:
            log_probabilities.append(replay_item.get_log_prob())
            action_batch.append(replay_item.get_action())
            rewards.append(replay_item.get_reward())

        action_batch = cuda_var(torch.from_numpy(np.array(action_batch)))
        rewards = cuda_var(torch.from_numpy(np.array(rewards))).float()

        num_states = int(action_batch.size()[0])
        model_prob_batch = torch.cat(log_probabilities, dim=0)

        chosen_log_probs = model_prob_batch.gather(1, action_batch.view(-1, 1))
        reward_log_probs = rewards * chosen_log_probs.view(-1)

        entropy = -torch.mean(torch.sum(model_prob_batch * torch.exp(model_prob_batch), 1))
        objective = torch.sum(reward_log_probs) / num_states
        loss = -(objective + self.entropy_coef * entropy)
        self.entropy = entropy

        return loss

    def do_train(self, agent, experiment_name):
        """ Perform training """
        print("in training")

        for epoch in range(1, self.max_epoch + 1):

            logging.info("Starting epoch %r", epoch)
            # Test on tuning data
            # switch instruction set to test
            agent.server.env.switch_instructions_set('test')
            agent.test(30, tensorboard=self.tensorboard)
            agent.server.env.switch_instructions_set('train')

            for i in range(0, 500):

                batch_replay_items = []
                num_actions = 0
                total_reward = 0

                instruction, image, metadata = agent.server.reset_receive_feedback()
                state = AgentObservedState(instruction=instruction,
                                           config=self.config,
                                           constants=self.constants,
                                           start_image=image,
                                           previous_action=None)

                model_state = None
                while True:

                    # Sample action using the policy
                    # Generate probabilities over actions
                    log_probabilities, model_state, _, _ = self.model.get_probs(state, model_state)
                    probabilities = list(torch.exp(log_probabilities.data))

                    # Use test policy to get the action
                    action = gp.sample_action_from_prob(probabilities[0])
                    # logging.info('Train: probabilities:' + str(probabilities[0].cpu().numpy()) + ' , action taken: ' + str(action))

                    # Send the action and get feedback
                    image, reward, done, metadata = agent.server.send_action_receive_feedback(action, num_actions)
                    total_reward += reward

                    # Store it in the replay memory list
                    replay_item = ReplayMemoryItem(state, action, reward, log_prob=log_probabilities)
                    batch_replay_items.append(replay_item)

                    # Update the agent state
                    state = state.update(image, action)

                    num_actions += 1
                    if done:
                        break

                # Perform update
                loss_val = self.do_update(batch_replay_items)
                entropy_val = float(self.entropy.data[0])
                self.tensorboard.log(entropy_val, loss_val, total_reward)

            # Save the model
            self.model.save_model(experiment_name + "/contextual_bandit_epoch_" + str(epoch))
