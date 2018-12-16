import logging
import torch
import time
import numpy as np

from agent_observed_state import AgentObservedState
from models.model.abstract_model import AbstractModel
from models.incremental_model.abstract_incremental_model import AbstractIncrementalModel


class VizDoomAgent:

    def __init__(self, server, model, test_policy, config, constants):
        self.server = server
        self.model = model
        self.test_policy = test_policy
        self.config = config
        self.constants = constants

    def test(self, test_dataset_size, tensorboard=None):

        action_counts = [0] * self.config["num_actions"]  # self.action_space.num_actions()

        num_tasks_done = 0
        total_reward = 0

        reward_sum = 0
        done = True

        episode_length = 0
        rewards_list = []
        accuracy_list = []
        episode_length_list = []
        num_episode = 0
        best_reward = 0.0
        test_freq = 50

        start_time = time.time()

        for data_point_ix in range(test_dataset_size):
            instruction, image, metadata = self.server.reset_receive_feedback()
            state = AgentObservedState(instruction=instruction,
                                       config=self.config,
                                       constants=self.constants,
                                       start_image=image,
                                       previous_action=None)
            # state.start_read_pointer, state.end_read_pointer = data_point.get_instruction_indices()
            num_actions = 0
            model_state = None

            while True:

                episode_length += 1

                # Generate probabilities over actions
                if isinstance(self.model, AbstractModel):
                    probabilities = list(torch.exp(self.model.get_probs(state).data))
                elif isinstance(self.model, AbstractIncrementalModel):
                    log_probabilities, model_state, _, _ = self.model.get_probs(state, model_state, volatile=True)
                    probabilities = list(torch.exp(log_probabilities.data))[0]
                else:
                    probabilities = list(torch.exp(self.model.get_probs(state).data))


                # Use test policy to get the action
                action = self.test_policy(probabilities.cpu().numpy())
                logging.info('Test: probabilities:' + str(probabilities.cpu().numpy()) + ' , action taken: ' + str(action))

                # DONT FORGET TO REMOVE
                # action = np.random.randint(0, 2)
                action_counts[action] += 1

                # Send the action and get feedback
                image, reward, done, metadata = self.server.send_action_receive_feedback(action)
                total_reward += reward

                # Update the agent state
                state = state.update(image, action)
                num_actions += 1

                reward_sum += reward

                if done:

                    num_episode += 1
                    rewards_list.append(reward_sum)

                    print("Total reward: {}".format(reward_sum))

                    episode_length_list.append(episode_length)
                    if metadata["succeeded"]:
                        accuracy = 1
                    else:
                        accuracy = 0
                    accuracy_list.append(accuracy)
                    if (len(rewards_list) >= test_freq):
                        print(" ".join([
                            "Time {},".format(time.strftime("%Hh %Mm %Ss",
                                                            time.gmtime(time.time() - start_time))),
                            "Avg Reward {},".format(np.mean(rewards_list)),
                            "Avg Accuracy {},".format(np.mean(accuracy_list)),
                            "Avg Ep length {},".format(np.mean(episode_length_list)),
                            "Best Reward {}".format(best_reward)]))
                        logging.info(" ".join([
                            "Time {},".format(time.strftime("%Hh %Mm %Ss",
                                                            time.gmtime(time.time() - start_time))),
                            "Avg Reward {},".format(np.mean(rewards_list)),
                            "Avg Accuracy {},".format(np.mean(accuracy_list)),
                            "Avg Ep length {},".format(np.mean(episode_length_list)),
                            "Best Reward {}".format(best_reward)]))

                        rewards_list = []
                        accuracy_list = []
                        episode_length_list = []
                    reward_sum = 0
                    episode_length = 0

                    # Print instruction while evaluating and visualizing
                    print("Instruction: {} ".format(metadata["instruction_plaintext"]))
                    print("Succeeded?: {} ".format(metadata["succeeded"]))

                    if tensorboard is not None:
                        tensorboard.log_all_test_errors(
                            0,0,metadata["stop_dist_error"])
                    break

            num_tasks_done += 1

        logging.info("Overall test result: ")
        logging.info("Number of tasks done %r and total reward %r", num_tasks_done, total_reward)
        logging.info("Testing data action counts %r", action_counts)
