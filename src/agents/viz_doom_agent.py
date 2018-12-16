import logging
import torch

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
                if done:
                    if tensorboard is not None:
                        tensorboard.log_all_test_errors(
                            0,0,metadata["stop_dist_error"])
                    break

            num_tasks_done += 1

        logging.info("Overall test result: ")
        logging.info("Number of tasks done %r and total reward %r", num_tasks_done, total_reward)
        logging.info("Testing data action counts %r", action_counts)
