import json
import logging

import time
import torch
import scipy.misc
import os
import math
import random
import numpy as np

from agents.agent_observed_state import AgentObservedState
from utils.debug_nav_drone_instruction import instruction_to_string
from models.model.abstract_model import AbstractModel
from models.incremental_model.abstract_incremental_model import AbstractIncrementalModel


NO_BUCKETS = 48
BUCKET_WIDTH = 360.0/(1.0*NO_BUCKETS)


class HumanControlledAgent:

    def __init__(self, server, model, test_policy, action_space, meta_data_util,
                 config, constants):
        self.server = server
        self.model = model
        self.test_policy = test_policy
        self.action_space = action_space
        self.meta_data_util = meta_data_util
        self.config = config
        self.constants = constants
        self.vocab = {}
        vocab_path = config["vocab_file"]
        word_index = 0
        with open(vocab_path) as f:
            for line in f.readlines():
                token = line.strip()
                self.vocab[token] = word_index
                word_index += 1

    def convert_to_id(self, instruction):
        tk_seq = instruction.split()
        token_ids = []
        for tk in tk_seq:
            if tk in self.vocab:
                token_ids.append(self.vocab[tk])
            else:
                print("Out of vocabulary word. Ignoring ", tk)
        return token_ids

    def test(self, test_dataset, tensorboard=None, logger=None,
             pushover_logger=None):

        self.server.clear_metadata()
        action_counts = [0] * self.action_space.num_actions()
        task_completion_accuracy = 0
        print("Reached Test")
        test_dataset_size = len(test_dataset)

        metadata = {"feedback": ""}
        data_point = random.sample(test_dataset, 1)[0]
        while True:

            print("Please enter an instruction. For sample see:")
            # data_point = random.sample(test_dataset, 1)[0]
            image, metadata = self.server.reset_receive_feedback(data_point)
            print("Sample instruction: ", instruction_to_string(data_point.get_instruction(), self.config))
            input_instruction = input("Enter an instruction or enter q to quit ")
            if input_instruction == "q" or input_instruction == "quit":
                break
            input_instruction_ids = self.convert_to_id(input_instruction)

            pose = int(metadata["y_angle"] / 15.0)
            position_orientation = (metadata["x_pos"], metadata["z_pos"],
                                    metadata["y_angle"])
            state = AgentObservedState(instruction=input_instruction_ids,
                                       config=self.config,
                                       constants=self.constants,
                                       start_image=image,
                                       previous_action=None,
                                       pose=pose,
                                       position_orientation=position_orientation,
                                       data_point=data_point,
                                       prev_instruction=data_point.get_prev_instruction(),
                                       next_instruction=data_point.get_next_instruction())
            # state.start_read_pointer, state.end_read_pointer = data_point.get_instruction_indices()
            num_actions = 0
            max_num_actions = self.constants["horizon"]
            model_state = None
            # print "Model state is new "
            while True:

                time.sleep(0.3)

                # Generate probabilities over actions
                if isinstance(self.model, AbstractModel):
                    probabilities = list(torch.exp(self.model.get_probs(state).data))
                elif isinstance(self.model, AbstractIncrementalModel):
                    log_probabilities, model_state, _, _ = self.model.get_probs(state, model_state, volatile=True)
                    probabilities = list(torch.exp(log_probabilities.data))[0]
                else:
                    # print "Num action is " + str(num_actions) + " and max is " + str(max_num_actions)
                    log_probabilities, model_state = self.model.get_probs(state, model_state)
                    probabilities = list(torch.exp(log_probabilities.data))
                    # raise AssertionError("Unhandled Model type.")

                # Use test policy to get the action
                action = self.test_policy(probabilities)
                # DONT FORGET TO REMOVE
                # action = np.random.randint(0, 2)
                action_counts[action] += 1

                if action == self.action_space.get_stop_action_index() or num_actions >= max_num_actions:
                    # Send the action and get feedback
                    image, reward, metadata = self.server.halt_and_receive_feedback()
                    if tensorboard is not None:
                        tensorboard.log_all_test_errors(
                            metadata["edit_dist_error"],
                            metadata["closest_dist_error"],
                            metadata["stop_dist_error"])

                    if metadata["stop_dist_error"] < 5.0:
                        task_completion_accuracy += 1

                    # Update the scores based on meta_data
                    self.meta_data_util.log_results(metadata)
                    break
                else:
                    # Send the action and get feedback
                    image, reward, metadata = self.server.send_action_receive_feedback(action)
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

        print("Finished testing. Now logging.")
        task_completion_accuracy = (task_completion_accuracy * 100.0) /float(max(len(test_dataset), 1))
        self.log("Overall test results:", logger)
        self.log("Testing: Task completion accuracy is: %r" % task_completion_accuracy, logger)
        self.log("Testing: Final Metadata: %r" % metadata, logger)
        self.log("Testing: Action Distribution: %r" % action_counts, logger)
        self.meta_data_util.log_results(metadata, logger)
        self.log("Testing data action counts %r" % action_counts, logger)
        if pushover_logger is not None:
            pushover_feedback = str(metadata["feedback"]) + " --- " + "task_completion_accuracy=%r" % task_completion_accuracy
            pushover_logger.log(pushover_feedback)

    def log(self, message, logger=None):
        if logger is not None:
            logger.log(message)
        else:
            logging.info(message)
