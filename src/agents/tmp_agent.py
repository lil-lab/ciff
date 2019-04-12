import logging
import torch
import nltk
import os
import numpy as np

from agents.agent_observed_state import AgentObservedState
from utils.debug_nav_drone_instruction import instruction_to_string
from models.model.abstract_model import AbstractModel
from models.incremental_model.abstract_incremental_model import AbstractIncrementalModel


NO_BUCKETS = 48
BUCKET_WIDTH = 360.0/(1.0*NO_BUCKETS)


class TmpBlockAgent:

    def __init__(self, server, model, test_policy, action_space, meta_data_util,
                 config, constants):
        self.server = server
        self.model = model
        self.test_policy = test_policy
        self.action_space = action_space
        self.meta_data_util = meta_data_util
        self.config = config
        self.constants = constants

    @staticmethod
    def convert_text_to_indices(text, vocab, ignore_case=True):

        # Tokenize the text
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

    def test(self, test_dataset, tensorboard=None, logger=None,
             pushover_logger=None):

        self.server.clear_metadata()
        action_counts = [0] * self.action_space.num_actions()
        task_completion_accuracy = 0

        metadata = {"feedback": ""}
        sum_bisk_metric = 0
        for data_point_ix, data_point in enumerate(test_dataset):
            image, metadata = self.server.reset_receive_feedback(data_point)
            sum_bisk_metric += metadata["metric"]
            instruction = self.convert_text_to_indices(metadata["instruction"], vocab)
            state = AgentObservedState(instruction=instruction,
                                       config=self.config,
                                       constants=self.constants,
                                       start_image=image,
                                       previous_action=None,
                                       data_point=data_point)
            # state.start_read_pointer, state.end_read_pointer = data_point.get_instruction_indices()
            num_actions = 0
            max_num_actions = self.constants["horizon"]
            model_state = None

            while True:

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
                action_counts[action] += 1

                if action == self.action_space.get_stop_action_index() or num_actions >= max_num_actions:
                    # Send the action and get feedback
                    image, reward, metadata = self.server.halt_and_receive_feedback()
                    # if tensorboard is not None:
                    #     tensorboard.log_all_test_errors(
                    #         metadata["edit_dist_error"],
                    #         metadata["closest_dist_error"],
                    #         metadata["stop_dist_error"])

                    # if metadata["stop_dist_error"] < 5.0:
                    #     task_completion_accuracy += 1

                    # Update the scores based on meta_data
                    # self.meta_data_util.log_results(metadata, logger)
                    break
                else:
                    # Send the action and get feedback
                    image, reward, metadata = self.server.send_action_receive_feedback(action)
                    # Update the agent state
                    state = state.update(
                        image, action, data_point=data_point)
                    num_actions += 1

        self.log("Overall test results:", logger)
        self.log("Mean Bisk Metric %r" % (sum_bisk_metric/float(len(test_dataset))), logger)
        # self.log("Testing: Final Metadata: %r" % metadata, logger)
        self.log("Testing: Action Distribution: %r" % action_counts, logger)
        # self.meta_data_util.log_results(metadata, logger)
        self.log("Testing data action counts %r" % action_counts, logger)
        if pushover_logger is not None:
            pushover_feedback = str(metadata["feedback"])
            pushover_logger.log(pushover_feedback)

    def save_numpy_image(self, test_dataset, vocab, folder):

        self.server.clear_metadata()

        os.mkdir("./block_world_%s_image_data" % folder)
        for data_point_ix, data_point in enumerate(test_dataset):
            image, metadata = self.server.reset_receive_feedback(data_point)
            instruction_string = metadata["instruction"]
            instruction = self.convert_text_to_indices(metadata["instruction"], vocab)
            image, reward, metadata = self.server.halt_and_receive_feedback()
            folder_name = "./block_world_%s_image_data/example_%r" % (folder, data_point_ix + 1)
            os.mkdir(folder_name)
            np.save(folder_name + "/image.npy", image)
            f = open(folder_name + "/instruction.txt", "w")
            f.write(instruction_string + "\n")
            f.write(str(instruction))
            f.flush()
            f.close()

    def test_auto_segmented(self, test_dataset, tensorboard=None,
                            segmenting_type="auto"):
        assert segmenting_type in ("auto", "oracle")
        self.server.clear_metadata()
        action_counts = [0] * self.action_space.num_actions()

        metadata = ""

        for data_point in test_dataset:
            if segmenting_type == "auto":
                segmented_instruction = data_point.get_instruction_auto_segmented()
            else:
                segmented_instruction = data_point.get_instruction_oracle_segmented()
            num_segments = len(segmented_instruction)
            gold_num_actions = len(data_point.get_trajectory())
            horizon = gold_num_actions // num_segments
            horizon += self.constants["max_extra_horizon_auto_segmented"]

            image, metadata = self.server.reset_receive_feedback(data_point)

            instruction = instruction_to_string(
                data_point.get_instruction(), self.config)
            print ("TEST INSTRUCTION: %r" % instruction)
            print ("")

            for instruction_i, instruction in enumerate(segmented_instruction):

                state = AgentObservedState(instruction=instruction,
                                           config=self.config,
                                           constants=self.constants,
                                           start_image=image,
                                           previous_action=None,
                                           prev_instruction=data_point.get_prev_instruction(),
                                           next_instruction=data_point.get_next_instruction)

                num_actions = 0
                # self._save_agent_state(state, num_actions)

                while True:

                    # Generate probabilities over actions
                    probabilities = list(torch.exp(self.model.get_probs(state).data))
                    # print "test probs:", probabilities

                    # Use test policy to get the action
                    action = self.test_policy(probabilities)
                    action_counts[action] += 1

                    # logging.info("Taking action-num=%d horizon=%d action=%s from %s",
                    #              num_actions, max_num_actions, str(action), str(probabilities))

                    if action == self.action_space.get_stop_action_index() or num_actions >= horizon:
                        break

                    else:
                        # Send the action and get feedback
                        image, reward, metadata = self.server.send_action_receive_feedback(action)

                        # Update the agent state
                        state = state.update(image, action)
                        num_actions += 1

            _,  _, metadata = self.server.halt_and_receive_feedback()
            if tensorboard is not None:
                tensorboard.log_test_error(metadata["error"])

        self.meta_data_util.log_results(metadata)
        logging.info("Testing data action counts %r", action_counts)

    def log(self, message, logger=None):
        if logger is not None:
            logger.log(message)
        else:
            logging.info(message)
