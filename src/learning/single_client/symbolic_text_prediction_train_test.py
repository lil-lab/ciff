import logging
import random
import numpy as np
import torch
import torch.optim as optim
import collections
import utils.generic_policy as gp
import utils.debug_nav_drone_instruction as debug
import utils.nav_drone_symbolic_instructions as nav_drone_symbolic_instructions

from agents.agent_observed_state import AgentObservedState
from agents.symbolic_text_replay_memory_item import SymbolicTextReplayMemoryItem
from abstract_learning import AbstractLearning
from utils.cuda import cuda_var

#For error analysis/printing
from dataset_agreement_nav_drone.nav_drone_dataset_parser import make_vocab_map
from utils.nav_drone_landmarks import get_all_landmark_names
LANDMARK_NAMES = get_all_landmark_names()

NO_BUCKETS = 48
BUCKET_WIDTH = 7.5


class SymbolicTextPredictionTrainTest(AbstractLearning):
    """ Trains model to predict symbolic form from the text """

    def __init__(self, model, action_space, meta_data_util, config, constants, tensorboard):
        self.max_epoch = constants["max_epochs"]
        self.model = model
        self.action_space = action_space
        self.meta_data_util = meta_data_util
        self.config = config
        self.constants = constants
        self.tensorboard = tensorboard
        self.global_replay_memory = collections.deque(maxlen=2000)
        self.optimizer = optim.Adam(model.get_parameters(),
                                    lr=constants["learning_rate"])

        theta_values = []
        for i in range(0, 48):
            theta_values.append([i * 7.5])
        self.theta_values = cuda_var(torch.from_numpy(np.array(theta_values))).float()

        AbstractLearning.__init__(self, self.model, self.calc_loss,
                                  self.optimizer, self.config, self.constants, self.tensorboard)

    def calc_loss(self, batch_replay_items):

        agent_observation_state_ls = []
        landmark = []
        theta_1 = []
        theta_2 = []
        r = []
        for replay_item in batch_replay_items:
            agent_observation_state_ls.append(replay_item.get_agent_observed_state())
            landmark_, theta_1_, theta_2_, r_ = replay_item.get_symbolic_text()
            landmark.append(landmark_)
            theta_1.append(theta_1_)
            theta_2.append(theta_2_)
            r.append(r_)

        num_states = len(agent_observation_state_ls)

        landmark_batch = cuda_var(torch.from_numpy(np.array(landmark)))
        theta_1_batch = cuda_var(torch.from_numpy(np.array(theta_1)))
        theta_2_batch = cuda_var(torch.from_numpy(np.array(theta_2)))
        r_batch = cuda_var(torch.from_numpy(np.array(r)))

        model_prob_landmark, model_prob_theta_1, model_prob_theta_2, model_prob_r \
            = self.model.get_symbolic_text_batch(agent_observation_state_ls)

        # compute expected theta
        model_prob_theta_1_ = torch.exp(model_prob_theta_1)
        model_prob_theta_2_ = torch.exp(model_prob_theta_2)
        expected_theta_1 = torch.matmul(model_prob_theta_1_, self.theta_values)  # batch
        expected_theta_2 = torch.matmul(model_prob_theta_2_, self.theta_values)  # batch

        gold_theta_1 = self.theta_values.gather(0, theta_1_batch.view(-1, 1))
        gold_theta_2 = self.theta_values.gather(0, theta_2_batch.view(-1, 1))

        theta_1_diff_1 = torch.remainder(gold_theta_1 - expected_theta_1, 360)
        theta_1_diff_2 = torch.remainder(expected_theta_1 - gold_theta_1, 360)
        theta_1_diff = torch.min(theta_1_diff_1, theta_1_diff_2)
        theta_1_loss = torch.mean(theta_1_diff ** 2)

        theta_2_diff_1 = torch.remainder(gold_theta_2 - expected_theta_2, 360)
        theta_2_diff_2 = torch.remainder(expected_theta_2 - gold_theta_2, 360)
        theta_2_diff = torch.min(theta_2_diff_1, theta_2_diff_2)
        theta_2_loss = torch.mean(theta_2_diff ** 2)

        chosen_log_probs_landmark = model_prob_landmark.gather(1, landmark_batch.view(-1, 1))
        # chosen_log_probs_theta_1 = model_prob_theta_1.gather(1, theta_1_batch.view(-1, 1))
        # chosen_log_probs_theta_2 = model_prob_theta_2.gather(1, theta_2_batch.view(-1, 1))
        chosen_log_probs_r = model_prob_r.gather(1, r_batch.view(-1, 1))

        cross_entropy_loss_objective = torch.sum(chosen_log_probs_landmark) / num_states \
                    + torch.sum(chosen_log_probs_r) / num_states
        loss = -cross_entropy_loss_objective + 0.0002 * theta_1_loss + 0.0002 * theta_2_loss

        return loss

    def calc_loss_cross_entropy(self, batch_replay_items):

        agent_observation_state_ls = []
        landmark = []
        theta_1 = []
        theta_2 = []
        r = []
        for replay_item in batch_replay_items:
            agent_observation_state_ls.append(replay_item.get_agent_observed_state())
            landmark_, theta_1_, theta_2_, r_ = replay_item.get_symbolic_text()
            landmark.append(landmark_)
            theta_1.append(theta_1_)
            theta_2.append(theta_2_)
            r.append(r_)

        num_states = len(agent_observation_state_ls)

        landmark_batch = cuda_var(torch.from_numpy(np.array(landmark)))
        theta_1_batch = cuda_var(torch.from_numpy(np.array(theta_1)))
        theta_2_batch = cuda_var(torch.from_numpy(np.array(theta_2)))
        r_batch = cuda_var(torch.from_numpy(np.array(r)))

        model_prob_landmark, model_prob_theta_1, model_prob_theta_2, model_prob_r \
            = self.model.get_symbolic_text_batch(agent_observation_state_ls)

        chosen_log_probs_landmark = model_prob_landmark.gather(1, landmark_batch.view(-1, 1))
        chosen_log_probs_theta_1 = model_prob_theta_1.gather(1, theta_1_batch.view(-1, 1))
        chosen_log_probs_theta_2 = model_prob_theta_2.gather(1, theta_2_batch.view(-1, 1))
        chosen_log_probs_r = model_prob_r.gather(1, r_batch.view(-1, 1))

        objective = torch.sum(chosen_log_probs_landmark) / num_states \
                    + torch.sum(chosen_log_probs_theta_1) / num_states \
                    + torch.sum(chosen_log_probs_theta_2) / num_states \
                    + torch.sum(chosen_log_probs_r) / num_states
        loss = -objective

        return loss

    @staticmethod
    def _sub_sample(batch_replay_items):

        sub_sampled_replay_items = []
        for replay_item in batch_replay_items:
            if replay_item.get_action() == 0:  # Forward action
                # reject the sample with 0.75 probability
                if random.random() < 0.75:
                    continue
                else:
                    sub_sampled_replay_items.append(replay_item)
            else:
                sub_sampled_replay_items.append(replay_item)

        return sub_sampled_replay_items

    def sample_from_global_memory(self):
        size = min(32, len(self.global_replay_memory))
        return random.sample(self.global_replay_memory, size)

    def test_classifier(self, agent, test_dataset):

        accuracy = 0
        landmark_accuracy = 0
        theta_1_accuracy = 0
        theta_2_accuracy = 0
        theta_1_regression_accuracy = 0
        theta_2_regression_accuracy = 0
        r_accuracy = 0
        cmatrix_landmark = np.zeros((67, 67))
        cmatrix_theta1 = np.zeros((NO_BUCKETS, NO_BUCKETS))
        cmatrix_theta2 = np.zeros((NO_BUCKETS, NO_BUCKETS))
        cmatrix_range = np.zeros((15, 15))

        for data_point_ix, data_point in enumerate(test_dataset):
            state = AgentObservedState(instruction=data_point.instruction,
                                       config=self.config,
                                       constants=self.constants,
                                       start_image=None,
                                       previous_action=None)

            prob_landmark, prob_theta_1, prob_theta_2, prob_r = self.model.get_symbolic_text_batch([state])
            prob_landmark_float = list(torch.exp(prob_landmark.data)[0])
            prob_theta_1_float = list(torch.exp(prob_theta_1.data)[0])
            prob_theta_2_float = list(torch.exp(prob_theta_2.data)[0])
            prob_r_float = list(torch.exp(prob_r.data)[0])

            landmark = gp.get_argmax_action(prob_landmark_float)
            theta_1 = gp.get_argmax_action(prob_theta_1_float)
            theta_2 = gp.get_argmax_action(prob_theta_2_float)
            r = gp.get_argmax_action(prob_r_float)

            gold_landmark, gold_theta_1, gold_theta_2, gold_r = \
                nav_drone_symbolic_instructions.get_nav_drone_symbolic_instruction_segment(data_point)

            plaintext_sentence = self.get_sentence(data_point.instruction)
            sentence_printed = False
            def direction(angle_binned):
                direction = "FAIL"
                if 42 <= angle_binned or 0 <= angle_binned < 6:
                    direction = "BEHIND OF"
                if 6 <= angle_binned < 18:
                    direction = "RIGHT OF"
                if 18 <= angle_binned < 30:
                    direction = "FRONT OF"
                if 30 <= angle_binned < 42:
                    direction = "LEFT OF"
                return direction

            if gold_landmark == landmark:
                landmark_accuracy += 1
            #else:
                #if not sentence_printed:
                #    print "SENTENCE IS: " + plaintext_sentence
                #    sentence_printed = True
                #print "INCORRECT LANDMARK: " + LANDMARK_NAMES[landmark] + " instead of " + LANDMARK_NAMES[gold_landmark]

            if gold_theta_1 == theta_1:
                theta_1_accuracy += 1

            if gold_theta_2 == theta_2:
                theta_2_accuracy += 1
            elif direction(theta_2) != direction(gold_theta_2):
                if not sentence_printed:
                    print "SENTENCE IS: " + plaintext_sentence
                    sentence_printed = True
                print "INCORRECT THETA: " + direction(theta_2) + "(" + str(theta_2) + ")" + " instead of " + direction(gold_theta_2) + "(" + str(gold_theta_2) + ")"

            theta_1_regression_accuracy += min((gold_theta_1 - theta_1) % NO_BUCKETS, NO_BUCKETS - (gold_theta_1 - theta_1) % NO_BUCKETS)
            theta_2_regression_accuracy += min((gold_theta_2 - theta_2) % NO_BUCKETS, NO_BUCKETS - (gold_theta_2 - theta_2) % NO_BUCKETS)

            if gold_r == r:
                r_accuracy += 1

            if gold_landmark == landmark and gold_theta_1 == theta_1 and gold_theta_2 == theta_2 and gold_r == r:
                accuracy += 1

            # update confusion matrix
            cmatrix_landmark[gold_landmark][landmark] += 1
            cmatrix_theta1[gold_theta_1][theta_1] += 1
            cmatrix_theta2[gold_theta_2][theta_2] += 1
            cmatrix_range[gold_r][r] += 1

        dataset_size = len(test_dataset)
        landmark_accuracy = (landmark_accuracy * 100) / float(max(1, dataset_size))
        theta_1_accuracy = (theta_1_accuracy * 100) / float(max(1, dataset_size))
        theta_2_accuracy = (theta_2_accuracy * 100) / float(max(1, dataset_size))
        theta_1_regression_accuracy = (BUCKET_WIDTH * theta_1_regression_accuracy) / float(max(1, dataset_size))
        theta_2_regression_accuracy = (BUCKET_WIDTH * theta_2_regression_accuracy) / float(max(1, dataset_size))
        r_accuracy = (r_accuracy * 100) / float(max(1, dataset_size))
        accuracy = (accuracy * 100) / float(max(1, dataset_size))

        logging.info(
            "Test accuracy on dataset of size %r is landmark %r, theta1 %r %r angle, theta2 %r %r angle, "
            "r %r, total acc %r",
            dataset_size, landmark_accuracy, theta_1_accuracy, theta_1_regression_accuracy,
            theta_2_accuracy, theta_2_regression_accuracy, r_accuracy, accuracy)
        # logging.info("Landmark Confusion Matrix = ", cmatrix_landmark)
        # logging.info("Theta1 Confusion Matrix = ", cmatrix_theta1)
        # logging.info("Theta2 Confusion Matrix = ", cmatrix_theta2)
        # logging.info("Range Confusion Matrix = ", cmatrix_range)


    def dump_data(self, train_dataset, test_dataset):

        landmark_distribution = [0] * 63
        theta_1_distribution = [0] * NO_BUCKETS
        theta_2_distribution = [0] * NO_BUCKETS
        r_distribution = [0] * 15

        for data_point in train_dataset:
            symbolic_form = nav_drone_symbolic_instructions.get_nav_drone_symbolic_instruction_segment(data_point)
            landmark, theta_1, theta_2, r = symbolic_form
            landmark_distribution[landmark] += 1
            theta_1_distribution[theta_1] += 1
            theta_2_distribution[theta_2] += 1
            r_distribution[r] += 1
            landmark_string = nav_drone_symbolic_instructions.LANDMARK_NAMES[landmark]
            instruction = data_point.get_instruction()
            logging.info("Instruction %r, Symbolic form %r %r %r %r",
                         debug.instruction_to_string(instruction, self.config), landmark_string, theta_1, theta_2, r)

        logging.info("Landmark Distribution %r", landmark_distribution)
        logging.info("Theta 1 Distribution %r", theta_1_distribution)
        logging.info("Theta 2 Distribution %r", theta_2_distribution)
        logging.info("R Distribution %r", r_distribution)

    def get_sentence(self, encoded_sentence):

        #Print out the plain text sentence by recovering vocab indices
        vocab_map = make_vocab_map(self.config["vocab_file"])
        inv_vocab_map = {v: k for k, v in vocab_map.iteritems()}

        plaintext_sentence = [inv_vocab_map[idx] for idx in encoded_sentence]

        return " ".join(plaintext_sentence)

    def do_train(self, agent, train_dataset, test_dataset, experiment_name):
        """ Perform training """

        dataset_size = len(train_dataset)
        clock = 0
        clock_max = 1  # 32

        for epoch in range(1, self.max_epoch + 1):

            logging.info("Starting epoch %d", epoch)
            self.test_classifier(agent, test_dataset)

            for data_point_ix, data_point in enumerate(train_dataset):

                if (data_point_ix + 1) % 100 == 0:
                    logging.info("Done %d out of %d", data_point_ix, dataset_size)

                batch_replay_items = []

                state = AgentObservedState(instruction=data_point.instruction,
                                           config=self.config,
                                           constants=self.constants,
                                           start_image=None,  # image,
                                           previous_action=None)

                # Store it in the replay memory list
                symbolic_form = nav_drone_symbolic_instructions.get_nav_drone_symbolic_instruction_segment(data_point)
                replay_item = SymbolicTextReplayMemoryItem(state, symbolic_form)
                batch_replay_items.append(replay_item)

                # Update the scores based on meta_data
                # self.meta_data_util.log_results(metadata)

                # Subsample the batch replay items
                # batch_replay_items = self._sub_sample(batch_replay_items)

                # if data_point_ix == 2000:
                #     self.test_classifier(agent, test_dataset)
                #
                # if data_point_ix == 6000:
                #     self.test_classifier(agent, test_dataset)

                # Global
                for replay_item in batch_replay_items:
                    self.global_replay_memory.append(replay_item)

                clock += 1
                if clock % clock_max == 0:
                    batch_replay_items = self.sample_from_global_memory()
                    self.global_replay_memory.clear()
                    clock = 0
                    # Perform update
                    loss_val = self.do_update(batch_replay_items)
                    self.tensorboard.log_loglikelihood_position(loss_val)

            # Save the model
            self.model.save_model(experiment_name + "/ml_learning_symbolic_text_prediction_epoch_" + str(epoch))
