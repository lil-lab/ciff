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


class ParagraphSegmentationTrainTest(AbstractLearning):
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
        AbstractLearning.__init__(self, self.model, self.calc_loss,
                                  self.optimizer, self.config, self.constants,
                                  self.tensorboard)

    def calc_loss(self, batch_replay_items):
        aos_list = [s for (s, _) in batch_replay_items]
        action_list = [a for (_, a) in batch_replay_items]
        action_batch = cuda_var(torch.from_numpy(np.array(action_list)))
        model_log_probs = self.model.get_segmentation_probs(aos_list)
        chosen_log_probs = model_log_probs.gather(1, action_batch.view(-1, 1))
        loss = -torch.sum(chosen_log_probs) / len(batch_replay_items)
        return loss

    def sample_from_global_memory(self):
        size = min(32, len(self.global_replay_memory))
        return random.sample(self.global_replay_memory, size)

    def test_classifier(self, agent, test_dataset):
        fp, fn, tp, tn = 0, 0, 0, 0
        fn_examples = []
        fp_examples = []
        perfect_segmented_examples = []

        for data_point_ix, data_point in enumerate(test_dataset):
            state = AgentObservedState(instruction=data_point.instruction,
                                       config=self.config,
                                       constants=self.constants,
                                       start_image=None,  # image,
                                       previous_action=None)
            segments = data_point.get_instruction_oracle_segmented()
            segment_lens = [len(s) for s in segments]
            num_mistakes = 0
            for i, seg_len in enumerate(segment_lens):
                segment_instruction = debug.instruction_to_string(segments[i], self.config)
                num_read = 0
                while num_read < seg_len:
                    state = state.update_on_read()
                    num_read += 1
                    candidate_instruction = debug.instruction_to_string(segments[i][:num_read], self.config)
                    model_log_probs = list(self.model.get_segmentation_probs([state]).view(-1).data)
                    pred_action = gp.get_argmax_action(model_log_probs)
                    if num_read < seg_len and pred_action == 0:
                        tn += 1
                    elif num_read < seg_len and pred_action == 1:
                        fp += 1
                        num_mistakes += 1
                        fp_examples.append((candidate_instruction, segment_instruction))
                    elif num_read == seg_len and pred_action == 0:
                        fn += 1
                        num_mistakes += 1
                        fn_examples.append((candidate_instruction, segment_instruction))
                    elif num_read == seg_len and pred_action == 1:
                        tp += 1
                state = state.update_on_act_halt()

            if num_mistakes == 0:
                instruction_strings = []
                for seg in segments:
                    instruction_strings.append(debug.instruction_to_string(seg, self.config))
                perfect_segmented_examples.append(" ----- ".join(instruction_strings))

        # calculate precision
        if fp + tp > 0:
            precision = (tp * 1.0) / (fp + tp)
        else:
            precision = 1.0

        # calculate recall
        if fn + tp > 0:
            recall = (tp * 1.0) / (fn + tp)
        else:
            recall = 1.0

        if precision + recall > 0:
            f1 = (2.0 * precision * recall) / (precision + recall)
        else:
            f1 = 0.0

        # print FP examples
        random.shuffle(fp_examples)
        logging.info("FP EXAMPLES:")
        for ex in fp_examples[:20]:
            logging.info(ex)

        # print FN examples
        random.shuffle(fn_examples)
        logging.info("FN EXAMPLES:")
        for ex in fn_examples[:20]:
            logging.info(ex)

        # print perfect segmented examples
        random.shuffle(perfect_segmented_examples)
        logging.info("PERFECT SEGMENTED EXAMPLES:")
        for ex in perfect_segmented_examples[:20]:
            logging.info(ex)


        logging.info("testing results: precision=%.2f; recall=%f; f1=%.2f" %
                     (precision, recall, f1))

    def test_classifier_baseline(self, agent, test_dataset):
        vocab = {}
        with open(self.config["vocab_file"]) as f:
            for line in f.xreadlines():
                token = line.strip().decode("utf-8")
                token_i = len(vocab)
                vocab[token_i] = token
        fp, fn, tp, tn = 0, 0, 0, 0

        for data_point_ix, data_point in enumerate(test_dataset):
            instruction = data_point.get_instruction()
            state = AgentObservedState(instruction=data_point.instruction,
                                       config=self.config,
                                       constants=self.constants,
                                       start_image=None,  # image,
                                       previous_action=None)
            segments = data_point.get_instruction_oracle_segmented()
            segment_lens = [len(s) for s in segments]
            for seg_len in segment_lens:
                num_read = 0
                while num_read < seg_len:
                    state = state.update_on_read()
                    num_read += 1
                    end_pointer = state.end_read_pointer
                    end_token_id = instruction[end_pointer - 1]
                    end_token = vocab[end_token_id]
                    if end_token == ".":
                        pred_action = 1
                    else:
                        pred_action = 0

                    if num_read < seg_len and pred_action == 0:
                        tn += 1
                    elif num_read < seg_len and pred_action == 1:
                        fp += 1
                    elif num_read == seg_len and pred_action == 0:
                        fn += 1
                    elif num_read == seg_len and pred_action == 1:
                        tp += 1
                state = state.update_on_act_halt()

        # calculate precision
        if fp + tp > 0:
            precision = (tp * 1.0) / (fp + tp)
        else:
            precision = 1.0

        # calculate recall
        if fn + tp > 0:
            recall = (tp * 1.0) / (fn + tp)
        else:
            recall = 1.0

        if precision + recall > 0:
            f1 = (2.0 * precision * recall) / (precision + recall)
        else:
            f1 = 0.0

        logging.info("BASELINE testing results: precision=%.2f; recall=%f; f1=%.2f" %
                     (precision, recall, f1))

    def do_train(self, agent, train_dataset, test_dataset, experiment_name):
        """ Perform training """

        dataset_size = len(train_dataset)
        clock = 0
        clock_max = 1 #32

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
                segments = data_point.get_instruction_oracle_segmented()
                segment_lens = [len(s) for s in segments]
                for seg_len in segment_lens:
                    num_read = 0
                    while num_read < seg_len:
                        state = state.update_on_read()
                        num_read += 1
                        if num_read < seg_len:
                            batch_replay_items.append((state, 0))
                        else:
                            batch_replay_items.append((state, 1))
                    state = state.update_on_act_halt()

                # add to global memory
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
            self.model.save_model(experiment_name + "/mle_segmentation_prediction_epoch_" + str(epoch))
