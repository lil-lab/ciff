import torch
import numpy as np

from utils.cuda import cuda_var


class SymbolicLanguagePrediction:
    """ Predict Action from Pair of Temporally Consecutive Image Features """

    def __init__(self, model):
        self.model = model
        self.wrong = [0] * 4
        self.correct = [0] * 4

    def calc_loss(self, batch_replay_items):

        if len(batch_replay_items) == 0:
            return None

        landmark_batch = []
        theta_1_batch = []
        theta_2_batch = []
        r_batch = []
        for replay_item in batch_replay_items:
            landmark_, theta_1_, theta_2_, r_ = replay_item.symbolic_text
            landmark_batch.append(landmark_)
            theta_1_batch.append(theta_1_)
            theta_2_batch.append(theta_2_)
            r_batch.append(r_)

        landmark_batch = cuda_var(torch.from_numpy(np.array(landmark_batch)))
        theta_1_batch = cuda_var(torch.from_numpy(np.array(theta_1_batch)))
        theta_2_batch = cuda_var(torch.from_numpy(np.array(theta_2_batch)))
        r_batch = cuda_var(torch.from_numpy(np.array(r_batch)))

        batch_input = [replay_item.get_text_emb() for replay_item in batch_replay_items]
        batch_input = torch.cat(batch_input)
        landmark_log_prob, theta_1_log_prob, theta_2_log_prob, r_log_prob = \
            self.model.get_language_prediction_probs(batch_input)

        chosen_landmark_log_probs = landmark_log_prob.gather(1, landmark_batch.view(-1, 1))
        chosen_theta_1_log_probs = theta_1_log_prob.gather(1, theta_1_batch.view(-1, 1))
        chosen_theta_2_log_probs = theta_2_log_prob.gather(1, theta_2_batch.view(-1, 1))
        chosen_r_log_probs = r_log_prob.gather(1, r_batch.view(-1, 1))

        symbolic_language_prediction_loss = -torch.sum(chosen_landmark_log_probs)\
                                            -torch.sum(chosen_theta_1_log_probs)\
                                            -torch.sum(chosen_theta_2_log_probs)\
                                            -torch.sum(chosen_r_log_probs)
        num_states = float(len(batch_replay_items))
        symbolic_language_prediction_loss = symbolic_language_prediction_loss / num_states

        return symbolic_language_prediction_loss
