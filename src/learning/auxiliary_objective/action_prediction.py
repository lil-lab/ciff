import torch
import logging
import numpy as np
import utils.generic_policy as gp

from utils.cuda import cuda_var


class ActionPrediction:
    """ Predict Action from Pair of Temporally Consecutive Image Features """

    def __init__(self, model):
        self.model = model
        self.wrong = [0] * 4
        self.correct = [0] * 4

    def predict_action(self, batch_replay_items):

        if len(batch_replay_items) <= 1:
            return None

        num_items = len(batch_replay_items)
        action_batch = []
        batch_input = []

        for replay_item in batch_replay_items:
            next_image_emb = replay_item.get_next_image_emb()
            if next_image_emb is None:  # sometimes it can None for the last item in a rollout
                continue
            action_batch.append(replay_item.get_action())
            image_emb = replay_item.get_image_emb()
            x = torch.cat([image_emb, next_image_emb], 2)
            batch_input.append(x)

        batch_input = torch.cat(batch_input)
        model_log_prob_batch = self.model.action_prediction_log_prob(batch_input)

        log_prob = list(model_log_prob_batch.data)
        for i in range(0, num_items - 1):
            predicted_action = gp.get_argmax_action(log_prob[i])
            if action_batch[i] != predicted_action:
                self.wrong[action_batch[i]] += 1
            else:
                self.correct[action_batch[i]] += 1
            logging.info("Was %r and predicted %r, wrong %r, correct %r",
                         action_batch[i], predicted_action, self.wrong, self.correct)

    def calc_loss(self, batch_replay_items):

        action_batch = []
        batch_input = []

        for replay_item in batch_replay_items:
            next_image_emb = replay_item.get_next_image_emb()
            if next_image_emb is None:  # sometimes it can None for the last item in a rollout
                continue
            action_batch.append(replay_item.get_action())
            image_emb = replay_item.get_image_emb()
            x = torch.cat([image_emb, next_image_emb], 2)
            batch_input.append(x)

        batch_input = torch.cat(batch_input)
        action_batch = cuda_var(torch.from_numpy(np.array(action_batch)))
        model_log_prob_batch = self.model.action_prediction_log_prob(batch_input)
        chosen_log_probs = model_log_prob_batch.gather(1, action_batch.view(-1, 1))
        action_prediction_loss = -torch.mean(chosen_log_probs)

        return action_prediction_loss
