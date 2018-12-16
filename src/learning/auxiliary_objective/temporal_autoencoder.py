import torch
import numpy as np

from utils.cuda import cuda_var


class TemporalAutoEncoder:

    def __init__(self, model):
        self.model = model

    def calc_loss(self, batch_replay_items):

        if len(batch_replay_items) <= 1:
            return None

        action_batch = []
        batch_image_feature = []
        batch_next_image_feature = []

        for replay_item in batch_replay_items:
            next_image_emb = replay_item.get_next_image_emb()
            if next_image_emb is None:  # sometimes it can None for the last item in a rollout
                continue
            action_batch.append(replay_item.get_action())
            batch_image_feature.append(replay_item.get_image_emb())
            batch_next_image_feature.append(next_image_emb)

        action_batch = cuda_var(torch.from_numpy(np.array(action_batch)))
        batch_image_feature = torch.cat(batch_image_feature)
        batch_next_image_feature = torch.cat(batch_next_image_feature)

        # Predict the feature of next image
        batch_predicted_next_image_feature = self.model.predict_action_result(batch_image_feature, action_batch)

        # Compute the squared mean loss
        diff = (batch_predicted_next_image_feature - batch_next_image_feature)
        temporal_autoencoding_loss = torch.mean(diff ** 2)

        return temporal_autoencoding_loss
