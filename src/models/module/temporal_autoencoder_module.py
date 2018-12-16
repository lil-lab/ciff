import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalAutoencoderModule(nn.Module):
    """
    pytorch module for predicting the feature of the next image from feature of the current image and the action.
    """
    def __init__(self, action_embedding_module, image_emb_size, action_emb_size, hidden_dim):
        super(TemporalAutoencoderModule, self).__init__()
        self.action_embedding_module = action_embedding_module
        self.dense_1 = nn.Linear(image_emb_size + action_emb_size, hidden_dim)
        self.dense_2 = nn.Linear(hidden_dim, image_emb_size)

    def forward(self, current_image_embedding, action):

        action_embedding = self.action_embedding_module(action)
        current_image_embedding = current_image_embedding.view(current_image_embedding.size()[0], -1)

        x = torch.cat([current_image_embedding, action_embedding], 1)
        x = F.relu(self.dense_1(x))
        x = self.dense_2(x)
        return x
