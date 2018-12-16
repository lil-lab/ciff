import torch.nn as nn
import torch.nn.functional as F


class ActionPredictionModule(nn.Module):
    """
    pytorch module for predicting the action taken from batch of pair of image embedding and the next embededing
    resulting from taking the action.
    """
    def __init__(self, total_emb_size, hidden_dim, num_actions):
        super(ActionPredictionModule, self).__init__()
        self.dense_1 = nn.Linear(total_emb_size, hidden_dim)
        self.dense_2 = nn.Linear(hidden_dim, num_actions)

    def forward(self, embedding):
        embedding = embedding.view(embedding.size()[0], -1)
        x = F.relu(self.dense_1(embedding))
        x = self.dense_2(x)
        return F.log_softmax(x)
