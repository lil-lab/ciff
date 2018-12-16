import torch
import torch.nn as nn
import torch.nn.functional as F

from agents.agent_with_read import ReadPointerAgent


class GoalPredictionModule(nn.Module):
    """
    pytorch module for predicting the goal from state representation
    """
    def __init__(self, total_emb_size):
        super(GoalPredictionModule, self).__init__()
        self.dense_theta = nn.Linear(total_emb_size, 24)
        self.dense_r = nn.Linear(total_emb_size, 15)

    def forward(self, state_representation):
        return F.log_softmax(self.dense_theta(state_representation)), \
               F.log_softmax(self.dense_r(state_representation))