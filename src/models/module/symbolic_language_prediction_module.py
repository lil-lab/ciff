import torch
import torch.nn as nn
import torch.nn.functional as F

from agents.agent_with_read import ReadPointerAgent


NO_BUCKETS = 48
BUCKET_WIDTH = 7.5


class SymbolicLanguagePredictionModule(nn.Module):
    """
    pytorch module for final part of model
    combines embeddings of image, text, and previous action
    """
    def __init__(self, total_emb_size):
        super(SymbolicLanguagePredictionModule, self).__init__()
        self.dense_landmark = nn.Linear(total_emb_size, 67) #63)
        self.dense_theta_1 = nn.Linear(total_emb_size, NO_BUCKETS)
        self.dense_theta_2 = nn.Linear(total_emb_size, NO_BUCKETS)
        self.dense_r = nn.Linear(total_emb_size, 15)

    def forward(self, text_emb):
        return F.log_softmax(self.dense_landmark(text_emb)), \
               F.log_softmax(self.dense_theta_1(text_emb)), \
               F.log_softmax(self.dense_theta_2(text_emb)), \
               F.log_softmax(self.dense_r(text_emb))
