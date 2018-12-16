import torch
import torch.nn as nn
import torch.nn.functional as F

from agents.agent_with_read import ReadPointerAgent


NO_BUCKETS = 48
BUCKET_WIDTH = 7.5

class TextClassificationModule(nn.Module):
    """
    pytorch module for final part of model
    combines embeddings of image, text, and previous action
    """
    def __init__(self, text_module, total_emb_size):
        super(TextClassificationModule, self).__init__()
        self.text_module = text_module
        self.dense_landmark = nn.Linear(total_emb_size, 67)
        self.dense_theta_1 = nn.Linear(total_emb_size, NO_BUCKETS)
        self.dense_theta_2 = nn.Linear(total_emb_size, NO_BUCKETS)
        self.dense_r = nn.Linear(total_emb_size, 15)

    def forward(self, instructions):
        text_emb = self.text_module(instructions)
        return F.log_softmax(self.dense_landmark(text_emb)), \
               F.log_softmax(self.dense_theta_1(text_emb)), \
               F.log_softmax(self.dense_theta_2(text_emb)), \
               F.log_softmax(self.dense_r(text_emb))
