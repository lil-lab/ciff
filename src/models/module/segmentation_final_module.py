import torch.nn as nn
import torch.nn.functional as F


class SegmentationFinalModule(nn.Module):
    """
    pytorch module for final part of model for text segmentation prediction
    """
    def __init__(self, text_module, text_emb_size):
        super(SegmentationFinalModule, self).__init__()
        self.text_module = text_module
        self.dense1 = nn.Linear(text_emb_size, 512)
        self.dense2 = nn.Linear(512, 2)

    def forward(self, instructions):
        x = self.text_module(instructions)
        x = F.relu(self.dense1(x))
        return F.log_softmax(self.dense2(x))
