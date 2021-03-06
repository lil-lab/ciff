import torch
import torch.nn as nn
import torch.nn.functional as F

from agents.agent_with_read import ReadPointerAgent


class DistMultimodalRecurrentSimpleModule(nn.Module):
    """
    pytorch module for final part of model
    combines embeddings of image, text, and previous action
    """
    def __init__(self, image_module, text_module, action_module,
                 image_recurrence_module,
                 total_emb_size, num_actions):
        super(DistMultimodalRecurrentSimpleModule, self).__init__()
        self.image_module = image_module
        self.image_recurrence_module = image_recurrence_module
        self.text_module = text_module
        self.action_module = action_module
        self.dense1 = nn.Linear(total_emb_size, 120)
        self.dense2 = nn.Linear(120, 1)

    def forward(self, image, image_lens, instructions, prev_action, mode):
        image_emb_seq = self.image_module(image)
        image_emb = self.image_recurrence_module(image_emb_seq, image_lens)
        text_emb = self.text_module(instructions)
        x = torch.cat([image_emb, text_emb], dim=1)
        x = F.relu(self.dense1(x))
        x = F.sigmoid(self.dense2(x))
        return x
