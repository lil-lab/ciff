import torch
import torch.nn as nn
import torch.nn.functional as F

from agents.agent_with_read import ReadPointerAgent


class MultimodalSimpleWithStopModule(nn.Module):
    """
    pytorch module for final part of model
    combines embeddings of image, text, and previous action
    """
    def __init__(self, image_module, text_module, action_module,
                 total_emb_size, num_actions):
        super(MultimodalSimpleWithStopModule, self).__init__()
        self.image_module = image_module
        self.text_module = text_module
        self.action_module = action_module
        self.dense1 = nn.Linear(total_emb_size, 512)
        self.dense2 = nn.Linear(512, num_actions)
        self.dense_penul_stop = nn.Linear(total_emb_size, 256)
        self.dense_stop = nn.Linear(256, 2)
        self.dense_read = nn.Linear(512, 2)

    def forward(self, image, instructions, prev_action, mode):
        image_emb = self.image_module(image)
        text_emb = self.text_module(instructions)
        action_emb = self.action_module(prev_action)
        x = torch.cat([image_emb, text_emb, action_emb], dim=1)
        y = F.relu(self.dense1(x))
        z = F.relu(self.dense_penul_stop(x))
        if mode is None or mode == ReadPointerAgent.ACT_MODE:
            return F.log_softmax(self.dense2(y)), F.log_softmax(self.dense_stop(z))
        elif mode == ReadPointerAgent.READ_MODE:
            return F.log_softmax(self.dense_read(y))
        else:
            raise ValueError("invalid mode for model: %r" % mode)
