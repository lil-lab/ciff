import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from agents.agent_with_read import ReadPointerAgent


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = np.prod(weight_shape[1:4])
        fan_out = np.prod(weight_shape[2:4]) * weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = weight_shape[1]
        fan_out = weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)


class TmpHouseIncrementalMisraFinal(nn.Module):
    """
    pytorch module for final part of model
    combines embeddings of image, text, and previous action
    """
    def __init__(self, image_module, text_module):
        super(TmpHouseIncrementalMisraFinal, self).__init__()
        self.image_module = image_module
        self.text_module = text_module
        self.linear = nn.Linear(256 * 256, 256)
        self.navigation_layer = nn.Linear(256, 5)
        self.interaction_layer = nn.Linear(256, 32*32)

    @staticmethod
    def normalized_columns_initializer(weights, std=1.0):
        out = torch.randn(weights.size())
        out *= std / torch.sqrt(out.pow(2).sum(1, keepdim=True).expand_as(out))
        return out

    def init_weights(self):

        # Initializing weights
        self.apply(weights_init)

    def forward(self, image, instructions, tx, mode, model_state):

        image_emb = self.image_module(image)

        if model_state is None:
            text_emb, text_emb_raw = self.text_module(instructions)
            model_state = (text_emb, text_emb_raw)
        else:
            text_emb, text_emb_raw = model_state

        x = torch.cat([text_emb_raw.view(1, -1), image_emb.view(1, -1)])
        x = F.relu(self.linear(x))

        navigation_log_prob = F.log_softmax(self.navigation_layer(x).view(-1))
        interaction_log_prob = F.log_softmax(self.interaction_layer(x).view(-1))

        x = navigation_log_prob[0:4]  # forward, left, right and stop
        y = torch.mul(navigation_log_prob[5], interaction_log_prob)
        x = torch.cat([x, y]).view(-1)

        return x, model_state, image_emb, x

