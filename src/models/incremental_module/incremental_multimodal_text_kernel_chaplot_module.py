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


class IncrementalMultimodalTextKernelChaplotModule(nn.Module):
    """
    pytorch module for final part of model
    combines embeddings of image, text, and previous action
    """
    def __init__(self, image_module, text_module, image_recurrence_module,
                 max_episode_length, final_image_height, final_image_width):
        super(IncrementalMultimodalTextKernelChaplotModule, self).__init__()
        self.image_module = image_module
        self.image_recurrence_module = image_recurrence_module
        self.text_module = text_module
        self.dense_read = nn.Linear(512, 2)

        # Time embedding layer, helps in stabilizing value prediction
        self.time_emb_dim = 32
        self.time_emb_layer = nn.Embedding(max_episode_length+1, self.time_emb_dim)

        # Text kernel reshaping term
        self.text_kernel = nn.Linear(64 * 64 * 3 * 3, 256, bias=True)
        self.text_kernel_bias = nn.Linear(64, 256, bias=True)

        # A3C-LSTM layers
        self.final_image_height = final_image_height
        self.final_image_width = final_image_width
        self.linear = nn.Linear(64 * self.final_image_height * self.final_image_width, 256)
        self.critic_linear = nn.Linear(256 + self.time_emb_dim, 1)
        self.actor_linear = nn.Linear(256 + self.time_emb_dim, 4)

    @staticmethod
    def normalized_columns_initializer(weights, std=1.0):
        out = torch.randn(weights.size())
        out *= std / torch.sqrt(out.pow(2).sum(1, keepdim=True).expand_as(out))
        return out

    def init_weights(self):

        # Initializing weights
        self.apply(weights_init)
        # self.actor_linear.weight.data = self.normalized_columns_initializer(
        #     self.actor_linear.weight.data, 0.01)
        # self.actor_linear.bias.data.fill_(0)
        # self.critic_linear.weight.data = self.normalized_columns_initializer(
        #     self.critic_linear.weight.data, 1.0)
        # self.critic_linear.bias.data.fill_(0)

    def forward(self, image, instructions, tx, mode, model_state):

        image_emb_seq = self.image_module(image)
        image_emb = image_emb_seq[:, 0, :, :, :]

        if model_state is None:
            text_emb, text_emb_raw = self.text_module(instructions)
            text_weight = torch.matmul(text_emb_raw, self.text_kernel.weight)
            text_bias = torch.matmul(text_emb_raw, self.text_kernel_bias.weight)
            text_weight = text_weight.view(64, 64, 3, 3)
            text_bias = text_bias.view(64)
            image_hidden_states = None
        else:
            text_weight, text_bias, image_hidden_states = model_state

        # apply a non-strided convolution to the map
        x = F.conv2d(image_emb, weight=text_weight, bias=text_bias, stride=1, padding=1)
        x = x.view(x.size(0), -1)

        # A3C-LSTM
        x = F.relu(self.linear(x))
        new_image_hidden_states = self.image_recurrence_module(x, image_hidden_states)
        hx, cx = new_image_hidden_states

        time_emb = self.time_emb_layer(tx)
        x = torch.cat((hx, time_emb.view(-1, self.time_emb_dim)), 1)

        new_model_state = (text_weight, text_bias, new_image_hidden_states)

        actor_linear = self.actor_linear(x)
        critic_linear = self.critic_linear(x)
        volatile = dict()

        if mode is None or mode == ReadPointerAgent.ACT_MODE:
            return F.log_softmax(actor_linear, dim=1), new_model_state, image_emb_seq, volatile
        elif mode == ReadPointerAgent.READ_MODE:
            return F.log_softmax(self.dense_read(x)), new_model_state, image_emb_seq, volatile
        else:
            raise ValueError("invalid mode for model: %r" % mode)

