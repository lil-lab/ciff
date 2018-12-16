import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


def normalized_columns_initializer(weights, std=1.0):
    out = torch.randn(weights.size())
    out *= std / torch.sqrt(out.pow(2).sum(1, keepdim=True).expand_as(out))
    return out


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


class a3c_lstm_ga_concat_gavector(torch.nn.Module):

    def __init__(self, args, config=None):
        super(a3c_lstm_ga_concat_gavector, self).__init__()
        self.config = config

        # Image Processing
        self.conv1 = nn.Conv2d(3, 128, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(128, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=4, stride=2)

        # Instruction Processing
        self.gru_hidden_size = 256
        self.input_size = args.input_size
        self.embedding = nn.Embedding(self.input_size, 32)
        self.gru = nn.GRU(32, self.gru_hidden_size)

        # Gated-Attention layers
        self.attn_linear = nn.Linear(self.gru_hidden_size, 64)

        # Time embedding layer, helps in stabilizing value prediction
        self.time_emb_dim = 32
        self.time_emb_layer = nn.Embedding(
                args.max_episode_length+1,
                self.time_emb_dim)

        # A3C-LSTM layers
        self.final_image_height = 6
        self.final_image_width = 6
        self.linear = nn.Linear(64 * self.final_image_height * self.final_image_width, 256)
        self.lstm = nn.LSTMCell(256 * 3, 256 * 3)
        self.critic_linear = nn.Linear(256 * 3 + self.time_emb_dim, 1)
        self.actor_linear = nn.Linear(256 * 3 + self.time_emb_dim, 4)

        # Initializing weights
        self.apply(weights_init)
        self.actor_linear.weight.data = normalized_columns_initializer(
            self.actor_linear.weight.data, 0.01)
        self.actor_linear.bias.data.fill_(0)
        self.critic_linear.weight.data = normalized_columns_initializer(
            self.critic_linear.weight.data, 1.0)
        self.critic_linear.bias.data.fill_(0)

        self.lstm.bias_ih.data.fill_(0)
        self.lstm.bias_hh.data.fill_(0)
        self.train()

    def forward(self, inputs, cached_computation=None):
        x, curr_instr, prev_instr, next_instr, (tx, hx, cx) = inputs

        # Get the image representation
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x_image_rep = F.relu(self.conv3(x))

        xs = []
        if cached_computation is None:
            cached_computation = dict()

            i = 0
            for input_inst in (curr_instr, prev_instr, next_instr):
                # Get the instruction representation
                encoder_hidden = Variable(torch.zeros(1, 1, self.gru_hidden_size).cuda())
                for i in range(input_inst.data.size(1)):
                    word_embedding = self.embedding(input_inst[0, i]).unsqueeze(0)
                    _, encoder_hidden = self.gru(word_embedding, encoder_hidden)
                x_instr_rep = encoder_hidden.view(encoder_hidden.size(1), -1)

                # Get the attention vector from the instruction representation
                x_attention = F.sigmoid(self.attn_linear(x_instr_rep))

                # Gated-Attention
                x_attention = x_attention.unsqueeze(2).unsqueeze(3)
                x_attention = x_attention.expand(1, 64, self.final_image_height, self.final_image_width)

                if i == 0:
                    cached_computation["curr_x_attention"] = x_attention
                elif i == 1:
                    cached_computation["prev_x_attention"] = x_attention
                elif i == 2:
                    cached_computation["next_x_attention"] = x_attention
                else:
                    raise AssertionError("i cannot exceed 2")

        curr_x_attention = cached_computation["curr_x_attention"]
        prev_x_attention = cached_computation["curr_x_attention"]
        next_x_attention = cached_computation["curr_x_attention"]

        curr_x = x_image_rep * curr_x_attention
        prev_x = x_image_rep * prev_x_attention
        next_x = x_image_rep * next_x_attention
        curr_x = curr_x.view(curr_x.size(0), -1)
        prev_x = prev_x.view(prev_x.size(0), -1)
        next_x = next_x.view(next_x.size(0), -1)
        xs.append(F.relu(self.linear(curr_x)))
        xs.append(F.relu(self.linear(prev_x)))
        xs.append(F.relu(self.linear(next_x)))

        # A3C-LSTM
        x = torch.cat(xs, 1)
        hx, cx = self.lstm(x, (hx, cx))
        time_emb = self.time_emb_layer(tx)
        x = torch.cat((hx, time_emb.view(-1, self.time_emb_dim)), 1)

        return self.critic_linear(x), self.actor_linear(x), (hx, cx), cached_computation
