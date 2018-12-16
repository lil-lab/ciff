import numpy as np
import scipy.misc
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
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


class a3c_lstm_ga_directionalgating(torch.nn.Module):

    def __init__(self, args, config=None):
        super(a3c_lstm_ga_directionalgating, self).__init__()
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
        self.dir_attn_linear = nn.Linear(self.gru_hidden_size, 6)

        # Time embedding layer, helps in stabilizing value prediction
        self.time_emb_dim = 32
        self.time_emb_layer = nn.Embedding(
                args.max_episode_length+1,
                self.time_emb_dim)

        # A3C-LSTM layers
        self.final_image_height = 6
        self.final_image_width = 6
        self.linear = nn.Linear(64 * self.final_image_height * self.final_image_width, 256)
        self.lstm = nn.LSTMCell(256, 256)
        self.critic_linear = nn.Linear(256 + self.time_emb_dim, 1)
        self.actor_linear = nn.Linear(256 + self.time_emb_dim, 4)

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
        self.global_id = 0

    def save_image(self, image, pre_attention, attention):
        self.global_id += 1
        pre_attention = pre_attention.cpu().data.numpy()   # 64 x 6 x 6
        image_flipped = image[0].cpu().data.numpy().swapaxes(0, 1).swapaxes(1, 2)
        value = attention.cpu().data.numpy()[0]

        values = []
        for i in range(0, 64):
            values.append(value[i][0][0])

        print("Value is ", values)
        max_value = max(values)
        print("Max Value is ", max_value)
        print("Index of max value is ", values.index(max_value))

        ix = values.index(max_value)

        for i in [ix]:#range(0, 64):
            resized_kernel = scipy.misc.imresize(pre_attention[0][i], (128, 128))
            plt.imshow(image_flipped)
            plt.imshow(resized_kernel, cmap='jet', alpha=0.5)
            plt.savefig("./kernels/" + str(self.global_id) + "_kernel_" + str(i) + "_pre_attention.png")
            plt.clf()

    def forward(self, inputs, cached_computation=None):
        x, input_inst, _, _, (tx, hx, cx) = inputs
        image = x

        # Get the image representation
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x_image_rep = F.relu(self.conv3(x))

        if cached_computation is None:
            # Get the instruction representation
            encoder_hidden = Variable(torch.zeros(1, 1, self.gru_hidden_size).cuda())
            for i in range(input_inst.data.size(1)):
                word_embedding = self.embedding(input_inst[0, i]).unsqueeze(0)
                _, encoder_hidden = self.gru(word_embedding, encoder_hidden)
            x_instr_rep = encoder_hidden.view(encoder_hidden.size(1), -1)

            # Get the attention vector from the instruction representation
            x_attention = F.sigmoid(self.attn_linear(x_instr_rep))
            x_dir_attention = F.sigmoid(self.dir_attn_linear(x_instr_rep))

            # Gated-Attention
            x_attention = x_attention.unsqueeze(2).unsqueeze(3)
            x_attention = x_attention.expand(1, 64, self.final_image_height, self.final_image_width)
            x_dir_attention = x_dir_attention.unsqueeze(1).unsqueeze(2)
            x_dir_attention = x_dir_attention.expand(1, 64, self.final_image_height, self.final_image_width)
            cached_computation = dict()
            cached_computation["x_attention"] = x_attention
            cached_computation["x_dir_attention"] = x_dir_attention
        else:
            x_attention = cached_computation["x_attention"]
            x_dir_attention = cached_computation["x_dir_attention"]

        assert x_image_rep.size() == x_attention.size()
        assert x_image_rep.size() == x_dir_attention.size()
        x = x_image_rep*x_attention*x_dir_attention
        # self.save_image(image, x_image_rep, x_attention)
        x = x.view(x.size(0), -1)

        # A3C-LSTM
        x = F.relu(self.linear(x))
        hx, cx = self.lstm(x, (hx, cx))
        time_emb = self.time_emb_layer(tx)
        x = torch.cat((hx, time_emb.view(-1, self.time_emb_dim)), 1)

        return self.critic_linear(x), self.actor_linear(x), (hx, cx), cached_computation
