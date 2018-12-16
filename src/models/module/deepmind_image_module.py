import torch
import torch.nn as nn
import numpy as np
import scipy.misc
import matplotlib.pyplot as plt
import torch.nn.functional as F


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


class DeepMindImageModule(torch.nn.Module):

    def __init__(self, image_emb_size, input_num_channels,
                 image_height, image_width, using_recurrence=False):
        super(DeepMindImageModule, self).__init__()

        self.input_num_channels = input_num_channels
        self.norm1 = nn.InstanceNorm2d(input_num_channels)
        self.input_dims = (input_num_channels, image_height, image_width)
        self.conv1 = nn.Conv2d(input_num_channels, 16, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2)
        self.linear = nn.Linear(32 * 9 * 9, image_emb_size)
        self.using_recurrence = using_recurrence
        self.global_id = 0

    def init_weights(self):
        self.apply(weights_init)

    def fix(self):
        parameters = self.parameters()
        for parameter in parameters:
            parameter.require_grad = False

    def forward(self, x):
        b, n, c, h, w = x.size()

        if self.using_recurrence:
            x = x.view(b * n, 1, c, h, w)
            num_batches = b * n
            num_channels = self.input_num_channels
        else:
            num_batches = b
            num_channels = n * c
        x = x.view(num_batches, num_channels, h, w)

        x = self.norm1(x)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x_image_rep = self.linear(x.view(num_batches, -1))

        if self.using_recurrence:
            x_image_rep = x_image_rep.view(b, n, -1)
        else:
            x_image_rep = x_image_rep.view(b, -1)

        return x_image_rep
