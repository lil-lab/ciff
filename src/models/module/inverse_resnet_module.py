import torch
import torch.nn as nn
import scipy.misc
import matplotlib.pyplot as plt

import models.module.resnet_blocks as blocks


class InverseResnetModule(torch.nn.Module):

    def __init__(self, image_emb_size, input_num_channels,
                 image_height, image_width):
        super(InverseResnetModule, self).__init__()
        h, w = image_height / 16, image_width / 16
        num_channels = 32
        self.c, self.h, self.w = num_channels, h, w
        self.dense = nn.Linear(image_emb_size, h * w * num_channels)
        block1 = blocks.InvResBlockStrided(num_channels, num_channels)
        block2 = nn.Sequential(
            blocks.InvResBlock(num_channels),
            blocks.InvResBlockStrided(num_channels, num_channels),
        )
        block3 = nn.Sequential(
            blocks.InvResBlock(num_channels),
            blocks.InvResBlockStrided(num_channels, num_channels),
        )
        self.inv_resnet_blocks = nn.Sequential(block1, block2, block3)
        self.deconv = nn.ConvTranspose2d(num_channels, input_num_channels,
                                         7, stride=2, padding=3,
                                         output_padding=1)

    def init_weights(self):

        for block in self.resnet_blocks:
            for element in block:
                element.init_weights()

        torch.nn.init.kaiming_uniform(self.deconv.weight)
        self.deconv.bias.data.fill_(0)

    def forward(self, x):
        """
        takes a batch of image embeddings (batch x embedding-dim)
        returns a batch of images (batch x channel x height x width)

        :param x:
        :return:
        """
        batch_size = int(x.data.shape[0])
        x = self.dense(x).view(batch_size, self.c, self.h, self.w)
        x = self.inv_resnet_blocks(x)
        x = self.deconv(x)
        return x

