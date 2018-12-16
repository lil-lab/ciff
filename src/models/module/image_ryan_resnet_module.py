import torch
import torch.nn as nn
import numpy as np
import scipy.misc
import matplotlib.pyplot as plt

import models.module.ryan_resnet_blocks as blocks


class ImageRyanResnetModule(torch.nn.Module):

    def __init__(self, image_emb_size, input_num_channels,
                 image_height, image_width, using_recurrence=False):
        super(ImageRyanResnetModule, self).__init__()
        self.input_dims = (input_num_channels, image_height, image_width)
        self.using_recurrence = using_recurrence

        num_channels = 32
        self.norm1 = nn.InstanceNorm2d(input_num_channels)
        self.conv1 = nn.Conv2d(input_num_channels, num_channels, 7, stride=2, padding=3)
        self.act1 = nn.PReLU(init=0.2)
        h, w = (image_height / 2, image_width / 2)

        block1 = nn.Sequential(
            blocks.ResBlockStrided(num_channels, num_channels),
            blocks.ResBlock(num_channels),
            blocks.ResBlock(num_channels)
        )
        h, w = h / 2, w / 2

        block2 = nn.Sequential(
            blocks.ResBlockStrided(num_channels, num_channels),
            blocks.ResBlock(num_channels),
            blocks.ResBlock(num_channels)
        )
        h, w = h / 2, w / 2

        block3 = nn.Sequential(
            blocks.ResBlockStrided(num_channels, num_channels),
            blocks.ResBlock(num_channels),
            blocks.ResBlock(num_channels)
        )
        h, w = h / 2, w / 2

        self.resnet_blocks = nn.Sequential(block1, block2, block3)
        self.norm2 = nn.InstanceNorm2d(num_channels)
        self.dense = nn.Linear(h * w * num_channels, image_emb_size)
        self.global_id = 0

        self.average_mean = torch.nn.AvgPool2d(kernel_size=8)

    def init_weights(self):

        for block in self.resnet_blocks:
            for element in block:
                element.init_weights()

        torch.nn.init.kaiming_uniform(self.conv1.weight)
        self.conv1.bias.data.fill_(0)

    def fix_resnet(self):
        parameters = self.parameters()
        for parameter in parameters:
            parameter.require_grad = False

    def forward(self, x):
        b, n, c, h, w = x.size()
        if self.using_recurrence:
            if c == 3:
                x = x.view(b * n, 1, c, h, w)
                num_batches = b * n
            else:
                x = x.view(b * n, 1, c, h, w)
                assert c == 18
                side_1 = x[:, :, 0:3, :, :]
                side_2 = x[:, :, 3:6, :, :]
                side_3 = x[:, :, 6:9, :, :]
                side_4 = x[:, :, 9:12, :, :]
                side_5 = x[:, :, 12:15, :, :]
                side_6 = x[:, :, 15:18, :, :]
                x = torch.cat([side_1, side_2, side_3, side_4, side_5, side_6], dim=0)
                num_batches = b * n * 6
            num_channels = 3
        else:
            num_batches = b
            num_channels = n * c
        x = x.view(num_batches, num_channels, h, w)
        x = self.norm1(x)
        x = self.conv1(x)
        x = self.act1(x)
        x = self.resnet_blocks(x)
        x = self.norm2(x)  # batch x 32 x 8 x 8
        x = x.view(num_batches, 32, 8, 8)

        if c == 3:
            x = self.dense(x.view(x.size(0), -1))

        if self.using_recurrence:
            if c == 3:
                x = x.view(b, n, -1)
            else:
                left_half = x[:, :, :, 0:4]
                right_half = x[:, :, :, 4:8]
                left_half = left_half.mean(2).mean(2)  # batch x 32
                right_half = right_half.mean(2).mean(2)  # batch x 32

                bucket_1 = right_half[0 * b * n: 1 * b * n, :]  # bucket 0 is 3rd image's right half
                bucket_2 = left_half[1 * b * n: 2 * b * n, :]
                bucket_3 = right_half[1 * b * n: 2 * b * n, :]
                bucket_4 = left_half[2 * b * n: 3 * b * n, :]
                bucket_5 = right_half[2 * b * n: 3 * b * n, :]
                bucket_6 = left_half[3 * b * n: 4 * b * n, :]
                bucket_7 = right_half[3 * b * n: 4 * b * n, :]
                bucket_8 = left_half[4 * b * n: 5 * b * n, :]
                bucket_9 = right_half[4 * b * n: 5 * b * n, :]
                bucket_10 = left_half[5 * b * n: 6 * b * n, :]
                bucket_11 = right_half[5 * b * n: 6 * b * n, :]
                bucket_12 = left_half[0 * b * n: 1 * b * n, :]  # bucket 12 is 3rd image's left half

                # bucket_1 = right_half[3 * b * n: 4 * b * n, :]  # bucket 0 is 3rd image's right half
                # bucket_2 = left_half[4 * b * n: 5 * b * n, :]
                # bucket_3 = right_half[4 * b * n: 5 * b * n, :]
                # bucket_4 = left_half[5 * b * n: 6 * b * n, :]
                # bucket_5 = right_half[5 * b * n: 6 * b * n, :]
                # bucket_6 = left_half[0 * b * n: 1 * b * n, :]
                # bucket_7 = right_half[0 * b * n: 1 * b * n, :]
                # bucket_8 = left_half[1 * b * n: 2 * b * n, :]
                # bucket_9 = right_half[1 * b * n: 2 * b * n, :]
                # bucket_10 = left_half[2 * b * n: 3 * b * n, :]
                # bucket_11 = right_half[2 * b * n: 3 * b * n, :]
                # bucket_12 = left_half[3 * b * n: 4 * b * n, :]  # bucket 12 is 3rd image's left half

                x = torch.stack([
                    bucket_1, bucket_2, bucket_3, bucket_4, bucket_5, bucket_6,
                    bucket_7, bucket_8, bucket_9, bucket_10, bucket_11, bucket_12], dim=1)  # batch x 12 x 32
                # x = x.view(b, n, -1)
        else:
            x = x.view(b, -1)
        return x
