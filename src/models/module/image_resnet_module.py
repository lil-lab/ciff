import torch
import torch.nn as nn
import numpy as np
import scipy.misc
import matplotlib.pyplot as plt

import models.module.resnet_blocks as blocks


class ImageResnetModule(torch.nn.Module):

    def __init__(self, image_emb_size, input_num_channels,
                 image_height, image_width, using_recurrence=False):
        super(ImageResnetModule, self).__init__()
        self.input_dims = (input_num_channels, image_height, image_width)
        self.using_recurrence = using_recurrence

        num_channels = 32
        self.norm1 = nn.InstanceNorm2d(input_num_channels)
        self.conv1 = nn.Conv2d(input_num_channels, num_channels, 7, stride=2, padding=3)
        h, w = (image_height // 2, image_width // 2)

        block1 = nn.Sequential(
            blocks.ResBlockStrided(num_channels, num_channels),
            blocks.ResBlock(num_channels)
        )
        h, w = h // 2, w // 2

        block2 = nn.Sequential(
            blocks.ResBlockStrided(num_channels, num_channels),
            blocks.ResBlock(num_channels)
        )
        h, w = h // 2, w // 2

        block3 = nn.Sequential(
            blocks.ResBlockStrided(num_channels, num_channels),
        )
        h, w = h // 2, w // 2

        self.resnet_blocks = nn.Sequential(block1, block2, block3)
        self.norm2 = nn.InstanceNorm2d(num_channels)
        self.dense = nn.Linear(h * w * num_channels, image_emb_size)
        self.global_id = 0

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
        image = x
        x = self.norm1(x)
        x = self.conv1(x)
        x = self.resnet_blocks(x)
        x = self.norm2(x)
        top_layer = x

        x = self.dense(x.view(x.size(0), -1))

        ################################################
        # self.global_id += 1
        # top_layer = torch.log(top_layer)
        # numpy_ar = top_layer.cpu().data.numpy()
        # print "Feature Map shape ", numpy_ar.shape
        # print "IMAGE SHAPE ", image.size()
        # image_flipped = image[-1].cpu().data.numpy().swapaxes(0, 1).swapaxes(1, 2)
        # for i in range(0, 32):
        #     fmap = numpy_ar[-1][i]
        #     fmap = (fmap - np.mean(fmap)) / np.std(fmap)
        #     resized_kernel = scipy.misc.imresize(fmap, (128, 128))
        #     plt.imshow(image_flipped)
        #     plt.imshow(resized_kernel, cmap='jet', alpha=0.5)
        #     plt.savefig("./kernels/" + str(self.global_id) + "_kernel_" + str(i) + ".png")
        #     plt.clf()
        ################################################

        if self.using_recurrence:
            if c == 3:
                x = x.view(b, n, -1)
            else:
                x_1 = x[0:b * n, :]
                x_2 = x[b * n: 2 * b * n, :]
                x_3 = x[2 * b * n: 3 * b * n, :]
                x_4 = x[3 * b * n: 4 * b * n, :]
                x_5 = x[4 * b * n: 5 * b * n, :]
                x_6 = x[5 * b * n: 6 * b * n, :]
                x = torch.cat([x_1, x_2, x_3, x_4, x_5, x_6], dim=1)
                x = x.view(b, n, -1)
        else:
            x = x.view(b, -1)
        return x

    def forward_old(self, x):
        b, n, c, h, w = x.size()
        if self.using_recurrence:
            x = x.view(b * n, 1, c, h, w)
            num_batches = b * n
            num_channels = c
        else:
            num_batches = b
            num_channels = n * c
        x = x.view(num_batches, num_channels, h, w)
        image = x
        x = self.norm1(x)
        x = self.conv1(x)
        x = self.resnet_blocks(x)
        x = self.norm2(x)
        top_layer = x

        x = self.dense(x.view(x.size(0), -1))

        ################################################
        # self.global_id += 1
        # numpy_ar = top_layer.cpu().data.numpy()
        # print "Shape " + str((num_batches, num_channels, h, w))
        # image_flipped = image[-1].cpu().data.numpy().swapaxes(0, 1).swapaxes(1, 2)
        # print image_flipped.shape
        # for i in range(0, 32):
        #     resized_kernel = scipy.misc.imresize(numpy_ar[-1][i], (128, 128))
        #     plt.imshow(image_flipped)
        #     plt.imshow(resized_kernel, cmap='jet', alpha=0.5)
        #     plt.savefig("./kernels_tmp/" + str(self.global_id) + "_kernel_" + str(i) + ".png")
        #     plt.clf()
        ################################################

        if self.using_recurrence:
            x = x.view(b, n, -1)
        else:
            x = x.view(b, -1)
        return x
