import torch
import torch.nn as nn
import numpy as np
import scipy.misc
import matplotlib.pyplot as plt

import models.module.resnet_blocks as blocks


class ImageChaplotResnetModule(torch.nn.Module):

    def __init__(self, image_emb_size, input_num_channels,
                 image_height, image_width, using_recurrence=False):
        super(ImageChaplotResnetModule, self).__init__()
        self.input_dims = (input_num_channels, image_height, image_width)
        self.using_recurrence = using_recurrence
        self.input_num_channels = input_num_channels

        num_channels = 64
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
            assert c == self.input_num_channels, "Expected %r channels. Got %r " % (self.input_num_channels, c)
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
        top_layer = x

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
            x = x.view(b, n, 64, 8, 8)
        else:
            x = x.view(b, 64, 6, 6)
        return x
