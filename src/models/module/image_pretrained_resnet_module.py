import torch
import torch.nn as nn

import models.module.resnet_blocks as blocks


class ImageResnetModule(torch.nn.Module):

    def __init__(self, image_emb_size, input_num_channels,
                 image_height, image_width, using_recurrence=False):
        super(ImageResnetModule, self).__init__()
        self.input_dims = (input_num_channels, image_height, image_width)
        self.using_recurrence = using_recurrence

        self.norm1 = nn.InstanceNorm2d(input_num_channels)
        self.conv1 = nn.Conv2d(input_num_channels, 32, 7, stride=2, padding=3)
        h, w = (image_height / 2, image_width / 2)

        block1 = nn.Sequential(
            blocks.ResBlockStrided(32, 32),
            blocks.ResBlock(32)
        )
        h, w = h / 2, w / 2

        block2 = nn.Sequential(
            blocks.ResBlockStrided(32, 32),
            blocks.ResBlock(32)
        )
        h, w = h / 2, w / 2

        block3 = nn.Sequential(
            blocks.ResBlockStrided(32, 32),
        )
        h, w = h / 2, w / 2

        self.resnet_blocks = nn.Sequential(block1, block2, block3)
        self.norm2 = nn.InstanceNorm2d(32)
        self.dense = nn.Linear(h * w * 32, image_emb_size)
        self.global_id = 0

    def init_weights(self):

        for block in self.resnet_blocks:
            for element in block:
                element.init_weights()

        torch.nn.init.kaiming_uniform(self.conv1.weight)
        self.conv1.bias.data.fill_(0)

    def forward(self, x):
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
        # image_flipped = image[-1].cpu().data.numpy().swapaxes(0, 1).swapaxes(1, 2)
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
