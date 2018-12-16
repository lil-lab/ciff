import torch
import torch.nn as nn
import torch.nn.functional as F

import models.module.resnet_blocks as blocks


class ImageGAResnetModule(torch.nn.Module):

    def __init__(self, image_emb_size, input_num_channels,
                 image_height, image_width, text_emb_size,
                 using_recurrence=False):
        super(ImageGAResnetModule, self).__init__()
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
        self.num_feature_maps = 32

        self.final_h = h
        self.final_w = w

        self.resnet_blocks = nn.Sequential(block1, block2, block3)
        self.norm2 = nn.InstanceNorm2d(self.num_feature_maps)

        self.text_dense = nn.Linear(text_emb_size, self.num_feature_maps)
        self.final_dense = nn.Linear(h * w * self.num_feature_maps, image_emb_size)
        self.global_id = 0

    def init_weights(self):

        for block in self.resnet_blocks:
            for element in block:
                element.init_weights()

        torch.nn.init.kaiming_uniform(self.conv1.weight)
        self.conv1.bias.data.fill_(0)

    def forward(self, x, text_vector):
        b, n, c, h, w = x.size()
        if self.using_recurrence:
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

        a_vectors = F.sigmoid(self.text_dense(text_vector))
        tensor_list = []
        for vec in a_vectors:
            if self.using_recurrence:
                num_repeat = n
            else:
                num_repeat = 1
            v = vec.repeat(num_repeat, self.final_h, self.final_w, 1)
            # v = v.permute(0, 3, 1, 2)
            tensor_list.append(v)
        a_tensor = torch.cat(tensor_list)
        # a_tensor change from H x W x C to C x H x W
        a_tensor = a_tensor.permute(0, 3, 1, 2)

        x = a_tensor * x

        x = self.final_dense(x.view(num_batches, -1))

        if self.using_recurrence:
            x = x.view(b, n, -1)
        else:
            x = x.view(b, -1)
        return x
