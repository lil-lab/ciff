import torch
import torch.nn as nn
import torch.nn.functional as F

import models.module.resnet_blocks as blocks


class ImageAttentionResnetModule(torch.nn.Module):

    def __init__(self, image_emb_size, input_num_channels,
                 image_height, image_width, text_emb_size,
                 using_recurrence=False):
        super(ImageAttentionResnetModule, self).__init__()
        self.input_dims = (input_num_channels, image_height, image_width)
        self.using_recurrence = using_recurrence
        self.num_attention_heads = 5

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

        self.attention_dense_layers = nn.ModuleList()
        for _ in xrange(self.num_attention_heads):
            dense_layer = nn.Linear(text_emb_size, self.final_h * self.final_w)
            self.attention_dense_layers.append(dense_layer)

        self.resnet_blocks = nn.Sequential(block1, block2, block3)

        self.final_dense = nn.Linear(h * w * self.num_attention_heads,
                                     image_emb_size)

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
        x = self.norm1(x)
        x = self.conv1(x)
        x = self.resnet_blocks(x)

        attention_maps = []
        for dense_layer in self.attention_dense_layers:
            # calculate tensor of repeated 8x8 kernels from text vector
            a_vectors = F.sigmoid(dense_layer(text_vector))
            a_vectors = a_vectors.view(b, self.final_h, self.final_w)
            tensor_list = []
            for vec in a_vectors:
                if self.using_recurrence:
                    num_repeat = n
                else:
                    num_repeat = 1
                v = vec.repeat(num_repeat, self.num_feature_maps, 1, 1)
                tensor_list.append(v)
            a_tensor = torch.cat(tensor_list)

            # take dot product of kernels with features maps,
            # and softmax
            probs = F.softmax((a_tensor * x).sum(3).sum(2))

            # add weighted average of feature maps to attention maps
            mask = probs.repeat(self.final_h, self.final_w, 1, 1)
            mask = mask.permute(2, 3, 0, 1)
            attention_map = (x * mask).sum(1).view(
                1, num_batches, self.final_h, self.final_w)
            attention_maps.append(attention_map)

        x = torch.cat(attention_maps).permute(1, 0, 2, 3)
        x = self.final_dense(x.contiguous().view(num_batches, -1))

        if self.using_recurrence:
            x = x.view(b, n, -1)
        else:
            x = x.view(b, -1)
        return x
