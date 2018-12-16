import torch
import torch.nn as nn
import torch.nn.functional as F

import models.module.resnet_blocks as blocks


class ImageTextKernelResnetModule(torch.nn.Module):
    """ Image Resnet module using a text based kernel Janner et al. arXiv 2017 paper"""

    def __init__(self, image_emb_size, input_num_channels, image_height,
                 image_width, text_emb_size, using_recurrence=False):
        super(ImageTextKernelResnetModule, self).__init__()
        self.input_dims = (input_num_channels, image_height, image_width)
        self.using_recurrence = using_recurrence

        # Convert text embedding into 16 kernel of size 7x7 specific shape and size
        self.dense_text_to_kernel = nn.Linear(text_emb_size, 16 * 3 * 7 * 7)
        self.global_kernel = nn.Parameter(torch.FloatTensor(16, 3, 7, 7))

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

    def forward(self, image_input, text_input):
        assert self.using_recurrence, "Currently code does not work with non-recurrent method"
        b, n, c, h, w = image_input.size()
        if self.using_recurrence:
            image_input = image_input.view(b * n, 1, c, h, w)
            num_batches = b * n
            num_channels = c
        else:
            num_batches = b
            num_channels = n * c
        image_input = image_input.view(num_batches, num_channels, h, w)
        image = image_input

        image_input = self.norm1(image_input)

        text_embedding = self.dense_text_to_kernel(text_input)  # batch x (16*3*7*7)
        text_kernel = text_embedding.view(b, 16, 3, 7, 7)

        image_outputs = []
        for i in range(0, num_batches):
            text_kernel_id = int(i / n)
            kernel = torch.cat((self.global_kernel, text_kernel[text_kernel_id, :, :, :, :]), 0)  # becomes 32 x 3 x 7 x 7
            image_input_example = image_input[i, :, :, :]
            image_input_example = image_input_example.unsqueeze(0)
            image_output = F.conv2d(image_input_example, kernel, stride=2, padding=3)
            image_outputs.append(image_output[0, :, :, :])

        image_input = torch.stack(image_outputs)
        # image_input = self.conv1(image_input)
        image_input = self.resnet_blocks(image_input)
        image_input = self.norm2(image_input)
        top_layer = image_input

        image_input = self.dense(image_input.view(image_input.size(0), -1))

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
            image_input = image_input.view(b, n, -1)
        else:
            image_input = image_input.view(b, -1)
        return image_input
