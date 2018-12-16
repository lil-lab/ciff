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


class UnetImageModule(torch.nn.Module):

    def __init__(self, image_emb_size, input_num_channels,
                 image_height, image_width, using_recurrence=False, final_dimension=None):
        super(UnetImageModule, self).__init__()

        self.input_dims = (input_num_channels, image_height, image_width)
        self.conv1 = nn.Conv2d(input_num_channels, 128, kernel_size=8, stride=4, padding=3)
        self.conv2 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.using_recurrence = using_recurrence
        if final_dimension is not None:
            self.final_dimension = final_dimension
        else:
            self.final_dimension = (64, 32, 32)
        self.global_id = 0

    def init_weights(self):
        self.apply(weights_init)

    def get_final_dimension(self):
        return self.final_dimension

    def fix(self):
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
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x_image_rep = x  # F.relu(self.conv3(x))
        top_layer = x_image_rep  # (batch x n) x 64 x 6 x 6

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
                x_image_rep = x_image_rep.view(b, n, x_image_rep.size(1), x_image_rep.size(2), x_image_rep.size(3))
            else:
                raise AssertionError("Not implemented")
        else:
            x_image_rep = x_image_rep.view(b, x_image_rep.size(1), x_image_rep.size(2), x_image_rep.size(3))
        return x_image_rep
