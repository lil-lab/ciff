import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import scipy.misc


class ImagePositionSimpleModule(nn.Module):
    """
    pytorch module for image part of model
    """

    def __init__(self, image_emb_size, input_num_channels,
                 image_height, image_width):
        super(ImagePositionSimpleModule, self).__init__()
        self.input_dims = (input_num_channels, image_height, image_width)

        k1, s1 = 8, 3
        self.conv1 = nn.Conv2d(input_num_channels, 32, kernel_size=k1, stride=s1)
        self.bn1 = nn.BatchNorm2d(32)
        h = round_up((image_height - k1 + 1) * 1.0 / s1)
        w = round_up((image_width - k1 + 1) * 1.0 / s1)

        k2, s2 = 4, 2
        self.conv2 = nn.Conv2d(32, 32, kernel_size=k2, stride=s2)
        self.bn2 = nn.BatchNorm2d(32)
        h = round_up((h - k2 + 1) * 1.0 / s2)
        w = round_up((w - k2 + 1) * 1.0 / s2)

        k3, s3 = 4, 2
        self.conv3 = nn.Conv2d(32, 32, kernel_size=k3, stride=s3)
        self.bn3 = nn.BatchNorm2d(32)
        h = round_up((h - k3 + 1) * 1.0 / s3)
        w = round_up((w - k3 + 1) * 1.0 / s3)

        k4, s4 = 2, 1
        self.conv4 = nn.Conv2d(32, 32, kernel_size=k4, stride=s4)
        self.bn4 = nn.BatchNorm2d(32)
        h = round_up((h - k4 + 1) * 1.0 / s4)
        w = round_up((w - k4 + 1) * 1.0 / s4)

        self.ryan_top_layer_conv = nn.Conv2d(32, 1, 1, stride=1)
        self.global_id = 0

    def forward(self, x):
        num_batches = x.size()[0]
        input_size = (num_batches,) + self.input_dims
        x = x.view(input_size)
        image = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        # x = F.relu(self.bn4(self.conv4(x)))
        x = self.ryan_top_layer_conv(x)

        ################################################
        # self.global_id += 1
        # softmax = F.softmax(x.view(num_batches, -1)).view(num_batches, 1, 8, 8)
        # numpy_ar = softmax.cpu().data.numpy()
        # image_flipped = image[-1].cpu().data.numpy().swapaxes(0, 1).swapaxes(1, 2)
        # for i in range(0, 1):
        #     resized_kernel = scipy.misc.imresize(numpy_ar[-1][i], (128, 128))
        #     plt.imshow(image_flipped)
        #     plt.imshow(resized_kernel, cmap='jet', alpha=0.5)
        #     plt.savefig("./kernels_tmp/" + str(self.global_id) + "_kernel_" + str(i) + ".png")
        #     plt.clf()
        ################################################

        return x.view(num_batches, -1)

def round_up(x):
    return int(x + 0.99999)
