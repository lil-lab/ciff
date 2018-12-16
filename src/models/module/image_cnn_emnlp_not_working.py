import torch.nn as nn
import torch.nn.functional as F


class ImageCnnEmnlp(nn.Module):
    """
    pytorch module for image part of model
    """

    def __init__(self, image_emb_size, input_num_channels,
                 image_height, image_width):
        super(ImageCnnEmnlp, self).__init__()
        self.input_dims = (input_num_channels, image_height, image_width)

        k1, s1 = 8, 4
        self.conv1 = nn.Conv2d(input_num_channels, 32, kernel_size=k1, stride=s1, padding=2)
        #self.bn1 = nn.BatchNorm2d(32)
        h = round_up((image_height) * 1.0 / s1)# - k1 + 1) * 1.0 / s1)
        w = round_up((image_width) * 1.0 / s1)# - k1 + 1) * 1.0 / s1)

        #print("aabc")
        #print(h)

        k2, s2 = 8, 4
        self.conv2 = nn.Conv2d(32, 32, kernel_size=k2, stride=s2, padding=2)
        #self.bn2 = nn.BatchNorm2d(32)
        h = round_up((h) * 1.0 / s2)#  - k2 + 1) * 1.0 / s2)
        w = round_up((w) * 1.0 / s2)#  - k2 + 1) * 1.0 / s2)

        #print("aabc")
        #print(h)

        k3, s3 = 4, 2
        self.conv3 = nn.Conv2d(32, 32, kernel_size=k3, stride=s3, padding=1)
        #self.bn3 = nn.BatchNorm2d(32)
        h = round_up((h) * 1.0 / s3)# - k3 + 1) * 1.0 / s3)
        w = round_up((w) * 1.0 / s3)# - k3 + 1) * 1.0 / s3)

        #print("aabc")
        #print(h * w * 32)
        #print(self.conv3.size())

        # k4, s4 = 4, 4
        # self.conv4 = nn.Conv2d(32, 32, kernel_size=k4, stride=s4, padding=1)
        # #self.bn4 = nn.BatchNorm2d(32)
        # h = round_up((h) * 1.0 / s4)# - k4 + 1) * 1.0 / s4)
        # w = round_up((w) * 1.0 / s4)# - k4 + 1) * 1.0 / s4)

        # sanity chek we're in correcaltion with original code
        #print("aabc")
        #print(h*w*32)
        assert(h * w * 32 == 512)
        self.head = nn.Linear(h * w * 32, image_emb_size)

    def forward(self, x):
        num_batches = x.size()[0]
        input_size = (num_batches,) + self.input_dims
        x = x.view(input_size)

        ### FROM ORIGINAL CODDE
        #        self.output_size = output_size
        #        height = image_dim
        #        width = image_dim
        #        channels = 3 * 5

        # conv + affine + relu
        # conv + affine + relu
        # conv + affine + relu

        # affine transform

        # Create 4 images for padding

        ### END FROM ORIGINAL CODDE


        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        reshaped_x = x.view(num_batches, -1)
        #print(reshaped_x.size())
        return self.head(reshaped_x)

# what the f is this?
def round_up(x):
    return int(x + 0.99999)
