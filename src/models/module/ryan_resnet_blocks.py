import torch
import torch.nn as nn
from utils.cuda import cuda_var


class ResBlock(torch.nn.Module):
    def __init__(self, c_in=16):
        super(ResBlock, self).__init__()
        # order should be conv, then normalization, then activation
        # stride
        self.conv1 = nn.Conv2d(c_in, c_in, 3, padding=1)
        self.norm1 = nn.InstanceNorm2d(c_in, affine=True)
        self.act1 = nn.PReLU(init=0.2)
        self.conv2 = nn.Conv2d(c_in, c_in, 3, padding=1)
        self.norm2 = nn.InstanceNorm2d(c_in, affine=True)
        self.act2 = nn.PReLU(init=0.2)

    def init_weights(self):
        torch.nn.init.kaiming_uniform(self.conv1.weight)
        torch.nn.init.kaiming_uniform(self.conv2.weight)

        self.conv1.bias.data.fill_(0)
        self.conv2.bias.data.fill_(0)

    def forward(self, images):
        x = self.act1(self.conv1(self.norm1(images)))
        x = self.act2(self.conv2(self.norm2(x)))
        out = x + images
        return out


class InvResBlock(torch.nn.Module):
    def __init__(self, c_in=16):
        super(InvResBlock, self).__init__()
        self.deconv1 = nn.ConvTranspose2d(c_in, c_in, 3, padding=1)
        self.norm1 = nn.InstanceNorm2d(c_in, affine=True)
        self.act1 = nn.PReLU(init=0.2)
        self.deconv2 = nn.ConvTranspose2d(c_in, c_in, 3, padding=1)
        self.norm2 = nn.InstanceNorm2d(c_in, affine=True)
        self.act2 = nn.PReLU(init=0.2)

    def init_weights(self):
        torch.nn.init.kaiming_uniform(self.deconv1.weight)
        torch.nn.init.kaiming_uniform(self.deconv2.weight)

        self.deconv1.bias.data.fill_(0)
        self.deconv2.bias.data.fill_(0)

    def forward(self, images):
        x = self.act1(self.norm1(self.deconv1(images)))
        x = self.act2(self.norm2(self.deconv2(x)))
        out = x + images
        return out


class ResBlockStrided(torch.nn.Module):
    def __init__(self, c_in=16, c_out=32):
        super(ResBlockStrided, self).__init__()
        assert c_out >= c_in
        self.c_in = c_in
        self.c_out = c_out
        self.conv1 = nn.Conv2d(c_in, c_out, 3, stride=2, padding=1)
        self.norm1 = nn.InstanceNorm2d(c_out, affine=True)
        self.act1 = nn.PReLU(init=0.2)
        self.conv2 = nn.Conv2d(c_out, c_out, 3, padding=1)
        self.norm2 = nn.InstanceNorm2d(c_out, affine=True)
        self.act2 = nn.PReLU(init=0.2)

        self.avg_pool = nn.AvgPool2d(2)

    def init_weights(self):
        torch.nn.init.kaiming_uniform(self.conv1.weight)
        torch.nn.init.kaiming_uniform(self.conv2.weight)

        self.conv1.bias.data.fill_(0)
        self.conv2.bias.data.fill_(0)

    def forward(self, images):
        x = self.act1(self.conv1(self.norm1(images)))
        x_out = self.act2(self.conv2(self.norm2(x)))

        x_in = self.avg_pool(images)
        b, _, h, w = x_in.data.shape
        c_diff = self.c_out - self.c_in
        if c_diff > 0:
            zeros = cuda_var(torch.zeros(b, c_diff, h, w))
            x_in = torch.cat([x_in, zeros], dim=1)

        return x_in + x_out


class InvResBlockStrided(torch.nn.Module):
    def __init__(self, c_in=32, c_out=16):
        super(InvResBlockStrided, self).__init__()
        assert c_out <= c_in
        self.c_in = c_in
        self.c_out = c_out
        self.deconv1 = nn.ConvTranspose2d(c_in, c_out, 3, stride=2,
                                          padding=1, output_padding=1)
        self.norm1 = nn.InstanceNorm2d(c_out, affine=True)
        self.act1 = nn.PReLU(init=0.2)
        self.deconv2 = nn.ConvTranspose2d(c_out, c_out, 3, padding=1)
        self.norm2 = nn.InstanceNorm2d(c_out, affine=True)
        self.act2 = nn.PReLU(init=0.2)

        self.shrink = nn.Conv2d(c_in, c_out, 1)
        self.avg_pool = nn.AvgPool2d(2)

    def init_weights(self):
        torch.nn.init.kaiming_uniform(self.deconv1.weight)
        torch.nn.init.kaiming_uniform(self.deconv2.weight)
        torch.nn.init.kaiming_uniform(self.shrink.weight)

        self.deconv1.bias.data.fill_(0)
        self.deconv2.bias.data.fill_(0)
        self.shrink.bias.data.fill_(0)

    def forward(self, images):
        x = self.act1(self.norm1(self.deconv1(images)))
        x_out = self.act2(self.norm2(self.deconv2(x)))

        x_in = inverse_avg_pool(images)
        c_diff = self.c_in - self.c_out
        if c_diff > 0:
            x_in = self.shrink(x_in)

        return x_in + x_out


def inverse_avg_pool(images):
    b, c, h, w = images.data.shape
    images = images.view(b, c, h, w, 1).expand(b, c, h, w, 2).contiguous()
    images = images.view(b, c, h, 2*w).transpose(2, 3).contiguous()
    images = images.view(b, c, 2*w, h, 1).expand(b, c, 2*w, h, 2).contiguous()
    images = images.view(b, c, 2*w, 2*h).transpose(2, 3).contiguous()
    return images
