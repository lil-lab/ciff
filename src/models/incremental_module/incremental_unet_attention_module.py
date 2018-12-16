import logging
import torch
import scipy.misc
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np

from torch import nn as nn
from learning.auxiliary_objective.goal_prediction import GoalPrediction
from utils.cuda import cuda_var
from textwrap import wrap


class IncrementalUnetAttentionModuleCatSpatialInfo(torch.nn.Module):
    """ Unet architecture designed by Valts Blukis, based on original Unet paper, for predicting the image. """

    def __init__(self,  image_module, text_module, image_recurrence_module,
                 max_episode_length, final_image_height, final_image_width,
                 in_channels, out_channels, embedding_size, hc1=32, hc2=16,
                 k=5, stride=2, split_embedding=False):
        super(IncrementalUnetAttentionModuleCatSpatialInfo, self).__init__()

        self.image_module = image_module
        self.image_recurrence_module = image_recurrence_module
        self.text_module = text_module

        pad = int(k / 2)
        self.hc1 = hc1
        self.hc2 = hc2

        self.split_embedding = split_embedding

        self.embedding_size = embedding_size
        if split_embedding:
            self.emb_block_size = int(embedding_size / 5)
        else:
            self.emb_block_size = embedding_size

        # inchannels, outchannels, kernel size
        self.conv1 = nn.Conv2d(in_channels, hc1, k, stride=stride, padding=pad)
        self.conv2 = nn.Conv2d(hc1, hc1, k, stride=stride, padding=pad)
        self.conv3 = nn.Conv2d(hc1, hc1, k, stride=stride, padding=pad)
        self.conv4 = nn.Conv2d(hc1, hc1, k, stride=stride, padding=pad)
        self.conv5 = nn.Conv2d(hc1, hc1, k, stride=stride, padding=pad)

        self.deconv1 = nn.ConvTranspose2d(hc1, hc1, k, stride=stride, padding=pad)
        self.deconv2 = nn.ConvTranspose2d(2 * hc1, hc1, k, stride=stride, padding=pad)
        self.deconv3 = nn.ConvTranspose2d(2 * hc1, hc1, k, stride=stride, padding=pad)
        self.deconv4 = nn.ConvTranspose2d(2 * hc1, hc2, k, stride=stride, padding=pad)
        self.deconv5 = nn.ConvTranspose2d(hc1 + hc2, out_channels, k, stride=stride, padding=pad)

        self.act = nn.LeakyReLU()
        self.dropout = nn.Dropout(0.5)
        self.norm2 = nn.InstanceNorm2d(hc1)
        self.norm3 = nn.InstanceNorm2d(hc1)
        self.norm4 = nn.InstanceNorm2d(hc1)
        self.norm5 = nn.InstanceNorm2d(hc1)
        # self.dnorm1 = nn.InstanceNorm2d(in_channels * 4)
        self.dnorm2 = nn.InstanceNorm2d(hc1)
        self.dnorm3 = nn.InstanceNorm2d(hc1)
        self.dnorm4 = nn.InstanceNorm2d(hc1)
        self.dnorm5 = nn.InstanceNorm2d(hc2)

        self.lang19 = nn.Linear(self.emb_block_size, hc1 * hc1)
        self.lang28 = nn.Linear(self.emb_block_size, hc1 * hc1)
        self.lang37 = nn.Linear(self.emb_block_size, hc1 * hc1)
        self.lang46 = nn.Linear(self.emb_block_size, hc1 * hc1)
        self.lang55 = nn.Linear(self.emb_block_size, hc1 * hc1)

        self.attention_spatial_compressor1 = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3)

        # Time embedding layer, helps in stabilizing value prediction
        self.time_emb_dim = 32
        self.time_emb_layer = nn.Embedding(max_episode_length + 1, self.time_emb_dim)

        # A3C-LSTM layers
        self.final_channels = in_channels
        self.final_image_height = final_image_height
        self.final_image_width = final_image_width

        self.bias = nn.Linear(1, 1)
        self.linear = nn.Linear(32 * 32 + 1 + 30 * 30, 256)
        # self.linear = nn.Linear(self.final_channels, 256)
        self.critic_linear = nn.Linear(256 + self.time_emb_dim, 1)
        self.actor_linear = nn.Linear(256 + self.time_emb_dim, 4)

        # For logs
        self.global_id = 0

    def init_weights(self):
        torch.nn.init.kaiming_uniform(self.conv1.weight)
        self.conv1.bias.data.fill_(0)
        torch.nn.init.kaiming_uniform(self.conv2.weight)
        self.conv2.bias.data.fill_(0)
        torch.nn.init.kaiming_uniform(self.conv3.weight)
        self.conv3.bias.data.fill_(0)
        torch.nn.init.kaiming_uniform(self.conv4.weight)
        self.conv4.bias.data.fill_(0)
        torch.nn.init.kaiming_uniform(self.conv5.weight)
        self.conv5.bias.data.fill_(0)

        torch.nn.init.kaiming_uniform(self.deconv1.weight)
        self.deconv1.bias.data.fill_(0)
        torch.nn.init.kaiming_uniform(self.deconv2.weight)
        self.deconv2.bias.data.fill_(0)
        torch.nn.init.kaiming_uniform(self.deconv3.weight)
        self.deconv3.bias.data.fill_(0)
        torch.nn.init.kaiming_uniform(self.deconv4.weight)
        self.deconv4.bias.data.fill_(0)
        torch.nn.init.kaiming_uniform(self.deconv5.weight)
        self.deconv5.bias.data.fill_(0)

    def save_attention_prob(self, image, attention_prob, instruction, goal_prob=None):
        self.global_id += 1

        image_flipped = image[0, 0, :, :, :].cpu().data.numpy().swapaxes(0, 1).swapaxes(1, 2)
        attention_prob = attention_prob.cpu().data.numpy()
        resized_kernel = scipy.misc.imresize(attention_prob, (128, 128))
        if goal_prob is not None:
            goal_location = goal_prob.cpu().data.numpy()
            if np.sum(goal_location) > 0.01:
                for i in range(0, self.final_image_height):
                    for j in range(0, self.final_image_width):
                            if goal_location[i][j] < 0.01:
                                goal_location[i][j] = 0.0
                goal_location = scipy.misc.imresize(goal_location, (128, 128))
        else:
            goal_location = None

        f, axarr = plt.subplots(1, 2)
        if instruction is not None:
            f.suptitle(instruction)
        axarr[0].set_title("Predicted Attention")
        axarr[0].imshow(image_flipped)
        axarr[0].imshow(resized_kernel, cmap='jet', alpha=0.5)
        axarr[1].set_title("Gold Attention (Goal)")
        axarr[1].imshow(image_flipped)
        if goal_location is not None:
            axarr[1].imshow(goal_location, cmap='jet', alpha=0.5)
        plt.savefig("./attention_prob/image_" + str(self.global_id) + ".png")
        plt.clf()

    def forward(self, image, instructions, tx, mode, model_state, instruction, goal):

        if model_state is None:
            _, sentence_embedding = self.text_module(instructions)
            image_hidden_states = None
        else:
            sentence_embedding, image_hidden_states = model_state

        image_emb_seq = self.image_module(image)
        image_input = image_emb_seq[:, 0, :, :, :]  # 1 x num_channels x height x width

        x1 = self.norm2(self.act(self.conv1(image_input)))
        x2 = self.norm3(self.act(self.conv2(x1)))
        x3 = self.norm4(self.act(self.conv3(x2)))
        x4 = self.norm5(self.act(self.conv4(x3)))
        x5 = self.act(self.conv5(x4))

        if sentence_embedding is not None:
            if self.split_embedding:
                block_size = self.emb_block_size
                emb1 = sentence_embedding[:, 0 * block_size:1 * block_size]
                emb2 = sentence_embedding[:, 1 * block_size:2 * block_size]
                emb3 = sentence_embedding[:, 2 * block_size:3 * block_size]
                emb4 = sentence_embedding[:, 3 * block_size:4 * block_size]
                emb5 = sentence_embedding[:, 4 * block_size:5 * block_size]
            else:
                emb1 = emb2 = emb3 = emb4 = emb5 = sentence_embedding

            lf1 = F.normalize(self.lang19(emb1)).view([self.hc1, self.hc1, 1, 1])
            lf2 = F.normalize(self.lang28(emb2)).view([self.hc1, self.hc1, 1, 1])
            lf3 = F.normalize(self.lang37(emb3)).view([self.hc1, self.hc1, 1, 1])
            lf4 = F.normalize(self.lang46(emb4)).view([self.hc1, self.hc1, 1, 1])
            lf5 = F.normalize(self.lang55(emb5)).view([self.hc1, self.hc1, 1, 1])

            x1f = F.conv2d(x1, lf1)
            x2f = F.conv2d(x2, lf2)
            x3f = F.conv2d(x3, lf3)
            x4f = F.conv2d(x4, lf4)
            x5f = F.conv2d(x5, lf5)

            x5f = self.dropout(x5f)
        else:
            raise AssertionError("Embedding should not be none.")

        x6 = self.act(self.deconv1(x5f, output_size=x4.size()))
        x46 = torch.cat([x4f, x6], 1)
        x7 = self.dnorm3(self.act(self.deconv2(x46, output_size=x3.size())))
        x37 = torch.cat([x3f, x7], 1)
        x8 = self.dnorm4(self.act(self.deconv3(x37, output_size=x2.size())))
        x28 = torch.cat([x2f, x8], 1)
        x9 = self.dnorm5(self.act(self.deconv4(x28, output_size=x1.size())))
        x19 = torch.cat([x1f, x9], 1)
        out = self.deconv5(x19, output_size=image_input.size())

        attention_logits = out.view(-1)
        attention_logits = torch.cat([attention_logits, self.bias.bias.view(-1)], dim=0)
        attention_prob = F.softmax(attention_logits, dim=0)  # (height x width + 1)

        volatile = dict()
        volatile["image_emb"] = image_input
        volatile["attention_logits"] = attention_logits  # 32x32 + 1
        volatile["attention_probs"] = attention_prob

        # if True:
        #     goal_prob = GoalPrediction.generate_gold_prob(goal, self.final_image_height, self.final_image_width)
        #     self.save_attention_prob(image, attention_prob[:-1].view(self.final_image_height, self.final_image_width),
        #                              instruction, goal_prob[:-1].view(self.final_image_height, self.final_image_width))

        # Compute two-spatial
        spatial_information = F.sigmoid(self.attention_spatial_compressor1(image_input))  # 1 x 16 x self.height, self.width
        spatial_information = F.dropout(spatial_information)

        # Compute the probabilities
        x = torch.cat([spatial_information.view(1, -1), attention_prob.view(1, -1)], dim=1)
        x = F.relu(self.linear(x))
        new_image_hidden_states = self.image_recurrence_module(x, image_hidden_states)
        hx, cx = new_image_hidden_states

        time_emb = self.time_emb_layer(tx)
        x = torch.cat((hx, time_emb.view(-1, self.time_emb_dim)), 1)

        actor_linear = self.actor_linear(x)
        critic_linear = self.critic_linear(x)

        new_model_state = (sentence_embedding, new_image_hidden_states)

        return F.log_softmax(actor_linear, dim=1), new_model_state, None, volatile

    def get_attention_prob(self, image, instructions, instruction, goal):

        _, sentence_embedding = self.text_module(instructions)

        image_emb_seq = self.image_module(image)
        image_input = image_emb_seq[:, 0, :, :, :]  # 1 x num_channels x height x width

        x1 = self.norm2(self.act(self.conv1(image_input)))
        x2 = self.norm3(self.act(self.conv2(x1)))
        x3 = self.norm4(self.act(self.conv3(x2)))
        x4 = self.norm5(self.act(self.conv4(x3)))
        x5 = self.act(self.conv5(x4))

        if sentence_embedding is not None:
            if self.split_embedding:
                block_size = self.emb_block_size
                emb1 = sentence_embedding[:, 0 * block_size:1 * block_size]
                emb2 = sentence_embedding[:, 1 * block_size:2 * block_size]
                emb3 = sentence_embedding[:, 2 * block_size:3 * block_size]
                emb4 = sentence_embedding[:, 3 * block_size:4 * block_size]
                emb5 = sentence_embedding[:, 4 * block_size:5 * block_size]
            else:
                emb1 = emb2 = emb3 = emb4 = emb5 = sentence_embedding

            lf1 = F.normalize(self.lang19(emb1)).view([self.hc1, self.hc1, 1, 1])
            lf2 = F.normalize(self.lang28(emb2)).view([self.hc1, self.hc1, 1, 1])
            lf3 = F.normalize(self.lang37(emb3)).view([self.hc1, self.hc1, 1, 1])
            lf4 = F.normalize(self.lang46(emb4)).view([self.hc1, self.hc1, 1, 1])
            lf5 = F.normalize(self.lang55(emb5)).view([self.hc1, self.hc1, 1, 1])

            x1f = F.conv2d(x1, lf1)
            x2f = F.conv2d(x2, lf2)
            x3f = F.conv2d(x3, lf3)
            x4f = F.conv2d(x4, lf4)
            x5f = F.conv2d(x5, lf5)

            x5f = self.dropout(x5f)
        else:
            raise AssertionError("Embedding should not be none.")

        x6 = self.act(self.deconv1(x5f, output_size=x4.size()))
        x46 = torch.cat([x4f, x6], 1)
        x7 = self.dnorm3(self.act(self.deconv2(x46, output_size=x3.size())))
        x37 = torch.cat([x3f, x7], 1)
        x8 = self.dnorm4(self.act(self.deconv3(x37, output_size=x2.size())))
        x28 = torch.cat([x2f, x8], 1)
        x9 = self.dnorm5(self.act(self.deconv4(x28, output_size=x1.size())))
        x19 = torch.cat([x1f, x9], 1)
        out = self.deconv5(x19, output_size=image_input.size())

        attention_logits = out.view(-1)
        attention_logits = torch.cat([attention_logits, self.bias.bias.view(-1)], dim=0)
        attention_prob = F.softmax(attention_logits, dim=0)  # (height x width + 1)

        volatile = dict()
        volatile["image_emb"] = image_input
        volatile["attention_logits"] = attention_logits  # 32x32 + 1
        volatile["attention_probs"] = attention_prob

        # if True:
        #     goal_prob = GoalPrediction.generate_gold_prob(goal, self.final_image_height, self.final_image_width)
        #     self.save_attention_prob(image, attention_prob[:-1].view(self.final_image_height, self.final_image_width),
        #                              instruction, goal_prob[:-1].view(self.final_image_height, self.final_image_width))

        return volatile


class IncrementalUnetAttentionModule(torch.nn.Module):
    """ Unet architecture designed by Valts Blukis, based on original Unet paper, for predicting the image. """

    def __init__(self,  image_module, text_module, image_recurrence_module,
                 max_episode_length, final_image_height, final_image_width,
                 in_channels, out_channels, embedding_size, hc1=32, hc2=16,
                 k=5, stride=2, split_embedding=False):
        super(IncrementalUnetAttentionModule, self).__init__()

        self.image_module = image_module
        self.image_recurrence_module = image_recurrence_module
        self.text_module = text_module

        pad = int(k / 2)
        self.hc1 = hc1
        self.hc2 = hc2

        self.split_embedding = split_embedding

        self.embedding_size = embedding_size
        if split_embedding:
            self.emb_block_size = int(embedding_size / 5)
        else:
            self.emb_block_size = embedding_size

        # inchannels, outchannels, kernel size
        self.conv1 = nn.Conv2d(in_channels, hc1, k, stride=stride, padding=pad)
        self.conv2 = nn.Conv2d(hc1, hc1, k, stride=stride, padding=pad)
        self.conv3 = nn.Conv2d(hc1, hc1, k, stride=stride, padding=pad)
        self.conv4 = nn.Conv2d(hc1, hc1, k, stride=stride, padding=pad)
        self.conv5 = nn.Conv2d(hc1, hc1, k, stride=stride, padding=pad)

        self.deconv1 = nn.ConvTranspose2d(hc1, hc1, k, stride=stride, padding=pad)
        self.deconv2 = nn.ConvTranspose2d(2 * hc1, hc1, k, stride=stride, padding=pad)
        self.deconv3 = nn.ConvTranspose2d(2 * hc1, hc1, k, stride=stride, padding=pad)
        self.deconv4 = nn.ConvTranspose2d(2 * hc1, hc2, k, stride=stride, padding=pad)
        self.deconv5 = nn.ConvTranspose2d(hc1 + hc2, out_channels, k, stride=stride, padding=pad)

        self.act = nn.LeakyReLU()
        self.dropout = nn.Dropout(0.5)
        self.norm2 = nn.InstanceNorm2d(hc1)
        self.norm3 = nn.InstanceNorm2d(hc1)
        self.norm4 = nn.InstanceNorm2d(hc1)
        self.norm5 = nn.InstanceNorm2d(hc1)
        # self.dnorm1 = nn.InstanceNorm2d(in_channels * 4)
        self.dnorm2 = nn.InstanceNorm2d(hc1)
        self.dnorm3 = nn.InstanceNorm2d(hc1)
        self.dnorm4 = nn.InstanceNorm2d(hc1)
        self.dnorm5 = nn.InstanceNorm2d(hc2)

        self.lang19 = nn.Linear(self.emb_block_size, hc1 * hc1)
        self.lang28 = nn.Linear(self.emb_block_size, hc1 * hc1)
        self.lang37 = nn.Linear(self.emb_block_size, hc1 * hc1)
        self.lang46 = nn.Linear(self.emb_block_size, hc1 * hc1)
        self.lang55 = nn.Linear(self.emb_block_size, hc1 * hc1)

        # Time embedding layer, helps in stabilizing value prediction
        self.time_emb_dim = 32
        self.time_emb_layer = nn.Embedding(max_episode_length + 1, self.time_emb_dim)

        # Attention mixing layers
        self.final_channels = in_channels
        self.final_image_height = final_image_height
        self.final_image_width = final_image_width
        # self.attention_mix_conv1 = nn.Conv2d(in_channels=(in_channels + 1), out_channels=32, kernel_size=3, stride=1)
        # self.attention_mix_conv2 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, stride=1)
        # self.attention_mix_conv3 = nn.Conv2d(in_channels=16, out_channels=1, kernel_size=3, stride=1)
        self.attention_spatial_compressor1 = nn.Conv2d(in_channels=in_channels, out_channels=16, kernel_size=3, stride=1)
        self.attention_spatial_compressor2 = nn.Conv2d(in_channels=16, out_channels=8, kernel_size=3, stride=1)
        self.attention_spatial_compressor3 = nn.Conv2d(in_channels=8, out_channels=1, kernel_size=3, stride=1)
        self.attention_mixer = nn.Linear(1024 + 1 + 676, 1024)

        # A3C-LSTM layers
        self.bias = nn.Linear(1, 1)
        self.linear = nn.Linear(1024, 256)
        self.critic_linear = nn.Linear(256 + self.time_emb_dim, 1)
        self.actor_linear = nn.Linear(256 + self.time_emb_dim, 4)

        # For logs
        self.global_id = 0

    def init_weights(self):
        torch.nn.init.kaiming_uniform(self.conv1.weight)
        self.conv1.bias.data.fill_(0)
        torch.nn.init.kaiming_uniform(self.conv2.weight)
        self.conv2.bias.data.fill_(0)
        torch.nn.init.kaiming_uniform(self.conv3.weight)
        self.conv3.bias.data.fill_(0)
        torch.nn.init.kaiming_uniform(self.conv4.weight)
        self.conv4.bias.data.fill_(0)
        torch.nn.init.kaiming_uniform(self.conv5.weight)
        self.conv5.bias.data.fill_(0)

        torch.nn.init.kaiming_uniform(self.deconv1.weight)
        self.deconv1.bias.data.fill_(0)
        torch.nn.init.kaiming_uniform(self.deconv2.weight)
        self.deconv2.bias.data.fill_(0)
        torch.nn.init.kaiming_uniform(self.deconv3.weight)
        self.deconv3.bias.data.fill_(0)
        torch.nn.init.kaiming_uniform(self.deconv4.weight)
        self.deconv4.bias.data.fill_(0)
        torch.nn.init.kaiming_uniform(self.deconv5.weight)
        self.deconv5.bias.data.fill_(0)

    def save_attention_prob(self, image, attention_prob, instruction, goal_prob=None):
        self.global_id += 1

        image_flipped = image[0, 0, :, :, :].cpu().data.numpy().swapaxes(0, 1).swapaxes(1, 2)
        attention_prob = attention_prob.cpu().data.numpy()
        resized_kernel = scipy.misc.imresize(attention_prob, (128, 128))
        if goal_prob is not None:
            goal_location = goal_prob.cpu().data.numpy()
            if np.sum(goal_location) > 0.01:
                for i in range(0, self.final_image_height):
                    for j in range(0, self.final_image_width):
                            if goal_location[i][j] < 0.01:
                                goal_location[i][j] = 0.0
                goal_location = scipy.misc.imresize(goal_location, (128, 128))
        else:
            goal_location = None

        f, axarr = plt.subplots(1, 2)
        if instruction is not None:
            f.suptitle(instruction)
        axarr[0].set_title("Predicted Attention")
        axarr[0].imshow(image_flipped)
        axarr[0].imshow(resized_kernel, cmap='jet', alpha=0.5)
        axarr[1].set_title("Gold Attention (Goal)")
        axarr[1].imshow(image_flipped)
        if goal_location is not None:
            axarr[1].imshow(goal_location, cmap='jet', alpha=0.5)
        plt.savefig("./attention_prob/image_" + str(self.global_id) + ".png")
        plt.clf()

    def forward(self, image, instructions, tx, mode, model_state, instruction, goal):

        if model_state is None:
            _, sentence_embedding = self.text_module(instructions)
            image_hidden_states = None
        else:
            sentence_embedding, image_hidden_states = model_state

        image_emb_seq = self.image_module(image)
        image_input = image_emb_seq[:, 0, :, :, :]  # 1 x num_channels x height x width

        x1 = self.norm2(self.act(self.conv1(image_input)))
        x2 = self.norm3(self.act(self.conv2(x1)))
        x3 = self.norm4(self.act(self.conv3(x2)))
        x4 = self.norm5(self.act(self.conv4(x3)))
        x5 = self.act(self.conv5(x4))

        if sentence_embedding is not None:
            if self.split_embedding:
                block_size = self.emb_block_size
                emb1 = sentence_embedding[:, 0 * block_size:1 * block_size]
                emb2 = sentence_embedding[:, 1 * block_size:2 * block_size]
                emb3 = sentence_embedding[:, 2 * block_size:3 * block_size]
                emb4 = sentence_embedding[:, 3 * block_size:4 * block_size]
                emb5 = sentence_embedding[:, 4 * block_size:5 * block_size]
            else:
                emb1 = emb2 = emb3 = emb4 = emb5 = sentence_embedding

            lf1 = F.normalize(self.lang19(emb1)).view([self.hc1, self.hc1, 1, 1])
            lf2 = F.normalize(self.lang28(emb2)).view([self.hc1, self.hc1, 1, 1])
            lf3 = F.normalize(self.lang37(emb3)).view([self.hc1, self.hc1, 1, 1])
            lf4 = F.normalize(self.lang46(emb4)).view([self.hc1, self.hc1, 1, 1])
            lf5 = F.normalize(self.lang55(emb5)).view([self.hc1, self.hc1, 1, 1])

            x1f = F.conv2d(x1, lf1)
            x2f = F.conv2d(x2, lf2)
            x3f = F.conv2d(x3, lf3)
            x4f = F.conv2d(x4, lf4)
            x5f = F.conv2d(x5, lf5)

            x5f = self.dropout(x5f)
        else:
            raise AssertionError("Embedding should not be none.")

        x6 = self.act(self.deconv1(x5f, output_size=x4.size()))
        x46 = torch.cat([x4f, x6], 1)
        x7 = self.dnorm3(self.act(self.deconv2(x46, output_size=x3.size())))
        x37 = torch.cat([x3f, x7], 1)
        x8 = self.dnorm4(self.act(self.deconv3(x37, output_size=x2.size())))
        x28 = torch.cat([x2f, x8], 1)
        x9 = self.dnorm5(self.act(self.deconv4(x28, output_size=x1.size())))
        x19 = torch.cat([x1f, x9], 1)
        out = self.deconv5(x19, output_size=image_input.size())

        attention_logits = out.view(-1)
        attention_logits = torch.cat([attention_logits, self.bias.bias.view(-1)], dim=0)
        attention_prob = F.softmax(attention_logits, dim=0)  # (height x width + 1)

        volatile = dict()
        volatile["image_emb"] = image_input
        volatile["attention_logits"] = attention_logits  # 32x32 + 1
        volatile["attention_probs"] = attention_prob

        # if True:
        #     goal_prob = GoalPrediction.generate_gold_prob(goal, self.final_image_height, self.final_image_width)
        #     self.save_attention_prob(image, attention_prob[:-1].view(self.final_image_height, self.final_image_width),
        #                              instruction, goal_prob[:-1].view(self.final_image_height, self.final_image_width))

        # attention_prob_present = attention_prob[:-1].view(1, 1, self.final_image_height, self.final_image_width)
        # x = torch.cat([image_input, attention_prob_present], dim=1)  # 1 x (num_channels + 1) x height x width
        # x = self.act(self.attention_mix_conv1(x))
        # x = self.act(self.attention_mix_conv2(x))
        # x = self.attention_mix_conv3(x)  # 1 x 1 x height x width
        # x = x.view(1, -1)

        spatial_information = self.act(self.attention_spatial_compressor1(image_input))  # 1 x 16 x self.height, self.width
        spatial_information = self.act(
            self.attention_spatial_compressor2(spatial_information))  # 1 x 8 x self.height, self.width
        spatial_information = self.attention_spatial_compressor3(spatial_information)  # 1 x 1 x self.height, self.width
        spatial_information = spatial_information.view(1, -1)

        # Concatenate the probabilities as they are useful
        x = torch.cat([spatial_information, attention_logits.view(1, -1)], dim=1)
        x = self.act(self.attention_mixer(x))
        # 1 x (height x width + height' x width' + 1)

        # Compute the probabilities
        # x = attention_prob.view(1, -1)
        x = F.relu(self.linear(x))
        new_image_hidden_states = self.image_recurrence_module(x, image_hidden_states)
        hx, cx = new_image_hidden_states

        time_emb = self.time_emb_layer(tx)
        x = torch.cat((hx, time_emb.view(-1, self.time_emb_dim)), 1)

        actor_linear = self.actor_linear(x)
        critic_linear = self.critic_linear(x)

        new_model_state = (sentence_embedding, new_image_hidden_states)

        return F.log_softmax(actor_linear, dim=1), new_model_state, None, volatile

    def get_attention_prob(self, image, instructions, instruction, goal):

        _, sentence_embedding = self.text_module(instructions)

        image_emb_seq = self.image_module(image)
        image_input = image_emb_seq[:, 0, :, :, :]  # 1 x num_channels x height x width

        x1 = self.norm2(self.act(self.conv1(image_input)))
        x2 = self.norm3(self.act(self.conv2(x1)))
        x3 = self.norm4(self.act(self.conv3(x2)))
        x4 = self.norm5(self.act(self.conv4(x3)))
        x5 = self.act(self.conv5(x4))

        if sentence_embedding is not None:
            if self.split_embedding:
                block_size = self.emb_block_size
                emb1 = sentence_embedding[:, 0 * block_size:1 * block_size]
                emb2 = sentence_embedding[:, 1 * block_size:2 * block_size]
                emb3 = sentence_embedding[:, 2 * block_size:3 * block_size]
                emb4 = sentence_embedding[:, 3 * block_size:4 * block_size]
                emb5 = sentence_embedding[:, 4 * block_size:5 * block_size]
            else:
                emb1 = emb2 = emb3 = emb4 = emb5 = sentence_embedding

            lf1 = F.normalize(self.lang19(emb1)).view([self.hc1, self.hc1, 1, 1])
            lf2 = F.normalize(self.lang28(emb2)).view([self.hc1, self.hc1, 1, 1])
            lf3 = F.normalize(self.lang37(emb3)).view([self.hc1, self.hc1, 1, 1])
            lf4 = F.normalize(self.lang46(emb4)).view([self.hc1, self.hc1, 1, 1])
            lf5 = F.normalize(self.lang55(emb5)).view([self.hc1, self.hc1, 1, 1])

            x1f = F.conv2d(x1, lf1)
            x2f = F.conv2d(x2, lf2)
            x3f = F.conv2d(x3, lf3)
            x4f = F.conv2d(x4, lf4)
            x5f = F.conv2d(x5, lf5)

            x5f = self.dropout(x5f)
        else:
            raise AssertionError("Embedding should not be none.")

        x6 = self.act(self.deconv1(x5f, output_size=x4.size()))
        x46 = torch.cat([x4f, x6], 1)
        x7 = self.dnorm3(self.act(self.deconv2(x46, output_size=x3.size())))
        x37 = torch.cat([x3f, x7], 1)
        x8 = self.dnorm4(self.act(self.deconv3(x37, output_size=x2.size())))
        x28 = torch.cat([x2f, x8], 1)
        x9 = self.dnorm5(self.act(self.deconv4(x28, output_size=x1.size())))
        x19 = torch.cat([x1f, x9], 1)
        out = self.deconv5(x19, output_size=image_input.size())

        attention_logits = out.view(-1)
        attention_logits = torch.cat([attention_logits, self.bias.bias.view(-1)], dim=0)
        attention_prob = F.softmax(attention_logits, dim=0)  # (height x width + 1)

        volatile = dict()
        volatile["image_emb"] = image_input
        volatile["attention_logits"] = attention_logits  # 32x32 + 1
        volatile["attention_probs"] = attention_prob

        # if True:
        #     goal_prob = GoalPrediction.generate_gold_prob(goal, self.final_image_height, self.final_image_width)
        #     self.save_attention_prob(image, attention_prob[:-1].view(self.final_image_height, self.final_image_width),
        #                              instruction, goal_prob[:-1].view(self.final_image_height, self.final_image_width))

        return volatile


class IncrementalUnetAttentionModuleJustProb(torch.nn.Module):
    """ Unet architecture designed by Valts Blukis, based on original Unet paper, for predicting the image. """

    def __init__(self,  image_module, text_module, image_recurrence_module,
                 max_episode_length, final_image_height, final_image_width,
                 in_channels, out_channels, embedding_size, hc1=32, hc2=16,
                 k=5, stride=2, split_embedding=False):
        super(IncrementalUnetAttentionModuleJustProb, self).__init__()

        self.image_module = image_module
        self.image_recurrence_module = image_recurrence_module
        self.text_module = text_module

        pad = int(k / 2)
        self.hc1 = hc1
        self.hc2 = hc2

        self.split_embedding = split_embedding

        self.embedding_size = embedding_size
        if split_embedding:
            self.emb_block_size = int(embedding_size / 5)
        else:
            self.emb_block_size = embedding_size

        # inchannels, outchannels, kernel size
        self.conv1 = nn.Conv2d(in_channels, hc1, k, stride=stride, padding=pad)
        self.conv2 = nn.Conv2d(hc1, hc1, k, stride=stride, padding=pad)
        self.conv3 = nn.Conv2d(hc1, hc1, k, stride=stride, padding=pad)
        self.conv4 = nn.Conv2d(hc1, hc1, k, stride=stride, padding=pad)
        self.conv5 = nn.Conv2d(hc1, hc1, k, stride=stride, padding=pad)

        self.deconv1 = nn.ConvTranspose2d(hc1, hc1, k, stride=stride, padding=pad)
        self.deconv2 = nn.ConvTranspose2d(2 * hc1, hc1, k, stride=stride, padding=pad)
        self.deconv3 = nn.ConvTranspose2d(2 * hc1, hc1, k, stride=stride, padding=pad)
        self.deconv4 = nn.ConvTranspose2d(2 * hc1, hc2, k, stride=stride, padding=pad)
        self.deconv5 = nn.ConvTranspose2d(hc1 + hc2, out_channels, k, stride=stride, padding=pad)

        self.act = nn.LeakyReLU()
        self.dropout = nn.Dropout(0.5)
        self.norm2 = nn.InstanceNorm2d(hc1)
        self.norm3 = nn.InstanceNorm2d(hc1)
        self.norm4 = nn.InstanceNorm2d(hc1)
        self.norm5 = nn.InstanceNorm2d(hc1)
        # self.dnorm1 = nn.InstanceNorm2d(in_channels * 4)
        self.dnorm2 = nn.InstanceNorm2d(hc1)
        self.dnorm3 = nn.InstanceNorm2d(hc1)
        self.dnorm4 = nn.InstanceNorm2d(hc1)
        self.dnorm5 = nn.InstanceNorm2d(hc2)

        self.lang19 = nn.Linear(self.emb_block_size, hc1 * hc1)
        self.lang28 = nn.Linear(self.emb_block_size, hc1 * hc1)
        self.lang37 = nn.Linear(self.emb_block_size, hc1 * hc1)
        self.lang46 = nn.Linear(self.emb_block_size, hc1 * hc1)
        self.lang55 = nn.Linear(self.emb_block_size, hc1 * hc1)

        # Time embedding layer, helps in stabilizing value prediction
        self.time_emb_dim = 32
        self.time_emb_layer = nn.Embedding(max_episode_length + 1, self.time_emb_dim)

        # A3C-LSTM layers
        self.final_channels = in_channels
        self.final_image_height = final_image_height
        self.final_image_width = final_image_width

        final_channel, height, width = self.image_module.get_final_dimension()
        self.bias = nn.Linear(1, 1)
        self.linear = nn.Linear(height * width + 1, 256)
        # self.linear = nn.Linear(self.final_channels, 256)
        self.critic_linear = nn.Linear(256 + self.time_emb_dim, 1)
        self.actor_linear = nn.Linear(256 + self.time_emb_dim, 4)

        # For logs
        self.global_id = 0

    def init_weights(self):
        torch.nn.init.kaiming_uniform(self.conv1.weight)
        self.conv1.bias.data.fill_(0)
        torch.nn.init.kaiming_uniform(self.conv2.weight)
        self.conv2.bias.data.fill_(0)
        torch.nn.init.kaiming_uniform(self.conv3.weight)
        self.conv3.bias.data.fill_(0)
        torch.nn.init.kaiming_uniform(self.conv4.weight)
        self.conv4.bias.data.fill_(0)
        torch.nn.init.kaiming_uniform(self.conv5.weight)
        self.conv5.bias.data.fill_(0)

        torch.nn.init.kaiming_uniform(self.deconv1.weight)
        self.deconv1.bias.data.fill_(0)
        torch.nn.init.kaiming_uniform(self.deconv2.weight)
        self.deconv2.bias.data.fill_(0)
        torch.nn.init.kaiming_uniform(self.deconv3.weight)
        self.deconv3.bias.data.fill_(0)
        torch.nn.init.kaiming_uniform(self.deconv4.weight)
        self.deconv4.bias.data.fill_(0)
        torch.nn.init.kaiming_uniform(self.deconv5.weight)
        self.deconv5.bias.data.fill_(0)

    def save_attention_prob(self, image, attention_prob, instruction, goal_prob=None):
        self.global_id += 1

        image_flipped = image[0, 0, :, :, :].cpu().data.numpy().swapaxes(0, 1).swapaxes(1, 2)
        image_flipped = scipy.misc.imresize(image_flipped, (128, 128 * 6))
        attention_prob = attention_prob.cpu().data.numpy()
        # resized_kernel = scipy.misc.imresize(attention_prob, (128*5, 128*6 * 5))
        resized_kernel = scipy.misc.imresize(attention_prob, (128, 128*6))
        if goal_prob is not None:
            goal_location = goal_prob.cpu().data.numpy()
            if np.sum(goal_location) > 0.01:
                for i in range(0, self.final_image_height):
                    for j in range(0, self.final_image_width):
                            if goal_location[i][j] < 0.01:
                                goal_location[i][j] = 0.0
                goal_location = scipy.misc.imresize(goal_location, (128, 128*6))  # (128*5, 128*6*5))
        else:
            goal_location = None

        f, axarr = plt.subplots(1, 2)
        if instruction is not None:
            f.suptitle(instruction)
        axarr[0].set_title("Predicted Attention")
        axarr[0].imshow(image_flipped)
        axarr[0].imshow(resized_kernel, cmap='jet', alpha=0.5)
        axarr[1].set_title("Gold Attention (Goal)")
        axarr[1].imshow(image_flipped)
        if goal_location is not None:
            axarr[1].imshow(goal_location, cmap='jet', alpha=0.5)
        plt.savefig("./attention_prob/image_" + str(self.global_id) + ".png")
        plt.clf()

    def forward(self, image, instructions, tx, mode, model_state, instruction, goal):

        if model_state is None:
            _, sentence_embedding = self.text_module(instructions)
            image_hidden_states = None
        else:
            sentence_embedding, image_hidden_states = model_state

        image_emb_seq = self.image_module(image)
        image_input = image_emb_seq[:, 0, :, :, :]  # 1 x num_channels x height x width

        x1 = self.norm2(self.act(self.conv1(image_input)))
        x2 = self.norm3(self.act(self.conv2(x1)))
        x3 = self.norm4(self.act(self.conv3(x2)))
        x4 = self.norm5(self.act(self.conv4(x3)))
        x5 = self.act(self.conv5(x4))

        if sentence_embedding is not None:
            if self.split_embedding:
                block_size = self.emb_block_size
                emb1 = sentence_embedding[:, 0 * block_size:1 * block_size]
                emb2 = sentence_embedding[:, 1 * block_size:2 * block_size]
                emb3 = sentence_embedding[:, 2 * block_size:3 * block_size]
                emb4 = sentence_embedding[:, 3 * block_size:4 * block_size]
                emb5 = sentence_embedding[:, 4 * block_size:5 * block_size]
            else:
                emb1 = emb2 = emb3 = emb4 = emb5 = sentence_embedding

            lf1 = F.normalize(self.lang19(emb1)).view([self.hc1, self.hc1, 1, 1])
            lf2 = F.normalize(self.lang28(emb2)).view([self.hc1, self.hc1, 1, 1])
            lf3 = F.normalize(self.lang37(emb3)).view([self.hc1, self.hc1, 1, 1])
            lf4 = F.normalize(self.lang46(emb4)).view([self.hc1, self.hc1, 1, 1])
            lf5 = F.normalize(self.lang55(emb5)).view([self.hc1, self.hc1, 1, 1])

            x1f = F.conv2d(x1, lf1)
            x2f = F.conv2d(x2, lf2)
            x3f = F.conv2d(x3, lf3)
            x4f = F.conv2d(x4, lf4)
            x5f = F.conv2d(x5, lf5)

            x5f = self.dropout(x5f)
        else:
            raise AssertionError("Embedding should not be none.")

        x6 = self.act(self.deconv1(x5f, output_size=x4.size()))
        x46 = torch.cat([x4f, x6], 1)
        x7 = self.dnorm3(self.act(self.deconv2(x46, output_size=x3.size())))
        x37 = torch.cat([x3f, x7], 1)
        x8 = self.dnorm4(self.act(self.deconv3(x37, output_size=x2.size())))
        x28 = torch.cat([x2f, x8], 1)
        x9 = self.dnorm5(self.act(self.deconv4(x28, output_size=x1.size())))
        x19 = torch.cat([x1f, x9], 1)
        out = self.deconv5(x19, output_size=image_input.size())

        attention_logits = out.view(-1)
        attention_logits = torch.cat([attention_logits, self.bias.bias.view(-1)], dim=0)
        attention_prob = F.softmax(attention_logits, dim=0)  # (height x width + 1)

        volatile = dict()
        volatile["image_emb"] = image_input
        volatile["attention_logits"] = attention_logits  # 32x32 + 1
        volatile["attention_probs"] = attention_prob

        if True:
            goal_prob = GoalPrediction.generate_gold_prob(goal, self.final_image_height, self.final_image_width)
            self.save_attention_prob(image, attention_prob[:-1].view(self.final_image_height, self.final_image_width),
                                     instruction, goal_prob[:-1].view(self.final_image_height, self.final_image_width))

        # Compute the probabilities
        x = attention_prob.view(1, -1)
        x = F.relu(self.linear(x))
        new_image_hidden_states = self.image_recurrence_module(x, image_hidden_states)
        hx, cx = new_image_hidden_states

        time_emb = self.time_emb_layer(tx)
        x = torch.cat((hx, time_emb.view(-1, self.time_emb_dim)), 1)

        actor_linear = self.actor_linear(x)
        critic_linear = self.critic_linear(x)
        volatile["state_value"] = critic_linear

        new_model_state = (sentence_embedding, new_image_hidden_states)

        return F.log_softmax(actor_linear, dim=1), new_model_state, None, volatile

    def get_attention_prob(self, image, instructions, instruction, goal):

        _, sentence_embedding = self.text_module(instructions)

        image_emb_seq = self.image_module(image)
        image_input = image_emb_seq[:, 0, :, :, :]  # 1 x num_channels x height x width

        x1 = self.norm2(self.act(self.conv1(image_input)))
        x2 = self.norm3(self.act(self.conv2(x1)))
        x3 = self.norm4(self.act(self.conv3(x2)))
        x4 = self.norm5(self.act(self.conv4(x3)))
        x5 = self.act(self.conv5(x4))

        if sentence_embedding is not None:
            if self.split_embedding:
                block_size = self.emb_block_size
                emb1 = sentence_embedding[:, 0 * block_size:1 * block_size]
                emb2 = sentence_embedding[:, 1 * block_size:2 * block_size]
                emb3 = sentence_embedding[:, 2 * block_size:3 * block_size]
                emb4 = sentence_embedding[:, 3 * block_size:4 * block_size]
                emb5 = sentence_embedding[:, 4 * block_size:5 * block_size]
            else:
                emb1 = emb2 = emb3 = emb4 = emb5 = sentence_embedding

            lf1 = F.normalize(self.lang19(emb1)).view([self.hc1, self.hc1, 1, 1])
            lf2 = F.normalize(self.lang28(emb2)).view([self.hc1, self.hc1, 1, 1])
            lf3 = F.normalize(self.lang37(emb3)).view([self.hc1, self.hc1, 1, 1])
            lf4 = F.normalize(self.lang46(emb4)).view([self.hc1, self.hc1, 1, 1])
            lf5 = F.normalize(self.lang55(emb5)).view([self.hc1, self.hc1, 1, 1])

            x1f = F.conv2d(x1, lf1)
            x2f = F.conv2d(x2, lf2)
            x3f = F.conv2d(x3, lf3)
            x4f = F.conv2d(x4, lf4)
            x5f = F.conv2d(x5, lf5)

            x5f = self.dropout(x5f)
        else:
            raise AssertionError("Embedding should not be none.")

        x6 = self.act(self.deconv1(x5f, output_size=x4.size()))
        x46 = torch.cat([x4f, x6], 1)
        x7 = self.dnorm3(self.act(self.deconv2(x46, output_size=x3.size())))
        x37 = torch.cat([x3f, x7], 1)
        x8 = self.dnorm4(self.act(self.deconv3(x37, output_size=x2.size())))
        x28 = torch.cat([x2f, x8], 1)
        x9 = self.dnorm5(self.act(self.deconv4(x28, output_size=x1.size())))
        x19 = torch.cat([x1f, x9], 1)
        out = self.deconv5(x19, output_size=image_input.size())

        attention_logits = out.view(-1)
        attention_logits = torch.cat([attention_logits, self.bias.bias.view(-1)], dim=0)
        attention_prob = F.softmax(attention_logits, dim=0)  # (height x width + 1)

        volatile = dict()
        volatile["image_emb"] = image_input
        volatile["attention_logits"] = attention_logits
        volatile["attention_probs"] = attention_prob

        logging.info("Instruction is ", instruction)
        # if True:
        #     goal_prob = GoalPrediction.generate_gold_prob(goal, self.final_image_height, self.final_image_width)
        #     self.save_attention_prob(image, attention_prob[:-1].view(self.final_image_height, self.final_image_width),
        #                              instruction, goal_prob[:-1].view(self.final_image_height, self.final_image_width))

        return volatile


class IncrementalUnetAttentionModuleJustProbSpatialEncoding(torch.nn.Module):
    """ Unet architecture designed by Valts Blukis, based on original Unet paper, for predicting the image. """

    def __init__(self,  image_module, text_module, image_recurrence_module,
                 max_episode_length, final_image_height, final_image_width,
                 in_channels, out_channels, embedding_size, hc1=32, hc2=16,
                 k=5, stride=2, split_embedding=False):
        super(IncrementalUnetAttentionModuleJustProbSpatialEncoding, self).__init__()

        self.image_module = image_module
        self.image_recurrence_module = image_recurrence_module
        self.text_module = text_module

        pad = int(k / 2)
        self.hc1 = hc1
        self.hc2 = hc2

        self.split_embedding = split_embedding

        self.embedding_size = embedding_size
        if split_embedding:
            self.emb_block_size = int(embedding_size / 5)
        else:
            self.emb_block_size = embedding_size

        # Positional encoding
        num_positional_encodings = 6
        in_channels = in_channels + num_positional_encodings
        self.encodings = self.get_positional_encoding()

        # inchannels, outchannels, kernel size
        self.conv1 = nn.Conv2d(in_channels, hc1, k, stride=stride, padding=pad)
        self.conv2 = nn.Conv2d(hc1, hc1, k, stride=stride, padding=pad)
        self.conv3 = nn.Conv2d(hc1, hc1, k, stride=stride, padding=pad)
        self.conv4 = nn.Conv2d(hc1, hc1, k, stride=stride, padding=pad)
        self.conv5 = nn.Conv2d(hc1, hc1, k, stride=stride, padding=pad)

        self.deconv1 = nn.ConvTranspose2d(hc1, hc1, k, stride=stride, padding=pad)
        self.deconv2 = nn.ConvTranspose2d(2 * hc1, hc1, k, stride=stride, padding=pad)
        self.deconv3 = nn.ConvTranspose2d(2 * hc1, hc1, k, stride=stride, padding=pad)
        self.deconv4 = nn.ConvTranspose2d(2 * hc1, hc2, k, stride=stride, padding=pad)
        self.deconv5 = nn.ConvTranspose2d(hc1 + hc2, out_channels, k, stride=stride, padding=pad)

        self.act = nn.LeakyReLU()
        self.dropout = nn.Dropout(0.5)
        self.norm2 = nn.InstanceNorm2d(hc1)
        self.norm3 = nn.InstanceNorm2d(hc1)
        self.norm4 = nn.InstanceNorm2d(hc1)
        self.norm5 = nn.InstanceNorm2d(hc1)
        # self.dnorm1 = nn.InstanceNorm2d(in_channels * 4)
        self.dnorm2 = nn.InstanceNorm2d(hc1)
        self.dnorm3 = nn.InstanceNorm2d(hc1)
        self.dnorm4 = nn.InstanceNorm2d(hc1)
        self.dnorm5 = nn.InstanceNorm2d(hc2)

        self.lang19 = nn.Linear(self.emb_block_size, hc1 * hc1)
        self.lang28 = nn.Linear(self.emb_block_size, hc1 * hc1)
        self.lang37 = nn.Linear(self.emb_block_size, hc1 * hc1)
        self.lang46 = nn.Linear(self.emb_block_size, hc1 * hc1)
        self.lang55 = nn.Linear(self.emb_block_size, hc1 * hc1)

        # Time embedding layer, helps in stabilizing value prediction
        self.time_emb_dim = 32
        self.time_emb_layer = nn.Embedding(max_episode_length + 1, self.time_emb_dim)

        # A3C-LSTM layers
        self.final_channels = in_channels
        self.final_image_height = final_image_height
        self.final_image_width = final_image_width

        final_channel, height, width = self.image_module.get_final_dimension()
        self.bias = nn.Linear(1, 1)
        self.linear = nn.Linear(height * width + 1, 256)
        # self.linear = nn.Linear(self.final_channels, 256)
        self.critic_linear = nn.Linear(256 + self.time_emb_dim, 1)
        self.actor_linear = nn.Linear(256 + self.time_emb_dim, 4)

        # For logs
        self.global_id = 0

    def get_positional_encoding(self):

        zeroes = np.zeros((32, 32))
        ones = np.ones((32, 32))
        encodings = []
        for i in range(0, 6):
            val = [zeroes, zeroes, zeroes, zeroes, zeroes]
            val.insert(i, ones)
            val = np.hstack(val)
            tensor = torch.from_numpy(val)  # 32 x 192
            encoding = cuda_var(tensor).float()  # 32 x 192
            encoding = encoding.view(1, 1, 32, 192)
            encodings.append(encoding)

        encodings = torch.cat(encodings, dim=1)  # 1 x 6 x 32 x 192
        return encodings

    def init_weights(self):
        torch.nn.init.kaiming_uniform(self.conv1.weight)
        self.conv1.bias.data.fill_(0)
        torch.nn.init.kaiming_uniform(self.conv2.weight)
        self.conv2.bias.data.fill_(0)
        torch.nn.init.kaiming_uniform(self.conv3.weight)
        self.conv3.bias.data.fill_(0)
        torch.nn.init.kaiming_uniform(self.conv4.weight)
        self.conv4.bias.data.fill_(0)
        torch.nn.init.kaiming_uniform(self.conv5.weight)
        self.conv5.bias.data.fill_(0)

        torch.nn.init.kaiming_uniform(self.deconv1.weight)
        self.deconv1.bias.data.fill_(0)
        torch.nn.init.kaiming_uniform(self.deconv2.weight)
        self.deconv2.bias.data.fill_(0)
        torch.nn.init.kaiming_uniform(self.deconv3.weight)
        self.deconv3.bias.data.fill_(0)
        torch.nn.init.kaiming_uniform(self.deconv4.weight)
        self.deconv4.bias.data.fill_(0)
        torch.nn.init.kaiming_uniform(self.deconv5.weight)
        self.deconv5.bias.data.fill_(0)

    def save_attention_prob(self, image, attention_prob, instruction, goal_prob=None):
        # self.global_id += 1

        image_flipped = image[0, 0, :, :, :].cpu().data.numpy().swapaxes(0, 1).swapaxes(1, 2)
        image_flipped = scipy.misc.imresize(image_flipped, (128 * 5, 128 * 6 * 5))
        attention_prob = attention_prob.cpu().data.numpy()
        resized_kernel = scipy.misc.imresize(attention_prob, (128 * 5, 128 * 6 * 5))
        if goal_prob is not None:
            goal_location = goal_prob.cpu().data.numpy()
            if np.sum(goal_location) > 0.01:
                for i in range(0, self.final_image_height):
                    for j in range(0, self.final_image_width):
                            if goal_location[i][j] < 0.01:
                                goal_location[i][j] = 0.0
                goal_location = scipy.misc.imresize(goal_location, (128 * 5, 128 * 6 * 5))
        else:
            goal_location = None

        f, axarr = plt.subplots(1, 2)
        if instruction is not None:
            f.suptitle("\n".join(wrap(instruction)))
        axarr[0].set_title("Predicted Attention")
        axarr[0].imshow(image_flipped)
        axarr[0].imshow(resized_kernel, cmap='jet', alpha=0.5)
        axarr[1].set_title("Gold Attention (Goal)")
        axarr[1].imshow(image_flipped)
        if goal_location is not None:
            axarr[1].imshow(goal_location, cmap='jet', alpha=0.5)
        plt.savefig("./attention_prob/image_" + str(self.global_id) + ".png")
        # np.save("./attention_prob/image_" + str(self.global_id) + ".npy", image_flipped)
        plt.clf()

    def forward(self, image, instructions, tx, mode, model_state, instruction, goal):

        if model_state is None:
            _, sentence_embedding = self.text_module(instructions)
            image_hidden_states = None
        else:
            sentence_embedding, image_hidden_states = model_state

        image_emb_seq = self.image_module(image)
        image_input = image_emb_seq[:, 0, :, :, :]  # 1 x num_channels x height x width

        x1 = self.norm2(self.act(self.conv1(image_input)))
        x2 = self.norm3(self.act(self.conv2(x1)))
        x3 = self.norm4(self.act(self.conv3(x2)))
        x4 = self.norm5(self.act(self.conv4(x3)))
        x5 = self.act(self.conv5(x4))

        if sentence_embedding is not None:
            if self.split_embedding:
                block_size = self.emb_block_size
                emb1 = sentence_embedding[:, 0 * block_size:1 * block_size]
                emb2 = sentence_embedding[:, 1 * block_size:2 * block_size]
                emb3 = sentence_embedding[:, 2 * block_size:3 * block_size]
                emb4 = sentence_embedding[:, 3 * block_size:4 * block_size]
                emb5 = sentence_embedding[:, 4 * block_size:5 * block_size]
            else:
                emb1 = emb2 = emb3 = emb4 = emb5 = sentence_embedding

            lf1 = F.normalize(self.lang19(emb1)).view([self.hc1, self.hc1, 1, 1])
            lf2 = F.normalize(self.lang28(emb2)).view([self.hc1, self.hc1, 1, 1])
            lf3 = F.normalize(self.lang37(emb3)).view([self.hc1, self.hc1, 1, 1])
            lf4 = F.normalize(self.lang46(emb4)).view([self.hc1, self.hc1, 1, 1])
            lf5 = F.normalize(self.lang55(emb5)).view([self.hc1, self.hc1, 1, 1])

            x1f = F.conv2d(x1, lf1)
            x2f = F.conv2d(x2, lf2)
            x3f = F.conv2d(x3, lf3)
            x4f = F.conv2d(x4, lf4)
            x5f = F.conv2d(x5, lf5)

            x5f = self.dropout(x5f)
        else:
            raise AssertionError("Embedding should not be none.")

        x6 = self.act(self.deconv1(x5f, output_size=x4.size()))
        x46 = torch.cat([x4f, x6], 1)
        x7 = self.dnorm3(self.act(self.deconv2(x46, output_size=x3.size())))
        x37 = torch.cat([x3f, x7], 1)
        x8 = self.dnorm4(self.act(self.deconv3(x37, output_size=x2.size())))
        x28 = torch.cat([x2f, x8], 1)
        x9 = self.dnorm5(self.act(self.deconv4(x28, output_size=x1.size())))
        x19 = torch.cat([x1f, x9], 1)
        out = self.deconv5(x19, output_size=image_input.size())

        attention_logits = out.view(-1)
        attention_logits = torch.cat([attention_logits, self.bias.bias.view(-1)], dim=0)
        attention_prob = F.softmax(attention_logits, dim=0)  # (height x width + 1)

        volatile = dict()
        volatile["image_emb"] = image_input
        volatile["attention_logits"] = attention_logits  # 32x32 + 1
        volatile["attention_probs"] = attention_prob

        # if True:
        #     goal_prob = GoalPrediction.generate_gold_prob(goal, self.final_image_height, self.final_image_width)
        #     self.save_attention_prob(image, attention_prob[:-1].view(self.final_image_height, self.final_image_width),
        #                              instruction, goal_prob[:-1].view(self.final_image_height, self.final_image_width))

        # Compute the probabilities
        x = attention_prob.view(1, -1)
        x = F.relu(self.linear(x))
        new_image_hidden_states = self.image_recurrence_module(x, image_hidden_states)
        hx, cx = new_image_hidden_states

        time_emb = self.time_emb_layer(tx)
        x = torch.cat((hx, time_emb.view(-1, self.time_emb_dim)), 1)

        actor_linear = self.actor_linear(x)
        critic_linear = self.critic_linear(x)
        volatile["state_value"] = critic_linear

        new_model_state = (sentence_embedding, new_image_hidden_states)

        return F.log_softmax(actor_linear, dim=1), new_model_state, None, volatile

    def get_attention_prob(self, image, instructions, instruction, goal):

        _, sentence_embedding = self.text_module(instructions)

        image_emb_seq = self.image_module(image)
        image_input = image_emb_seq[:, 0, :, :, :]  # 1 x num_channels x height x width
        image_input = torch.cat([image_input, self.encodings], dim=1)    # 1 x (num_channels + encodings) x height x width

        x1 = self.norm2(self.act(self.conv1(image_input)))
        x2 = self.norm3(self.act(self.conv2(x1)))
        x3 = self.norm4(self.act(self.conv3(x2)))
        x4 = self.norm5(self.act(self.conv4(x3)))
        x5 = self.act(self.conv5(x4))

        if sentence_embedding is not None:
            if self.split_embedding:
                block_size = self.emb_block_size
                emb1 = sentence_embedding[:, 0 * block_size:1 * block_size]
                emb2 = sentence_embedding[:, 1 * block_size:2 * block_size]
                emb3 = sentence_embedding[:, 2 * block_size:3 * block_size]
                emb4 = sentence_embedding[:, 3 * block_size:4 * block_size]
                emb5 = sentence_embedding[:, 4 * block_size:5 * block_size]
            else:
                emb1 = emb2 = emb3 = emb4 = emb5 = sentence_embedding

            lf1 = F.normalize(self.lang19(emb1)).view([self.hc1, self.hc1, 1, 1])
            lf2 = F.normalize(self.lang28(emb2)).view([self.hc1, self.hc1, 1, 1])
            lf3 = F.normalize(self.lang37(emb3)).view([self.hc1, self.hc1, 1, 1])
            lf4 = F.normalize(self.lang46(emb4)).view([self.hc1, self.hc1, 1, 1])
            lf5 = F.normalize(self.lang55(emb5)).view([self.hc1, self.hc1, 1, 1])

            x1f = F.conv2d(x1, lf1)
            x2f = F.conv2d(x2, lf2)
            x3f = F.conv2d(x3, lf3)
            x4f = F.conv2d(x4, lf4)
            x5f = F.conv2d(x5, lf5)

            x5f = self.dropout(x5f)
        else:
            raise AssertionError("Embedding should not be none.")

        x6 = self.act(self.deconv1(x5f, output_size=x4.size()))
        x46 = torch.cat([x4f, x6], 1)
        x7 = self.dnorm3(self.act(self.deconv2(x46, output_size=x3.size())))
        x37 = torch.cat([x3f, x7], 1)
        x8 = self.dnorm4(self.act(self.deconv3(x37, output_size=x2.size())))
        x28 = torch.cat([x2f, x8], 1)
        x9 = self.dnorm5(self.act(self.deconv4(x28, output_size=x1.size())))
        x19 = torch.cat([x1f, x9], 1)
        out = self.deconv5(x19, output_size=image_input.size())

        attention_logits = out.view(-1)
        attention_logits = torch.cat([attention_logits, self.bias.bias.view(-1)], dim=0)
        attention_prob = F.softmax(attention_logits, dim=0)  # (height x width + 1)

        volatile = dict()
        volatile["image_emb"] = image_input
        volatile["attention_logits"] = attention_logits
        volatile["attention_probs"] = attention_prob

        self.global_id += 1
        if False: # self.global_id % 500 == 0:
            if goal is not None:
                goal_prob = GoalPrediction.generate_gold_prob(goal, self.final_image_height, self.final_image_width)
                goal_prob = goal_prob[:-1].view(self.final_image_height, self.final_image_width)
            else:
                goal_prob = None
                instruction = instruction + ": Goal None"
            self.save_attention_prob(image, attention_prob[:-1].view(self.final_image_height, self.final_image_width),
                                     instruction, goal_prob)

        return volatile
