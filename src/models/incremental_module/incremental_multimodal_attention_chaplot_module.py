import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import scipy.misc

from utils.cuda import cuda_var
from agents.agent_with_read import ReadPointerAgent
from learning.auxiliary_objective.goal_prediction import GoalPrediction


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


class IncrementalMultimodalAttentionChaplotModule(nn.Module):
    """
    pytorch module for final part of model
    combines embeddings of image, text, and previous action
    """
    def __init__(self, image_module, text_module, image_recurrence_module,
                 max_episode_length, final_image_height, final_image_width):
        super(IncrementalMultimodalAttentionChaplotModule, self).__init__()
        self.image_module = image_module
        self.image_recurrence_module = image_recurrence_module
        self.text_module = text_module
        self.dense_read = nn.Linear(512, 2)

        # Time embedding layer, helps in stabilizing value prediction
        self.time_emb_dim = 32
        self.time_emb_layer = nn.Embedding(max_episode_length+1, self.time_emb_dim)

        # Positional Encoding
        position_hor_1 = np.zeros(shape=(8, 8))
        position_hor_2 = np.zeros(shape=(8, 8))
        position_hor_3 = np.zeros(shape=(8, 8))
        position_ver_1 = np.zeros(shape=(8, 8))
        position_ver_2 = np.zeros(shape=(8, 8))
        position_ver_3 = np.zeros(shape=(8, 8))

        for i in range(0, 8):  # Y axis
            for j in range(0, 8):  # X axis
                if j <= 2:
                    position_hor_1[i, j] = 1.0
                elif 3 <= j <= 4:
                    position_hor_2[i, j] = 1.0
                else:
                    position_hor_3[i, j] = 1.0

                if i <= 2:
                    position_ver_1[i, j] = 1.0
                elif 3 <= j <= 4:
                    position_ver_2[i, j] = 1.0
                else:
                    position_ver_3[i, j] = 1.0

        positional_encodings = [position_hor_1, position_hor_2, position_hor_3,
                                position_ver_1, position_ver_2, position_ver_3]

        self.positional_encodings = []
        for positional_encoding in positional_encodings:
            position_var = cuda_var(torch.from_numpy(positional_encoding).float()).view(1, 1, 8, 8)
            self.positional_encodings.append(position_var)
        self.num_positional_encodings = len(self.positional_encodings)
        self.positional_encodings = torch.cat(self.positional_encodings, dim=1)  # 1 x num_pos x 8 x 8

        text_dim = 2 * 40
        self.language_filter1 = nn.Linear(64 * 3 * 3, text_dim - self.num_positional_encodings)
        self.language_filter2 = nn.Linear(3 * 3 * 3, text_dim - self.num_positional_encodings)
        self.language_filter3 = nn.Linear(12 * 3 * 3, text_dim - self.num_positional_encodings)

        # Convolution layer
        self.conv0 = nn.Conv2d(2, 2, 3, stride=1, padding=1)
        self.conv1 = nn.Conv2d(2 + 1, 3, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(1 + 1, 6, 3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(6 + 1, 12, 3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(1 + 1, 6, 3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(6 + 1, 3, 3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(3 + 1, 1, 3, stride=1, padding=1)

        self.final_channel = 256 + self.num_positional_encodings
        self.bias = nn.Linear(1, 1)  # self.final_channel)
        self.attention = nn.Linear(self.final_channel, 256)

        # A3C-LSTM layers
        self.final_image_height = final_image_height
        self.final_image_width = final_image_width
        self.linear = nn.Linear(self.final_channel * self.final_image_height * self.final_image_width, 256)
        self.critic_linear = nn.Linear(256 + self.time_emb_dim, 1)
        self.actor_linear = nn.Linear(256 + self.time_emb_dim, 4)
        self.global_id = 1

    @staticmethod
    def normalized_columns_initializer(weights, std=1.0):
        out = torch.randn(weights.size())
        out *= std / torch.sqrt(out.pow(2).sum(1, keepdim=True).expand_as(out))
        return out

    def init_weights(self):

        # Initializing weights
        self.apply(weights_init)
        # self.actor_linear.weight.data = self.normalized_columns_initializer(
        #     self.actor_linear.weight.data, 0.01)
        # self.actor_linear.bias.data.fill_(0)
        # self.critic_linear.weight.data = self.normalized_columns_initializer(
        #     self.critic_linear.weight.data, 1.0)
        # self.critic_linear.bias.data.fill_(0)

    def save_attention_prob(self, image, attention_prob, instruction, goal_prob=None):
        self.global_id += 1

        image_flipped = image[0, 0, :, :, :].cpu().data.numpy().swapaxes(0, 1).swapaxes(1, 2)
        attention_prob = attention_prob.cpu().data.numpy()
        resized_kernel = scipy.misc.imresize(attention_prob, (128, 128))
        if goal_prob is not None:
            goal_location = goal_prob.cpu().data.numpy()
            if np.sum(goal_location) > 0.01:
                for i in range(0, 8):
                    for j in range(0, 8):
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
            text_emb, text_emb_raw = self.text_module(instructions)
            text_emb_raw_local, text_emb_raw_global = \
                text_emb_raw[:, :-self.num_positional_encodings], text_emb_raw[:, -self.num_positional_encodings:]
            text_emb_raw_global = text_emb_raw_global.view(text_emb_raw_global.size(1), 1)

            text_emb_filter1 = torch.matmul(text_emb_raw_local, self.language_filter1.weight).view(1, 64, 3, 3)
            text_emb_filter2 = torch.matmul(text_emb_raw_local, self.language_filter2.weight).view(1, 3, 3, 3)
            text_emb_filter3 = torch.matmul(text_emb_raw_local, self.language_filter3.weight).view(1, 12, 3, 3)
            # text_emb_key = torch.matmul(text_emb_raw, self.attention.weight)
        else:
            text_emb_raw_global, text_emb_filter1, text_emb_filter2, text_emb_filter3, image_hidden_states = model_state

        image_emb_seq = self.image_module(image)
        image_emb = image_emb_seq[:, 0, :, :, :]  # 1 x num_channels x height x width

        # Language filters
        image_emb_language = F.conv2d(image_emb, weight=text_emb_filter1, bias=None, padding=1)  # 1 x 1 x height x width

        # Weighted positional encoding
        weighted_positional_encodings = \
            text_emb_raw_global * self.positional_encodings.view(self.num_positional_encodings, -1)
        weighted_positional_encodings = weighted_positional_encodings.view(
            1, self.num_positional_encodings, self.final_image_height, self.final_image_width)
        weighted_positional_encodings = torch.sum(weighted_positional_encodings, dim=1).view(
            1, 1, self.final_image_height, self.final_image_width)

        x_1 = torch.cat([image_emb_language, weighted_positional_encodings], dim=1)  # 1 x (num_pos + 1) x height x width

        # Few layers of non-strided convolution layers
        x_2 = self.conv0(x_1)  # 1 x 2 x height x width
        x_2 = F.leaky_relu(x_2)
        x_2 = torch.cat([x_2, weighted_positional_encodings], dim=1)
        x_3 = self.conv1(x_2)   # 1 x 3 x height x width
        x_4 = F.leaky_relu(x_3)
        x_4 = F.conv2d(x_4, weight=text_emb_filter2, bias=None, padding=1)
        x_5 = F.leaky_relu(x_4)
        x_5 = torch.cat([x_5, weighted_positional_encodings], dim=1)
        x_6 = self.conv2(x_5)   # 1 x 6 x height x width
        x_6 = F.leaky_relu(x_6)
        x_6 = torch.cat([x_6, weighted_positional_encodings], dim=1)
        x_7 = self.conv3(x_6)   # 1 x 12 x height x width
        x_8 = F.leaky_relu(x_7)
        x_8 = F.conv2d(x_8, weight=text_emb_filter3, bias=None, padding=1)
        x_8 = F.leaky_relu(x_8)
        x_8 = torch.cat([x_8, weighted_positional_encodings], dim=1)
        x_9 = self.conv4(x_8)  # 1 x 6 x height x width
        x_10 = F.leaky_relu(x_9)
        x_10 = torch.cat([x_10, weighted_positional_encodings], dim=1)
        x_11 = self.conv5(x_10)  # 1 x 3 x height x width
        x_12 = F.leaky_relu(x_11)
        x_12 = torch.cat([x_12, weighted_positional_encodings], dim=1)
        x_13 = self.conv6(x_12)  # 1 x 1 x height x width

        attention_logits = x_13.view(-1)
        # torch.matmul(text_emb_key, self.bias.weight).view(-1)
        # attention_logits = torch.cat([attention_logits, not_present_logit], dim=0)
        attention_logits = torch.cat([attention_logits, self.bias.bias.view(-1)], dim=0)
        attention_probs = F.softmax(attention_logits, dim=0)  # (height x width + 1)

        # Apply bilinear embedding
        # image_emb_reshaped = x_7.view(x_7.size(1), -1)  # num_channels x (hght x wdth)
        # attention_logits = torch.matmul(text_emb_key, image_emb_reshaped).view(-1)  # (height x width)
        # not_present_logit = torch.matmul(text_emb_key, self.bias.weight).view(-1)
        # attention_logits = torch.cat([attention_logits, not_present_logit], dim=0)
        # attention_probs = F.softmax(attention_logits, dim=0)  # (height x width + 1)

        # if goal[0] is not None:
        #     gold_prob = GoalPrediction.generate_gold_prob(goal)
        #     self.save_attention_prob(image, attention_probs[:-1].view(8, 8),
        #                              instruction, goal_prob=gold_prob[:-1].view(8, 8))

        volatile = dict()
        volatile["image_emb"] = image_emb
        volatile["attention_probs"] = attention_probs  # .view(self.final_image_height, self.final_image_width)
        volatile["attention_logits"] = attention_logits  # .view(self.final_image_height, self.final_image_width)
        return None, (text_emb_raw_global, text_emb_filter1, text_emb_filter2, text_emb_filter3, None), None, volatile


class IncrementalMultimodalAttentionChaplotModuleM5Andrew(nn.Module):
    """
    pytorch module for final part of model
    combines embeddings of image, text, and previous action
    """
    def __init__(self, image_module, text_module, image_recurrence_module,
                 max_episode_length, final_image_height, final_image_width,  normalize_filters=True):
        super(IncrementalMultimodalAttentionChaplotModuleM5Andrew, self).__init__()
        self.image_module = image_module
        self.image_recurrence_module = image_recurrence_module
        self.text_module = text_module
        self.dense_read = nn.Linear(512, 2)
        self.normalize_filters = normalize_filters

        # Time embedding layer, helps in stabilizing value prediction
        self.time_emb_dim = 32
        self.time_emb_layer = nn.Embedding(max_episode_length+1, self.time_emb_dim)

        # positional_encodings = self.generate_positional_encoding(final_image_height, final_image_width)
        positional_encodings = self.generate_360_positional_encoding(final_image_height, final_image_width//6)

        self.positional_encodings = []
        for positional_encoding in positional_encodings:
            position_var = cuda_var(torch.from_numpy(positional_encoding).float()).view(
                1, 1, final_image_height, final_image_width)
            self.positional_encodings.append(position_var)
        self.num_positional_encodings = len(self.positional_encodings)
        self.positional_encodings = torch.cat(self.positional_encodings, dim=1)  # 1 x num_pos x height x width

        text_dim = 256  # 2 * 40
        self.language_filter = nn.Linear(64 * 8, text_dim - self.num_positional_encodings)
        self.language_filter_2 = nn.Linear(16 * 4, text_dim - self.num_positional_encodings)
        self.language_filter_3 = nn.Linear(32 * 2, text_dim - self.num_positional_encodings)
        self.language_filter_4 = nn.Linear(64 * 1, text_dim - self.num_positional_encodings)
        self.language_filter_5 = nn.Linear(32 * 2, text_dim - self.num_positional_encodings)
        self.language_filter_6 = nn.Linear(16 * 4, text_dim - self.num_positional_encodings)

        # Convolution layer
        k = 5
        pad = int(k / 2)
        self.conv1 = nn.Conv2d(8 + 1, 16, k, stride=1, padding=pad)
        self.conv2 = nn.Conv2d(16 + 1, 32, k, stride=1, padding=pad)
        self.conv3 = nn.Conv2d(32 + 4 + 1, 64, k, stride=1, padding=pad)
        self.conv4 = nn.Conv2d(64 + 2 + 1, 32, k, stride=1, padding=pad)
        self.conv5 = nn.Conv2d(32 + 1 + 1, 16, k, stride=1, padding=pad)
        self.conv6 = nn.Conv2d(16 + 2 + 1, 8, k, stride=1, padding=pad)
        self.conv_final = nn.Conv2d(8 + 4 + 1, 1, 1, stride=1, padding=0)

        self.final_channel = 64  # 256 + self.num_positional_encodings
        self.bias = nn.Linear(1, 1)  # self.final_channel)
        self.attention = nn.Linear(256 + self.num_positional_encodings, 256)  # self.final_channel, 256)

        # A3C-LSTM layers
        self.final_image_height = final_image_height
        self.final_image_width = final_image_width
        self.linear = nn.Linear(
            (256 + self.num_positional_encodings) * self.final_image_height * self.final_image_width, 256)
        self.critic_linear = nn.Linear(256 + self.time_emb_dim, 1)
        self.actor_linear = nn.Linear(256 + self.time_emb_dim, 4)

        # self.linear1 = nn.Linear(self.final_channel * self.final_image_height * self.final_image_width, 256)
        self.linear1 = nn.Linear(32 * 32 + 1, 256)
        self.critic_linear1 = nn.Linear(256 + self.time_emb_dim, 1)
        self.actor_linear1 = nn.Linear(256 + self.time_emb_dim, 4)

        self.global_id = 1
        self.attention_model = None
        self.run_with_gold = False
        self.acc = 0
        self.count = 0

    @staticmethod
    def generate_positional_encoding(final_image_height, final_image_width):

        # Positional Encoding
        position_hor_1 = np.zeros(shape=(final_image_height, final_image_width))
        position_hor_2 = np.zeros(shape=(final_image_height, final_image_width))
        position_hor_3 = np.zeros(shape=(final_image_height, final_image_width))
        position_ver_1 = np.zeros(shape=(final_image_height, final_image_width))
        position_ver_2 = np.zeros(shape=(final_image_height, final_image_width))
        position_ver_3 = np.zeros(shape=(final_image_height, final_image_width))

        for i in range(0, final_image_height):  # Y axis
            for j in range(0, final_image_width):  # X axis
                if j <= final_image_width // 3:
                    position_hor_1[i, j] = 1.0
                elif final_image_width // 3 <= j <= (2 * final_image_width) // 3:
                    position_hor_2[i, j] = 1.0
                else:
                    position_hor_3[i, j] = 1.0

                if i <= final_image_height // 3:
                    position_ver_1[i, j] = 1.0
                elif final_image_height // 3 <= j <= (2 * final_image_height) // 3:
                    position_ver_2[i, j] = 1.0
                else:
                    position_ver_3[i, j] = 1.0

        positional_encodings = [position_hor_1, position_hor_2, position_hor_3,
                                position_ver_1, position_ver_2, position_ver_3]

        return positional_encodings

    @staticmethod
    def generate_360_positional_encoding(final_image_height, final_image_width):

        # Positional Encoding
        position_zero = np.zeros(shape=(final_image_height, final_image_width))
        position_hor_1 = np.zeros(shape=(final_image_height, final_image_width))
        position_hor_2 = np.zeros(shape=(final_image_height, final_image_width))
        position_hor_3 = np.zeros(shape=(final_image_height, final_image_width))
        position_ver_1 = np.zeros(shape=(final_image_height, final_image_width))
        position_ver_2 = np.zeros(shape=(final_image_height, final_image_width))
        position_ver_3 = np.zeros(shape=(final_image_height, final_image_width))

        for i in range(0, final_image_height):  # Y axis
            for j in range(0, final_image_width):  # X axis
                if j <= final_image_width // 3:
                    position_hor_1[i, j] = 1.0
                elif final_image_width // 3 <= j <= (2 * final_image_width) // 3:
                    position_hor_2[i, j] = 1.0
                else:
                    position_hor_3[i, j] = 1.0

                if i <= final_image_height // 3:
                    position_ver_1[i, j] = 1.0
                elif final_image_height // 3 <= j <= (2 * final_image_height) // 3:
                    position_ver_2[i, j] = 1.0
                else:
                    position_ver_3[i, j] = 1.0

        positional_encodings = [position_hor_1, position_hor_2, position_hor_3,
                                position_ver_1, position_ver_2, position_ver_3]

        all_around_positional_encodings = []
        for i in range(0, 6):
            for positional_encoding in positional_encodings:
                all_around_positional_encoding = [position_zero, position_zero, position_zero, position_zero, position_zero]
                all_around_positional_encoding.insert(i, positional_encoding)
                all_around_positional_encoding = np.hstack(all_around_positional_encoding)
                all_around_positional_encodings.append(all_around_positional_encoding)

        return all_around_positional_encodings

    @staticmethod
    def normalized_columns_initializer(weights, std=1.0):
        out = torch.randn(weights.size())
        out *= std / torch.sqrt(out.pow(2).sum(1, keepdim=True).expand_as(out))
        return out

    def set_attention_model(self, attention_model):
        self.attention_model = attention_model

    def init_weights(self):

        # Initializing weights
        self.apply(weights_init)
        # self.actor_linear.weight.data = self.normalized_columns_initializer(
        #     self.actor_linear.weight.data, 0.01)
        # self.actor_linear.bias.data.fill_(0)
        # self.critic_linear.weight.data = self.normalized_columns_initializer(
        #     self.critic_linear.weight.data, 1.0)
        # self.critic_linear.bias.data.fill_(0)

    def save_attention_prob(self, image, attention_prob, instruction, goal_prob=None):
        self.global_id += 1

        image_flipped = image[0, 0, :, :, :].cpu().data.numpy().swapaxes(0, 1).swapaxes(1, 2)
        attention_prob = attention_prob.cpu().data.numpy()
        resized_kernel = scipy.misc.imresize(attention_prob, (128, 128*6))
        if goal_prob is not None:
            goal_location = goal_prob.cpu().data.numpy()
            if np.sum(goal_location) > 0.01:
                for i in range(0, self.final_image_height):
                    for j in range(0, self.final_image_width):
                            if goal_location[i][j] < 0.01:
                                goal_location[i][j] = 0.0
                goal_location = scipy.misc.imresize(goal_location, (128, 128*6))
        else:
            goal_location = None

        f, axarr = plt.subplots(1, 2)
        if instruction is not None:
            f.suptitle(instruction)
        axarr[0].set_title("Predicted Attention")
        axarr[0].imshow(image_flipped)
        # axarr[0].imshow(resized_kernel, cmap='jet', alpha=0.5)
        axarr[1].set_title("Gold Attention (Goal)")
        axarr[1].imshow(image_flipped)
        if goal_location is not None:
            axarr[1].imshow(goal_location, cmap='jet', alpha=0.5)
        plt.savefig("./attention_prob/image_" + str(self.global_id) + ".png")
        plt.clf()

    def forward(self, image, instructions, tx, mode, model_state, instruction, goal):
        # return self.forward_both(image, instructions, tx, mode, model_state, instruction, goal)
        return self.forward_attention(image, instructions, tx, mode, model_state, instruction, goal)

    def forward_attention(self, image, instructions, tx, mode, model_state, instruction, goal):

        if model_state is None:
            text_emb, text_emb_raw = self.text_module(instructions)
            text_emb_raw_local, text_emb_raw_global = \
                text_emb_raw[:, :-self.num_positional_encodings], text_emb_raw[:, -self.num_positional_encodings:]
            text_emb_filter = torch.matmul(text_emb_raw_local, self.language_filter.weight).view(8, 64, 1, 1)
            text_emb_filter_2 = torch.matmul(text_emb_raw_local, self.language_filter.weight).view(4, 16, 1, 1)
            text_emb_filter_3 = torch.matmul(text_emb_raw_local, self.language_filter.weight).view(2, 32, 1, 1)
            text_emb_filter_4 = torch.matmul(text_emb_raw_local, self.language_filter.weight).view(1, 64, 1, 1)
            text_emb_filter_5 = torch.matmul(text_emb_raw_local, self.language_filter.weight).view(2, 32, 1, 1)
            text_emb_filter_6 = torch.matmul(text_emb_raw_local, self.language_filter.weight).view(4, 16, 1, 1)
            text_emb_filter_list = [text_emb_filter, text_emb_filter_2,
                                    text_emb_filter_3, text_emb_filter_4,
                                    text_emb_filter_5, text_emb_filter_6]
            text_emb_raw_global = text_emb_raw_global.view(text_emb_raw_global.size(1), 1)
            # text_emb_key = torch.matmul(text_emb_raw, self.attention.weight)
            image_hidden_states = None
        else:
            text_emb, text_emb_raw_global, text_emb_filter_list, image_hidden_states = model_state
            text_emb_filter, text_emb_filter_2, text_emb_filter_3, text_emb_filter_4, text_emb_filter_5, text_emb_filter_6 = text_emb_filter_list

        image_emb_seq = self.image_module(image)
        image_emb = image_emb_seq[:, 0, :, :, :]  # 1 x num_channels x height x width

        # # Language filters
        image_emb_language = F.conv2d(image_emb, weight=text_emb_filter, bias=None, padding=1)  # 1 x 8 x height x width

        # Weighted positional encoding
        weighted_positional_encodings = \
            text_emb_raw_global * self.positional_encodings.view(self.num_positional_encodings, -1)
        weighted_positional_encodings = weighted_positional_encodings.view(
            1, self.num_positional_encodings, self.final_image_height, self.final_image_width)
        weighted_positional_encodings = torch.sum(weighted_positional_encodings, dim=1).view(
            1, 1, self.final_image_height, self.final_image_width)

        # Few layers of non-strided convolution layers
        x_1 = F.leaky_relu(image_emb_language) # 1 x 8 x height x width
        x_2 = torch.cat([x_1, weighted_positional_encodings], dim=1)
        x_3 = self.conv1(x_2)   # 1 x 16 x height x width
        x_4 = F.leaky_relu(x_3)
        x_4_lf = F.conv2d(x_4, weight=text_emb_filter_2, bias=None, padding=1) # 1 x 4 x height x width
        x_4 = torch.cat([x_4, weighted_positional_encodings], dim=1)
        x_5 = self.conv2(x_4)   # 1 x 32 x height x width
        x_6 = F.leaky_relu(x_5)
        x_6_lf = F.conv2d(x_6, weight=text_emb_filter_3, bias=None, padding=1) # 1 x 2 x height x width
        x_6 = torch.cat([x_6, x_4_lf, weighted_positional_encodings], dim=1)
        x_7 = self.conv3(x_6)   # 1 x 64 x height x width
        x_8 = F.leaky_relu(x_7)
        x_8_lf = F.conv2d(x_8, weight=text_emb_filter_4, bias=None, padding=1) # 1 x 1 x height x width
        x_8 = torch.cat([x_8, x_6_lf, weighted_positional_encodings], dim=1)
        x_9 = self.conv4(x_8)  # 1 x 32 x height x width
        x_10 = F.leaky_relu(x_9)
        x_10_lf = F.conv2d(x_10, weight=text_emb_filter_5, bias=None, padding=1) # 1 x 2 x height x width
        x_10 = torch.cat([x_10, x_8_lf, weighted_positional_encodings], dim=1)
        x_11 = self.conv5(x_10)  # 1 x 16 x height x width
        x_12 = F.leaky_relu(x_11)
        x_12_lf = F.conv2d(x_12, weight=text_emb_filter_6, bias=None, padding=1) # 1 x 4 x height x width
        x_12 = torch.cat([x_12, x_10_lf, weighted_positional_encodings], dim=1)
        x_13 = self.conv6(x_12)  # 1 x 8 x height x width
        x_14 = F.leaky_relu(x_13)
        x_14 = torch.cat([x_14, x_12_lf, weighted_positional_encodings], dim=1)
        x_15 = self.conv_final(x_14)

        attention_logits = x_15.view(-1)
        attention_logits = torch.cat([attention_logits, self.bias.bias.view(-1)], dim=0)
        attention_probs = F.softmax(attention_logits, dim=0)  # (height x width + 1)

        volatile = dict()
        volatile["image_emb"] = image_emb
        volatile["attention_probs"] = attention_probs
        volatile["attention_logits"] = attention_logits

        # if True:  # goal[0] is not None:
        #     gold_prob = GoalPrediction.generate_gold_prob(goal, self.final_image_height, self.final_image_width)
        #     self.save_attention_prob(image,
        #                              attention_probs[:-1].view(self.final_image_height, self.final_image_width),
        #                              instruction,
        #                              goal_prob=gold_prob[:-1].view(self.final_image_height, self.final_image_width))

        # A3C-LSTM
        x = attention_probs.view(1, -1)
        x = F.relu(self.linear1(x))
        new_image_hidden_states = self.image_recurrence_module(x, image_hidden_states)
        hx, cx = new_image_hidden_states

        time_emb = self.time_emb_layer(tx)
        x = torch.cat((hx, time_emb.view(-1, self.time_emb_dim)), 1)

        actor_linear = self.actor_linear1(x)
        critic_linear = self.critic_linear1(x)

        return F.log_softmax(actor_linear, dim=1), (
        text_emb, text_emb_raw_global, text_emb_filter_list, new_image_hidden_states), None, volatile

    def forward_with_attention(self, image, instructions, tx, mode, model_state, instruction, goal, attention_probs=None):

        if model_state is None:
            text_emb, text_emb_raw = self.text_module(instructions)
            text_emb_raw_local, text_emb_raw_global = \
                text_emb_raw[:, :-self.num_positional_encodings], text_emb_raw[:, -self.num_positional_encodings:]
            text_emb_filter = torch.matmul(text_emb_raw_local, self.language_filter.weight).view(1, 64, 3, 3)
            text_emb_raw_global = text_emb_raw_global.view(text_emb_raw_global.size(1), 1)
            # text_emb_key = torch.matmul(text_emb_raw, self.attention.weight)
            image_hidden_states = None
        else:
            text_emb, text_emb_raw_global, text_emb_filter, image_hidden_states = model_state

        image_emb_seq = self.image_module(image)
        image_emb = image_emb_seq[:, 0, :, :, :]  # 1 x num_channels x height x width

        attention_logits = None
        if attention_probs is None:
            attention_probs = GoalPrediction.generate_gold_prob(goal)

        volatile = dict()
        volatile["image_emb"] = image_emb
        volatile["attention_probs"] = attention_probs  # .view(self.final_image_height, self.final_image_width)
        volatile["attention_logits"] = attention_logits  # .view(self.final_image_height, self.final_image_width)

        image_emb_reshaped = image_emb.view(image_emb.size(1), -1)
        x = attention_probs[:-1] * image_emb_reshaped  # num_channels x (height x width)
        x = x.view(self.final_channel, self.final_image_height, self.final_image_width)

        # assert image_emb.size() == text_emb.size()
        # x = x + (image_emb * text_emb)[0]  # Add residual connection with other connection with Chaplot gating
        x = x + image_emb[0]
        x = x.view(1, -1)

        # A3C-LSTM
        x = F.relu(self.linear1(x))
        new_image_hidden_states = self.image_recurrence_module(x, image_hidden_states)
        hx, cx = new_image_hidden_states

        time_emb = self.time_emb_layer(tx)
        x = torch.cat((hx, time_emb.view(-1, self.time_emb_dim)), 1)

        actor_linear = self.actor_linear1(x)
        critic_linear = self.critic_linear1(x)

        return F.log_softmax(actor_linear, dim=1), \
               (text_emb, text_emb_raw_global, text_emb_filter, new_image_hidden_states), None, volatile

    def forward_both(self, image, instructions, tx, mode, model_state, instruction, goal):
        _, _, _, volatile = self.attention_model.final_module.forward_attention(
            image, instructions, tx, mode, model_state, instruction, goal)

        inferred_ix = int(torch.max(volatile["attention_logits"], 0)[1].data.cpu().numpy()[0])
        gold_ix = self.final_image_height * self.final_image_width
        if goal[0] is not None:
            gold_ix = goal[0] * self.final_image_width + goal[1]
        if inferred_ix == gold_ix:
            self.acc += 1
        self.count += 1
        logging.info("Acc is %r out of %r = %r ", self.acc, self.count, (100.0 * self.acc)/float(self.count))

        if inferred_ix == self.final_image_height * self.final_image_width:
            my_goal = None, None, None, None
        else:
            row = inferred_ix // self.final_image_width
            col = inferred_ix % self.final_image_width
            my_goal = row, col, float(row), float(col)
        attention_probs = GoalPrediction.generate_gold_prob(my_goal)  # volatile["attention_probs"]

        return self.forward_with_attention(
            image, instructions, tx, mode, model_state, instruction, goal, attention_probs)

    def get_attention_prob(self, image, instructions, instruction, goal):

        _, text_emb_raw = self.text_module(instructions)
        text_emb_raw_local, text_emb_raw_global = \
            text_emb_raw[:, :-self.num_positional_encodings], text_emb_raw[:, -self.num_positional_encodings:]
        if self.normalize_filters:
            norm_ = F.normalize
        else:
            norm_ = lambda x_: x_
        text_emb_filter = norm_(torch.matmul(text_emb_raw_local, self.language_filter.weight).view(8, 64, 1, 1))
        text_emb_filter_2 = norm_(torch.matmul(text_emb_raw_local, self.language_filter_2.weight).view(4, 16, 1, 1))
        text_emb_filter_3 = norm_(torch.matmul(text_emb_raw_local, self.language_filter_3.weight).view(2, 32, 1, 1))
        text_emb_filter_4 = norm_(torch.matmul(text_emb_raw_local, self.language_filter_4.weight).view(1, 64, 1, 1))
        text_emb_filter_5 = norm_(torch.matmul(text_emb_raw_local, self.language_filter_5.weight).view(2, 32, 1, 1))
        text_emb_filter_6 = norm_(torch.matmul(text_emb_raw_local, self.language_filter_6.weight).view(4, 16, 1, 1))
        text_emb_raw_global = text_emb_raw_global.view(text_emb_raw_global.size(1), 1)

        image_emb_seq = self.image_module(image)
        image_emb = image_emb_seq[:, 0, :, :, :]  # 1 x num_channels x height x width

        # # Language filters
        image_emb_language = F.conv2d(image_emb, weight=text_emb_filter, bias=None, padding=0)  # 1 x 1 x height x width

        # Weighted positional encoding
        weighted_positional_encodings = \
            text_emb_raw_global * self.positional_encodings.view(self.num_positional_encodings, -1)
        weighted_positional_encodings = weighted_positional_encodings.view(
            1, self.num_positional_encodings, self.final_image_height, self.final_image_width)
        weighted_positional_encodings = torch.sum(weighted_positional_encodings, dim=1).view(
            1, 1, self.final_image_height, self.final_image_width)

        x_1 = F.leaky_relu(image_emb_language) # 1 x 8 x height x width
        x_2 = torch.cat([x_1, weighted_positional_encodings], dim=1)
        x_3 = self.conv1(x_2)   # 1 x 16 x height x width
        x_4 = F.leaky_relu(x_3)
        x_4_lf = F.conv2d(x_4, weight=text_emb_filter_2, bias=None, padding=0) # 1 x 4 x height x width
        x_4 = torch.cat([x_4, weighted_positional_encodings], dim=1)
        x_5 = self.conv2(x_4)   # 1 x 32 x height x width
        x_6 = F.leaky_relu(x_5)
        x_6_lf = F.conv2d(x_6, weight=text_emb_filter_3, bias=None, padding=0) # 1 x 2 x height x width
        x_6 = torch.cat([x_6, x_4_lf, weighted_positional_encodings], dim=1)
        x_7 = self.conv3(x_6)   # 1 x 64 x height x width
        x_8 = F.leaky_relu(x_7)
        x_8_lf = F.conv2d(x_8, weight=text_emb_filter_4, bias=None, padding=0) # 1 x 1 x height x width
        x_8 = torch.cat([x_8, x_6_lf, weighted_positional_encodings], dim=1)
        x_9 = self.conv4(x_8)  # 1 x 32 x height x width
        x_10 = F.leaky_relu(x_9)
        x_10_lf = F.conv2d(x_10, weight=text_emb_filter_5, bias=None, padding=0) # 1 x 2 x height x width
        x_10 = torch.cat([x_10, x_8_lf, weighted_positional_encodings], dim=1)
        x_11 = self.conv5(x_10)  # 1 x 16 x height x width
        x_12 = F.leaky_relu(x_11)
        x_12_lf = F.conv2d(x_12, weight=text_emb_filter_6, bias=None, padding=0) # 1 x 4 x height x width
        x_12 = torch.cat([x_12, x_10_lf, weighted_positional_encodings], dim=1)
        x_13 = self.conv6(x_12)  # 1 x 8 x height x width
        x_14 = F.leaky_relu(x_13)
        x_14 = torch.cat([x_14, x_12_lf, weighted_positional_encodings], dim=1)
        x_15 = self.conv_final(x_14)

        attention_logits = x_15.view(-1)
        attention_logits = torch.cat([attention_logits, self.bias.bias.view(-1)], dim=0)
        attention_probs = F.softmax(attention_logits, dim=0)  # (height x width + 1)

        volatile = dict()
        volatile["image_emb"] = image_emb
        volatile["attention_probs"] = attention_probs
        volatile["attention_logits"] = attention_logits


        # if goal is not None:
        #     gold_prob = GoalPrediction.generate_gold_prob(goal, self.final_image_height, self.final_image_width)
        #     self.save_attention_prob(image,
        #                              attention_probs[:-1].view(self.final_image_height, self.final_image_width),
        #                              instruction,
        #                              goal_prob=gold_prob[:-1].view(self.final_image_height, self.final_image_width))

        return volatile


class IncrementalMultimodalAttentionChaplotModuleM5AndrewV2(nn.Module):
    """
    pytorch module for final part of model
    combines embeddings of image, text, and previous action
    """
    def __init__(self, image_module, text_module, image_recurrence_module,
                 max_episode_length, final_image_height, final_image_width, normalize_filters=True):
        super(IncrementalMultimodalAttentionChaplotModuleM5AndrewV2, self).__init__()
        self.image_module = image_module
        self.image_recurrence_module = image_recurrence_module
        self.text_module = text_module
        self.dense_read = nn.Linear(512, 2)
        self.normalize_filters = normalize_filters

        # Time embedding layer, helps in stabilizing value prediction
        self.time_emb_dim = 32
        self.time_emb_layer = nn.Embedding(max_episode_length+1, self.time_emb_dim)

        # positional_encodings = self.generate_positional_encoding(final_image_height, final_image_width)
        positional_encodings = self.generate_360_positional_encoding(final_image_height, final_image_width//6)

        self.positional_encodings = []
        for positional_encoding in positional_encodings:
            position_var = cuda_var(torch.from_numpy(positional_encoding).float()).view(
                1, 1, final_image_height, final_image_width)
            self.positional_encodings.append(position_var)
        self.num_positional_encodings = len(self.positional_encodings)
        self.positional_encodings = torch.cat(self.positional_encodings, dim=1)  # 1 x num_pos x height x width

        text_dim = 256  # 2 * 40
        self.language_filter = nn.Linear(64 * 1, text_dim - self.num_positional_encodings)
        self.language_filter_2 = nn.Linear(8 * 1, text_dim - self.num_positional_encodings)
        self.language_filter_3 = nn.Linear(16 * 1, text_dim - self.num_positional_encodings)
        self.language_filter_4 = nn.Linear(32 * 1, text_dim - self.num_positional_encodings)
        self.language_filter_5 = nn.Linear(16 * 1, text_dim - self.num_positional_encodings)
        self.language_filter_6 = nn.Linear(8 * 1, text_dim - self.num_positional_encodings)

        # Convolution layer
        k = 5
        pad = int(k / 2)
        self.conv1 = nn.Conv2d(1 + 1, 8, k, stride=1, padding=pad)
        self.conv2 = nn.Conv2d(8 + 1, 16, k, stride=1, padding=pad)
        self.conv3 = nn.Conv2d(16 + 1 + 1, 32, k, stride=1, padding=pad)
        self.conv4 = nn.Conv2d(32 + 1 + 1, 16, k, stride=1, padding=pad)
        self.conv5 = nn.Conv2d(16 + 1 + 1, 8, k, stride=1, padding=pad)
        self.conv6 = nn.Conv2d(8 + 1 + 1, 4, k, stride=1, padding=pad)
        self.conv_final = nn.Conv2d(4 + 1 + 1, 1, 1, stride=1, padding=0)

        self.final_channel = 64  # 256 + self.num_positional_encodings
        self.bias = nn.Linear(1, 1)  # self.final_channel)
        self.attention = nn.Linear(256 + self.num_positional_encodings, 256)  # self.final_channel, 256)

        # A3C-LSTM layers
        self.final_image_height = final_image_height
        self.final_image_width = final_image_width
        self.linear = nn.Linear(
            (256 + self.num_positional_encodings) * self.final_image_height * self.final_image_width, 256)
        self.critic_linear = nn.Linear(256 + self.time_emb_dim, 1)
        self.actor_linear = nn.Linear(256 + self.time_emb_dim, 4)

        # self.linear1 = nn.Linear(self.final_channel * self.final_image_height * self.final_image_width, 256)
        self.linear1 = nn.Linear(32 * 32 + 1, 256)
        self.critic_linear1 = nn.Linear(256 + self.time_emb_dim, 1)
        self.actor_linear1 = nn.Linear(256 + self.time_emb_dim, 4)

        self.global_id = 1
        self.attention_model = None
        self.run_with_gold = False
        self.acc = 0
        self.count = 0

    @staticmethod
    def generate_positional_encoding(final_image_height, final_image_width):

        # Positional Encoding
        position_hor_1 = np.zeros(shape=(final_image_height, final_image_width))
        position_hor_2 = np.zeros(shape=(final_image_height, final_image_width))
        position_hor_3 = np.zeros(shape=(final_image_height, final_image_width))
        position_ver_1 = np.zeros(shape=(final_image_height, final_image_width))
        position_ver_2 = np.zeros(shape=(final_image_height, final_image_width))
        position_ver_3 = np.zeros(shape=(final_image_height, final_image_width))

        for i in range(0, final_image_height):  # Y axis
            for j in range(0, final_image_width):  # X axis
                if j <= final_image_width // 3:
                    position_hor_1[i, j] = 1.0
                elif final_image_width // 3 <= j <= (2 * final_image_width) // 3:
                    position_hor_2[i, j] = 1.0
                else:
                    position_hor_3[i, j] = 1.0

                if i <= final_image_height // 3:
                    position_ver_1[i, j] = 1.0
                elif final_image_height // 3 <= j <= (2 * final_image_height) // 3:
                    position_ver_2[i, j] = 1.0
                else:
                    position_ver_3[i, j] = 1.0

        positional_encodings = [position_hor_1, position_hor_2, position_hor_3,
                                position_ver_1, position_ver_2, position_ver_3]

        return positional_encodings

    @staticmethod
    def generate_360_positional_encoding(final_image_height, final_image_width):

        # Positional Encoding
        position_zero = np.zeros(shape=(final_image_height, final_image_width))
        position_hor_1 = np.zeros(shape=(final_image_height, final_image_width))
        position_hor_2 = np.zeros(shape=(final_image_height, final_image_width))
        position_hor_3 = np.zeros(shape=(final_image_height, final_image_width))
        position_ver_1 = np.zeros(shape=(final_image_height, final_image_width))
        position_ver_2 = np.zeros(shape=(final_image_height, final_image_width))
        position_ver_3 = np.zeros(shape=(final_image_height, final_image_width))

        for i in range(0, final_image_height):  # Y axis
            for j in range(0, final_image_width):  # X axis
                if j <= final_image_width // 3:
                    position_hor_1[i, j] = 1.0
                elif final_image_width // 3 <= j <= (2 * final_image_width) // 3:
                    position_hor_2[i, j] = 1.0
                else:
                    position_hor_3[i, j] = 1.0

                if i <= final_image_height // 3:
                    position_ver_1[i, j] = 1.0
                elif final_image_height // 3 <= j <= (2 * final_image_height) // 3:
                    position_ver_2[i, j] = 1.0
                else:
                    position_ver_3[i, j] = 1.0

        positional_encodings = [position_hor_1, position_hor_2, position_hor_3,
                                position_ver_1, position_ver_2, position_ver_3]

        all_around_positional_encodings = []
        for i in range(0, 6):
            for positional_encoding in positional_encodings:
                all_around_positional_encoding = [position_zero, position_zero, position_zero, position_zero, position_zero]
                all_around_positional_encoding.insert(i, positional_encoding)
                all_around_positional_encoding = np.hstack(all_around_positional_encoding)
                all_around_positional_encodings.append(all_around_positional_encoding)

        return all_around_positional_encodings

    @staticmethod
    def normalized_columns_initializer(weights, std=1.0):
        out = torch.randn(weights.size())
        out *= std / torch.sqrt(out.pow(2).sum(1, keepdim=True).expand_as(out))
        return out

    def set_attention_model(self, attention_model):
        self.attention_model = attention_model

    def init_weights(self):

        # Initializing weights
        self.apply(weights_init)
        # self.actor_linear.weight.data = self.normalized_columns_initializer(
        #     self.actor_linear.weight.data, 0.01)
        # self.actor_linear.bias.data.fill_(0)
        # self.critic_linear.weight.data = self.normalized_columns_initializer(
        #     self.critic_linear.weight.data, 1.0)
        # self.critic_linear.bias.data.fill_(0)

    def save_attention_prob(self, image, attention_prob, instruction, goal_prob=None):
        self.global_id += 1

        image_flipped = image[0, 0, :, :, :].cpu().data.numpy().swapaxes(0, 1).swapaxes(1, 2)
        attention_prob = attention_prob.cpu().data.numpy()
        resized_kernel = scipy.misc.imresize(attention_prob, (128, 128*6))
        if goal_prob is not None:
            goal_location = goal_prob.cpu().data.numpy()
            if np.sum(goal_location) > 0.01:
                for i in range(0, self.final_image_height):
                    for j in range(0, self.final_image_width):
                            if goal_location[i][j] < 0.01:
                                goal_location[i][j] = 0.0
                goal_location = scipy.misc.imresize(goal_location, (128, 128*6))
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
        # return self.forward_both(image, instructions, tx, mode, model_state, instruction, goal)
        return self.forward_attention(image, instructions, tx, mode, model_state, instruction, goal)

    def forward_attention(self, image, instructions, tx, mode, model_state, instruction, goal):

        if model_state is None:
            text_emb, text_emb_raw = self.text_module(instructions)
            text_emb_raw_local, text_emb_raw_global = \
                text_emb_raw[:, :-self.num_positional_encodings], text_emb_raw[:, -self.num_positional_encodings:]
            text_emb_filter = torch.matmul(text_emb_raw_local, self.language_filter.weight).view(8, 64, 1, 1)
            text_emb_filter_2 = torch.matmul(text_emb_raw_local, self.language_filter.weight).view(4, 16, 1, 1)
            text_emb_filter_3 = torch.matmul(text_emb_raw_local, self.language_filter.weight).view(2, 32, 1, 1)
            text_emb_filter_4 = torch.matmul(text_emb_raw_local, self.language_filter.weight).view(1, 64, 1, 1)
            text_emb_filter_5 = torch.matmul(text_emb_raw_local, self.language_filter.weight).view(2, 32, 1, 1)
            text_emb_filter_6 = torch.matmul(text_emb_raw_local, self.language_filter.weight).view(4, 16, 1, 1)
            text_emb_filter_list = [text_emb_filter, text_emb_filter_2,
                                    text_emb_filter_3, text_emb_filter_4,
                                    text_emb_filter_5, text_emb_filter_6]
            text_emb_raw_global = text_emb_raw_global.view(text_emb_raw_global.size(1), 1)
            # text_emb_key = torch.matmul(text_emb_raw, self.attention.weight)
            image_hidden_states = None
        else:
            text_emb, text_emb_raw_global, text_emb_filter_list, image_hidden_states = model_state
            text_emb_filter, text_emb_filter_2, text_emb_filter_3, text_emb_filter_4, text_emb_filter_5, text_emb_filter_6 = text_emb_filter_list

        image_emb_seq = self.image_module(image)
        image_emb = image_emb_seq[:, 0, :, :, :]  # 1 x num_channels x height x width

        # # Language filters
        image_emb_language = F.conv2d(image_emb, weight=text_emb_filter, bias=None, padding=1)  # 1 x 8 x height x width

        # Weighted positional encoding
        weighted_positional_encodings = \
            text_emb_raw_global * self.positional_encodings.view(self.num_positional_encodings, -1)
        weighted_positional_encodings = weighted_positional_encodings.view(
            1, self.num_positional_encodings, self.final_image_height, self.final_image_width)
        weighted_positional_encodings = torch.sum(weighted_positional_encodings, dim=1).view(
            1, 1, self.final_image_height, self.final_image_width)

        # Few layers of non-strided convolution layers
        x_1 = F.leaky_relu(image_emb_language) # 1 x 8 x height x width
        x_2 = torch.cat([x_1, weighted_positional_encodings], dim=1)
        x_3 = self.conv1(x_2)   # 1 x 16 x height x width
        x_4 = F.leaky_relu(x_3)
        x_4_lf = F.conv2d(x_4, weight=text_emb_filter_2, bias=None, padding=1) # 1 x 4 x height x width
        x_4 = torch.cat([x_4, weighted_positional_encodings], dim=1)
        x_5 = self.conv2(x_4)   # 1 x 32 x height x width
        x_6 = F.leaky_relu(x_5)
        x_6_lf = F.conv2d(x_6, weight=text_emb_filter_3, bias=None, padding=1) # 1 x 2 x height x width
        x_6 = torch.cat([x_6, x_4_lf, weighted_positional_encodings], dim=1)
        x_7 = self.conv3(x_6)   # 1 x 64 x height x width
        x_8 = F.leaky_relu(x_7)
        x_8_lf = F.conv2d(x_8, weight=text_emb_filter_4, bias=None, padding=1) # 1 x 1 x height x width
        x_8 = torch.cat([x_8, x_6_lf, weighted_positional_encodings], dim=1)
        x_9 = self.conv4(x_8)  # 1 x 32 x height x width
        x_10 = F.leaky_relu(x_9)
        x_10_lf = F.conv2d(x_10, weight=text_emb_filter_5, bias=None, padding=1) # 1 x 2 x height x width
        x_10 = torch.cat([x_10, x_8_lf, weighted_positional_encodings], dim=1)
        x_11 = self.conv5(x_10)  # 1 x 16 x height x width
        x_12 = F.leaky_relu(x_11)
        x_12_lf = F.conv2d(x_12, weight=text_emb_filter_6, bias=None, padding=1) # 1 x 4 x height x width
        x_12 = torch.cat([x_12, x_10_lf, weighted_positional_encodings], dim=1)
        x_13 = self.conv6(x_12)  # 1 x 8 x height x width
        x_14 = F.leaky_relu(x_13)
        x_14 = torch.cat([x_14, x_12_lf, weighted_positional_encodings], dim=1)
        x_15 = self.conv_final(x_14)

        attention_logits = x_15.view(-1)
        attention_logits = torch.cat([attention_logits, self.bias.bias.view(-1)], dim=0)
        attention_probs = F.softmax(attention_logits, dim=0)  # (height x width + 1)

        volatile = dict()
        volatile["image_emb"] = image_emb
        volatile["attention_probs"] = attention_probs
        volatile["attention_logits"] = attention_logits

        # if True:  # goal[0] is not None:
        #     gold_prob = GoalPrediction.generate_gold_prob(goal, self.final_image_height, self.final_image_width)
        #     self.save_attention_prob(image,
        #                              attention_probs[:-1].view(self.final_image_height, self.final_image_width),
        #                              instruction,
        #                              goal_prob=gold_prob[:-1].view(self.final_image_height, self.final_image_width))

        # A3C-LSTM
        x = attention_probs.view(1, -1)
        x = F.relu(self.linear1(x))
        new_image_hidden_states = self.image_recurrence_module(x, image_hidden_states)
        hx, cx = new_image_hidden_states

        time_emb = self.time_emb_layer(tx)
        x = torch.cat((hx, time_emb.view(-1, self.time_emb_dim)), 1)

        actor_linear = self.actor_linear1(x)
        critic_linear = self.critic_linear1(x)

        return F.log_softmax(actor_linear, dim=1), (
        text_emb, text_emb_raw_global, text_emb_filter_list, new_image_hidden_states), None, volatile

    def forward_with_attention(self, image, instructions, tx, mode, model_state, instruction, goal, attention_probs=None):

        if model_state is None:
            text_emb, text_emb_raw = self.text_module(instructions)
            text_emb_raw_local, text_emb_raw_global = \
                text_emb_raw[:, :-self.num_positional_encodings], text_emb_raw[:, -self.num_positional_encodings:]
            text_emb_filter = torch.matmul(text_emb_raw_local, self.language_filter.weight).view(1, 64, 3, 3)
            text_emb_raw_global = text_emb_raw_global.view(text_emb_raw_global.size(1), 1)
            # text_emb_key = torch.matmul(text_emb_raw, self.attention.weight)
            image_hidden_states = None
        else:
            text_emb, text_emb_raw_global, text_emb_filter, image_hidden_states = model_state

        image_emb_seq = self.image_module(image)
        image_emb = image_emb_seq[:, 0, :, :, :]  # 1 x num_channels x height x width

        attention_logits = None
        if attention_probs is None:
            attention_probs = GoalPrediction.generate_gold_prob(goal)

        volatile = dict()
        volatile["image_emb"] = image_emb
        volatile["attention_probs"] = attention_probs  # .view(self.final_image_height, self.final_image_width)
        volatile["attention_logits"] = attention_logits  # .view(self.final_image_height, self.final_image_width)

        image_emb_reshaped = image_emb.view(image_emb.size(1), -1)
        x = attention_probs[:-1] * image_emb_reshaped  # num_channels x (height x width)
        x = x.view(self.final_channel, self.final_image_height, self.final_image_width)

        # assert image_emb.size() == text_emb.size()
        # x = x + (image_emb * text_emb)[0]  # Add residual connection with other connection with Chaplot gating
        x = x + image_emb[0]
        x = x.view(1, -1)

        # A3C-LSTM
        x = F.relu(self.linear1(x))
        new_image_hidden_states = self.image_recurrence_module(x, image_hidden_states)
        hx, cx = new_image_hidden_states

        time_emb = self.time_emb_layer(tx)
        x = torch.cat((hx, time_emb.view(-1, self.time_emb_dim)), 1)

        actor_linear = self.actor_linear1(x)
        critic_linear = self.critic_linear1(x)

        return F.log_softmax(actor_linear, dim=1), \
               (text_emb, text_emb_raw_global, text_emb_filter, new_image_hidden_states), None, volatile

    def forward_both(self, image, instructions, tx, mode, model_state, instruction, goal):
        _, _, _, volatile = self.attention_model.final_module.forward_attention(
            image, instructions, tx, mode, model_state, instruction, goal)

        inferred_ix = int(torch.max(volatile["attention_logits"], 0)[1].data.cpu().numpy()[0])
        gold_ix = self.final_image_height * self.final_image_width
        if goal[0] is not None:
            gold_ix = goal[0] * self.final_image_width + goal[1]
        if inferred_ix == gold_ix:
            self.acc += 1
        self.count += 1
        logging.info("Acc is %r out of %r = %r ", self.acc, self.count, (100.0 * self.acc)/float(self.count))

        if inferred_ix == self.final_image_height * self.final_image_width:
            my_goal = None, None, None, None
        else:
            row = inferred_ix // self.final_image_width
            col = inferred_ix % self.final_image_width
            my_goal = row, col, float(row), float(col)
        attention_probs = GoalPrediction.generate_gold_prob(my_goal)  # volatile["attention_probs"]

        return self.forward_with_attention(
            image, instructions, tx, mode, model_state, instruction, goal, attention_probs)

    def get_attention_prob(self, image, instructions, instruction, goal):

        _, text_emb_raw = self.text_module(instructions)
        text_emb_raw_local, text_emb_raw_global = \
            text_emb_raw[:, :-self.num_positional_encodings], text_emb_raw[:, -self.num_positional_encodings:]
        if self.normalize_filters:
            norm_ = F.normalize
        else:
            norm_ = lambda x_: x_
        text_emb_filter = norm_(torch.matmul(text_emb_raw_local, self.language_filter.weight).view(1, 64, 1, 1))
        text_emb_filter_2 = norm_(torch.matmul(text_emb_raw_local, self.language_filter_2.weight).view(1, 8, 1, 1))
        text_emb_filter_3 = norm_(torch.matmul(text_emb_raw_local, self.language_filter_3.weight).view(1, 16, 1, 1))
        text_emb_filter_4 = norm_(torch.matmul(text_emb_raw_local, self.language_filter_4.weight).view(1, 32, 1, 1))
        text_emb_filter_5 = norm_(torch.matmul(text_emb_raw_local, self.language_filter_5.weight).view(1, 16, 1, 1))
        text_emb_filter_6 = norm_(torch.matmul(text_emb_raw_local, self.language_filter_6.weight).view(1, 8, 1, 1))
        text_emb_raw_global = text_emb_raw_global.view(text_emb_raw_global.size(1), 1)

        image_emb_seq = self.image_module(image)
        image_emb = image_emb_seq[:, 0, :, :, :]  # 1 x num_channels x height x width

        # # Language filters
        image_emb_language = F.conv2d(image_emb, weight=text_emb_filter, bias=None, padding=0)  # 1 x 1 x height x width

        # Weighted positional encoding
        weighted_positional_encodings = \
            text_emb_raw_global * self.positional_encodings.view(self.num_positional_encodings, -1)
        weighted_positional_encodings = weighted_positional_encodings.view(
            1, self.num_positional_encodings, self.final_image_height, self.final_image_width)
        weighted_positional_encodings = torch.sum(weighted_positional_encodings, dim=1).view(
            1, 1, self.final_image_height, self.final_image_width)

        x_1 = F.leaky_relu(image_emb_language) # 1 x 1 x height x width
        x_2 = torch.cat([x_1, weighted_positional_encodings], dim=1)
        x_3 = self.conv1(x_2)   # 1 x 8 x height x width
        x_4 = F.leaky_relu(x_3)
        x_4_lf = F.conv2d(x_4, weight=text_emb_filter_2, bias=None, padding=0) # 1 x 1 x height x width
        x_4 = torch.cat([x_4, weighted_positional_encodings], dim=1)
        x_5 = self.conv2(x_4)   # 1 x 16 x height x width
        x_6 = F.leaky_relu(x_5)
        x_6_lf = F.conv2d(x_6, weight=text_emb_filter_3, bias=None, padding=0) # 1 x 1 x height x width
        x_6 = torch.cat([x_6, x_4_lf, weighted_positional_encodings], dim=1)
        x_7 = self.conv3(x_6)   # 1 x 32 x height x width
        x_8 = F.leaky_relu(x_7)
        x_8_lf = F.conv2d(x_8, weight=text_emb_filter_4, bias=None, padding=0) # 1 x 1 x height x width
        x_8 = torch.cat([x_8, x_6_lf, weighted_positional_encodings], dim=1)
        x_9 = self.conv4(x_8)  # 1 x 16 x height x width
        x_10 = F.leaky_relu(x_9)
        x_10_lf = F.conv2d(x_10, weight=text_emb_filter_5, bias=None, padding=0) # 1 x 1 x height x width
        x_10 = torch.cat([x_10, x_8_lf, weighted_positional_encodings], dim=1)
        x_11 = self.conv5(x_10)  # 1 x 8 x height x width
        x_12 = F.leaky_relu(x_11)
        x_12_lf = F.conv2d(x_12, weight=text_emb_filter_6, bias=None, padding=0) # 1 x 1 x height x width
        x_12 = torch.cat([x_12, x_10_lf, weighted_positional_encodings], dim=1)
        x_13 = self.conv6(x_12)  # 1 x 4 x height x width
        x_14 = F.leaky_relu(x_13)
        x_14 = torch.cat([x_14, x_12_lf, weighted_positional_encodings], dim=1)
        x_15 = self.conv_final(x_14)

        attention_logits = x_15.view(-1)
        attention_logits = torch.cat([attention_logits, self.bias.bias.view(-1)], dim=0)
        attention_probs = F.softmax(attention_logits, dim=0)  # (height x width + 1)

        volatile = dict()
        volatile["image_emb"] = image_emb
        volatile["attention_probs"] = attention_probs
        volatile["attention_logits"] = attention_logits

        # if goal is not None:
        #     gold_prob = GoalPrediction.generate_gold_prob(goal, self.final_image_height, self.final_image_width)
        #     self.save_attention_prob(image,
        #                              attention_probs[:-1].view(self.final_image_height, self.final_image_width),
        #                              instruction,
        #                              goal_prob=gold_prob[:-1].view(self.final_image_height, self.final_image_width))

        return volatile


class IncrementalMultimodalAttentionChaplotModuleM5AndrewV3(nn.Module):
    """
    pytorch module for final part of model
    combines embeddings of image, text, and previous action
    """
    def __init__(self, image_module, text_module, image_recurrence_module,
                 max_episode_length, final_image_height, final_image_width):
        super(IncrementalMultimodalAttentionChaplotModuleM5AndrewV3, self).__init__()
        self.image_module = image_module
        self.image_recurrence_module = image_recurrence_module
        self.text_module = text_module
        self.dense_read = nn.Linear(512, 2)

        # Time embedding layer, helps in stabilizing value prediction
        self.time_emb_dim = 32
        self.time_emb_layer = nn.Embedding(max_episode_length+1, self.time_emb_dim)

        # positional_encodings = self.generate_positional_encoding(final_image_height, final_image_width)
        positional_encodings = self.generate_360_positional_encoding(final_image_height, final_image_width//6)

        self.positional_encodings = []
        for positional_encoding in positional_encodings:
            position_var = cuda_var(torch.from_numpy(positional_encoding).float()).view(
                1, 1, final_image_height, final_image_width)
            self.positional_encodings.append(position_var)
        self.num_positional_encodings = len(self.positional_encodings)
        self.positional_encodings = torch.cat(self.positional_encodings, dim=1)  # 1 x num_pos x height x width

        text_dim = 256  # 2 * 40
        self.language_filter = nn.Linear(64 * 3 * 3, text_dim - self.num_positional_encodings)

        # Convolution layer
        k = 5
        pad = int(k / 2)
        self.conv1 = nn.Conv2d(2 + 1, 3, k, stride=1, padding=pad)
        self.conv2 = nn.Conv2d(3 + 1, 6, k, stride=1, padding=pad)
        self.conv3 = nn.Conv2d(6 + 1, 12, k, stride=1, padding=pad)
        self.conv4 = nn.Conv2d(12 + 1, 6, k, stride=1, padding=pad)
        self.conv5 = nn.Conv2d(6 + 1, 3, k, stride=1, padding=pad)
        self.conv6 = nn.Conv2d(3 + 1, 1, k, stride=1, padding=pad)

        self.final_channel = 64  # 256 + self.num_positional_encodings
        self.bias = nn.Linear(1, 1)  # self.final_channel)
        self.attention = nn.Linear(256 + self.num_positional_encodings, 256)  # self.final_channel, 256)

        # A3C-LSTM layers
        self.final_image_height = final_image_height
        self.final_image_width = final_image_width
        self.linear = nn.Linear(
            (256 + self.num_positional_encodings) * self.final_image_height * self.final_image_width, 256)
        self.critic_linear = nn.Linear(256 + self.time_emb_dim, 1)
        self.actor_linear = nn.Linear(256 + self.time_emb_dim, 4)

        # self.linear1 = nn.Linear(self.final_channel * self.final_image_height * self.final_image_width, 256)
        self.linear1 = nn.Linear(32 * 32 + 1, 256)
        self.critic_linear1 = nn.Linear(256 + self.time_emb_dim, 1)
        self.actor_linear1 = nn.Linear(256 + self.time_emb_dim, 4)

        self.global_id = 1
        self.attention_model = None
        self.run_with_gold = False
        self.acc = 0
        self.count = 0

    @staticmethod
    def generate_positional_encoding(final_image_height, final_image_width):

        # Positional Encoding
        position_hor_1 = np.zeros(shape=(final_image_height, final_image_width))
        position_hor_2 = np.zeros(shape=(final_image_height, final_image_width))
        position_hor_3 = np.zeros(shape=(final_image_height, final_image_width))
        position_ver_1 = np.zeros(shape=(final_image_height, final_image_width))
        position_ver_2 = np.zeros(shape=(final_image_height, final_image_width))
        position_ver_3 = np.zeros(shape=(final_image_height, final_image_width))

        for i in range(0, final_image_height):  # Y axis
            for j in range(0, final_image_width):  # X axis
                if j <= final_image_width // 3:
                    position_hor_1[i, j] = 1.0
                elif final_image_width // 3 <= j <= (2 * final_image_width) // 3:
                    position_hor_2[i, j] = 1.0
                else:
                    position_hor_3[i, j] = 1.0

                if i <= final_image_height // 3:
                    position_ver_1[i, j] = 1.0
                elif final_image_height // 3 <= j <= (2 * final_image_height) // 3:
                    position_ver_2[i, j] = 1.0
                else:
                    position_ver_3[i, j] = 1.0

        positional_encodings = [position_hor_1, position_hor_2, position_hor_3,
                                position_ver_1, position_ver_2, position_ver_3]

        return positional_encodings

    @staticmethod
    def generate_360_positional_encoding(final_image_height, final_image_width):

        # Positional Encoding
        position_zero = np.zeros(shape=(final_image_height, final_image_width))
        position_hor_1 = np.zeros(shape=(final_image_height, final_image_width))
        position_hor_2 = np.zeros(shape=(final_image_height, final_image_width))
        position_hor_3 = np.zeros(shape=(final_image_height, final_image_width))
        position_ver_1 = np.zeros(shape=(final_image_height, final_image_width))
        position_ver_2 = np.zeros(shape=(final_image_height, final_image_width))
        position_ver_3 = np.zeros(shape=(final_image_height, final_image_width))

        for i in range(0, final_image_height):  # Y axis
            for j in range(0, final_image_width):  # X axis
                if j <= final_image_width // 3:
                    position_hor_1[i, j] = 1.0
                elif final_image_width // 3 <= j <= (2 * final_image_width) // 3:
                    position_hor_2[i, j] = 1.0
                else:
                    position_hor_3[i, j] = 1.0

                if i <= final_image_height // 3:
                    position_ver_1[i, j] = 1.0
                elif final_image_height // 3 <= j <= (2 * final_image_height) // 3:
                    position_ver_2[i, j] = 1.0
                else:
                    position_ver_3[i, j] = 1.0

        positional_encodings = [position_hor_1, position_hor_2, position_hor_3,
                                position_ver_1, position_ver_2, position_ver_3]

        all_around_positional_encodings = []
        for i in range(0, 6):
            for positional_encoding in positional_encodings:
                all_around_positional_encoding = [position_zero, position_zero, position_zero, position_zero, position_zero]
                all_around_positional_encoding.insert(i, positional_encoding)
                all_around_positional_encoding = np.hstack(all_around_positional_encoding)
                all_around_positional_encodings.append(all_around_positional_encoding)

        return all_around_positional_encodings

    @staticmethod
    def normalized_columns_initializer(weights, std=1.0):
        out = torch.randn(weights.size())
        out *= std / torch.sqrt(out.pow(2).sum(1, keepdim=True).expand_as(out))
        return out

    def set_attention_model(self, attention_model):
        self.attention_model = attention_model

    def init_weights(self):

        # Initializing weights
        self.apply(weights_init)
        # self.actor_linear.weight.data = self.normalized_columns_initializer(
        #     self.actor_linear.weight.data, 0.01)
        # self.actor_linear.bias.data.fill_(0)
        # self.critic_linear.weight.data = self.normalized_columns_initializer(
        #     self.critic_linear.weight.data, 1.0)
        # self.critic_linear.bias.data.fill_(0)

    def save_attention_prob(self, image, attention_prob, instruction, goal_prob=None):
        self.global_id += 1

        image_flipped = image[0, 0, :, :, :].cpu().data.numpy().swapaxes(0, 1).swapaxes(1, 2)
        attention_prob = attention_prob.cpu().data.numpy()
        resized_kernel = scipy.misc.imresize(attention_prob, (128, 128*6))
        if goal_prob is not None:
            goal_location = goal_prob.cpu().data.numpy()
            if np.sum(goal_location) > 0.01:
                for i in range(0, self.final_image_height):
                    for j in range(0, self.final_image_width):
                            if goal_location[i][j] < 0.01:
                                goal_location[i][j] = 0.0
                goal_location = scipy.misc.imresize(goal_location, (128, 128*6))
        else:
            goal_location = None

        f, axarr = plt.subplots(1, 2)
        if instruction is not None:
            f.suptitle(instruction)
        axarr[0].set_title("Predicted Attention")
        axarr[0].imshow(image_flipped)
        # axarr[0].imshow(resized_kernel, cmap='jet', alpha=0.5)
        axarr[1].set_title("Gold Attention (Goal)")
        axarr[1].imshow(image_flipped)
        if goal_location is not None:
            axarr[1].imshow(goal_location, cmap='jet', alpha=0.5)
        plt.savefig("./attention_prob/image_" + str(self.global_id) + ".png")
        plt.clf()

    def forward(self, image, instructions, tx, mode, model_state, instruction, goal):
        # return self.forward_both(image, instructions, tx, mode, model_state, instruction, goal)
        return self.forward_attention(image, instructions, tx, mode, model_state, instruction, goal)

    def forward_attention(self, image, instructions, tx, mode, model_state, instruction, goal):

        if model_state is None:
            text_emb, text_emb_raw = self.text_module(instructions)
            text_emb_raw_local, text_emb_raw_global = \
                text_emb_raw[:, :-self.num_positional_encodings], text_emb_raw[:, -self.num_positional_encodings:]
            text_emb_filter = torch.matmul(text_emb_raw_local, self.language_filter.weight).view(1, 64, 3, 3)
            text_emb_raw_global = text_emb_raw_global.view(text_emb_raw_global.size(1), 1)
            # text_emb_key = torch.matmul(text_emb_raw, self.attention.weight)
            image_hidden_states = None
        else:
            text_emb, text_emb_raw_global, text_emb_filter, image_hidden_states = model_state

        image_emb_seq = self.image_module(image)
        image_emb = image_emb_seq[:, 0, :, :, :]  # 1 x num_channels x height x width

        # # Language filters
        image_emb_language = F.conv2d(image_emb, weight=text_emb_filter, bias=None, padding=1)  # 1 x 1 x height x width

        # Weighted positional encoding
        weighted_positional_encodings = \
            text_emb_raw_global * self.positional_encodings.view(self.num_positional_encodings, -1)
        weighted_positional_encodings = weighted_positional_encodings.view(
            1, self.num_positional_encodings, self.final_image_height, self.final_image_width)
        weighted_positional_encodings = torch.sum(weighted_positional_encodings, dim=1).view(
            1, 1, self.final_image_height, self.final_image_width)

        x_1 = torch.cat([image_emb_language, weighted_positional_encodings], dim=1)  # 1 x (num_pos + 1) x height x width

        # Few layers of non-strided convolution layers
        x_2 = F.leaky_relu(x_1)
        x_2 = torch.cat([x_2, weighted_positional_encodings], dim=1)
        x_3 = self.conv1(x_2)   # 1 x 3 x height x width
        x_4 = F.leaky_relu(x_3)
        x_4 = torch.cat([x_4, weighted_positional_encodings], dim=1)
        x_5 = self.conv2(x_4)   # 1 x 6 x height x width
        x_6 = F.leaky_relu(x_5)
        x_6 = torch.cat([x_6, weighted_positional_encodings], dim=1)
        x_7 = self.conv3(x_6)   # 1 x 12 x height x width
        x_8 = F.leaky_relu(x_7)
        x_8 = torch.cat([x_8, weighted_positional_encodings], dim=1)
        x_9 = self.conv4(x_8)  # 1 x 6 x height x width
        x_10 = F.leaky_relu(x_9)
        x_10 = torch.cat([x_10, weighted_positional_encodings], dim=1)
        x_11 = self.conv5(x_10)  # 1 x 3 x height x width
        x_12 = F.leaky_relu(x_11)
        x_12 = torch.cat([x_12, weighted_positional_encodings], dim=1)
        x_13 = self.conv6(x_12)  # 1 x 1 x height x width

        attention_logits = x_13.view(-1)
        attention_logits = torch.cat([attention_logits, self.bias.bias.view(-1)], dim=0)
        attention_probs = F.softmax(attention_logits, dim=0)  # (height x width + 1)

        volatile = dict()
        volatile["image_emb"] = image_emb
        volatile["attention_probs"] = attention_probs
        volatile["attention_logits"] = attention_logits

        # if True:  # goal[0] is not None:
        #     gold_prob = GoalPrediction.generate_gold_prob(goal, self.final_image_height, self.final_image_width)
        #     self.save_attention_prob(image,
        #                              attention_probs[:-1].view(self.final_image_height, self.final_image_width),
        #                              instruction,
        #                              goal_prob=gold_prob[:-1].view(self.final_image_height, self.final_image_width))

        # A3C-LSTM
        x = attention_probs.view(1, -1)
        x = F.relu(self.linear1(x))
        new_image_hidden_states = self.image_recurrence_module(x, image_hidden_states)
        hx, cx = new_image_hidden_states

        time_emb = self.time_emb_layer(tx)
        x = torch.cat((hx, time_emb.view(-1, self.time_emb_dim)), 1)

        actor_linear = self.actor_linear1(x)
        critic_linear = self.critic_linear1(x)

        return F.log_softmax(actor_linear, dim=1), (
        text_emb, text_emb_raw_global, text_emb_filter, new_image_hidden_states), None, volatile

    def forward_with_attention(self, image, instructions, tx, mode, model_state, instruction, goal, attention_probs=None):

        if model_state is None:
            text_emb, text_emb_raw = self.text_module(instructions)
            text_emb_raw_local, text_emb_raw_global = \
                text_emb_raw[:, :-self.num_positional_encodings], text_emb_raw[:, -self.num_positional_encodings:]
            text_emb_filter = torch.matmul(text_emb_raw_local, self.language_filter.weight).view(1, 64, 3, 3)
            text_emb_raw_global = text_emb_raw_global.view(text_emb_raw_global.size(1), 1)
            # text_emb_key = torch.matmul(text_emb_raw, self.attention.weight)
            image_hidden_states = None
        else:
            text_emb, text_emb_raw_global, text_emb_filter, image_hidden_states = model_state

        image_emb_seq = self.image_module(image)
        image_emb = image_emb_seq[:, 0, :, :, :]  # 1 x num_channels x height x width

        attention_logits = None
        if attention_probs is None:
            attention_probs = GoalPrediction.generate_gold_prob(goal)

        volatile = dict()
        volatile["image_emb"] = image_emb
        volatile["attention_probs"] = attention_probs  # .view(self.final_image_height, self.final_image_width)
        volatile["attention_logits"] = attention_logits  # .view(self.final_image_height, self.final_image_width)

        image_emb_reshaped = image_emb.view(image_emb.size(1), -1)
        x = attention_probs[:-1] * image_emb_reshaped  # num_channels x (height x width)
        x = x.view(self.final_channel, self.final_image_height, self.final_image_width)

        # assert image_emb.size() == text_emb.size()
        # x = x + (image_emb * text_emb)[0]  # Add residual connection with other connection with Chaplot gating
        x = x + image_emb[0]
        x = x.view(1, -1)

        # A3C-LSTM
        x = F.relu(self.linear1(x))
        new_image_hidden_states = self.image_recurrence_module(x, image_hidden_states)
        hx, cx = new_image_hidden_states

        time_emb = self.time_emb_layer(tx)
        x = torch.cat((hx, time_emb.view(-1, self.time_emb_dim)), 1)

        actor_linear = self.actor_linear1(x)
        critic_linear = self.critic_linear1(x)

        return F.log_softmax(actor_linear, dim=1), \
               (text_emb, text_emb_raw_global, text_emb_filter, new_image_hidden_states), None, volatile

    def forward_both(self, image, instructions, tx, mode, model_state, instruction, goal):
        _, _, _, volatile = self.attention_model.final_module.forward_attention(
            image, instructions, tx, mode, model_state, instruction, goal)

        inferred_ix = int(torch.max(volatile["attention_logits"], 0)[1].data.cpu().numpy()[0])
        gold_ix = self.final_image_height * self.final_image_width
        if goal[0] is not None:
            gold_ix = goal[0] * self.final_image_width + goal[1]
        if inferred_ix == gold_ix:
            self.acc += 1
        self.count += 1
        logging.info("Acc is %r out of %r = %r ", self.acc, self.count, (100.0 * self.acc)/float(self.count))

        if inferred_ix == self.final_image_height * self.final_image_width:
            my_goal = None, None, None, None
        else:
            row = inferred_ix // self.final_image_width
            col = inferred_ix % self.final_image_width
            my_goal = row, col, float(row), float(col)
        attention_probs = GoalPrediction.generate_gold_prob(my_goal)  # volatile["attention_probs"]

        return self.forward_with_attention(
            image, instructions, tx, mode, model_state, instruction, goal, attention_probs)

    def get_attention_prob(self, image, instructions, instruction, goal):

        _, text_emb_raw = self.text_module(instructions)
        text_emb_raw_local, text_emb_raw_global = \
            text_emb_raw[:, :-self.num_positional_encodings], text_emb_raw[:, -self.num_positional_encodings:]
        text_emb_filter = torch.matmul(text_emb_raw_local, self.language_filter.weight).view(1, 64, 3, 3)
        text_emb_raw_global = text_emb_raw_global.view(text_emb_raw_global.size(1), 1)

        image_emb_seq = self.image_module(image)
        image_emb = image_emb_seq[:, 0, :, :, :]  # 1 x num_channels x height x width

        # # Language filters
        image_emb_language = F.conv2d(image_emb, weight=text_emb_filter, bias=None, padding=1)  # 1 x 1 x height x width

        # Weighted positional encoding
        weighted_positional_encodings = \
            text_emb_raw_global * self.positional_encodings.view(self.num_positional_encodings, -1)
        weighted_positional_encodings = weighted_positional_encodings.view(
            1, self.num_positional_encodings, self.final_image_height, self.final_image_width)
        weighted_positional_encodings = torch.sum(weighted_positional_encodings, dim=1).view(
            1, 1, self.final_image_height, self.final_image_width)

        x_1 = torch.cat([image_emb_language, weighted_positional_encodings], dim=1)  # 1 x (num_pos + 1) x height x width

        # Few layers of non-strided convolution layers
        x_2 = F.leaky_relu(x_1)
        x_2 = torch.cat([x_2, weighted_positional_encodings], dim=1)
        x_3 = self.conv1(x_2)   # 1 x 3 x height x width
        x_4 = F.leaky_relu(x_3)
        x_4 = torch.cat([x_4, weighted_positional_encodings], dim=1)
        x_5 = self.conv2(x_4)   # 1 x 6 x height x width
        x_6 = F.leaky_relu(x_5)
        x_6 = torch.cat([x_6, weighted_positional_encodings], dim=1)
        x_7 = self.conv3(x_6)   # 1 x 12 x height x width
        x_8 = F.leaky_relu(x_7)
        x_8 = torch.cat([x_8, weighted_positional_encodings], dim=1)
        x_9 = self.conv4(x_8)  # 1 x 6 x height x width
        x_10 = F.leaky_relu(x_9)
        x_10 = torch.cat([x_10, weighted_positional_encodings], dim=1)
        x_11 = self.conv5(x_10)  # 1 x 3 x height x width
        x_12 = F.leaky_relu(x_11)
        x_12 = torch.cat([x_12, weighted_positional_encodings], dim=1)
        x_13 = self.conv6(x_12)  # 1 x 1 x height x width

        attention_logits = x_13.view(-1)
        attention_logits = torch.cat([attention_logits, self.bias.bias.view(-1)], dim=0)
        attention_probs = F.softmax(attention_logits, dim=0)  # (height x width + 1)

        volatile = dict()
        volatile["image_emb"] = image_emb
        volatile["attention_probs"] = attention_probs
        volatile["attention_logits"] = attention_logits

        # if goal is not None:
        #     gold_prob = GoalPrediction.generate_gold_prob(goal, self.final_image_height, self.final_image_width)
        #     self.save_attention_prob(image,
        #                              attention_probs[:-1].view(self.final_image_height, self.final_image_width),
        #                              instruction,
        #                              goal_prob=gold_prob[:-1].view(self.final_image_height, self.final_image_width))

        return volatile


class IncrementalMultimodalAttentionChaplotModuleM4JKSUM1(nn.Module):
    """
    pytorch module for final part of model
    combines embeddings of image, text, and previous action
    """
    def __init__(self, image_module, text_module, image_recurrence_module,
                 max_episode_length, final_image_height, final_image_width):
        super(IncrementalMultimodalAttentionChaplotModuleM4JKSUM1, self).__init__()
        self.image_module = image_module
        self.image_recurrence_module = image_recurrence_module
        self.text_module = text_module
        self.dense_read = nn.Linear(512, 2)

        # Time embedding layer, helps in stabilizing value prediction
        self.time_emb_dim = 32
        self.time_emb_layer = nn.Embedding(max_episode_length+1, self.time_emb_dim)

        # positional_encodings = self.generate_positional_encoding(final_image_height, final_image_width)
        positional_encodings = self.generate_360_positional_encoding(final_image_height, final_image_width//6)

        self.positional_encodings = []
        for positional_encoding in positional_encodings:
            position_var = cuda_var(torch.from_numpy(positional_encoding).float()).view(
                1, 1, final_image_height, final_image_width)
            self.positional_encodings.append(position_var)
        self.num_positional_encodings = len(self.positional_encodings)
        self.positional_encodings = torch.cat(self.positional_encodings, dim=1)  # 1 x num_pos x height x width

        text_dim = 256  # 2 * 40
        self.language_filter = nn.Linear(64 * 3 * 3, text_dim - self.num_positional_encodings)

        # Convolution layer
        self.conv1 = nn.Conv2d(2 + 1, 3, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(3 + 1, 6, 3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(6 + 1, 12, 3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(12 + 1, 6, 3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(6 + 1, 3, 3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(3 + 1, 1, 3, stride=1, padding=1)

        self.final_channel = 64  # 256 + self.num_positional_encodings
        self.bias = nn.Linear(1, 1)  # self.final_channel)
        self.attention = nn.Linear(256 + self.num_positional_encodings, 256)  # self.final_channel, 256)

        # A3C-LSTM layers
        self.final_image_height = final_image_height
        self.final_image_width = final_image_width
        self.linear = nn.Linear(
            (256 + self.num_positional_encodings) * self.final_image_height * self.final_image_width, 256)
        self.critic_linear = nn.Linear(256 + self.time_emb_dim, 1)
        self.actor_linear = nn.Linear(256 + self.time_emb_dim, 4)

        # self.linear1 = nn.Linear(self.final_channel * self.final_image_height * self.final_image_width, 256)
        self.linear1 = nn.Linear(32 * 32 + 1, 256)
        self.critic_linear1 = nn.Linear(256 + self.time_emb_dim, 1)
        self.actor_linear1 = nn.Linear(256 + self.time_emb_dim, 4)

        self.global_id = 1
        self.attention_model = None
        self.run_with_gold = False
        self.acc = 0
        self.count = 0

    @staticmethod
    def generate_positional_encoding(final_image_height, final_image_width):

        # Positional Encoding
        position_hor_1 = np.zeros(shape=(final_image_height, final_image_width))
        position_hor_2 = np.zeros(shape=(final_image_height, final_image_width))
        position_hor_3 = np.zeros(shape=(final_image_height, final_image_width))
        position_ver_1 = np.zeros(shape=(final_image_height, final_image_width))
        position_ver_2 = np.zeros(shape=(final_image_height, final_image_width))
        position_ver_3 = np.zeros(shape=(final_image_height, final_image_width))

        for i in range(0, final_image_height):  # Y axis
            for j in range(0, final_image_width):  # X axis
                if j <= final_image_width // 3:
                    position_hor_1[i, j] = 1.0
                elif final_image_width // 3 <= j <= (2 * final_image_width) // 3:
                    position_hor_2[i, j] = 1.0
                else:
                    position_hor_3[i, j] = 1.0

                if i <= final_image_height // 3:
                    position_ver_1[i, j] = 1.0
                elif final_image_height // 3 <= j <= (2 * final_image_height) // 3:
                    position_ver_2[i, j] = 1.0
                else:
                    position_ver_3[i, j] = 1.0

        positional_encodings = [position_hor_1, position_hor_2, position_hor_3,
                                position_ver_1, position_ver_2, position_ver_3]

        return positional_encodings

    @staticmethod
    def generate_360_positional_encoding(final_image_height, final_image_width):

        # Positional Encoding
        position_zero = np.zeros(shape=(final_image_height, final_image_width))
        position_hor_1 = np.zeros(shape=(final_image_height, final_image_width))
        position_hor_2 = np.zeros(shape=(final_image_height, final_image_width))
        position_hor_3 = np.zeros(shape=(final_image_height, final_image_width))
        position_ver_1 = np.zeros(shape=(final_image_height, final_image_width))
        position_ver_2 = np.zeros(shape=(final_image_height, final_image_width))
        position_ver_3 = np.zeros(shape=(final_image_height, final_image_width))

        for i in range(0, final_image_height):  # Y axis
            for j in range(0, final_image_width):  # X axis
                if j <= final_image_width // 3:
                    position_hor_1[i, j] = 1.0
                elif final_image_width // 3 <= j <= (2 * final_image_width) // 3:
                    position_hor_2[i, j] = 1.0
                else:
                    position_hor_3[i, j] = 1.0

                if i <= final_image_height // 3:
                    position_ver_1[i, j] = 1.0
                elif final_image_height // 3 <= j <= (2 * final_image_height) // 3:
                    position_ver_2[i, j] = 1.0
                else:
                    position_ver_3[i, j] = 1.0

        positional_encodings = [position_hor_1, position_hor_2, position_hor_3,
                                position_ver_1, position_ver_2, position_ver_3]

        all_around_positional_encodings = []
        for i in range(0, 6):
            for positional_encoding in positional_encodings:
                all_around_positional_encoding = [position_zero, position_zero, position_zero, position_zero, position_zero]
                all_around_positional_encoding.insert(i, positional_encoding)
                all_around_positional_encoding = np.hstack(all_around_positional_encoding)
                all_around_positional_encodings.append(all_around_positional_encoding)

        return all_around_positional_encodings

    @staticmethod
    def normalized_columns_initializer(weights, std=1.0):
        out = torch.randn(weights.size())
        out *= std / torch.sqrt(out.pow(2).sum(1, keepdim=True).expand_as(out))
        return out

    def set_attention_model(self, attention_model):
        self.attention_model = attention_model

    def init_weights(self):

        # Initializing weights
        self.apply(weights_init)
        # self.actor_linear.weight.data = self.normalized_columns_initializer(
        #     self.actor_linear.weight.data, 0.01)
        # self.actor_linear.bias.data.fill_(0)
        # self.critic_linear.weight.data = self.normalized_columns_initializer(
        #     self.critic_linear.weight.data, 1.0)
        # self.critic_linear.bias.data.fill_(0)

    def save_attention_prob(self, image, attention_prob, instruction, goal_prob=None):
        self.global_id += 1

        image_flipped = image[0, 0, :, :, :].cpu().data.numpy().swapaxes(0, 1).swapaxes(1, 2)
        attention_prob = attention_prob.cpu().data.numpy()
        resized_kernel = scipy.misc.imresize(attention_prob, (128, 128*6))
        if goal_prob is not None:
            goal_location = goal_prob.cpu().data.numpy()
            if np.sum(goal_location) > 0.01:
                for i in range(0, self.final_image_height):
                    for j in range(0, self.final_image_width):
                            if goal_location[i][j] < 0.01:
                                goal_location[i][j] = 0.0
                goal_location = scipy.misc.imresize(goal_location, (128, 128*6))
        else:
            goal_location = None

        f, axarr = plt.subplots(1, 2)
        if instruction is not None:
            f.suptitle(instruction)
        axarr[0].set_title("Predicted Attention")
        axarr[0].imshow(image_flipped)
        # axarr[0].imshow(resized_kernel, cmap='jet', alpha=0.5)
        axarr[1].set_title("Gold Attention (Goal)")
        axarr[1].imshow(image_flipped)
        if goal_location is not None:
            axarr[1].imshow(goal_location, cmap='jet', alpha=0.5)
        plt.savefig("./attention_prob/image_" + str(self.global_id) + ".png")
        plt.clf()

    def forward(self, image, instructions, tx, mode, model_state, instruction, goal):
        # return self.forward_both(image, instructions, tx, mode, model_state, instruction, goal)
        return self.forward_attention(image, instructions, tx, mode, model_state, instruction, goal)

    def forward_attention(self, image, instructions, tx, mode, model_state, instruction, goal):

        if model_state is None:
            text_emb, text_emb_raw = self.text_module(instructions)
            text_emb_raw_local, text_emb_raw_global = \
                text_emb_raw[:, :-self.num_positional_encodings], text_emb_raw[:, -self.num_positional_encodings:]
            text_emb_filter = torch.matmul(text_emb_raw_local, self.language_filter.weight).view(1, 64, 3, 3)
            text_emb_raw_global = text_emb_raw_global.view(text_emb_raw_global.size(1), 1)
            # text_emb_key = torch.matmul(text_emb_raw, self.attention.weight)
            image_hidden_states = None
        else:
            text_emb, text_emb_raw_global, text_emb_filter, image_hidden_states = model_state

        image_emb_seq = self.image_module(image)
        image_emb = image_emb_seq[:, 0, :, :, :]  # 1 x num_channels x height x width

        # # Language filters
        image_emb_language = F.conv2d(image_emb, weight=text_emb_filter, bias=None, padding=1)  # 1 x 1 x height x width

        # Weighted positional encoding
        weighted_positional_encodings = \
            text_emb_raw_global * self.positional_encodings.view(self.num_positional_encodings, -1)
        weighted_positional_encodings = weighted_positional_encodings.view(
            1, self.num_positional_encodings, self.final_image_height, self.final_image_width)
        weighted_positional_encodings = torch.sum(weighted_positional_encodings, dim=1).view(
            1, 1, self.final_image_height, self.final_image_width)

        x_1 = torch.cat([image_emb_language, weighted_positional_encodings], dim=1)  # 1 x (num_pos + 1) x height x width

        # Few layers of non-strided convolution layers
        x_2 = F.leaky_relu(x_1)
        x_2 = torch.cat([x_2, weighted_positional_encodings], dim=1)
        x_3 = self.conv1(x_2)   # 1 x 3 x height x width
        x_4 = F.leaky_relu(x_3)
        x_4 = torch.cat([x_4, weighted_positional_encodings], dim=1)
        x_5 = self.conv2(x_4)   # 1 x 6 x height x width
        x_6 = F.leaky_relu(x_5)
        x_6 = torch.cat([x_6, weighted_positional_encodings], dim=1)
        x_7 = self.conv3(x_6)   # 1 x 12 x height x width
        x_8 = F.leaky_relu(x_7)
        x_8 = torch.cat([x_8, weighted_positional_encodings], dim=1)
        x_9 = self.conv4(x_8)  # 1 x 6 x height x width
        x_10 = F.leaky_relu(x_9)
        x_10 = torch.cat([x_10, weighted_positional_encodings], dim=1)
        x_11 = self.conv5(x_10)  # 1 x 3 x height x width
        x_12 = F.leaky_relu(x_11)
        x_12 = torch.cat([x_12, weighted_positional_encodings], dim=1)
        x_13 = self.conv6(x_12)  # 1 x 1 x height x width

        attention_logits = x_13.view(-1)
        attention_logits = torch.cat([attention_logits, self.bias.bias.view(-1)], dim=0)
        attention_probs = F.softmax(attention_logits, dim=0)  # (height x width + 1)

        volatile = dict()
        volatile["image_emb"] = image_emb
        volatile["attention_probs"] = attention_probs
        volatile["attention_logits"] = attention_logits

        # if True:  # goal[0] is not None:
        #     gold_prob = GoalPrediction.generate_gold_prob(goal, self.final_image_height, self.final_image_width)
        #     self.save_attention_prob(image,
        #                              attention_probs[:-1].view(self.final_image_height, self.final_image_width),
        #                              instruction,
        #                              goal_prob=gold_prob[:-1].view(self.final_image_height, self.final_image_width))

        # A3C-LSTM
        x = attention_probs.view(1, -1)
        x = F.relu(self.linear1(x))
        new_image_hidden_states = self.image_recurrence_module(x, image_hidden_states)
        hx, cx = new_image_hidden_states

        time_emb = self.time_emb_layer(tx)
        x = torch.cat((hx, time_emb.view(-1, self.time_emb_dim)), 1)

        actor_linear = self.actor_linear1(x)
        critic_linear = self.critic_linear1(x)

        return F.log_softmax(actor_linear, dim=1), (
        text_emb, text_emb_raw_global, text_emb_filter, new_image_hidden_states), None, volatile

    def forward_with_attention(self, image, instructions, tx, mode, model_state, instruction, goal, attention_probs=None):

        if model_state is None:
            text_emb, text_emb_raw = self.text_module(instructions)
            text_emb_raw_local, text_emb_raw_global = \
                text_emb_raw[:, :-self.num_positional_encodings], text_emb_raw[:, -self.num_positional_encodings:]
            text_emb_filter = torch.matmul(text_emb_raw_local, self.language_filter.weight).view(1, 64, 3, 3)
            text_emb_raw_global = text_emb_raw_global.view(text_emb_raw_global.size(1), 1)
            # text_emb_key = torch.matmul(text_emb_raw, self.attention.weight)
            image_hidden_states = None
        else:
            text_emb, text_emb_raw_global, text_emb_filter, image_hidden_states = model_state

        image_emb_seq = self.image_module(image)
        image_emb = image_emb_seq[:, 0, :, :, :]  # 1 x num_channels x height x width

        attention_logits = None
        if attention_probs is None:
            attention_probs = GoalPrediction.generate_gold_prob(goal)

        volatile = dict()
        volatile["image_emb"] = image_emb
        volatile["attention_probs"] = attention_probs  # .view(self.final_image_height, self.final_image_width)
        volatile["attention_logits"] = attention_logits  # .view(self.final_image_height, self.final_image_width)

        image_emb_reshaped = image_emb.view(image_emb.size(1), -1)
        x = attention_probs[:-1] * image_emb_reshaped  # num_channels x (height x width)
        x = x.view(self.final_channel, self.final_image_height, self.final_image_width)

        # assert image_emb.size() == text_emb.size()
        # x = x + (image_emb * text_emb)[0]  # Add residual connection with other connection with Chaplot gating
        x = x + image_emb[0]
        x = x.view(1, -1)

        # A3C-LSTM
        x = F.relu(self.linear1(x))
        new_image_hidden_states = self.image_recurrence_module(x, image_hidden_states)
        hx, cx = new_image_hidden_states

        time_emb = self.time_emb_layer(tx)
        x = torch.cat((hx, time_emb.view(-1, self.time_emb_dim)), 1)

        actor_linear = self.actor_linear1(x)
        critic_linear = self.critic_linear1(x)

        return F.log_softmax(actor_linear, dim=1), \
               (text_emb, text_emb_raw_global, text_emb_filter, new_image_hidden_states), None, volatile

    def forward_both(self, image, instructions, tx, mode, model_state, instruction, goal):
        _, _, _, volatile = self.attention_model.final_module.forward_attention(
            image, instructions, tx, mode, model_state, instruction, goal)

        inferred_ix = int(torch.max(volatile["attention_logits"], 0)[1].data.cpu().numpy()[0])
        gold_ix = self.final_image_height * self.final_image_width
        if goal[0] is not None:
            gold_ix = goal[0] * self.final_image_width + goal[1]
        if inferred_ix == gold_ix:
            self.acc += 1
        self.count += 1
        logging.info("Acc is %r out of %r = %r ", self.acc, self.count, (100.0 * self.acc)/float(self.count))

        if inferred_ix == self.final_image_height * self.final_image_width:
            my_goal = None, None, None, None
        else:
            row = inferred_ix // self.final_image_width
            col = inferred_ix % self.final_image_width
            my_goal = row, col, float(row), float(col)
        attention_probs = GoalPrediction.generate_gold_prob(my_goal)  # volatile["attention_probs"]

        return self.forward_with_attention(
            image, instructions, tx, mode, model_state, instruction, goal, attention_probs)

    def get_attention_prob(self, image, instructions, instruction, goal):

        _, text_emb_raw = self.text_module(instructions)
        text_emb_raw_local, text_emb_raw_global = \
            text_emb_raw[:, :-self.num_positional_encodings], text_emb_raw[:, -self.num_positional_encodings:]
        text_emb_filter = torch.matmul(text_emb_raw_local, self.language_filter.weight).view(1, 64, 3, 3)
        text_emb_raw_global = text_emb_raw_global.view(text_emb_raw_global.size(1), 1)

        image_emb_seq = self.image_module(image)
        image_emb = image_emb_seq[:, 0, :, :, :]  # 1 x num_channels x height x width

        # # Language filters
        image_emb_language = F.conv2d(image_emb, weight=text_emb_filter, bias=None, padding=1)  # 1 x 1 x height x width

        # Weighted positional encoding
        weighted_positional_encodings = \
            text_emb_raw_global * self.positional_encodings.view(self.num_positional_encodings, -1)
        weighted_positional_encodings = weighted_positional_encodings.view(
            1, self.num_positional_encodings, self.final_image_height, self.final_image_width)
        weighted_positional_encodings = torch.sum(weighted_positional_encodings, dim=1).view(
            1, 1, self.final_image_height, self.final_image_width)

        x_1 = torch.cat([image_emb_language, weighted_positional_encodings], dim=1)  # 1 x (num_pos + 1) x height x width

        # Few layers of non-strided convolution layers
        x_2 = F.leaky_relu(x_1)
        x_2 = torch.cat([x_2, weighted_positional_encodings], dim=1)
        x_3 = self.conv1(x_2)   # 1 x 3 x height x width
        x_4 = F.leaky_relu(x_3)
        x_4 = torch.cat([x_4, weighted_positional_encodings], dim=1)
        x_5 = self.conv2(x_4)   # 1 x 6 x height x width
        x_6 = F.leaky_relu(x_5)
        x_6 = torch.cat([x_6, weighted_positional_encodings], dim=1)
        x_7 = self.conv3(x_6)   # 1 x 12 x height x width
        x_8 = F.leaky_relu(x_7)
        x_8 = torch.cat([x_8, weighted_positional_encodings], dim=1)
        x_9 = self.conv4(x_8)  # 1 x 6 x height x width
        x_10 = F.leaky_relu(x_9)
        x_10 = torch.cat([x_10, weighted_positional_encodings], dim=1)
        x_11 = self.conv5(x_10)  # 1 x 3 x height x width
        x_12 = F.leaky_relu(x_11)
        x_12 = torch.cat([x_12, weighted_positional_encodings], dim=1)
        x_13 = self.conv6(x_12)  # 1 x 1 x height x width

        attention_logits = x_13.view(-1)
        attention_logits = torch.cat([attention_logits, self.bias.bias.view(-1)], dim=0)
        attention_probs = F.softmax(attention_logits, dim=0)  # (height x width + 1)

        volatile = dict()
        volatile["image_emb"] = image_emb
        volatile["attention_probs"] = attention_probs
        volatile["attention_logits"] = attention_logits

        # if goal is not None:
        #     gold_prob = GoalPrediction.generate_gold_prob(goal, self.final_image_height, self.final_image_width)
        #     self.save_attention_prob(image,
        #                              attention_probs[:-1].view(self.final_image_height, self.final_image_width),
        #                              instruction,
        #                              goal_prob=gold_prob[:-1].view(self.final_image_height, self.final_image_width))

        return volatile


class IncrementalMultimodalAttentionChaplotModuleM4JKSUM(nn.Module):
    """
    pytorch module for final part of model
    combines embeddings of image, text, and previous action
    """
    def __init__(self, image_module, text_module, image_recurrence_module,
                 max_episode_length, final_image_height, final_image_width):
        super(IncrementalMultimodalAttentionChaplotModuleM4JKSUM, self).__init__()
        self.image_module = image_module
        self.image_recurrence_module = image_recurrence_module
        self.text_module = text_module
        self.dense_read = nn.Linear(512, 2)

        # Time embedding layer, helps in stabilizing value prediction
        self.time_emb_dim = 32
        self.time_emb_layer = nn.Embedding(max_episode_length+1, self.time_emb_dim)

        # Positional Encoding
        position_hor_1 = np.zeros(shape=(8, 8))
        position_hor_2 = np.zeros(shape=(8, 8))
        position_hor_3 = np.zeros(shape=(8, 8))
        position_ver_1 = np.zeros(shape=(8, 8))
        position_ver_2 = np.zeros(shape=(8, 8))
        position_ver_3 = np.zeros(shape=(8, 8))

        for i in range(0, 8):  # Y axis
            for j in range(0, 8):  # X axis
                if j <= 2:
                    position_hor_1[i, j] = 1.0
                elif 3 <= j <= 4:
                    position_hor_2[i, j] = 1.0
                else:
                    position_hor_3[i, j] = 1.0

                if i <= 2:
                    position_ver_1[i, j] = 1.0
                elif 3 <= j <= 4:
                    position_ver_2[i, j] = 1.0
                else:
                    position_ver_3[i, j] = 1.0

        positional_encodings = [position_hor_1, position_hor_2, position_hor_3,
                                position_ver_1, position_ver_2, position_ver_3]

        self.positional_encodings = []
        for positional_encoding in positional_encodings:
            position_var = cuda_var(torch.from_numpy(positional_encoding).float()).view(1, 1, 8, 8)
            self.positional_encodings.append(position_var)
        self.num_positional_encodings = len(self.positional_encodings)
        self.positional_encodings = torch.cat(self.positional_encodings, dim=1)  # 1 x num_pos x 8 x 8

        self.language_filter = nn.Linear(64 * 3 * 3, 256 - self.num_positional_encodings)

        #################
        # self.language_filter2 = nn.Linear(64 * 3 * 3, 256 - self.num_positional_encodings)
        # self.language_filter3 = nn.Linear(64 * 3 * 3, 256 - self.num_positional_encodings)
        # self.language_filter4 = nn.Linear(64 * 3 * 3, 256 - self.num_positional_encodings)
        # self.language_filter5 = nn.Linear(64 * 3 * 3, 256 - self.num_positional_encodings)
        # self.language_filter6 = nn.Linear(64 * 3 * 3, 256 - self.num_positional_encodings)
        # self.language_filter7 = nn.Linear(64 * 3 * 3, 256 - self.num_positional_encodings)


        # Convolution layer
        self.conv1 = nn.Conv2d(2, 3, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(3, 6, 3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(6, 12, 3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(12, 6, 3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(6, 3, 3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(3, 1, 3, stride=1, padding=1)

        self.final_channel = 256 + self.num_positional_encodings
        self.bias = nn.Linear(1, 1)  # self.final_channel)
        self.attention = nn.Linear(self.final_channel, 256)

        # A3C-LSTM layers
        self.final_image_height = final_image_height
        self.final_image_width = final_image_width
        self.linear = nn.Linear(self.final_channel * self.final_image_height * self.final_image_width, 256)
        self.critic_linear = nn.Linear(256 + self.time_emb_dim, 1)
        self.actor_linear = nn.Linear(256 + self.time_emb_dim, 4)
        self.global_id = 1

    @staticmethod
    def normalized_columns_initializer(weights, std=1.0):
        out = torch.randn(weights.size())
        out *= std / torch.sqrt(out.pow(2).sum(1, keepdim=True).expand_as(out))
        return out

    def init_weights(self):

        # Initializing weights
        self.apply(weights_init)
        # self.actor_linear.weight.data = self.normalized_columns_initializer(
        #     self.actor_linear.weight.data, 0.01)
        # self.actor_linear.bias.data.fill_(0)
        # self.critic_linear.weight.data = self.normalized_columns_initializer(
        #     self.critic_linear.weight.data, 1.0)
        # self.critic_linear.bias.data.fill_(0)

    def save_attention_prob(self, image, attention_prob, instruction, goal_prob=None):
        self.global_id += 1

        image_flipped = image[0, 0, :, :, :].cpu().data.numpy().swapaxes(0, 1).swapaxes(1, 2)
        attention_prob = attention_prob.cpu().data.numpy()
        resized_kernel = scipy.misc.imresize(attention_prob, (128, 128))
        if goal_prob is not None:
            goal_location = goal_prob.cpu().data.numpy()
            if np.sum(goal_location) > 0.01:
                for i in range(0, 8):
                    for j in range(0, 8):
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
            text_emb, text_emb_raw = self.text_module(instructions)
            text_emb_raw_local, text_emb_raw_global = \
                text_emb_raw[:, :-self.num_positional_encodings], text_emb_raw[:, -self.num_positional_encodings:]
            text_emb_filter = torch.matmul(text_emb_raw_local, self.language_filter.weight).view(1, 64, 3, 3)
            text_emb_raw_global = text_emb_raw_global.view(text_emb_raw_global.size(1), 1)
            # text_emb_key = torch.matmul(text_emb_raw, self.attention.weight)
        else:
            text_emb_raw_global, text_emb_filter, image_hidden_states = model_state

        image_emb_seq = self.image_module(image)
        image_emb = image_emb_seq[:, 0, :, :, :]  # 1 x num_channels x height x width

        # Language filters
        image_emb_language = F.conv2d(image_emb, weight=text_emb_filter, bias=None, padding=1)  # 1 x 1 x height x width

        # Weighted positional encoding
        weighted_positional_encodings = \
            text_emb_raw_global * self.positional_encodings.view(self.num_positional_encodings, -1)
        weighted_positional_encodings = weighted_positional_encodings.view(
            1, self.num_positional_encodings, self.final_image_height, self.final_image_width)
        weighted_positional_encodings = torch.sum(weighted_positional_encodings, dim=1).view(
            1, 1, self.final_image_height, self.final_image_width)

        x_1 = torch.cat([image_emb_language, weighted_positional_encodings], dim=1)  # 1 x (num_pos + 1) x height x width

        # Few layers of non-strided convolution layers
        x_2 = F.leaky_relu(x_1)
        x_3 = self.conv1(x_2)   # 1 x 3 x height x width
        x_4 = F.leaky_relu(x_3)
        x_5 = self.conv2(x_4)   # 1 x 6 x height x width
        x_6 = F.leaky_relu(x_5)
        x_7 = self.conv3(x_6)   # 1 x 12 x height x width
        x_8 = F.leaky_relu(x_7)
        x_9 = self.conv4(x_8)  # 1 x 6 x height x width
        x_10 = F.leaky_relu(x_9)
        x_11 = self.conv5(x_10)  # 1 x 3 x height x width
        x_12 = F.leaky_relu(x_11)
        x_13 = self.conv6(x_12)  # 1 x 1 x height x width

        attention_logits = x_13.view(-1)
        # torch.matmul(text_emb_key, self.bias.weight).view(-1)
        # attention_logits = torch.cat([attention_logits, not_present_logit], dim=0)
        attention_logits = torch.cat([attention_logits, self.bias.bias.view(-1)], dim=0)
        attention_probs = F.softmax(attention_logits, dim=0)  # (height x width + 1)

        # Apply bilinear embedding
        # image_emb_reshaped = x_7.view(x_7.size(1), -1)  # num_channels x (hght x wdth)
        # attention_logits = torch.matmul(text_emb_key, image_emb_reshaped).view(-1)  # (height x width)
        # not_present_logit = torch.matmul(text_emb_key, self.bias.weight).view(-1)
        # attention_logits = torch.cat([attention_logits, not_present_logit], dim=0)
        # attention_probs = F.softmax(attention_logits, dim=0)  # (height x width + 1)

        # if goal[0] is not None:
        #     gold_prob = GoalPrediction.generate_gold_prob(goal)
        #     self.save_attention_prob(image, attention_probs[:-1].view(8, 8),
        #                              instruction, goal_prob=gold_prob[:-1].view(8, 8))

        volatile = dict()
        volatile["image_emb"] = image_emb
        volatile["attention_probs"] = attention_probs  # .view(self.final_image_height, self.final_image_width)
        volatile["attention_logits"] = attention_logits  # .view(self.final_image_height, self.final_image_width)
        return None, (text_emb_raw_global, text_emb_filter, None), None, volatile


class IncrementalMultimodalAttentionChaplotModuleM3(nn.Module):
    """
    pytorch module for final part of model
    combines embeddings of image, text, and previous action
    """
    def __init__(self, image_module, text_module, image_recurrence_module,
                 max_episode_length, final_image_height, final_image_width):
        super(IncrementalMultimodalAttentionChaplotModuleM3, self).__init__()
        self.image_module = image_module
        self.image_recurrence_module = image_recurrence_module
        self.text_module = text_module
        self.dense_read = nn.Linear(512, 2)

        # Time embedding layer, helps in stabilizing value prediction
        self.time_emb_dim = 32
        self.time_emb_layer = nn.Embedding(max_episode_length+1, self.time_emb_dim)

        # Positional Encoding
        position_hor_1 = np.zeros(shape=(8, 8))
        position_hor_2 = np.zeros(shape=(8, 8))
        position_hor_3 = np.zeros(shape=(8, 8))
        position_ver_1 = np.zeros(shape=(8, 8))
        position_ver_2 = np.zeros(shape=(8, 8))
        position_ver_3 = np.zeros(shape=(8, 8))

        for i in range(0, 8):  # Y axis
            for j in range(0, 8):  # X axis
                if j <= 2:
                    position_hor_1[i, j] = 1.0
                elif 3 <= j <= 4:
                    position_hor_2[i, j] = 1.0
                else:
                    position_hor_3[i, j] = 1.0

                if i <= 2:
                    position_ver_1[i, j] = 1.0
                elif 3 <= j <= 4:
                    position_ver_2[i, j] = 1.0
                else:
                    position_ver_3[i, j] = 1.0

        positional_encodings = [position_hor_1, position_hor_2, position_hor_3,
                                position_ver_1, position_ver_2, position_ver_3]

        self.positional_encodings = []
        for positional_encoding in positional_encodings:
            position_var = cuda_var(torch.from_numpy(positional_encoding).float()).view(1, 1, 8, 8)
            self.positional_encodings.append(position_var)
        num_positional_encodings = 6
        self.language_filter = nn.Linear(64 * 3, 256)

        # Attention layer
        self.conv1 = nn.Conv2d(64 , 128, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(128 + 3, 256, 3, stride=1, padding=1)  # 3 for language
        self.final_channel = 256 + num_positional_encodings
        self.bias = nn.Linear(1, self.final_channel)
        self.attention = nn.Linear(self.final_channel, 256)

        # A3C-LSTM layers
        self.final_image_height = final_image_height
        self.final_image_width = final_image_width
        self.linear = nn.Linear(self.final_channel * self.final_image_height * self.final_image_width, 256)
        self.critic_linear = nn.Linear(256 + self.time_emb_dim, 1)
        self.actor_linear = nn.Linear(256 + self.time_emb_dim, 4)
        self.global_id = 1

    @staticmethod
    def normalized_columns_initializer(weights, std=1.0):
        out = torch.randn(weights.size())
        out *= std / torch.sqrt(out.pow(2).sum(1, keepdim=True).expand_as(out))
        return out

    def init_weights(self):

        # Initializing weights
        self.apply(weights_init)
        # self.actor_linear.weight.data = self.normalized_columns_initializer(
        #     self.actor_linear.weight.data, 0.01)
        # self.actor_linear.bias.data.fill_(0)
        # self.critic_linear.weight.data = self.normalized_columns_initializer(
        #     self.critic_linear.weight.data, 1.0)
        # self.critic_linear.bias.data.fill_(0)

    def save_attention_prob(self, image, attention_prob, instruction, goal_prob=None):
        self.global_id += 1

        image_flipped = image[0, 0, :, :, :].cpu().data.numpy().swapaxes(0, 1).swapaxes(1, 2)
        attention_prob = attention_prob.cpu().data.numpy()
        resized_kernel = scipy.misc.imresize(attention_prob, (128, 128))
        if goal_prob is not None:
            goal_location = goal_prob.cpu().data.numpy()
            if np.sum(goal_location) > 0.01:
                for i in range(0, 8):
                    for j in range(0, 8):
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
            text_emb, text_emb_raw = self.text_module(instructions)
            text_emb_filter = torch.matmul(text_emb_raw, self.language_filter.weight).view(3, 64, 1, 1)
            text_emb_key = torch.matmul(text_emb_raw, self.attention.weight)
            image_hidden_states = None
        else:
            text_emb_key, text_emb_filter, image_hidden_states = model_state

        image_emb_seq = self.image_module(image)
        image_emb = image_emb_seq[:, 0, :, :, :]  # 1 x num_channels x height x width
        image_emb_convoluted = self.conv1(image_emb)  # 1 x num_channels x height x width

        # Language filters
        image_emb_language = F.conv2d(image_emb, weight=text_emb_filter, bias=None)
        x_1 = torch.cat([image_emb_convoluted, image_emb_language], dim=1)
        x_2 = F.leaky_relu(x_1)
        x_3 = self.conv2(x_2)
        image_emb_convoluted = torch.cat([x_3] + self.positional_encodings, dim=1)

        # Apply bilinear embedding
        image_emb_reshaped = image_emb_convoluted.view(image_emb_convoluted.size(1), -1)  # num_channels x (hght x wdth)
        attention_logits = torch.matmul(text_emb_key, image_emb_reshaped).view(-1)  # (height x width)
        not_present_logit = torch.matmul(text_emb_key, self.bias.weight).view(-1)
        attention_logits = torch.cat([attention_logits, not_present_logit], dim=0)
        attention_probs = F.softmax(attention_logits, dim=0)  # (height x width + 1)

        # if goal[0] is not None:
        #     gold_prob = GoalPrediction.generate_gold_prob(goal)
        #     self.save_attention_prob(image, attention_probs[:-1].view(8, 8),
        #                              instruction, goal_prob=gold_prob[:-1].view(8, 8))

        ########################################################################################
        volatile = dict()
        volatile["image_emb"] = image_emb
        volatile["attention_probs"] = attention_probs  # .view(self.final_image_height, self.final_image_width)
        volatile["attention_logits"] = attention_logits  # .view(self.final_image_height, self.final_image_width)
        return None, (text_emb_key, text_emb_filter, None), None, volatile
        ########################################################################################

        x = attention_probs[:-1] * image_emb_reshaped   # num_channels x (height x width)
        x = x.view(self.final_channel, self.final_image_height, self.final_image_width)
        # x = x + image_emb[0]  # Add residual connection
        x = x.view(1, -1)

        # A3C-LSTM
        x = F.relu(self.linear(x))
        new_image_hidden_states = self.image_recurrence_module(x, image_hidden_states)
        hx, cx = new_image_hidden_states

        time_emb = self.time_emb_layer(tx)
        x = torch.cat((hx, time_emb.view(-1, self.time_emb_dim)), 1)

        new_model_state = (text_emb_key, new_image_hidden_states)

        actor_linear = self.actor_linear(x)
        critic_linear = self.critic_linear(x)

        volatile = dict()
        volatile["image_emb"] = image_emb
        volatile["attention_probs"] = attention_probs  # .view(self.final_image_height, self.final_image_width)
        volatile["attention_logits"] = attention_logits  # .view(self.final_image_height, self.final_image_width)

        if mode is None or mode == ReadPointerAgent.ACT_MODE:
            return F.log_softmax(actor_linear, dim=1), new_model_state, image_emb_seq, volatile
        elif mode == ReadPointerAgent.READ_MODE:
            return F.log_softmax(self.dense_read(x)), new_model_state, image_emb_seq, volatile
        else:
            raise ValueError("invalid mode for model: %r" % mode)

    def forward_old(self, image, instructions, tx, mode, model_state):

        image_emb_seq = self.image_module(image)
        image_emb = image_emb_seq[:, 0, :, :, :]  # num_channels x height x width
        image_emb_convoluted = self.conv1(image_emb)  # num_channels x height x width

        if model_state is None:
            text_emb, text_emb_raw = self.text_module(instructions)
            text_emb_key = torch.matmul(text_emb_raw, self.attention.weight)
            image_hidden_states = None
        else:
            text_emb_key, image_hidden_states = model_state

        # Apply bilinear embedding
        image_emb_reshaped = image_emb_convoluted.view(image_emb_convoluted.size(1), -1)  # num_channels x (height x width)
        attention_logits = torch.matmul(text_emb_key, image_emb_reshaped).view(-1)  # (height x width)
        # attention_probs = F.softmax(attention_logits, dim=0)  # (height x width)
        attention_probs = F.sigmoid(attention_logits)

        x = attention_probs * image_emb_reshaped   # num_channels x (height x width)
        x = x.view(64, self.final_image_height, self.final_image_width)
        concatenation = torch.cat([image_emb[0], x], dim=0).view(1, 128, self.final_image_height, self.final_image_width)
        x = self.conv_mixer(concatenation)
        x = F.leaky_relu(x)
        x = x.view(1, -1)

        # A3C-LSTM
        x = F.relu(self.linear(x))
        new_image_hidden_states = self.image_recurrence_module(x, image_hidden_states)
        hx, cx = new_image_hidden_states

        time_emb = self.time_emb_layer(tx)
        x = torch.cat((hx, time_emb.view(-1, self.time_emb_dim)), 1)

        new_model_state = (text_emb_key, new_image_hidden_states)

        actor_linear = self.actor_linear(x)
        critic_linear = self.critic_linear(x)

        volatile = dict()
        volatile["image_emb"] = image_emb
        volatile["attention_probs"] = attention_probs.view(self.final_image_height, self.final_image_width)
        volatile["attention_logits"] = attention_logits.view(self.final_image_height, self.final_image_width)

        if mode is None or mode == ReadPointerAgent.ACT_MODE:
            return F.log_softmax(actor_linear, dim=1), new_model_state, image_emb_seq, volatile
        elif mode == ReadPointerAgent.READ_MODE:
            return F.log_softmax(self.dense_read(x)), new_model_state, image_emb_seq, volatile
        else:
            raise ValueError("invalid mode for model: %r" % mode)

