import torch
import scipy.misc
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np

from torch import nn as nn
from learning.auxiliary_objective.goal_prediction import GoalPrediction


class OracleGold(torch.nn.Module):
    """ Generates oracle probability based on the goal """

    def __init__(self,  image_recurrence_module, max_episode_length,
                 final_image_height, final_image_width):
        super(OracleGold, self).__init__()

        self.image_recurrence_module = image_recurrence_module

        # Time embedding layer, helps in stabilizing value prediction
        self.time_emb_dim = 32
        self.time_emb_layer = nn.Embedding(max_episode_length + 1, self.time_emb_dim)

        self.final_image_height = final_image_height
        self.final_image_width = final_image_width

        # A3C-LSTM layers
        self.linear = nn.Linear(final_image_height * final_image_width + 1, 256)
        self.critic_linear = nn.Linear(256 + self.time_emb_dim, 1)
        self.actor_linear = nn.Linear(256 + self.time_emb_dim, 4)

        # For logs
        self.global_id = 0

    def init_weights(self):
        return

    def forward(self, image, instructions, tx, mode, model_state, instruction, goal):

        if model_state is None:
            image_hidden_states = None
        else:
            image_hidden_states = model_state

        goal_prob = GoalPrediction.generate_gold_prob(goal, self.final_image_height, self.final_image_width)

        ############################
        # self.global_id += 1
        # if self.global_id % 10 == 0:
        #     attention_prob = goal_prob[:-1].view(32, 32).cpu().data.numpy()
        #     image_flipped = image[0][0].cpu().data.numpy().swapaxes(0, 1).swapaxes(1, 2)
        #     fmap = attention_prob
        #     fmap = (fmap - np.mean(fmap)) / np.std(fmap)
        #     resized_kernel = scipy.misc.imresize(fmap, (256, 256))
        #
        #     plt.title(instruction)
        #     plt.imshow(image_flipped)
        #     plt.imshow(resized_kernel, cmap='jet', alpha=0.5)
        #     plt.savefig("./train_gold_prob/image_" + str(self.global_id) + ".png")
        #     plt.clf()
        ############################

        volatile = dict()

        # Compute the probabilities
        x = goal_prob.view(1, -1)
        x = F.relu(self.linear(x))

        # Inserted the two lines below
        # hx = x
        # new_image_hidden_states = None
        new_image_hidden_states = self.image_recurrence_module(x, image_hidden_states)
        hx, cx = new_image_hidden_states

        time_emb = self.time_emb_layer(tx)
        x = torch.cat((hx, time_emb.view(-1, self.time_emb_dim)), 1)

        actor_linear = self.actor_linear(x)
        critic_linear = self.critic_linear(x)
        volatile["state_value"] = critic_linear

        new_model_state = new_image_hidden_states

        return F.log_softmax(actor_linear, dim=1), new_model_state, None, volatile


class OracleGoldWithImage(torch.nn.Module):
    """ Generates oracle probability based on the goal """

    def __init__(self,  image_recurrence_module, image_module,
                 max_episode_length, final_image_height, final_image_width):
        super(OracleGoldWithImage, self).__init__()

        self.image_recurrence_module = image_recurrence_module
        self.image_module = image_module

        # extra convolutions for image
        self.num_image_features = 16
        self.conv1 = nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2)
        self.conv2 = nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=2)
        self.conv3 = nn.Conv2d(256, 4, kernel_size=1)
        self.image_lin = nn.Linear(256, self.num_image_features)

        self.norm1 = nn.InstanceNorm2d(64, affine=True)
        self.norm2 = nn.InstanceNorm2d(128, affine=True)
        self.norm3 = nn.InstanceNorm2d(256, affine=True)
        self.norm_final = nn.InstanceNorm2d(4, affine=True)

        self.act1 = nn.PReLU(init=0.2)
        self.act2 = nn.PReLU(init=0.2)
        self.act3 = nn.PReLU(init=0.2)

        # Time embedding layer, helps in stabilizing value prediction
        self.time_emb_dim = 32
        self.time_emb_layer = nn.Embedding(max_episode_length + 1, self.time_emb_dim)

        self.final_image_height = final_image_height
        self.final_image_width = final_image_width

        # A3C-LSTM layers
        self.linear = nn.Linear(final_image_height * final_image_width + self.num_image_features + 1, 256)
        self.critic_linear = nn.Linear(256 + self.time_emb_dim, 1)
        self.actor_linear = nn.Linear(256 + self.time_emb_dim, 4)

        # For logs
        self.global_id = 0

    def init_weights(self):
        return

    def forward(self, image, instructions, tx, mode, model_state, instruction, goal):

        if model_state is None:
            image_hidden_states = None
        else:
            image_hidden_states = model_state

        img = self.image_module(image)
        if len(img.size()) == 5:
            size = img.size()
            img = img.view(size[0], size[2], size[3], size[4])
        b = img.size()[0]
        img = self.act1(self.conv1(self.norm1(img)))
        img = self.act2(self.conv2(self.norm2(img)))
        img = self.act3(self.conv3(self.norm3(img)))
        img = F.sigmoid(self.image_lin(self.norm_final(img).view(1, -1)))

        goal_prob = GoalPrediction.generate_gold_prob(goal, self.final_image_height, self.final_image_width)

        volatile = dict()

        # Compute the probabilities
        x = goal_prob.view(1, -1)
        x = torch.cat([x, img], dim=1)
        x = F.relu(self.linear(x))
        new_image_hidden_states = self.image_recurrence_module(x, image_hidden_states)
        hx, cx = new_image_hidden_states

        time_emb = self.time_emb_layer(tx)
        x = torch.cat((hx, time_emb.view(-1, self.time_emb_dim)), 1)

        actor_linear = self.actor_linear(x)
        critic_linear = self.critic_linear(x)
        volatile["state_value"] = critic_linear

        new_model_state = new_image_hidden_states

        return F.log_softmax(actor_linear, dim=1), new_model_state, None, volatile