import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


def normalized_columns_initializer(weights, std=1.0):
    out = torch.randn(weights.size())
    out *= std / torch.sqrt(out.pow(2).sum(1, keepdim=True).expand_as(out))
    return out


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


class a3c_lstm_ga_default_with_aux(torch.nn.Module):

    def __init__(self, args, config=None):
        super(a3c_lstm_ga_default_with_aux, self).__init__()
        self.config = config

        # Image Processing
        self.conv1 = nn.Conv2d(3, 128, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(128, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=4, stride=2)

        # Instruction Processing
        self.gru_hidden_size = 256
        self.input_size = args.input_size
        self.embedding = nn.Embedding(self.input_size, 32)
        self.gru = nn.GRU(32, self.gru_hidden_size)
        # self.instruction_lstm = nn.LSTMCell(48, self.gru_hidden_size)

        # Gated-Attention layers
        self.attn_linear = nn.Linear(self.gru_hidden_size, 64)

        # Time embedding layer, helps in stabilizing value prediction
        self.time_emb_dim = 32
        self.time_emb_layer = nn.Embedding(
                args.max_episode_length+1,
                self.time_emb_dim)

        # A3C-LSTM layers
        self.final_image_height = 6
        self.final_image_width = 6
        self.linear = nn.Linear(64 * self.final_image_height * self.final_image_width, 256)
        self.lstm = nn.LSTMCell(256, 256)
        self.critic_linear = nn.Linear(256 + self.time_emb_dim, 1)
        self.actor_linear = nn.Linear(256 + self.time_emb_dim, 4)

        # Auxiliary loss
        # Temporal autoencoding
        action_dim = 32
        dim = 256
        self.action_embedding = nn.Embedding(config["num_actions"], action_dim)
        self.tae_linear_1 = nn.Linear(256 + action_dim, dim)
        self.tae_linear_2 = nn.Linear(dim, 256)

        # Reward prediction loss
        self.rp_linear_1 = nn.Linear(256 + action_dim, 256)
        self.rp_linear_2 = nn.Linear(256, 1)

        # Alignment Auxiliary
        self.W_image_text_alignment = nn.Linear(64 * self.final_image_height * self.final_image_width,
                                                self.gru_hidden_size)

        # Goal prediction Auxiliary
        self.conv4 = nn.Conv2d(64, 256, kernel_size=3, stride=1, padding=1)
        self.goal_prediction_bilinear = nn.Linear(256, self.gru_hidden_size)

        # Initializing weights
        self.apply(weights_init)
        self.actor_linear.weight.data = normalized_columns_initializer(
            self.actor_linear.weight.data, 0.01)
        self.actor_linear.bias.data.fill_(0)
        self.critic_linear.weight.data = normalized_columns_initializer(
            self.critic_linear.weight.data, 1.0)
        self.critic_linear.bias.data.fill_(0)

        # TAE Loss
        self.tae_linear_1.weight.data = normalized_columns_initializer(
            self.tae_linear_1.weight.data, 1.0)
        self.tae_linear_1.bias.data.fill_(0)
        self.tae_linear_2.weight.data = normalized_columns_initializer(
            self.tae_linear_2.weight.data, 1.0)
        self.tae_linear_2.bias.data.fill_(0)

        # RP Linear
        self.rp_linear_1.weight.data = normalized_columns_initializer(
            self.rp_linear_1.weight.data, 1.0)
        self.rp_linear_1.bias.data.fill_(0)
        self.rp_linear_2.weight.data = normalized_columns_initializer(
            self.rp_linear_2.weight.data, 1.0)
        self.rp_linear_2.bias.data.fill_(0)

        # Alignment function
        self.W_image_text_alignment.weight.data = normalized_columns_initializer(
            self.W_image_text_alignment.weight.data, 0.01)
        self.W_image_text_alignment.bias.data.fill_(0)

        # Goal prediction bilinear function
        self.goal_prediction_bilinear.weight.data = normalized_columns_initializer(
            self.goal_prediction_bilinear.weight.data, 0.01)
        self.goal_prediction_bilinear.bias.data.fill_(0)

        # Image Auxiliary
        self.lstm.bias_ih.data.fill_(0)
        self.lstm.bias_hh.data.fill_(0)
        # self.instruction_lstm.bias_ih.data.fill_(0)
        # self.instruction_lstm.bias_hh.data.fill_(0)
        self.train()
        self.global_id = 0

    def forward(self, inputs, cached_computation=None):
        x, input_inst, _, _, (tx, hx, cx) = inputs
        image = x

        # Get the image representation
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x_image_rep = F.relu(self.conv3(x))

        if cached_computation is None:
            # Get the instruction representation
            encoder_hidden = Variable(torch.zeros(1, 1, self.gru_hidden_size).cuda())
            # encoder_cell = Variable(torch.zeros(1, 1, self.gru_hidden_size).cuda())
            for i in range(input_inst.data.size(1)):
                word_embedding = self.embedding(input_inst[0, i]).unsqueeze(0)
                _, encoder_hidden = self.gru(word_embedding, encoder_hidden)
                # encoder_hidden, encoder_cell = self.instruction_lstm(word_embedding, encoder_hidden)
            x_instr_rep = encoder_hidden.view(encoder_hidden.size(1), -1)

            # Get the attention vector from the instruction representation
            x_attention = F.sigmoid(self.attn_linear(x_instr_rep))

            # Gated-Attention
            x_attention = x_attention.unsqueeze(2).unsqueeze(3)
            x_attention = x_attention.expand(1, 64, self.final_image_height, self.final_image_width)
            cached_computation = dict()
            cached_computation["x_attention"] = x_attention
            cached_computation["text_rep"] = x_instr_rep
        else:
            x_attention = cached_computation["x_attention"]

        assert x_image_rep.size() == x_attention.size()
        x = x_image_rep*x_attention
        # self.save_image(image, x_image_rep, x_attention)
        x = x.view(x.size(0), -1)

        # A3C-LSTM
        cached_computation["image_rep"] = x_image_rep
        x = self.linear(x)
        x = F.relu(x)
        hx, cx = self.lstm(x, (hx, cx))
        time_emb = self.time_emb_layer(tx)
        cached_computation["lstm_rep"] = hx
        x = torch.cat((hx, time_emb.view(-1, self.time_emb_dim)), 1)

        return self.critic_linear(x), self.actor_linear(x), (hx, cx), cached_computation

    def get_tae_loss(self, image_rep, actions):

        if len(image_rep) == 1:
            return None

        actions = torch.cat(actions, dim=0).view(-1)
        image_rep = torch.cat(image_rep, dim=0)
        action_embedding = self.action_embedding(Variable(actions).cuda())

        image_action = torch.cat([image_rep, action_embedding], dim=1)
        x = self.tae_linear_1(image_action)
        x = F.relu(x)
        next_image_embedding = self.tae_linear_2(x)
        image_rep = image_rep[1:, :]
        next_image_embedding = next_image_embedding[:-1, :]

        x = (image_rep - next_image_embedding) ** 2
        x = torch.sqrt(torch.sum(x, dim=1))
        x = torch.mean(x)

        return x

    def get_reward_prediction_loss(self, lstm_rep, actions, rewards):

        actions = torch.cat(actions, dim=0).view(-1)
        action_embedding = self.action_embedding(Variable(actions).cuda())

        rewards = torch.from_numpy(np.array(rewards)).float()
        rewards = Variable(rewards).cuda()

        lstm_rep = torch.cat(lstm_rep, dim=0)

        x = torch.cat([lstm_rep, action_embedding], dim=1)

        x = self.rp_linear_1(x)
        x = F.relu(x)
        x = self.rp_linear_2(x)
        x = x.view(-1)

        loss = (x - rewards) ** 2
        mean_loss = torch.mean(loss)

        return mean_loss

    def alignment_auxiliary(self, image_rep, text_rep):

        last_image = torch.cat(image_rep, dim=0)  # batch x image_dimension
        text_bilinear = torch.matmul(text_rep, self.W_image_text_alignment.weight)  # 1 x image_dim
        text_image_logits = torch.matmul(last_image, text_bilinear.transpose(0, 1))  # batch x 1
        alignment_score = torch.mean(text_image_logits)

        alignment_loss = -F.logsigmoid(alignment_score)
        w_norm = torch.norm(self.W_image_text_alignment.weight)
        alignment_loss = alignment_loss + 0.1 * w_norm

        return alignment_loss, w_norm

    def calc_goal_prediction_loss(self, unattended_image_rep, text_rep, goal_location):

        weight_text = torch.matmul(text_rep, self.goal_prediction_bilinear.weight)  # 1 x num_kernels
        loss = None
        unattended_image_rep_concat = torch.cat(unattended_image_rep, dim=0)
        convoluted_image_rep = self.conv4(unattended_image_rep_concat)
        convoluted_image_rep = convoluted_image_rep.view(convoluted_image_rep.size(0), convoluted_image_rep.size(1), -1)
        for ix, _ in enumerate(unattended_image_rep):
            if goal_location[ix][0] is None:
                continue
            image_rep = convoluted_image_rep[ix, :, :]  # num_kernels x (height x width)
            logits = torch.matmul(weight_text, image_rep).view(6, 6)
            # max_height_logit = torch.max(logits, dim=0)[0]  # width
            logits = logits.view(36)
            log_prob = F.log_softmax(logits, dim=0).view(6, 6)
            if loss is None:
                loss = -log_prob[goal_location[ix][0], goal_location[ix][1]]
            else:
                loss = loss - log_prob[goal_location[ix][0], goal_location[ix][1]]
            '''target = Variable(torch.zeros(6, 6).float()).cuda()
            if goal_location[ix][0] is not None:
                target[goal_location[ix][0], goal_location[ix][1]] = 1.0
            target = target.view(-1)
            binary_cross_entropy_loss = F.binary_cross_entropy_with_logits(logits, target)
            if loss is None:
                loss = binary_cross_entropy_loss
            else:
                loss += binary_cross_entropy_loss'''

        return loss
