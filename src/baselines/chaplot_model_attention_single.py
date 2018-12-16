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


class a3c_lstm_ga_attention_single(torch.nn.Module):

    def __init__(self, args, config=None):
        super(a3c_lstm_ga_attention_single, self).__init__()
        self.config = config

        # Image Processing
        self.conv1 = nn.Conv2d(3, 128, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(128, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=4, stride=2)
        self.final_image_height = 6
        self.final_image_width = 6

        # Instruction Processing
        self.gru_hidden_size = 256
        self.input_size = args.input_size
        self.embedding = nn.Embedding(self.input_size, 32)
        self.gru_f = nn.GRU(32, self.gru_hidden_size)
        self.gru_b = nn.GRU(32, self.gru_hidden_size)

        # Time embedding layer, helps in stabilizing value prediction
        self.time_emb_dim = 32
        self.time_emb_layer = nn.Embedding(
                args.max_episode_length+1,
                self.time_emb_dim)

        # attention layers
        self.image_emb_layer = nn.Linear(64 * self.final_image_height * self.final_image_width, 256)
        self.attn_temp_size = 32
        self.attn_temporal_emb = nn.Embedding(2, self.attn_temp_size)
        self.attn_key_size = 256
        self.attention_key = nn.Linear(256 * 3, self.attn_key_size)
        self.attention_bilinear = nn.Bilinear(self.attn_key_size, 512 + self.attn_temp_size, 1)
        self.reduce_layer = nn.Linear(512 + self.attn_temp_size, 128)
        self.gating_key_layer = nn.Linear(512 + 128, 64)

        # A3C-LSTM layers
        self.linear = nn.Linear(64 * self.final_image_height * self.final_image_width, 256)
        self.lstm = nn.LSTMCell(256, 256)
        self.critic_linear = nn.Linear(256 + self.time_emb_dim, 1)
        self.actor_linear = nn.Linear(256 + self.time_emb_dim, 4)

        # Initializing weights
        self.apply(weights_init)
        self.actor_linear.weight.data = normalized_columns_initializer(
            self.actor_linear.weight.data, 0.01)
        self.actor_linear.bias.data.fill_(0)
        self.critic_linear.weight.data = normalized_columns_initializer(
            self.critic_linear.weight.data, 1.0)
        self.critic_linear.bias.data.fill_(0)

        self.lstm.bias_ih.data.fill_(0)
        self.lstm.bias_hh.data.fill_(0)
        self.train()

    def forward(self, inputs, cached_computation=None):
        x, curr_inst, prev_inst, next_inst, (tx, hx, cx) = inputs
        all_inst = torch.cat([prev_inst, curr_inst, next_inst], dim=1)
        num_prev = int(prev_inst.size(1))
        num_curr = int(curr_inst.size(1))
        num_next = int(next_inst.size(1))

        # Get the image representation
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x_image_rep = F.relu(self.conv3(x))

        if cached_computation is None:
            # run forward LSTM
            hidden_list_f = []
            encoder_hidden_f = Variable(torch.zeros(1, 1, self.gru_hidden_size).cuda())
            for i in range(all_inst.data.size(1)):
                word_embedding = self.embedding(all_inst[0, i]).unsqueeze(0)
                _, encoder_hidden_f = self.gru_f(word_embedding, encoder_hidden_f)
                hidden_list_f.append(encoder_hidden_f.view(encoder_hidden_f.size(1), -1))
            curr_instr_rep_f = hidden_list_f[num_prev + num_curr - 1]

            # run backwards LSTM
            hidden_list_b = []
            encoder_hidden_b = Variable(torch.zeros(1, 1, self.gru_hidden_size).cuda())
            for i in range(all_inst.data.size(1)):
                word_embedding = self.embedding(all_inst[0, i]).unsqueeze(0)
                _, encoder_hidden_b = self.gru_b(word_embedding, encoder_hidden_b)
                hidden_list_b.append(encoder_hidden_b.view(encoder_hidden_b.size(1), -1))
            hidden_list_b = hidden_list_b[::-1]
            curr_instr_rep_b = hidden_list_b[num_prev]

            # compute attention key inputs for prev/next instructions
            curr_instr_rep = torch.cat([curr_instr_rep_f, curr_instr_rep_b], dim=1)

            # do attention over previous/next instruction
            instr_rep_f = torch.cat(hidden_list_f[:num_prev] + hidden_list_f[-num_next:], dim=0)
            instr_rep_b = torch.cat(hidden_list_b[:num_prev] + hidden_list_b[-num_next:], dim=0)
            zeros = torch.from_numpy(np.array([0 for _ in range(num_prev)]))
            prev_temporal_tokens = Variable(zeros.cuda()).view(num_prev)
            ones = torch.from_numpy(np.array([1 for _ in range(num_next)]))
            next_temporal_tokens = Variable(ones.cuda()).view(num_next)
            temporal_tokens = torch.cat([prev_temporal_tokens, next_temporal_tokens], dim=0)
            temporal_emb = self.attn_temporal_emb(temporal_tokens)
            instr_rep = torch.cat([instr_rep_f, instr_rep_b, temporal_emb], dim=1)

            cached_computation = dict()
            cached_computation["curr_instr_rep"] = curr_instr_rep
            cached_computation["instr_rep"] = instr_rep
        else:
            curr_instr_rep = cached_computation["curr_instr_rep"]
            instr_rep = cached_computation["instr_rep"]

        image_emb_input = x_image_rep.view(x.size(0), -1)
        image_emb = self.image_emb_layer(image_emb_input)
        attention_key_input = torch.cat([curr_instr_rep, image_emb], dim=1)
        attn_key = self.attention_key(attention_key_input).repeat(num_prev + num_next, 1)
        attn_weights = self.attention_bilinear(attn_key, instr_rep)
        attn_weights = F.softmax(attn_weights, dim=0).repeat(1, self.gru_hidden_size * 2 + self.attn_temp_size)
        instruction_sum = (instr_rep * attn_weights).sum(0).view(1, self.gru_hidden_size * 2 + self.attn_temp_size)
        instr_gating_key = self.reduce_layer(instruction_sum)
        # calculate key for gating
        gate_key_input = torch.cat([curr_instr_rep, instr_gating_key], dim=1)
        gate_key = F.sigmoid(self.gating_key_layer(gate_key_input))

        # Gated-Attention
        x_attention = gate_key.unsqueeze(2).unsqueeze(3)
        x_attention = x_attention.expand(1, 64, self.final_image_height, self.final_image_width)
        assert x_image_rep.size() == x_attention.size()
        x = x_image_rep * x_attention
        x = x.view(x.size(0), -1)

        # A3C-LSTM
        x = F.relu(self.linear(x))
        hx, cx = self.lstm(x, (hx, cx))
        time_emb = self.time_emb_layer(tx)
        x = torch.cat((hx, time_emb.view(-1, self.time_emb_dim)), 1)

        return self.critic_linear(x), self.actor_linear(x), (hx, cx), cached_computation
