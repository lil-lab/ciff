import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from agents.agent_with_read import ReadPointerAgent


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


class IncrementalMultimodalEmnlp(nn.Module):
    """
    pytorch module for final part of model
    combines embeddings of image, text, and previous action
    """
    def __init__(self, image_module, text_module, previous_action_module, previous_block_module,
                 max_episode_length, input_embedding_size, output_hidden_size, blocks_hidden_size, directions_hidden_size):

        super(IncrementalMultimodalEmnlp, self).__init__()
        self.image_module = image_module
        self.text_module = text_module
        self.previous_action_module = previous_action_module
        self.previous_block_module = previous_block_module

        #layer that converts from the concatenated inputs to h^1
        self.final_embedder = nn.Linear(
            input_embedding_size,
            output_hidden_size
        )

        #final output
        self.blocks_layer = nn.Linear(output_hidden_size, blocks_hidden_size)
        self.directions_layer = nn.Linear(output_hidden_size, directions_hidden_size)

    def init_weights(self):

        # Initializing weights
        self.apply(weights_init)
        # self.actor_linear.weight.data = self.normalized_columns_initializer(
        #     self.actor_linear.weight.data, 0.01)
        # self.actor_linear.bias.data.fill_(0)
        # self.critic_linear.weight.data = self.normalized_columns_initializer(
        #     self.critic_linear.weight.data, 1.0)
        # self.critic_linear.bias.data.fill_(0)

    def forward(self, images, instruction, prev_block_id_batch, prev_direction_id_batch, model_state):


        images_embedded = self.image_module(images)
        # what is this? - iput should be batch x channel x height x width
        #output shoudl be batch x image_emb_size
#        image_emb = image_emb_seq[:, :, :, :]

        # Text embedding doesn't change, so we keep this across everything
        if model_state is None:
            text_emb= self.text_module(instruction)
            #print(text_emb.size())
        else:
            text_emb= model_state

        previous_direction_emb = self.previous_action_module(prev_direction_id_batch)
        previous_block_emb = self.previous_block_module(prev_block_id_batch)

        #combined
        action_emb = torch.cat((previous_direction_emb, previous_block_emb), 1)

        new_model_state = (text_emb)

        #concanate along embedding
        pre_h1 = torch.cat((images_embedded, text_emb, action_emb), 1)

        #embed to h1
        h1 = F.relu(self.final_embedder(pre_h1))

        blocks_logit = self.blocks_layer(h1)
        direction_logit = self.directions_layer(h1)

        #softmax over non-batch dimension
        block_logprob = F.log_softmax(blocks_logit, dim=1)  # 1 x num_block
        direction_logprob = F.log_softmax(direction_logit, dim=1)  # 1 x num_direction

        # this should work but i'm not convinced
        action_logprob = block_logprob.transpose(0, 1) + direction_logprob[:, :4]  # num_block x num_direction
        action_logprob = action_logprob.view(1, -1)
        stop_logprob = direction_logprob[:, 4:5]

        action_logprob = torch.cat([action_logprob, stop_logprob], dim=1)

        return action_logprob, new_model_state

