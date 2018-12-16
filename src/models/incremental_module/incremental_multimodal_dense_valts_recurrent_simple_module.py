import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from agents.agent_with_read import ReadPointerAgent
from utils.cuda import cuda_tensor


class IncrementalMultimodalDenseValtsRecurrentSimpleModule(nn.Module):
    """
    pytorch module for final part of model
    combines embeddings of image, text, and previous action.
    Valts Recurrence concatenates the embedding of current image and previous LSTM.
    """
    def __init__(self, image_module, text_module, action_module,
                 image_recurrence_module,
                 total_emb_size, num_actions):
        super(IncrementalMultimodalDenseValtsRecurrentSimpleModule, self).__init__()
        self.image_module = image_module
        self.image_recurrence_module = image_recurrence_module
        self.text_module = text_module
        self.action_module = action_module
        hidden_dim = 32
        self.dense1 = nn.Linear(total_emb_size, hidden_dim)
        self.dense2 = nn.Linear(total_emb_size + hidden_dim, hidden_dim)
        self.dense3 = nn.Linear(total_emb_size + 2 * hidden_dim, hidden_dim)
        self.dense_action = nn.Linear(hidden_dim, num_actions)
        self.dense_read = nn.Linear(hidden_dim, 2)
        self.total_emb_size = total_emb_size

    def forward(self, image, image_lens, instructions, prev_action, mode, model_state):

        image_emb_seq = self.image_module(image)
        num_states = image_emb_seq.size()[0]
        image_emb = image_emb_seq.view(num_states, -1)

        if model_state is None:
            text_emb = self.text_module(instructions)
            image_hidden_states = None
            dims = (num_states, self.image_recurrence_module.output_emb_dim)
            prev_image_memory_emb = Variable(cuda_tensor(torch.zeros(*dims)), requires_grad=False)
        else:
            text_emb, image_hidden_states, prev_image_memory_emb = model_state

        action_emb = self.action_module(prev_action)
        image_action_embedding = torch.cat([image_emb, action_emb], dim=1)
        image_action_embedding = image_action_embedding.view(num_states, 1, -1)

        # new_image_memory_emb, new_image_hidden_states = \
        #     self.image_recurrence_module(image_emb_seq, image_lens, image_hidden_states)
        new_image_memory_emb, new_image_hidden_states = \
            self.image_recurrence_module(image_action_embedding, image_lens, image_hidden_states)

        new_model_state = (text_emb, new_image_hidden_states, new_image_memory_emb)
        x_input = torch.cat([prev_image_memory_emb, image_emb, text_emb, action_emb], dim=1)

        x_1 = F.leaky_relu(self.dense1(x_input))
        x_2 = F.leaky_relu(self.dense2(torch.cat([x_input, x_1], dim=1)))
        x_3 = F.leaky_relu(self.dense3(torch.cat([x_input, x_1, x_2], dim=1)))

        if mode is None or mode == ReadPointerAgent.ACT_MODE:
            val = self.dense_action(x_3)
            # val = torch.clamp(val, min=-2, max=2)
            return F.log_softmax(val), new_model_state, image_emb_seq, x_3
        elif mode == ReadPointerAgent.READ_MODE:
            return F.log_softmax(self.dense_read(x_3)), new_model_state, image_emb_seq, x_3
        else:
            raise ValueError("invalid mode for model: %r" % mode)
