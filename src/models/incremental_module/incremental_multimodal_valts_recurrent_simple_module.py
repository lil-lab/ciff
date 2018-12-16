import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from agents.agent_with_read import ReadPointerAgent
from utils.cuda import cuda_var, cuda_tensor


class IncrementalMultimodalValtsRecurrentSimpleModule(nn.Module):
    """
    pytorch module for final part of model
    combines embeddings of image, text, and previous action.
    Valts Recurrence concatenates the embedding of current image and previous LSTM.
    """
    def __init__(self, image_module, text_module, action_module,
                 image_recurrence_module,
                 total_emb_size, num_actions):
        super(IncrementalMultimodalValtsRecurrentSimpleModule, self).__init__()
        self.image_module = image_module
        self.image_recurrence_module = image_recurrence_module
        self.text_module = text_module
        self.action_module = action_module
        self.dense1 = nn.Linear(total_emb_size, 512)
        self.dense2 = nn.Linear(512, num_actions)
        self.dense_read = nn.Linear(512, 2)
        self.hidden_dim = None
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

        new_image_memory_emb, new_image_hidden_states = \
            self.image_recurrence_module(image_emb_seq, image_lens, image_hidden_states)

        new_model_state = (text_emb, new_image_hidden_states, new_image_memory_emb)
        action_emb = self.action_module(prev_action)
        x = torch.cat([prev_image_memory_emb, image_emb, text_emb, action_emb], dim=1)
        x = F.leaky_relu(self.dense1(x))
        if mode is None or mode == ReadPointerAgent.ACT_MODE:
            return F.log_softmax(self.dense2(x)), new_model_state, image_emb_seq, x
        elif mode == ReadPointerAgent.READ_MODE:
            return F.log_softmax(self.dense_read(x)), new_model_state, image_emb_seq, x
        else:
            raise ValueError("invalid mode for model: %r" % mode)

