import torch
import torch.nn as nn
import torch.nn.functional as F

from agents.agent_with_read import ReadPointerAgent


class IncrementalMultimodalRecurrentSimpleRecurrentactionModule(nn.Module):
    """
    pytorch module for final part of model
    combines embeddings of image, text, and previous action
    """
    def __init__(self, image_module, text_module, action_module,
                 action_recurrence_module, image_recurrence_module,
                 total_emb_size, num_actions):
        super(IncrementalMultimodalRecurrentSimpleRecurrentactionModule, self).__init__()
        self.image_module = image_module
        self.image_recurrence_module = image_recurrence_module
        self.text_module = text_module
        self.action_module = action_module
        self.recurrent_action_module = action_recurrence_module
        self.dense1 = nn.Linear(total_emb_size, 512)
        self.dense2 = nn.Linear(512, num_actions)
        self.dense_read = nn.Linear(512, 2)
        self.total_emb_size = total_emb_size

    def forward(self, image, image_lens, instructions, prev_action, mode, model_state):

        image_emb_seq = self.image_module(image)

        if model_state is None:
            text_emb = self.text_module(instructions)
            image_hidden_states = None
            action_hidden_states = None
        else:
            text_emb, image_hidden_states, action_hidden_states = model_state
        image_emb, new_image_hidden_states = \
            self.image_recurrence_module(image_emb_seq, image_lens, image_hidden_states)

        action_emb = self.action_module(prev_action)
        action_seq_emb, new_action_hidden_states = \
            self.action_recurrence_module(action_emb, action_hidden_states)
        x = torch.cat([image_emb, text_emb, action_seq_emb], dim=1)
        x = F.relu(self.dense1(x))
        new_model_state = (text_emb, new_image_hidden_states,
                           new_action_hidden_states)
        if mode is None or mode == ReadPointerAgent.ACT_MODE:
            return F.log_softmax(self.dense2(x)), new_model_state, image_emb_seq
        elif mode == ReadPointerAgent.READ_MODE:
            return F.log_softmax(self.dense_read(x)), new_model_state, image_emb_seq
        else:
            raise ValueError("invalid mode for model: %r" % mode)

