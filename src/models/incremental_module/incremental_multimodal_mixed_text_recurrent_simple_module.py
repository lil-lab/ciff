import torch
import torch.nn as nn
import torch.nn.functional as F

from agents.agent_with_read import ReadPointerAgent


class IncrementalMultimodalMixedTextRecurrentSimpleModule(nn.Module):
    """
    pytorch module for final part of model
    combines embeddings of image, text, and previous action
    """
    def __init__(self, image_module, symbolic_text_module, lstm_text_module,
                 action_module, image_recurrence_module, total_emb_size, num_actions):
        super(IncrementalMultimodalMixedTextRecurrentSimpleModule, self).__init__()
        self.image_module = image_module
        self.image_recurrence_module = image_recurrence_module
        self.symbolic_text_module = symbolic_text_module
        self.lstm_text_module = lstm_text_module
        self.action_module = action_module
        self.dense1 = nn.Linear(total_emb_size, 512)
        self.dense2 = nn.Linear(512, num_actions)
        self.dense_read = nn.Linear(512, 2)
        self.total_emb_size = total_emb_size

    def forward(self, image, image_lens, symbolic_instructions, real_instructions,
                prev_action, mode, model_state):

        image_emb_seq = self.image_module(image)

        if model_state is None:
            symbolic_text_emb = self.symbolic_text_module(symbolic_instructions)
            lstm_text_emb = self.lstm_text_module(real_instructions)
            text_emb = torch.cat([symbolic_text_emb, lstm_text_emb], dim=1)
            image_hidden_states = None
        else:
            text_emb, image_hidden_states = model_state
        image_emb, new_image_hidden_states = \
            self.image_recurrence_module(image_emb_seq, image_lens, image_hidden_states)

        new_model_state = (text_emb, new_image_hidden_states)
        action_emb = self.action_module(prev_action)
        x = torch.cat([image_emb, text_emb, action_emb], dim=1)
        x = F.relu(self.dense1(x))
        if mode is None or mode == ReadPointerAgent.ACT_MODE:
            return F.log_softmax(self.dense2(x)), new_model_state, image_emb_seq, x
        elif mode == ReadPointerAgent.READ_MODE:
            return F.log_softmax(self.dense_read(x)), new_model_state, image_emb_seq, x
        else:
            raise ValueError("invalid mode for model: %r" % mode)

