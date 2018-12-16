import torch
import torch.nn as nn
import torch.nn.functional as F

from agents.agent_with_read import ReadPointerAgent


class IncrementalMultimodalRecurrentSimpleGoalImageModule(nn.Module):
    """
    pytorch module for final part of model
    combines embeddings of image, text, and previous action
    """
    def __init__(self, image_module, action_module, image_recurrence_module, total_emb_size, num_actions):
        super(IncrementalMultimodalRecurrentSimpleGoalImageModule, self).__init__()
        self.image_module = image_module
        self.image_recurrence_module = image_recurrence_module
        self.action_module = action_module
        self.dense1 = nn.Linear(total_emb_size, 512)
        self.dense2 = nn.Linear(512, num_actions)
        self.dense_read = nn.Linear(512, 2)
        self.total_emb_size = total_emb_size

    def forward(self, image, image_lens, goal_image, prev_action, mode, model_state):

        image_emb_seq = self.image_module(image)
        num_states = goal_image.size()[0]

        if model_state is None:
            last_goal_emb = self.image_module(goal_image).view(num_states, -1)
            image_hidden_states = None
        else:
            last_goal_emb, image_hidden_states = model_state
        # image_emb, new_image_hidden_states = \
        #     self.image_recurrence_module(image_emb_seq, image_lens, image_hidden_states)
        image_emb, new_image_hidden_states = image_emb_seq.view(num_states, -1), None

        new_model_state = (last_goal_emb, new_image_hidden_states)
        action_emb = self.action_module(prev_action)
        x = torch.cat([image_emb, last_goal_emb, action_emb], dim=1)
        x = F.relu(self.dense1(x))
        if mode is None or mode == ReadPointerAgent.ACT_MODE:
            return F.log_softmax(self.dense2(x)), new_model_state, image_emb_seq
        elif mode == ReadPointerAgent.READ_MODE:
            return F.log_softmax(self.dense_read(x)), new_model_state, image_emb_seq
        else:
            raise ValueError("invalid mode for model: %r" % mode)

    def forward1(self, image, image_lens, goal_image, prev_action, mode, model_state):

        image_emb_seq = self.image_module(image)

        if model_state is None:
            num_states = goal_image.size()[0]
            last_image_emb = self.image_module(goal_image).view(num_states, -1)
            image_hidden_states = None
        else:
            last_image_emb, image_hidden_states = model_state
        image_emb, new_image_hidden_states = \
            self.image_recurrence_module(image_emb_seq, image_lens, image_hidden_states)

        new_model_state = (last_image_emb, new_image_hidden_states)
        action_emb = self.action_module(prev_action)
        x = torch.cat([image_emb, last_image_emb, action_emb], dim=1)
        x = F.relu(self.dense1(x))
        if mode is None or mode == ReadPointerAgent.ACT_MODE:
            return F.log_softmax(self.dense2(x)), new_model_state, image_emb_seq
        elif mode == ReadPointerAgent.READ_MODE:
            return F.log_softmax(self.dense_read(x)), new_model_state, image_emb_seq
        else:
            raise ValueError("invalid mode for model: %r" % mode)

