import torch
import torch.nn as nn
import torch.nn.functional as F

from agents.agent_with_read import ReadPointerAgent


class IncrementalJointMultimodalRecurrentSimpleModule(nn.Module):
    """
    pytorch module for final part of model
    combines embeddings of image, text, and previous action
    """
    def __init__(self, image_module, text_module, action_module,
                 image_recurrence_module, angle_prediction_model, landmark_model,
                 total_emb_size, num_actions):
        super(IncrementalJointMultimodalRecurrentSimpleModule, self).__init__()
        self.image_module = image_module
        self.image_recurrence_module = image_recurrence_module
        self.angle_prediction_model = angle_prediction_model
        self.landmark_model = landmark_model
        self.text_module = text_module
        self.action_module = action_module
        self.dense1 = nn.Linear(total_emb_size, 512)
        self.dense2 = nn.Linear(512, num_actions)
        self.dense_read = nn.Linear(512, 2)
        self.total_emb_size = total_emb_size

    def get_landmark_prob(self, instructions):
        prob_landmark, prob_theta_1, prob_theta_2, prob_r = \
            self.landmark_model.final_module(instructions)
        return prob_landmark

    def get_angle_prob(self, start_images):
        log_prob_theta = self.angle_prediction_model.get_probs([start_images])
        return log_prob_theta

    def forward(self, image, image_lens, instructions, start_images, prev_action, mode, model_state):

        image_emb_seq = self.image_module(image)

        if model_state is None:
            prob_theta = self.get_angle_prob(start_images)
            prob_landmark = self.get_landmark_prob(instructions)
            text_emb = self.text_module(instructions, prob_landmark, prob_theta)
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

