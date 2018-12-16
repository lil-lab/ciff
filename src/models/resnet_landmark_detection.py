import math
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from agents.agent_observed_state import AgentObservedState
from models.module.image_resnet_module import ImageResnetModule
from models.module.image_ryan_resnet_module import ImageRyanResnetModule
from utils.cuda import cuda_var
from utils.nav_drone_landmarks import get_all_landmark_names
from utils.nav_drone_symbolic_instructions import NO_BUCKETS, NUM_LANDMARKS, BUCKET_WIDTH


class ResnetLandmarkDetection(object):
    def __init__(self, config, constants, image_module_path):
        self.none_action = config["num_actions"]
        self.landmark_names = get_all_landmark_names()
        self.image_module = ImageRyanResnetModule(
            image_emb_size=constants["image_emb_dim"],
            input_num_channels=3,
            image_height=config["image_height"],
            image_width=config["image_width"],
            using_recurrence=True)

        # TODO: load pre-trained image module from file
        # image_module_path = "path/to/saved/image/module"
        if torch.cuda.is_available():
            torch_load = torch.load
        else:
            torch_load = lambda f_: torch.load(f_, map_location=lambda s_, l_: s_)

        if image_module_path is not None:
            self.image_module.load_state_dict(torch_load(image_module_path))

        total_emb_size = constants["image_emb_dim"] * 6
        final_module = ImageRyanDetectionModule(
            image_module=self.image_module,
            image_emb_size=total_emb_size)
        self.final_module = final_module
        if torch.cuda.is_available():
            self.image_module.cuda()
            self.final_module.cuda()
        self.image_module.init_weights()

    def get_probs(self, images):

        image_batch = cuda_var(torch.from_numpy(np.array(images)).float())
        return self.final_module(image_batch)

    def get_probs_and_visible_objects(self, agent_observed_state_list):
        for aos in agent_observed_state_list:
            assert isinstance(aos, AgentObservedState)
        # print "batch size:", len(agent_observed_state_list)

        # sort list by instruction length
        agent_observed_state_list = sorted(
            agent_observed_state_list,
            key=lambda aos_: len(aos_.get_instruction()),
            reverse=True
        )

        images = [[aos.get_last_image()] for aos in agent_observed_state_list]
        image_batch = cuda_var(torch.from_numpy(np.array(images)).float())

        landmarks_visible = []
        for aos in agent_observed_state_list:
            x_pos, z_pos, y_angle = aos.get_position_orientation()
            landmark_pos_dict = aos.get_landmark_pos_dict()
            visible_landmarks = get_visible_landmark_r_theta(
                x_pos, z_pos, y_angle, landmark_pos_dict, self.landmark_names)
            landmarks_visible.append(visible_landmarks)

        # shape is BATCH_SIZE x 63 x 2
        probs_batch = self.final_module(image_batch)

        # landmarks_visible is list of length BATCH_SIZE, each item is a set containing landmark indices
        return probs_batch, landmarks_visible

    def load_saved_model(self, load_dir):
        if torch.cuda.is_available():
            torch_load = torch.load
        else:
            torch_load = lambda f_: torch.load(f_, map_location=lambda s_, l_: s_)
        image_module_path = os.path.join(load_dir, "image_module_state.bin")
        self.image_module.load_state_dict(torch_load(image_module_path))
        final_module_path = os.path.join(load_dir, "final_module_state.bin")
        self.final_module.load_state_dict(torch_load(final_module_path))

    def save_model(self, save_dir):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # save state file for image nn
        image_module_path = os.path.join(save_dir, "image_module_state.bin")
        torch.save(self.image_module.state_dict(), image_module_path)
        # save state file for final nn
        final_module_path = os.path.join(save_dir, "final_module_state.bin")
        torch.save(self.final_module.state_dict(), final_module_path)

    def get_parameters(self):
        parameters = list(self.image_module.parameters())
        parameters += list(self.final_module.parameters())
        return parameters


class ImageDetectionModule(nn.Module):
    def __init__(self, image_module, image_emb_size):
        super(ImageDetectionModule, self).__init__()
        self.image_module = image_module
        self.dense_landmark = nn.Linear(image_emb_size, NUM_LANDMARKS * 2)
        self.dense_distance = nn.Linear(image_emb_size, NUM_LANDMARKS * 15)
        self.dense_theta = nn.Linear(image_emb_size, NUM_LANDMARKS * NO_BUCKETS)

    def forward(self, image_batch):
        batch_size = image_batch.size(0)
        image_module_output = self.image_module(image_batch)
        image_module_output = image_module_output[:, -1, :]  # Take the last image in resnet

        x_landmark = self.dense_landmark(image_module_output)
        x_distance = self.dense_distance(image_module_output)
        x_theta = self.dense_theta(image_module_output)

        x_landmark = F.log_softmax(x_landmark.view(batch_size * NUM_LANDMARKS, 2))
        x_distance = F.log_softmax(x_distance.view(batch_size * NUM_LANDMARKS, 15))
        x_theta = F.log_softmax(x_theta.view(batch_size * NUM_LANDMARKS, NO_BUCKETS))
        return x_landmark.view(batch_size, NUM_LANDMARKS, 2), \
               x_distance.view(batch_size, NUM_LANDMARKS, 15), \
               x_theta.view(batch_size, NUM_LANDMARKS, NO_BUCKETS)


class ImageRyanDetectionModule(nn.Module):
    def __init__(self, image_module, image_emb_size):
        super(ImageRyanDetectionModule, self).__init__()
        image_emb_size = 32  # NO_BUCKETS * 32
        self.image_module = image_module
        self.dense_theta = nn.Linear(image_emb_size, NUM_LANDMARKS)

    def forward(self, image_batch):
        batch_size = image_batch.size(0)
        image_module_output = self.image_module(image_batch)  # batch x 12 x 32
        image_module_output = image_module_output.mean(1)
        image_module_output = image_module_output.view(batch_size, 32)
        x_theta = self.dense_theta(image_module_output).contiguous()  # batch x 67

        return x_theta.view(batch_size, NUM_LANDMARKS)


def get_visible_landmark_r_theta(x_pos, z_pos, y_angle, landmark_pos_dict,
                                 landmark_names):
    landmarks_in_view = set([])
    for landmark, (landmark_x, landmark_z) in landmark_pos_dict.iteritems():
        # get angle between drone's current orientation and landmark
        landmark_angle = 90.0 - np.arctan2(landmark_z - z_pos, landmark_x - x_pos) * 180.0 / math.pi
        angle_diff = landmark_angle - y_angle
        while angle_diff > 180.0:
            angle_diff -= 360.0
        while angle_diff < -180.0:
            angle_diff += 360.0
        if abs(angle_diff) <= 30.0:
            landmarks_in_view.add(landmark_names.index(landmark))

    return landmarks_in_view
