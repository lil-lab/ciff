import math
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models.resnet as resnet
import scipy.misc

from agents.agent_observed_state import AgentObservedState
from utils.cuda import cuda_var
from utils.nav_drone_landmarks import get_all_landmark_names


class StandardResnetImageDetection(object):
    def __init__(self, config, constants, image_module_path):
        self.none_action = config["num_actions"]
        self.landmark_names = get_all_landmark_names()
        self.image_module = resnet.resnet18(pretrained=True)

        # TODO: load pre-trained image module from file
        # image_module_path = "path/to/saved/image/module"
        if torch.cuda.is_available():
            torch_load = torch.load
        else:
            torch_load = lambda f_: torch.load(f_, map_location=lambda s_, l_: s_)

        if image_module_path is not None:
            self.image_module.load_state_dict(torch_load(image_module_path))

        total_emb_size = 1000
        final_module = ImageDetectionModule(
            image_module=self.image_module,
            image_emb_size=total_emb_size)
        self.final_module = final_module
        if torch.cuda.is_available():
            self.image_module.cuda()
            self.final_module.cuda()

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

        # Take the last image and resize them
        images = []
        for aos in agent_observed_state_list:
            img = aos.get_last_image().swapaxes(0, 1).swapaxes(1, 2)
            resized_img = scipy.misc.imresize(img, (224, 224))
            images.append(resized_img.swapaxes(1, 2).swapaxes(0, 1))
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
        self.dense = nn.Linear(image_emb_size, 63 * 2)

    def forward(self, image_batch):
        batch_size = image_batch.size(0)
        image_module_output = self.image_module(image_batch)
        x = self.dense(image_module_output)
        x = F.log_softmax(x.view(batch_size * 63, 2))
        return x.view(batch_size, 63, 2)


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
