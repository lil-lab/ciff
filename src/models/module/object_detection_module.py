import math
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class ObjectDetectionModule(nn.Module):
    def __init__(self, image_module, image_emb_size, num_objects):
        super(ObjectDetectionModule, self).__init__()
        self.image_module = image_module
        self.num_objects = num_objects
        self.dense_landmark = nn.Linear(image_emb_size, num_objects * 2)
        self.dense_distance = nn.Linear(image_emb_size, num_objects * 15)
        self.dense_theta = nn.Linear(image_emb_size, num_objects * 48)

    def forward(self, image_batch):

        batch_size = image_batch.size(0)
        # image_module_output = self.image_module(image_batch)
        image_module_output = image_batch
        image_module_output = image_module_output[:, -1, :]  # Take the last image in resnet

        x_landmark = self.dense_landmark(image_module_output)
        x_distance = self.dense_distance(image_module_output)
        x_theta = self.dense_theta(image_module_output)

        x_landmark = F.log_softmax(x_landmark.view(batch_size * self.num_objects, 2))
        x_distance = F.log_softmax(x_distance.view(batch_size * self.num_objects, 15))
        x_theta = F.log_softmax(x_theta.view(batch_size * self.num_objects, 48))
        return x_landmark.view(batch_size, self.num_objects, 2), \
               x_distance.view(batch_size, self.num_objects,15), \
               x_theta.view(batch_size, self.num_objects, 48)

    @staticmethod
    def get_visible_landmark_r_theta(x_pos, z_pos, y_angle, landmark_pos_dict):
        landmark_r_theta_dict = {}
        for landmark, (landmark_x, landmark_z) in landmark_pos_dict.iteritems():
            # get angle between drone's current orientation and landmark
            landmark_angle = 90.0 - np.arctan2(landmark_z - z_pos, landmark_x - x_pos) * 180.0 / math.pi
            angle_diff = landmark_angle - y_angle
            while angle_diff > 180.0:
                angle_diff -= 360.0
            while angle_diff < -180.0:
                angle_diff += 360.0
            if abs(angle_diff) <= 30.0:
                angle_discrete = int((angle_diff + 30.0) / 7.5)
            else:
                angle_discrete = -1

            # get discretized radius
            radius = ((landmark_x - x_pos) ** 2 + (landmark_z - z_pos) ** 2) ** 0.5
            radius_discrete = int(radius / 5.0)

            landmark_r_theta_dict[landmark] = (radius_discrete, angle_discrete)
        return landmark_r_theta_dict
