import math
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class PixelIdentificationModule(nn.Module):
    def __init__(self, num_channels, num_objects):
        super(PixelIdentificationModule, self).__init__()
        self.num_objects = num_objects
        self.dense_landmark = nn.Linear(num_channels, num_objects)

    def forward(self, image_batch):
        """ Expects image batch of size: batch x num_channel x height x width"""

        batch_size, num_channel, height, width = image_batch.size()
        image_batch = image_batch.transpose(1, 2)
        image_batch = image_batch.transpose(2, 3)
        image_batch = image_batch.contiguous().view(batch_size * height * width, -1)
        x = self.dense_landmark(image_batch)
        log_prob = F.log_softmax(x, dim=1).view(batch_size, height, width, self.num_objects)
        return log_prob

    @staticmethod
    def get_visible_landmark_r_theta(x_pos, z_pos, y_angle, landmark_pos_dict):
        landmark_r_theta_dict = {}
        for landmark, (landmark_x, landmark_z) in landmark_pos_dict.items():
            # get angle between drone's current orientation and landmark
            landmark_angle = 90.0 - np.arctan2(landmark_z - z_pos, landmark_x - x_pos) * 180.0 / math.pi
            angle_diff = landmark_angle - y_angle
            while angle_diff > 180.0:
                angle_diff -= 360.0
            while angle_diff < -180.0:
                angle_diff += 360.0
            # if abs(angle_diff) <= 30.0:
            #     angle_discrete = int((angle_diff + 30.0) / 7.5)
            # else:
            #     angle_discrete = -1

            # get discretized radius
            radius = ((landmark_x - x_pos) ** 2 + (landmark_z - z_pos) ** 2) ** 0.5
            # radius_discrete = int(radius / 5.0)

            landmark_r_theta_dict[landmark] = (radius, angle_diff)
        return landmark_r_theta_dict
