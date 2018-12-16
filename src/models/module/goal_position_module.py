import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.cuda import cuda_var


class GoalPositionModule(torch.nn.Module):
    def __init__(self, radius_module, angle_module, num_actions):
        super(GoalPositionModule, self).__init__()
        self.radius_module = radius_module
        self.angle_module = angle_module
        self.dense = nn.Linear(64, num_actions)

    def forward(self, agent_positions, goal_positions):
        x = []
        for agent_pos, goal_pos  in zip(agent_positions, goal_positions):
            agent_x, agent_z, pose = agent_pos
            goal_x, goal_z = goal_pos
            radius = ((goal_x - agent_x) ** 2 + (goal_z - agent_z) ** 2) ** 0.5
            radius_embedding = self.radius_module(int_to_cuda_var(int(radius / 5.0)))
            drone_to_goal_angle = 90.0 - float(np.arctan2(goal_z - agent_z, goal_x - agent_x)) * 180.0 / math.pi
            theta = get_angle_diff_discrete(pose, drone_to_goal_angle)
            theta_embedding = self.angle_module(int_to_cuda_var(theta))
            embedding = torch.cat([radius_embedding, theta_embedding], dim=1)
            x.append(embedding)
        embedding_batch = torch.cat(x)
        return F.log_softmax(self.dense(embedding_batch))


def get_angle_diff_discrete(angle_0, angle_1):
    # get angle from 0 to 1, between 0 to 360
    angle_diff = angle_1 - angle_0
    while angle_diff > 360.0:
        angle_diff -= 360.0
    while angle_diff < 0.0:
        angle_diff += 360.0
    return int(angle_diff / 7.5)


def int_to_cuda_var(int_val):
    return cuda_var(torch.from_numpy(np.array([int_val])))
