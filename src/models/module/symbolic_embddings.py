import torch.nn as nn


class AngleModule(nn.Module):
    def __init__(self, num_angles):
        super(AngleModule, self).__init__()
        self.theta_embedding = nn.Embedding(num_angles, 32)

    def forward(self, theta):
        return self.theta_embedding(theta)


class RadiusModule(nn.Module):
    def __init__(self, num_radius_vals):
        super(RadiusModule, self).__init__()
        self.radius_embedding = nn.Embedding(num_radius_vals, 32)

    def forward(self, radius):
        return self.radius_embedding(radius)


class LandmarkModule(nn.Module):
    def __init__(self, num_landmarks):
        super(LandmarkModule, self).__init__()
        self.landmark_embedding = nn.Embedding(num_landmarks, 32)

    def forward(self, landmark_i):
        return self.landmark_embedding(landmark_i)