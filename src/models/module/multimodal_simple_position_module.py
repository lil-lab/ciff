import torch
import torch.nn as nn
import torch.nn.functional as F

from agents.agent_with_read import ReadPointerAgent


class MultimodalSimplePositionModule(nn.Module):
    """
    pytorch module for final part of model
    combines embeddings of image and text
    """
    def __init__(self, image_module, text_module, total_emb_size, num_grid_x, num_grid_y, num_grid_pose):
        super(MultimodalSimplePositionModule, self).__init__()
        self.image_module = image_module
        self.text_module = text_module
        self.dense1 = nn.Linear(total_emb_size, 120)
        self.dense_x = nn.Linear(120, num_grid_x)
        self.dense_y = nn.Linear(120, num_grid_y)
        self.dense_pose = nn.Linear(120, num_grid_pose)

    def forward(self, image, instructions):
        image_emb = self.image_module(image)
        # text_emb = self.text_module(instructions)
        x = image_emb  # torch.cat([image_emb, text_emb], dim=1)
        x = F.log_softmax(x)
        return x
        # x = F.relu(self.dense1(x))
        # return F.log_softmax(self.dense_x(x)), F.log_softmax(self.dense_y(x)), F.log_softmax(self.dense_pose(x))
