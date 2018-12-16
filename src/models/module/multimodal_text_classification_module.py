import torch
import torch.nn as nn
import torch.nn.functional as F
import utils.nav_drone_symbolic_instructions as symbolic_instructions
from utils.cuda import cuda_tensor
from torch.autograd import Variable


class MultimodalTextClassificationModule(nn.Module):
    """
    pytorch module for final part of model
    combines embeddings of image, text, and previous action
    """
    def __init__(self, text_module, image_module, total_emb_size):
        super(MultimodalTextClassificationModule, self).__init__()
        self.text_module = text_module
        self.image_module = image_module
        hidden_dim = 250
        self.dense = nn.Linear(300, 300)
        self.dense_1 = nn.Linear(total_emb_size, hidden_dim)
        num_buckets = symbolic_instructions.NO_BUCKETS
        self.dense_landmark = nn.Linear(hidden_dim, 67)
        self.dense_theta_1 = nn.Linear(hidden_dim, num_buckets)
        self.dense_theta_2 = nn.Linear(hidden_dim, num_buckets)
        self.dense_r = nn.Linear(hidden_dim, 15)
        self.instance_norm1 = torch.nn.InstanceNorm1d(250, affine=True)
        self.instance_norm2 = torch.nn.InstanceNorm1d(300, affine=True)

    def forward(self, instructions):
        # Assume there is only 1 instruction
        text_emb = self.text_module(instructions)
        x = F.relu(self.dense_1(text_emb))
        return F.log_softmax(self.dense_landmark(x)), \
               F.log_softmax(self.dense_theta_1(x)), \
               F.log_softmax(self.dense_theta_2(x)), \
               F.log_softmax(self.dense_r(x))

    def forward_1(self, instructions, prev_instructions, next_instructions):
        # Assume there is only 1 instruction
        text_emb = self.text_module(instructions)
        if prev_instructions[0][0] is None:
            prev_text_emb = Variable(cuda_tensor(torch.zeros(text_emb.size())), requires_grad=False)
        else:
            prev_text_emb = self.text_module(prev_instructions)

        if next_instructions[0][0] is None:
            next_text_emb = Variable(cuda_tensor(torch.zeros(text_emb.size())), requires_grad=False)
        else:
            next_text_emb = self.text_module(next_instructions)
        x = torch.cat([prev_text_emb, text_emb, next_text_emb], dim=1)
        x = F.relu(self.dense_1(x))
        return F.log_softmax(self.dense_landmark(x)), \
               F.log_softmax(self.dense_theta_1(x)), \
               F.log_softmax(self.dense_theta_2(x)), \
               F.log_softmax(self.dense_r(x))

    def forward_old(self, instructions, images):
        text_emb = self.text_module(instructions)
        image_emb = self.image_module(images)
        batch_size = image_emb.size()[0]
        image_emb = image_emb.view(batch_size, -1)
        image_emb = image_emb * 0
        x = torch.cat([image_emb, text_emb], dim=1)
        x = F.relu(self.dense_1(x))
        return F.log_softmax(self.dense_landmark(x)), \
               F.log_softmax(self.dense_theta_1(x)), \
               F.log_softmax(self.dense_theta_2(x)), \
               F.log_softmax(self.dense_r(x))
