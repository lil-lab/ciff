import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from utils.cuda import cuda_tensor, cuda_var
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class SymbolicInstructionModuleProbabilities(nn.Module):
    """
    pytorch module for text part of model
    assumes input is two parts:
    (1) text_tokens is pytorch variable consisting of instructions in
        batch (dimensionality is BatchSize x SequenceLength
      - each element of sequence is integer giving word ID
      - SequenceLength is based on longest element of sequence
      - each non-longest sequence should be padded by 0's at the end,
        to make the tensor rectangular
      - sequences in batch should be sorted in descending-length order
    (2) text_length is pytorch tensor giving actual lengths of each sequence
        in batch (dimensionality is BatchSize)
    """
    def __init__(self, radius_embedding, theta_embedding, landmark_embedding):
        super(SymbolicInstructionModuleProbabilities, self).__init__()
        self.radius_embedding = radius_embedding
        self.theta_embedding = theta_embedding
        self.landmark_embedding = landmark_embedding

    def forward(self, symbolic_instructions_batch, probabilities_landmark, probabilities_theta):

        expected_landmark_embedding = torch.matmul(probabilities_landmark,
                                                   self.landmark_embedding.landmark_embedding.weight)

        b, l, t = probabilities_theta.size()
        probabilities_theta = probabilities_theta.transpose(1, 2)  # batch x theta x landmark
        probabilities_landmark = probabilities_landmark.view(b, l, 1)
        probabilities_marginalized_theta = torch.matmul(probabilities_theta, probabilities_landmark)
        probabilities_marginalized_theta = probabilities_marginalized_theta.view(b, t)

        expected_theta_embedding = torch.matmul(probabilities_marginalized_theta,
                                                self.theta_embedding.theta_embedding.weight)

        return torch.cat([expected_landmark_embedding, expected_theta_embedding], dim=1)


def int_to_cuda_var(int_val):
    return cuda_var(torch.from_numpy(np.array([int_val])))
