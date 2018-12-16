import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from utils.cuda import cuda_tensor, cuda_var
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class SymbolicInstructionLandmarkonlyModule(nn.Module):
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
    def __init__(self, landmark_embedding):
        super(SymbolicInstructionLandmarkonlyModule, self).__init__()
        self.landmark_embedding = landmark_embedding

    def forward(self, symbolic_instructions_batch):
        embedding_list = []
        for landmark_i, theta_1, theta_2, r in symbolic_instructions_batch:
            embedding = self.landmark_embedding(int_to_cuda_var(landmark_i))
            embedding_list.append(embedding.view(1, -1))
        return torch.cat(embedding_list)


def int_to_cuda_var(int_val):
    return cuda_var(torch.from_numpy(np.array([int_val])))
