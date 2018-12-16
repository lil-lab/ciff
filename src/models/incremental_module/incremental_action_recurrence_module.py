import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from utils.cuda import cuda_tensor
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class IncrementalActionRecurrenceModule(nn.Module):
    def __init__(self, input_emb_dim, output_emb_dim, num_layers=1):
        super(IncrementalActionRecurrenceModule, self).__init__()
        self.input_emb_dim = input_emb_dim
        self.output_emb_dim = output_emb_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_emb_dim, output_emb_dim, num_layers)

    def forward(self, actions_batch, hidden_vectors):
        """
        @param image_seq_batch: batch of sequence of image embedding
        @param seq_lengths: length of the sequence for each sequence in the batch
        @param hidden_vectors: hidden vectors for each batch """

        b, d = actions_batch.data.shape
        actions_batch = actions_batch.view(b, 1, d)
        batch_size = int(actions_batch.data.shape[0])
        if hidden_vectors is None:
            dims = (self.num_layers, batch_size, self.output_emb_dim)
            hidden_vectors = (Variable(cuda_tensor(torch.zeros(*dims)), requires_grad=False),
                              Variable(cuda_tensor(torch.zeros(*dims)), requires_grad=False))

        # swap so batch dimension is second, sequence dimension is first
        actions_batch = actions_batch.view(1, b, d)
        lstm_out, new_hidden_vector = self.lstm(actions_batch, hidden_vectors)
        # return output embeddings
        return lstm_out.view(batch_size, -1), new_hidden_vector

