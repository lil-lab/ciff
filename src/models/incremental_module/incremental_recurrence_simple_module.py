import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from utils.cuda import cuda_tensor
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class IncrementalRecurrenceSimpleModule(nn.Module):
    def __init__(self, input_emb_dim, output_emb_dim, num_layers=1):
        super(IncrementalRecurrenceSimpleModule, self).__init__()
        self.input_emb_dim = input_emb_dim
        self.output_emb_dim = output_emb_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_emb_dim, output_emb_dim, num_layers)

    def init_weights(self):
        for name, param in self.lstm.named_parameters():
            if 'bias' in name:
                nn.init.constant(param, 0.0)
            elif 'weight' in name:
                nn.init.xavier_normal(param)

    def forward(self, image_seq_batch, seq_lengths, hidden_vectors):
        """
        @param image_seq_batch: batch of sequence of image embedding
        @param seq_lengths: length of the sequence for each sequence in the batch
        @param hidden_vectors: hidden vectors for each batch """

        b, n, d = image_seq_batch.data.shape
        lengths = [(l, i) for i, l in enumerate(seq_lengths.cpu().numpy())]
        lengths.sort(reverse=True)
        sort_idx = [i for _, i in lengths]
        sort_idx_reverse = [sort_idx.index(i) for i in range(len(sort_idx))]

        image_seq_list = [images for images in image_seq_batch]
        image_seq_batch = torch.cat([image_seq_list[i].view(1, n, d) for i in sort_idx])
        lengths_np = np.array([l for l, _ in lengths])

        batch_size = int(image_seq_batch.data.shape[0])
        if hidden_vectors is None:
            dims = (self.num_layers, batch_size, self.output_emb_dim)
            hidden_vectors = (Variable(cuda_tensor(torch.zeros(*dims)), requires_grad=False),
                              Variable(cuda_tensor(torch.zeros(*dims)), requires_grad=False))

        # swap so batch dimension is second, sequence dimension is first
        image_seq_batch = image_seq_batch.transpose(0, 1)
        packed_input = pack_padded_sequence(image_seq_batch, lengths_np)
        lstm_out_packed, new_hidden_vector = self.lstm(packed_input, hidden_vectors)
        # return average output embedding
        lstm_out, seq_lengths = pad_packed_sequence(lstm_out_packed)
        lstm_out = lstm_out.transpose(0, 1)
        final_vectors = [lstm_out[i][int(seq_len) - 1]
                         for i, seq_len in enumerate(seq_lengths)]
        final_vectors = [final_vectors[i] for i in sort_idx_reverse]
        return torch.cat([vec.view(1, self.output_emb_dim)
                          for vec in final_vectors]), new_hidden_vector

    def forward_old(self, image_seq_batch, seq_lengths):
        b, n, d = image_seq_batch.data.shape
        lengths = [(l, i) for i, l in enumerate(seq_lengths.cpu().numpy())]
        lengths.sort(reverse=True)
        sort_idx = [i for _, i in lengths]
        sort_idx_reverse = [sort_idx.index(i) for i in range(len(sort_idx))]

        image_seq_list = [images for images in image_seq_batch]
        image_seq_batch = torch.cat([image_seq_list[i].view(1, n, d) for i in sort_idx])
        lengths_np = np.array([l for l, _ in lengths])

        batch_size = int(image_seq_batch.data.shape[0])
        dims = (self.num_layers, batch_size, self.output_emb_dim)
        hidden = (Variable(cuda_tensor(torch.zeros(*dims)),
                           requires_grad=False),
                  Variable(cuda_tensor(torch.zeros(*dims)),
                           requires_grad=False))

        # swap so batch dimension is second, sequence dimension is first
        image_seq_batch = image_seq_batch.transpose(0, 1)
        packed_input = pack_padded_sequence(image_seq_batch, lengths_np)
        lstm_out_packed, _ = self.lstm(packed_input, hidden)
        # return average output embedding
        lstm_out, seq_lengths = pad_packed_sequence(lstm_out_packed)
        lstm_out = lstm_out.transpose(0, 1)
        final_vectors = [lstm_out[i][int(seq_len) - 1]
                         for i, seq_len in enumerate(seq_lengths)]
        final_vectors = [final_vectors[i] for i in sort_idx_reverse]
        return torch.cat([vec.view(1, self.output_emb_dim)
                          for vec in final_vectors])
