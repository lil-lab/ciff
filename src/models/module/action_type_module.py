import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from utils.cuda import cuda_var


class ActionTypeModule(nn.Module):
    """
    Predict the action type
    """
    def __init__(self, emb_dim=32, hidden_dim=50, instruction_emb=256):
        super(ActionTypeModule, self).__init__()

        self.max_types = 5
        self.gru_hidden_size = hidden_dim
        self.instruction_emb = instruction_emb
        self.embedding = nn.Embedding(4, emb_dim)  # 0, Navigation; 1 for Manipulation, 2 for Halt and 3 for Start
        self.lstm = nn.LSTM(emb_dim, self.gru_hidden_size)
        self.attention = nn.Linear(self.gru_hidden_size, 4)
        self.linear_hidden = nn.Linear(self.instruction_emb, self.gru_hidden_size)
        self.linear_cell = nn.Linear(self.instruction_emb, self.gru_hidden_size)

    def decoding_from_indices_to_indices(self, instruction, text_embedding_model):

        instructions = [instruction]
        instructions_batch = cuda_var(torch.from_numpy(np.array(instructions)).long())
        _, text_emb = text_embedding_model(instructions_batch)
        text_emb = Variable(text_emb.data)

        token_ids = self.greedy_decoding(text_emb)
        token_ids_int = []
        for token_id in token_ids:

            token_id = token_id.data.cpu().numpy()[0]
            assert token_id in [0, 1, 2, 3]
            token_ids_int.append(token_id)

        return token_ids_int

    def get_loss(self, instruction_embedding, token_ids):

        token_ids = token_ids[0: self.max_types]

        encoder_hidden = self.linear_hidden(instruction_embedding).view(1, 1, self.gru_hidden_size)
        encoder_cell = self.linear_cell(instruction_embedding).view(1, 1, self.gru_hidden_size)

        prev_embedding_decoding = cuda_var(torch.from_numpy(np.array([3])).long())

        sum_log_prob = None

        for token_id in token_ids:

            word_embedding = self.embedding(prev_embedding_decoding).unsqueeze(0)
            _, (encoder_hidden, encoder_cell) = self.lstm(word_embedding, (encoder_hidden, encoder_cell))
            logits = encoder_hidden.view(encoder_hidden.size(1), -1)
            log_prob = F.log_softmax(self.attention(logits).view(-1))

            # Pick the argmax decoding
            if sum_log_prob is None:
                sum_log_prob = log_prob[token_id]
            else:
                sum_log_prob = sum_log_prob + log_prob[token_id]

        loss = -sum_log_prob/float(len(token_ids))

        return loss

    def greedy_decoding(self, instruction_embedding):

        encoder_hidden = self.linear_hidden(instruction_embedding).view(1, 1, self.gru_hidden_size)
        encoder_cell = self.linear_cell(instruction_embedding).view(1, 1, self.gru_hidden_size)

        decoding = []
        prev_embedding_decoding = cuda_var(torch.from_numpy(np.array([3])).long())

        for i in range(self.max_types):

            word_embedding = self.embedding(prev_embedding_decoding).unsqueeze(0)

            _, (encoder_hidden, encoder_cell) = self.lstm(word_embedding, (encoder_hidden, encoder_cell))
            logits = encoder_hidden.view(encoder_hidden.size(1), -1)

            prob = F.softmax(self.attention(logits).view(-1))

            # Pick the argmax decoding
            argmax_prob, indices = torch.max(prob, 0)
            prev_embedding_decoding = indices[0]
            indices_int = prev_embedding_decoding.cpu().data.numpy()[0]
            if indices_int == 4: # predicted 3
                break
            decoding.append(indices[0])

        return decoding
