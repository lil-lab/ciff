import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from utils.cuda import cuda_var


class SymbolicImageModule(torch.nn.Module):

    def __init__(self, landmark_names, radius_module, angle_module,
                 landmark_module):
        super(SymbolicImageModule, self).__init__()
        self.landmark_names = landmark_names

        self.landmark_embedding = landmark_module
        self.r_embedding = radius_module
        self.theta_embedding = angle_module

    def forward(self, landmark_r_theta_dict_list):
        x_list = []

        for landmark_r_theta_dict in landmark_r_theta_dict_list:
            x = []
            for landmark in self.landmark_names:

                if landmark in landmark_r_theta_dict:
                    (r, theta) = landmark_r_theta_dict[landmark]
                    if theta == -1:
                        # not visible
                        x.append(cuda_var(torch.zeros(1, 96)))
                        continue

                    # get landmark embedding
                    landmark_id = self.landmark_names.index(landmark)
                    landmark_var = cuda_var(torch.from_numpy(np.array([landmark_id])))
                    landmark_embedding = self.landmark_embedding(landmark_var)

                    # get r embedding
                    r_var = cuda_var(torch.from_numpy(np.array([r])))
                    r_embedding = self.r_embedding(r_var)

                    # get theta embedding
                    theta_var = cuda_var(torch.from_numpy(np.array([theta])))
                    theta_embedding = self.theta_embedding(theta_var)

                    embedding = torch.cat([landmark_embedding, r_embedding,
                                           theta_embedding], dim=1)
                    x.append(embedding)
                else:
                    x.append(cuda_var(torch.zeros(1, 96)))

            x = torch.cat(x).view(-1)
            x_list.append(x)

        embedding = torch.cat(x_list).view(len(x_list), -1)

        return embedding

    def forward_sum_symbolic(self, landmark_r_theta_dict_list):
        x_list = []
        for landmark_r_theta_dict in landmark_r_theta_dict_list:
            x = cuda_var(torch.zeros(1, self.image_emb_size))
            for landmark, (r, theta) in landmark_r_theta_dict.iteritems():
                if theta == -1:
                    # not visible
                    continue
                # get landmark embedding
                landmark_id = self.landmark_names.index(landmark)
                landmark_var = cuda_var(torch.from_numpy(np.array([landmark_id])))
                landmark_embedding = self.landmark_embedding(landmark_var)

                # get r embedding
                r_var = cuda_var(torch.from_numpy(np.array([r])))
                r_embedding = self.r_embedding(r_var)

                # get theta embedding
                theta_var = cuda_var(torch.from_numpy(np.array([theta])))
                theta_embedding = self.theta_embedding(theta_var)

                embedding = torch.cat([landmark_embedding, r_embedding,
                                       theta_embedding], dim=1)
                #embedding = F.relu(self.dense(embedding))
                x = x + embedding
            x_list.append(x)

        return torch.cat(x_list)
