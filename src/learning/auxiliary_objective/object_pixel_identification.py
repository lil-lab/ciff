import torch
import numpy as np
import matplotlib.pyplot as plt

from utils.camera_mapping import get_image_coordinates
from utils.nav_drone_landmarks import LANDMARK_NAMES


class ObjectPixelIdentification:

    def __init__(self, model, num_objects, camera_angle, image_height, image_width, object_height):
        self.num_objects = num_objects
        self.model = model
        self.camera_angle = camera_angle
        self.image_height = image_height
        self.image_width = image_width
        self.object_height = object_height
        self.global_id = 0

    def save_image(self, image, goal, landmark_name, r, theta):
        print("Saving image")
        self.global_id += 1
        image_flipped = image.swapaxes(0, 1).swapaxes(1, 2)
        kernel = np.zeros((128, 128))
        height = int(128/8)
        width = int(128/8)
        row, col = goal
        for i in range(- height // 2, height // 2 + 1):
            for j in range(-width //2, width // 2 + 1):
                kernel[row * height + i][col * width + j] = 1.0

        plt.imshow(image_flipped)
        plt.imshow(kernel, cmap='jet', alpha=0.5)
        plt.savefig("./landmarks/image_" + str(self.global_id) + "_" + str(landmark_name)
                    + "_" + str(r) + "_" + str(theta) + ".png")
        plt.clf()

    def calc_loss(self, batch_replay_items):

        batch_image_feature = []
        agent_observed_state_list = []
        for i, replay_item in enumerate(batch_replay_items):
            agent_observed_state_list.append(replay_item.get_agent_observed_state())
            batch_image_feature.append(replay_item.get_volatile_features()["image_emb"])

        batch_image_feature = torch.cat(batch_image_feature)
        object_pixel_log_prob, visible_objects = self.model.get_pixel_level_object_prob(
            agent_observed_state_list, batch_image_feature)
        # object_pixel_log_prob shape is batch x height x width x landmarks

        num_states = len(batch_image_feature)
        num_visible = 0.0
        loss = 0

        for i in range(0, num_states):
            visible_objects_example = visible_objects[i]
            for landmark, landmark_name in enumerate(LANDMARK_NAMES):
                # See if the landmark is present and visible in the agent's field of view
                if landmark_name in visible_objects_example:
                    r, theta = visible_objects_example[landmark_name]

                    # Convert to row and col
                    row, col = get_image_coordinates(r, theta, self.object_height,
                                                     self.camera_angle, self.image_height, self.image_width)

                    if row is not None and col is not None:
                        loss = loss - object_pixel_log_prob[i, row, col, landmark]
                        # self.save_image(batch_replay_items[i].get_agent_observed_state().get_last_image(), goal=(row, col),
                        #                 landmark_name=landmark_name, r=r, theta=theta)
                        num_visible += 1.0
                else:
                    pass

        if num_visible > 0:
            loss = loss / float(num_visible)
            return loss
        else:
            return None
