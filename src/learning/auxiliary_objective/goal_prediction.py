import math
import torch
import logging
import numpy as np
import scipy.misc
import matplotlib.pyplot as plt
import torch.nn.functional as F

from utils.cuda import cuda_var
from utils.geometry import get_turn_angle_from_metadata_datapoint, get_distance_from_metadata_datapoint


class GoalPrediction:

    LOGLOSS, LOGLOSS_DIST, CROSS_ENTROPY, DIST_LOSS, CROSS_ENTROPY_AND_DIST_LOSS = range(5)
    loss_type = LOGLOSS

    def __init__(self, model, final_height, final_width):
        self.model = model
        self.global_id = 1
        self.final_height, self.final_width = final_height, final_width

    @staticmethod
    def get_goal_location(metadata, datapoint, height, width, max_angle=30):

        angle_phi = get_turn_angle_from_metadata_datapoint(metadata, datapoint)  # angle is in -180 to 180 degree
        if angle_phi < -max_angle or angle_phi > max_angle:
            return None, None, None, None
        distance = get_distance_from_metadata_datapoint(metadata, datapoint)  # return distance
        tan_phi = math.tan(math.radians(angle_phi))
        cos_phi = math.cos(math.radians(angle_phi))
        tan_theta = math.tan(math.radians(30.0))  # camera width is 30.0
        height_drone = 2.5

        height_half = int(height / 2)
        width_half = int(width / 2)
        try:
            row_real = height_half + (height_half * height_drone) / (distance * cos_phi * tan_theta)
            col_real = width_half + (width_half * tan_phi) / tan_theta
            row = int(round(row_real))  # int(row_real)
            col = int(round(col_real))  # int(col_real)
        except:
            print("Found exception. Cosphi is %r, tan theta is %r, and distance is %r" % (cos_phi, tan_theta, distance))
            return None, None, None, None

        if row < 0:
            row = 0
        elif row >= height:
            row = height - 1

        if col < 0:
            col = 0
        elif col >= width:
            col = width - 1

        return row, col, row_real, col_real

    @staticmethod
    def generate_gold_prob(goal, final_height, final_width, sigma2=0.5):
        row, col, row_real, col_real = goal
        gold_prob = cuda_var(torch.zeros(final_height * final_width + 1)).float()

        if row is None or col is None:
            gold_prob[final_height * final_width] = 1.0  # last value indicates not present
            return gold_prob

        row_ = float(round(row_real)) + 0.5
        col_ = float(round(col_real)) + 0.5

        for i in range(0, final_height):
            for j in range(0, final_width):
                ix = i * final_width + j
                center = (i + 0.5, j + 0.5)
                # dist2 = (center[0] - row_real) * (center[0] - row_real) + \
                #         (center[1] - col_real) * (center[1] - col_real)
                dist2 = (center[0] - row_) * (center[0] - row_) + \
                        (center[1] - col_) * (center[1] - col_)
                gold_prob[ix] = -dist2/(2.0 * sigma2)

        gold_prob = torch.exp(gold_prob).float()
        gold_prob[final_height * final_width] = 0.0
        gold_prob = gold_prob/(gold_prob.sum() + 0.00001)

        return gold_prob

    @staticmethod
    def get_loss_and_prob(volatile_features, goal, final_height, final_width):
        attention_probs = volatile_features["attention_probs"]
        attention_logits = volatile_features["attention_logits"]
        attention_log_prob = F.log_softmax(attention_logits, dim=0)
        row, col, row_real, col_real = goal
        gold_prob = GoalPrediction.generate_gold_prob(goal, final_height, final_width)

        if row is None:
            cross_entropy_loss = -torch.sum(gold_prob * attention_log_prob)  # cross entropy loss
            meta = {"cross_entropy": cross_entropy_loss, "dist_loss": None}
            return cross_entropy_loss, attention_log_prob[final_height * final_width], meta

        row_, col_ = row + 0.5, col + 0.5

        position_height = cuda_var(torch.from_numpy(np.array(list(range(0, final_height))))).float().view(-1, 1) + 0.5
        position_width = cuda_var(torch.from_numpy(np.array(list(range(0, final_width))))).float().view(-1, 1) + 0.5
        attention_prob = attention_probs[:-1].view(final_height, final_width)

        expected_row = torch.sum(position_height * attention_prob)
        expected_col = torch.sum(position_width.view(1, -1) * attention_prob)

        dist_loss = torch.sqrt((expected_row - row_) * (expected_row - row_)
                               + (expected_col - col_) * (expected_col - col_))
        cross_entropy_loss = -torch.sum(gold_prob * attention_log_prob)  # cross entropy loss

        if row is None or col is None:
            ix = final_height * final_width
        else:
            ix = row * final_width + col

        if GoalPrediction.loss_type == GoalPrediction.LOGLOSS:
            loss = - attention_log_prob[ix]
        elif GoalPrediction.loss_type == GoalPrediction.LOGLOSS_DIST:
            loss = - attention_log_prob[ix] + dist_loss
        elif GoalPrediction.loss_type == GoalPrediction.CROSS_ENTROPY:
            loss = cross_entropy_loss
        elif GoalPrediction.loss_type == GoalPrediction.DIST_LOSS:
            loss = dist_loss
        elif GoalPrediction.loss_type == GoalPrediction.CROSS_ENTROPY_AND_DIST_LOSS:
            loss = cross_entropy_loss + dist_loss
        else:
            raise AssertionError("Unhandled loss type ", GoalPrediction.loss_type)

        prob = attention_log_prob[ix]

        meta = {"cross_entropy": cross_entropy_loss, "dist_loss": dist_loss}
        return loss, prob, meta

    def calc_loss(self, batch_replay_items):

        loss = None
        cross_entropy_loss = None
        dist_loss = None
        num_items = 0
        prob, goal_items = None, 0

        for replay_item in batch_replay_items:
            goal = replay_item.get_goal()

            num_items += 1
            volatile_features = replay_item.get_volatile_features()

            # loss_ = -torch.log(attention_prob[row, col])
            # loss_ = F.binary_cross_entropy_with_logits(input=attention_logits.view(-1), target=gold_prob.view(-1))
            # loss_ = -torch.log(prob)
            loss_, prob_, meta = self.get_loss_and_prob(volatile_features, goal, self.final_height, self.final_width)

            if loss is None:
                loss = loss_
                cross_entropy_loss = meta["cross_entropy"]

            else:
                loss = loss + loss_
                cross_entropy_loss = cross_entropy_loss + meta["cross_entropy"]

            if meta["dist_loss"] is not None:
                if dist_loss is None:
                    dist_loss = meta["dist_loss"]
                else:
                    dist_loss = dist_loss + meta["dist_loss"]
                goal_items += 1

            if prob is None:
                prob = prob_
            else:
                prob = prob + prob_

        if loss is not None:
            loss = loss / float(num_items)
            cross_entropy_loss = cross_entropy_loss / float(num_items)
            if dist_loss is not None:
                dist_loss = dist_loss / float(max(1, goal_items))
        if prob is not None:
            prob = prob / float(max(1, goal_items))

        meta = {"cross_entropy": cross_entropy_loss, "dist_loss": dist_loss}
        return loss, prob, meta

    def save_attention_prob(self, image, volatile, goal_prob=None):
        self.global_id += 1
        attention_prob = volatile["attention_probs"].cpu().data.numpy()
        image_flipped = image.swapaxes(0, 1).swapaxes(1, 2)
        fmap = attention_prob
        fmap = (fmap - np.mean(fmap)) / np.std(fmap)
        resized_kernel = scipy.misc.imresize(fmap, (128, 128))

        plt.imshow(image_flipped)
        plt.imshow(resized_kernel, cmap='jet', alpha=0.5)
        if goal_prob is not None:
            goal_location = goal_prob.cpu().data.numpy()
            plt.imshow(goal_location, cmap='Oranges', alpha=0.5)
        plt.savefig("./attention_prob/image_" + str(self.global_id) + ".png")
        plt.clf()

    def calc_loss_old(self, batch_replay_items):

        angle_batch = []
        distance_batch = []
        batch_next_state_feature = []

        for replay_item in batch_replay_items:
            angle, distance = replay_item.get_goal()
            angle_batch.append(angle)
            distance_batch.append(distance)
            batch_next_state_feature.append(replay_item.get_state_feature())

        angle_batch = cuda_var(torch.from_numpy(np.array(angle_batch)))
        distance_batch = cuda_var(torch.from_numpy(np.array(distance_batch)))
        batch_next_state_feature = torch.cat(batch_next_state_feature)

        # Compute the negative log probability loss
        goal_angle_log_probability, goal_distance_log_probability = self.model.predict_goal_result(
            batch_next_state_feature)

        chosen_angle_log_probs = goal_angle_log_probability.gather(1, angle_batch.view(-1, 1))
        chosen_distance_log_probs = goal_distance_log_probability.gather(1, distance_batch.view(-1, 1))

        goal_probability_loss = - torch.sum(chosen_angle_log_probs) - torch.sum(chosen_distance_log_probs)
        num_states = float(len(batch_replay_items))
        goal_probability_loss = goal_probability_loss / num_states

        return goal_probability_loss
