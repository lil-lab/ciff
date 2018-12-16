import math
import numpy as np


def current_pos_from_metadata(metadata):
    return metadata["x_pos"], metadata["z_pos"]


def current_pose_from_metadata(metadata):
    return metadata["y_angle"]


def goal_pos_from_datapoint(datapoint, seg_index=-1):
    x, z = datapoint.get_destination_list()[seg_index]
    return x, z


def start_pos_from_datapoint(datapoint, seg_index=0):
    x, z, _ = datapoint.get_start_pos_list()[seg_index]
    return x, z


def start_pose_from_datapoint(datapoint, seg_index=0):
    _, _, pose = datapoint.get_start_pos_list()[seg_index]
    return pose


def get_turn_angle(start_pos, start_pose, target_pos):
    x1, z1 = start_pos
    x2, z2 = target_pos
    target_angle = 90.0 - np.arctan2(z2 - z1, x2 - x1) * 180.0 / math.pi
    turn_angle = target_angle - start_pose
    while turn_angle > 180.0:
        turn_angle -= 360.0
    while turn_angle < -180.0:
        turn_angle += 360.0
    return turn_angle


def get_turn_angle_from_metadata_datapoint(metadata, datapoint, seg_index=-1):
    start_pos = current_pos_from_metadata(metadata)
    start_pose = current_pose_from_metadata(metadata)
    target_pos = goal_pos_from_datapoint(datapoint, seg_index=seg_index)
    return get_turn_angle(start_pos, start_pose, target_pos)


def get_distance(start_pos, target_pos):
    start_pos = np.array(start_pos)
    target_pos = np.array(target_pos)
    return ((target_pos - start_pos) ** 2).sum() ** 0.5


def get_distance_from_metadata_datapoint(metadata, datapoint, seg_index=-1):
    start_pos = current_pos_from_metadata(metadata)
    target_pos = goal_pos_from_datapoint(datapoint, seg_index=seg_index)
    return get_distance(start_pos, target_pos)
