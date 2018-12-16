import math
import numpy as np

from utils.nav_drone_landmarks import get_all_landmark_names
from dataset_agreement_nav_drone.nav_drone_datapoint import NavDroneDataPoint

LANDMARK_NAMES = get_all_landmark_names()
NUM_LANDMARKS = len(LANDMARK_NAMES)
NO_BUCKETS = 12
BUCKET_WIDTH = 360.0/float(NO_BUCKETS)



def get_nav_drone_symbolic_instruction_segment(data_point):
    assert isinstance(data_point, NavDroneDataPoint)
    num_seg = len(data_point.get_destination_list())
    if num_seg > 1:
        raise ValueError("data point is for a paragraph, not a segment")
    return get_nav_drone_symbolic_instruction(data_point)


def get_nav_drone_symbolic_instruction(data_point, seg_index=-1):
    assert isinstance(data_point, NavDroneDataPoint)
    drone_x, drone_z, drone_angle = data_point.get_start_pos_list()[seg_index]
    goal_x, goal_z = data_point.get_destination_list()[seg_index]
    landmark_pos_dict = data_point.get_landmark_pos_dict()

    # find closest landmark to end of segment
    landmark, dist = get_closest_landmark(landmark_pos_dict, goal_x, goal_z)
    landmark_x, landmark_z = landmark_pos_dict[landmark]
    drone_to_landmark_angle = get_absolute_angle(drone_x, drone_z,
                                                 landmark_x, landmark_z)
    landmark_to_goal_angle = get_absolute_angle(landmark_x, landmark_z,
                                                goal_x, goal_z)

    # get symbolic info
    landmark_i = LANDMARK_NAMES.index(landmark)
    theta_1 = get_angle_diff_discrete(drone_angle,
                                      drone_to_landmark_angle)
    theta_2 = get_angle_diff_discrete(drone_to_landmark_angle,
                                      landmark_to_goal_angle)
    r = int(dist / 5.0)
    return landmark_i, theta_1, theta_2, r


def get_closest_landmark(landmark_pos_dict, end_x, end_z):
    min_dist = float("Inf")
    closest_landmark = None
    for landmark, (landmark_x, landmark_z) in landmark_pos_dict.iteritems():
        dist = ((landmark_x - end_x) ** 2 + (landmark_z - end_z) ** 2) ** 0.5
        if dist < min_dist:
            min_dist = dist
            closest_landmark = landmark
    assert closest_landmark is not None
    return closest_landmark, min_dist


def get_angle_diff_discrete(angle_0, angle_1):
    # get angle from 0 to 1, between 0 to 360
    angle_diff = angle_1 - angle_0
    while angle_diff > 360.0:
        angle_diff -= 360.0
    while angle_diff < 0.0:
        angle_diff += 360.0
    return int(angle_diff / BUCKET_WIDTH)


def get_absolute_angle(x_0, z_0, x_1, z_1):
    return 90.0 - float(np.arctan2(z_1 - z_0, x_1 - x_0)) * 180.0 / math.pi
