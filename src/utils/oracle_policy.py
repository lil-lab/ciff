import math
import heapq
import numpy as np
from utils.nav_drone_landmarks import get_landmark_radius

MAX_SEARCH_NUM = 1500  # 100000
CORNER_LANDMARKS = {"NECorner", "SECorner", "NWCorner", "SWCorner"}


def oracle_policy(metadata, goal_x, goal_z, data_point):
    trajectory = get_oracle_trajectory(metadata, goal_x, goal_z, data_point)
    if len(trajectory) > 0:
        return trajectory[0]
    else:
        return "Stop"


def get_oracle_trajectory(metadata, goal_x, goal_z, data_point):
    drone_x, drone_z = metadata["x_pos"], metadata["z_pos"]
    drone_pose = metadata["y_angle"]
    landmark_pos_dict = data_point.get_landmark_pos_dict()
    start_state = (drone_x, drone_z, drone_pose)
    frontier = []
    heapq.heappush(frontier, (0, start_state))
    came_from = {start_state: None}
    cost_so_far = {start_state: 0}

    start_pos = np.array([drone_x, drone_z])
    goal_pos = np.array([goal_x, goal_z])
    closest_state = None
    closest_dist = ((goal_pos - start_pos) ** 2).sum() ** 0.5

    goal_state = None
    num_expanded = 0
    while len(frontier) > 0:
        current_val, current = heapq.heappop(frontier)
        num_expanded += 1
        if is_terminal(current, goal_x, goal_z):
            goal_state = current
            break

        if num_expanded > MAX_SEARCH_NUM:
            break

        for action in ("Forward", "TurnRight", "TurnLeft"):
            next_state = successor(current, action, landmark_pos_dict)
            if next_state is None:
                continue
            next_x, next_z, _ = next_state
            next_pos = np.array([next_x, next_z])
            next_dist = ((goal_pos - next_pos) ** 2).sum() ** 0.5
            if next_dist < closest_dist:
                closest_state = next_state

            new_cost = cost_so_far[current] + 1
            if next_state not in cost_so_far or new_cost < cost_so_far[next_state]:
                cost_so_far[next_state] = new_cost
                priority = new_cost + heuristic(next_state, goal_x, goal_z,
                                                landmark_pos_dict)
                heapq.heappush(frontier, (priority, next_state))
                came_from[next_state] = current, action

    # print num_expanded, "searched"
    actions_reverse = []
    s = goal_state if (goal_state is not None) else closest_state
    if s is not None:
        while came_from[s] is not None:
            s, action = came_from[s]
            actions_reverse.append(action)

    return actions_reverse[::-1]


def heuristic(state, goal_x, goal_z, landmark_pos_dict):
    x, z, pose = state
    # return heuristic_simple(x, z, pose, goal_x, goal_z, 1.5)
    start_pos = np.array([x, z])
    goal_pos = np.array([goal_x, goal_z])
    landmark, gamma = get_closest_obstacle(start_pos, goal_pos,
                                           landmark_pos_dict)
    if landmark is None:
        # calculate simple heuristic since there are no obstacles in the way
        return heuristic_simple(x, z, pose, goal_x, goal_z, 1.5)
    else:
        # calculate 2-part heuristic, passing obstacle on left vs right
        landmark_pos = np.array(landmark_pos_dict[landmark])
        landmark_angle = calc_target_angle(start_pos, landmark_pos)
        landmark_dist = ((landmark_pos - start_pos) ** 2).sum() ** 0.5
        landmark_radius = get_landmark_radius(landmark)
        angular_width = np.arcsin(landmark_radius / landmark_dist) * 180.0 / math.pi
        side_distance = (landmark_dist ** 2 - landmark_radius ** 2) ** 0.5
        landmark_goal_dist = ((goal_pos - landmark_pos) ** 2).sum() ** 0.5

        # calculate left heuristic
        left_angle = landmark_angle - angular_width
        l_vector = get_angle_unit_vector(left_angle)
        landmark_left = start_pos + l_vector * side_distance
        h1_left = heuristic_simple(
            x, z, pose,
            landmark_left[0], landmark_left[1], 0.0)
        left_pose = calc_target_angle(start_pos, landmark_left)
        h2_left = heuristic_simple(
            landmark_left[0], landmark_left[1], left_pose,
            goal_x, goal_z, 1.5)
        left_goal_dist = ((goal_pos - landmark_left) ** 2).sum() ** 0.5
        left_arc_cost = calc_extra_arc_distance(
            landmark_radius, landmark_goal_dist, left_goal_dist)
        h_left = h1_left + h2_left + left_arc_cost

        # calculate right heuristic
        right_angle = landmark_angle + angular_width
        r_vector = get_angle_unit_vector(right_angle)
        landmark_right = start_pos + r_vector * side_distance
        h1_right = heuristic_simple(
            x, z, pose,
            landmark_right[0], landmark_right[1], 0.0)
        right_pose = calc_target_angle(start_pos, landmark_right)
        h2_right = heuristic_simple(
            landmark_right[0], landmark_right[1], right_pose,
            goal_x, goal_z, 1.5)
        right_goal_dist = ((goal_pos - landmark_right) ** 2).sum() ** 0.5
        right_arc_cost = calc_extra_arc_distance(
            landmark_radius, landmark_goal_dist, right_goal_dist)
        h_right = h1_right + h2_right + right_arc_cost

        return min(h_left, h_right)


def heuristic_simple(start_x, start_z, start_pose, goal_x, goal_z, goal_radius):
    # get forward distance
    goal_dist = ((goal_x - start_x) ** 2 + (goal_z - start_z) ** 2) ** 0.5
    forward_dist = goal_dist / 1.5
    if forward_dist <= 1.5:
        return 0
    r_dist = forward_dist - 1.5

    # get turn distance
    start_pos = np.array([start_x, start_z])
    goal_pos = np.array([goal_x, goal_z])
    # simple case; no obstacle
    goal_angle = calc_target_angle(start_pos, goal_pos)
    turn_angle = abs_angle_diff(start_pose, goal_angle)
    turn_epsilon = np.arcsin(goal_radius / goal_dist) * 180.0 / math.pi
    theta_dist = max(0.0, turn_angle - turn_epsilon)

    return r_dist + theta_dist


def get_closest_obstacle(start_pos, goal_pos, landmark_pos_dict):
    sg_vec = goal_pos - start_pos
    sg_sq_norm = np.dot(sg_vec, sg_vec)
    min_gamma = 1.0
    best_landmark = None
    for landmark, (landmark_x, landmark_z) in landmark_pos_dict.items():
        if landmark in CORNER_LANDMARKS:
            continue
        landmark_pos = np.array([landmark_x, landmark_z])
        gamma = np.dot(landmark_pos - start_pos, sg_vec) / sg_sq_norm
        if 0.0 < gamma < 1.0 and gamma < min_gamma:
            # check if landmark is blocking straight path
            closest_pos = start_pos + gamma * sg_vec
            dist = ((landmark_pos - closest_pos) ** 2).sum() ** 0.5
            l_radius = get_landmark_radius(landmark)
            if dist < l_radius:
                min_gamma = gamma
                best_landmark = landmark
    return best_landmark, min_gamma


def calc_extra_arc_distance(landmark_radius, landmark_goal_dist, goal_dist):
    d1, d2, d3 = landmark_radius, landmark_goal_dist, goal_dist
    theta = np.arccos((d1 ** 2 + d2 ** 2 - d3 ** 2) / (2 * d1 * d2))
    arc_distance = theta * landmark_radius
    cut_distance = (2 * (landmark_radius ** 2) * (1 - np.cos(theta))) ** 0.5
    return arc_distance - cut_distance


def calc_target_angle(pos, target_pos):
    x_diff = target_pos[0] - pos[0]
    z_diff = target_pos[1] - pos[1]
    return 90.0 - np.arctan2(z_diff, x_diff) * 180.0 / math.pi


def get_angle_unit_vector(angle):
    x = np.cos((90.0 - angle) * math.pi / 180.0)
    z = np.sin((90.0 - angle) * math.pi / 180.0)
    return np.array([x, z])


def abs_angle_diff(angle1, angle2):
    angle_diff = angle2 - angle1
    while angle_diff > 180.0:
        angle_diff -= 360.0
    while angle_diff < -180.0:
        angle_diff += 360.0
    return abs(angle_diff)


def is_terminal(state, goal_x, goal_z):
    x, z, _ = state
    goal_dist = ((goal_x - x) ** 2 + (goal_z - z) ** 2) ** 0.5
    if goal_dist < 1.5:
        return True
    else:
        return False


def successor(state, action, landmark_pos_dict):
    x, z, pose = state
    if action == "Forward":
        # calculate new x and new z after moving forward
        pose_x = np.cos((90.0 - pose) * math.pi / 180.0)
        pose_z = np.sin((90.0 - pose) * math.pi / 180.0)
        new_x = x + 1.5 * pose_x
        new_z = z + 1.5 * pose_z
        # find out if next state is valid
        if new_x < 225.0 or new_x > 275.0:
            return None
        elif new_z < 225.0 or new_z > 275.0:
            return None
        for landmark, (l_x, l_z) in landmark_pos_dict.items():
            if landmark in CORNER_LANDMARKS:
                continue
            l_dist = ((new_x - l_x) ** 2 + (new_z - l_z) ** 2) ** 0.5
            l_radius = get_landmark_radius(landmark)
            if l_dist < l_radius:
                return None
        return new_x, new_z, pose

    elif action == "TurnRight":
        # calculate new pose after turning right
        new_pose = (pose + 15.0) % 360.0
        return x, z, new_pose

    elif action == "TurnLeft":
        # calculate new pose after turning left
        new_pose = (pose - 15.0) % 360.0
        return x, z, new_pose
