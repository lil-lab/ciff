import random
import math
import numpy as np
import sys
import json

from dataset_agreement_nav_drone.nav_drone_dataset_parser import \
    parse_resources_json, get_landmark_pos_dict
from utils.nav_drone_landmarks import map_landmark_name_human, \
    get_landmark_radius


class Sides:
    FRONT = 0
    RIGHT = 1
    LEFT = 2
    BACK = 3

GOTO_INSTRUCTIONS = (
    "go to the %s",
    "approach the %s",
    "travel until you reach the %s",
    "move close to the %s",
)

PASS_INSTRUCTIONS = (
    "pass to the right of the %s",
    "pass to the left of the %s",
)

DIRECTIONAL_LOCATIONS = (
    "back of the %s",
    "right of the %s",
    "left of the %s",
)

CORNER_LANDMARKS = ["NWCorner", "NECorner", "SWCorner", "SECorner"]

def main():
    if len(sys.argv) != 3:
        sys.stderr.write("USAGE: python %s <input-path> <output-path>\n\n"
                         % sys.argv[0])
        sys.exit(1)
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    with open(input_path) as f:
        dataset = json.load(f)
    with open("data/nav_drone/config_localmoves_4000.json") as f:
        config = json.load(f)

    new_dataset = []
    for d in dataset:
        new_item = make_synthetic_item(d, config)
        if new_item is not None:
            new_dataset.append(new_item)
    with open(output_path, 'w') as f:
        json.dump(new_dataset, f)


def make_synthetic_item(original_item, config):
    if not original_item["valid"]:
        return None
    keep_keys = ["config_file", "id", "path_file", "start_rot",
                 "start_x", "start_z", "valid"]
    segment_keys = ["instructions", "end_x", "end_z", "end_rot",
                    "landmark", "side", "moves"]
    new_item = {k: original_item[k] for k in keep_keys}
    for k in segment_keys:
        new_item[k] = []

    scene_config = parse_resources_json(new_item["config_file"], config)
    landmark_pos_dict = get_landmark_pos_dict(scene_config)
    for seg_i in xrange(len(original_item["instructions"])):
        start_x = new_item["start_x"][seg_i]
        start_z = new_item["start_z"][seg_i]
        start_rot = new_item["start_rot"][seg_i]
        new_goal = sample_new_goal(
            landmark_pos_dict, start_x, start_z, start_rot)
        if new_goal is None:
            return None
        instruction, end_x, end_z, end_rot, landmark, side = new_goal
        new_item["instructions"].append(instruction)
        new_item["end_x"].append(end_x)
        new_item["end_z"].append(end_z)
        new_item["end_rot"].append(end_rot)
        new_item["landmark"].append(landmark)
        new_item["side"].append(side)
        new_item["moves"].append("")

    return new_item


def sample_new_goal(landmark_pos_dict, start_x, start_z, start_rot):
    landmark = sample_landmark(landmark_pos_dict, start_x, start_z, start_rot)
    if landmark is None:
        return None
    instruction, side = sample_instruction(landmark)
    end_x, end_z, end_rot = sample_goal(landmark_pos_dict, landmark, side,
                                        start_x, start_z)
    if side == Sides.FRONT:
        side_str = "front"
    elif side == Sides.RIGHT:
        side_str = "right"
    elif side == Sides.LEFT:
        side_str = "left"
    elif side == Sides.BACK:
        side_str = "back"
    else:
        assert False
    return instruction, end_x, end_z, end_rot, landmark, side_str


def sample_landmark(landmark_pos_dict, start_x, start_z, start_rot):
    max_angle = 30.0
    max_distance = 22.5
    landmark_sample = []
    landmark_angles = {}
    for landmark, (landmark_x, landmark_z) in landmark_pos_dict.iteritems():
        if landmark in CORNER_LANDMARKS:
            continue
        # get angle between drone's current orientation and landmark
        landmark_angle = 90.0 - np.arctan2(landmark_z - start_z, landmark_x - start_x) * 180.0 / math.pi
        landmark_dist = math.sqrt((landmark_x - start_x) ** 2 + (landmark_z - start_z) ** 2)
        angle_diff = landmark_angle - start_rot
        while angle_diff > 180.0:
            angle_diff -= 360.0
        while angle_diff < -180.0:
            angle_diff += 360.0
        if landmark_dist > 5:
            landmark_angles[landmark] = abs(angle_diff)
            if abs(angle_diff) <= max_angle and landmark_dist <= max_distance:
                landmark_sample.append(landmark)
    if len(landmark_sample) > 0:
        return random.choice(landmark_sample)
    else:
        # return min(landmark_angles, key=landmark_angles.get)
        return None


def sample_instruction(landmark):
    landmark_name = map_landmark_name_human(landmark)
    die1 = random.random()
    if die1 < 0.75:
        # use goto instruction
        template = random.choice(GOTO_INSTRUCTIONS)
        die2 = random.random()
        if die2 < 0.333:
            # simple approach
            target = landmark_name
            side_token = "front"
        else:
            target = random.choice(DIRECTIONAL_LOCATIONS) % landmark_name
            side_token = target.split()[0]
    else:
        # use pass instruction
        template = random.choice(PASS_INSTRUCTIONS)
        target = landmark_name
        side_token = template.split()[3]

    if side_token == "front":
        side = Sides.FRONT
    elif side_token == "right":
        side = Sides.RIGHT
    elif side_token == "left":
        side = Sides.LEFT
    elif side_token == "back":
        side = Sides.BACK
    else:
        assert False
    return template % target, side



def sample_goal(landmark_pos_dict, landmark, side, start_x, start_z):
    landmark_x, landmark_z = landmark_pos_dict[landmark]
    radius = get_landmark_radius(landmark)

    # calculate F and R vectors
    start_to_landmark = np.array([landmark_x - start_x, landmark_z - start_z])
    landmark_dist = (start_to_landmark ** 2).sum() ** 0.5
    f_vector = start_to_landmark / landmark_dist * (radius + 2.0)
    r_vector = np.array([f_vector[1], -f_vector[0]])

    # calculate center of goal sample
    landmark_vector = np.array([landmark_x, landmark_z])
    if side == Sides.FRONT:
        center = landmark_vector - f_vector
    elif side == Sides.RIGHT:
        center = landmark_vector + r_vector
    elif side == Sides.LEFT:
        center = landmark_vector - r_vector
    elif side == Sides.BACK:
        center = landmark_vector + f_vector
    else:
        assert False

    noise = np.random.multivariate_normal([0.0, 0.0], [[1.5, 0.0], [0.0, 1.5]])
    goal = center + noise
    end_x, end_z = goal
    end_rot = 90.0 - np.arctan2(end_z - start_z, end_x - start_x) * 180.0 / math.pi
    return float(end_x), float(end_z), float(end_rot)


if __name__ == "__main__":
    main()
