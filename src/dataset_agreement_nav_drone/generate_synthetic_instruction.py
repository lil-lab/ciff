import os
import random
import math
import itertools
import numpy as np
import sys
import json

from dataset_agreement_nav_drone.nav_drone_dataset_parser import \
    parse_resources_json, get_landmark_pos_dict, DatasetParser
from utils.nav_drone_landmarks import map_landmark_name_human, \
    get_landmark_radius
from dataset_agreement_nav_drone.make_random_map import make_random_map
from make_random_map import X_RANGE, Y_RANGE

START_RANGE = (45, 955)


NUM_TRAIN = 7000
NUM_DEV = 1500
NUM_TEST = 1500
DATASET_PATH = "data/nav_drone/synthetic_v2_10k_%s.json"
DEBUG = False


class RestartException(Exception):
    def __init__(self):
        pass


class Direction:
    FRONT = 0
    RIGHT = 1
    LEFT = 2
    BACK = 3
    BETWEEN = 4

    @staticmethod
    def to_string(direction):
        if direction == Direction.FRONT:
            return "front"
        elif direction == Direction.RIGHT:
            return "right"
        elif direction == Direction.LEFT:
            return "left"
        elif direction == Direction.BACK:
            return "back"
        elif direction == Direction.BETWEEN:
            return "between"
        else:
            assert False

    @staticmethod
    def to_dir(dir_str):
        if dir_str == "front":
            return Direction.FRONT
        elif dir_str == "right":
            return Direction.RIGHT
        elif dir_str == "left":
            return Direction.LEFT
        elif dir_str == "back":
            return Direction.BACK
        elif dir_str == "between":
            return Direction.BETWEEN
        else:
            assert False





GOTO_INSTRUCTIONS = (
    "go to the %s",
    "approach the %s",
    "travel until you reach the %s",
    "move close to the %s",
)

BETWEEN_INSTRUCTIONS = (
    "go between the %s and the %s",
    "travel to the middle of the %s and the %s",
)

PASS_INSTRUCTIONS = (
    ("pass to the right of the %s", Direction.RIGHT),
    ("pass to the left of the %s", Direction.LEFT),
)

DIRECTIONAL_LOCATIONS = (
    ("back of the %s", Direction.BACK),
    ("right of the %s", Direction.RIGHT),
    ("left of the %s", Direction.LEFT),
)

CORNER_LANDMARKS = ["NWCorner", "NECorner", "SWCorner", "SECorner"]


def main():
    with open("data/nav_drone/config_localmoves_4000.json") as f:
        config = json.load(f)
    resources_dir = config["resources_dir"]
    path_dir_name = "synthetic_path_files"
    path_dir = os.path.join(resources_dir, path_dir_name)
    map_dir_name = "synthetic_map_files"
    map_dir = os.path.join(resources_dir, map_dir_name)
    if not os.path.exists(path_dir):
        os.makedirs(path_dir)
        os.makedirs(map_dir)
    counter = itertools.count()
    for which, num_items in (("train", NUM_TRAIN), ("dev", NUM_DEV), ("test", NUM_TEST)):
        items = []
        for _ in range(num_items):
            i = next(counter)
            map_file_name = os.path.join(map_dir_name, "map_%d.json") % i
            path_file_name = os.path.join(path_dir_name, "path_%d.json") % i
            item = make_synthetic_item(config, map_file_name, path_file_name, i)
            items.append(item)
        with open(DATASET_PATH % which, 'w') as f:
            json.dump(items, f)

    with open(DATASET_PATH % "train") as f:
        items = json.load(f)
    for input_item in items:
        scene_config = parse_resources_json(input_item["config_file"], config)
        landmark_pos_dict = get_landmark_pos_dict(scene_config)
        valid_landmarks = sample_valid_landmarks(
            landmark_pos_dict, input_item["start_x"][0], input_item["start_z"][0],
            input_item["start_rot"][0])
        print(valid_landmarks)


def debug():
    with open("data/nav_drone/config_localmoves_4000.json") as f:
        config = json.load(f)
    items = []
    for i in range(2):
        map_file_name = "tmp/map_%d.json" % i
        path_file_name = "tmp/path_%d.json" % i
        item_id = i
        item = make_synthetic_item(config, map_file_name, path_file_name, item_id)
        items.append(item)
    with open("tmp.json", 'w') as f:
        json.dump(items, f)
    parser = DatasetParser()

    dataset = parser.parse("tmp.json", config)
    print(len(dataset))


def make_synthetic_item(config, map_file_name, path_file_name, item_id):
    while True:
        try:
            resources_dir = config["resources_dir"]
            map_path = os.path.join(resources_dir, map_file_name)
            make_random_map(map_path)
            scene_config = parse_resources_json(map_file_name, config)
            landmark_pos_dict = get_landmark_pos_dict(scene_config)
            start_x, start_z, start_rot = sample_start_pos(landmark_pos_dict)
            instruction, end_x, end_z, end_rot, symbolic_instruction = sample_new_task(
                landmark_pos_dict, start_x, start_z, start_rot
            )
            path_path = os.path.join(resources_dir, path_file_name)
            make_path_file(path_path, start_x, start_z, end_x, end_z, start_rot)
            synthetic_item = {
                "start_x": [start_x],
                "start_z": [start_z],
                "start_rot": [start_rot],
                "end_x": [end_x],
                "end_z": [end_z],
                "end_rot": [end_rot],
                "config_file": map_file_name,
                "path_file": path_file_name,
                "id": item_id,
                "moves": [""],
                "valid": True,
                "instructions": [instruction],
                "symbolic_instruction": symbolic_instruction,
            }
            return synthetic_item
        except RestartException:
            continue


def make_path_file(save_path, start_x, start_z, end_x, end_z, start_rot):
    x_delta = float(np.cos(np.radians(90.0 - start_rot)))
    x1, x2, x3 = start_x, start_x + x_delta, end_x
    z_delta = float(np.sin(np.radians(90.0 - start_rot)))
    z1, z2, z3 = start_z, start_z + z_delta, end_z
    path = {"x_array": [x1, x2, x2, x2, x3], "z_array": [z1, z2, z2, z2, z3]}
    with open(save_path, 'w') as f:
        json.dump(path, f)


def sample_start_pos(landmark_pos_dict, min_dist=10.0):
    x, z, angle = None, None, None
    while True:
        x, z = (225.0 + 0.05 * random.randint(*START_RANGE),
                225.0 + 0.05 * random.randint(*START_RANGE))
        pos = np.array([x, z])
        fail = False
        for landmark, (landmark_x, landmark_z) in landmark_pos_dict.items():
            if landmark in CORNER_LANDMARKS:
                continue
            landmark_pos = np.array([landmark_x, landmark_z])
            radius = get_landmark_radius(landmark)
            dist = ((landmark_pos - pos) ** 2).sum() ** 0.5
            if dist < max(radius, min_dist):
                fail = True
                break
        if fail:
            continue
        angle = random.random() * 360.0
        valid_landmarks = sample_valid_landmarks(landmark_pos_dict, x, z, angle)
        if valid_landmarks is not None:
            break
    return x, z, angle


def sample_new_task(landmark_pos_dict, start_x, start_z, start_rot):
    landmarks = sample_valid_landmarks(landmark_pos_dict, start_x, start_z,
                                       start_rot)
    print(landmarks)
    if landmarks is None:
        return None
    instruction, direction = sample_instruction(landmarks)
    end_x, end_z, end_rot = sample_goal(landmark_pos_dict, landmarks, direction,
                                        start_x, start_z, start_rot)
    if direction != Direction.BETWEEN:
        symbolic_instruction = (landmarks[0], Direction.to_string(direction))
    else:
        symbolic_instruction = ((landmarks[0], landmarks[1]), "between")

    return instruction, end_x, end_z, end_rot, symbolic_instruction


def sample_valid_landmarks(landmark_pos_dict, start_x, start_z, start_rot):
    max_angle = 25.0
    min_distance = 10.0
    max_distance = 30.0
    min_sample_size = 2
    landmark_sample = []
    landmark_angles = {}
    for landmark, (landmark_x, landmark_z) in landmark_pos_dict.items():
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
            if abs(angle_diff) <= max_angle and min_distance <= landmark_dist <= max_distance:
                landmark_sample.append(landmark)
    landmark_names = {map_landmark_name_human(l) for l in landmark_sample}
    if len(landmark_sample) < min_sample_size:
        return None
    elif len(landmark_names) < len(landmark_sample):
        return None
    else:
        random.shuffle(landmark_sample)
        return landmark_sample


def sample_instruction(landmarks):
    landmark_names = [map_landmark_name_human(l) for l in landmarks]
    die1 = random.random()
    if die1 < 0.60:
        # use goto instruction
        template = random.choice(GOTO_INSTRUCTIONS)
        die2 = random.random()
        if die2 < 0.333:
            # simple approach
            target = landmark_names[0]
            direction = Direction.FRONT
        else:
            target_template, direction = random.choice(DIRECTIONAL_LOCATIONS)
            target = target_template % landmark_names[0]
    elif die1 < 0.80:
        # use pass instruction
        template, direction = random.choice(PASS_INSTRUCTIONS)
        target = landmark_names[0]
    else:
        # use between instruction
        template = random.choice(BETWEEN_INSTRUCTIONS)
        direction = Direction.BETWEEN
        target = (landmark_names[0], landmark_names[1])

    return template % target, direction


def sample_goal(landmark_pos_dict, landmarks, direction, start_x, start_z,
                start_rot):
    if direction != Direction.BETWEEN:
        # goal next to a landmark
        landmark_x, landmark_z = landmark_pos_dict[landmarks[0]]
        radius = get_landmark_radius(landmarks[0])

        # calculate F and R vectors
        start_to_landmark = np.array([landmark_x - start_x, landmark_z - start_z])
        landmark_dist = (start_to_landmark ** 2).sum() ** 0.5
        f_vector = start_to_landmark / landmark_dist * (radius + 2.0)
        r_vector = np.array([f_vector[1], -f_vector[0]])

        # calculate center of goal sample
        landmark_vector = np.array([landmark_x, landmark_z])
        if direction == Direction.FRONT:
            center = landmark_vector - f_vector
        elif direction == Direction.RIGHT:
            center = landmark_vector + r_vector
        elif direction == Direction.LEFT:
            center = landmark_vector - r_vector
        elif direction == Direction.BACK:
            center = landmark_vector + f_vector
        else:
            assert False
    else:
        # goal between two landmarks
        landmark_x_1, landmark_z_1 = landmark_pos_dict[landmarks[0]]
        landmark_x_2, landmark_z_2 = landmark_pos_dict[landmarks[1]]
        center = np.array([(landmark_x_1 + landmark_x_2) / 2.0,
                           (landmark_z_1 + landmark_z_2) / 2.0])

    noise = np.random.multivariate_normal([0.0, 0.0], [[1.5, 0.0], [0.0, 1.5]])
    goal = center + noise
    end_x, end_z = goal
    end_rot = 90.0 - np.arctan2(end_z - start_z, end_x - start_x) * 180.0 / math.pi

    # make sure goal is in range
    angle_diff = end_rot - start_rot
    while angle_diff > 180.0:
        angle_diff -= 360.0
    while angle_diff < -180.0:
        angle_diff += 360.0
    if abs(angle_diff) >= 30.0:
        raise RestartException()

    return float(end_x), float(end_z), float(end_rot)


if __name__ == "__main__":
    if DEBUG:
        debug()
    else:
        main()
