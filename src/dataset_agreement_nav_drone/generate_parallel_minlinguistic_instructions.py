import random
import sys
import json
from dataset_agreement_nav_drone.generate_synthetic_instruction import \
    sample_valid_landmarks, BETWEEN_INSTRUCTIONS, sample_instruction, Direction, \
    sample_goal, RestartException
from dataset_agreement_nav_drone.nav_drone_dataset_parser import \
    parse_resources_json, get_landmark_pos_dict
from utils.nav_drone_landmarks import map_landmark_name_human


DEBUG = False


def main():
    if len(sys.argv) != 3:
        sys.stderr.write("USAGE: python %s <input-path> <output-path>\n\n"
                         % sys.argv[0])
        sys.exit(1)
    input_path = sys.argv[1]
    output_path = sys.argv[2]

    with open(input_path) as f:
        input_items = json.load(f)
    with open("data/nav_drone/config_localmoves_4000.json") as f:
        config = json.load(f)
    output_items = []

    for input_item in input_items:
        while True:
            try:
                output_item = generate_min_linguistic_pair(input_item, config)
                break
            except RestartException:
                continue
        output_items.append(output_item)

    with open(output_path, 'w') as f:
        json.dump(output_items, f)


def debug():
    input_path = sys.argv[1]

    with open(input_path) as f:
        input_items = json.load(f)
    with open("data/nav_drone/config_localmoves_4000.json") as f:
        config = json.load(f)

    for input_item in input_items:
        while True:
            try:
                output_item = generate_min_linguistic_pair(input_item, config)
                break
            except RestartException:
                continue
        print(input_item["symbolic_instruction"], "---", input_item["instructions"][0])
        print(output_item["symbolic_instruction"], "---", output_item["instructions"][0])
        print("")


def generate_min_linguistic_pair(input_item, config):
    start_x, start_z, start_rot = (input_item["start_x"][0],
                                   input_item["start_z"][0],
                                   input_item["start_rot"][0])
    scene_config = parse_resources_json(input_item["config_file"], config)
    landmark_pos_dict = get_landmark_pos_dict(scene_config)

    valid_landmarks = sample_valid_landmarks(
        landmark_pos_dict, start_x, start_z, start_rot)
    landmark, direction = input_item["symbolic_instruction"]
    old_instruction = input_item["instructions"][0]
    if direction != "between":
        old_landmark_str = map_landmark_name_human(landmark)
        # decide whether to switch landmark or direction
        die = random.random()
        if die < 0.25:
            # change to between instruction
            new_landmarks = random.sample(valid_landmarks, 2)
            new_landmarks_strs = (map_landmark_name_human(new_landmarks[0]),
                                  map_landmark_name_human(new_landmarks[1]))
            new_instruction_template = random.choice(BETWEEN_INSTRUCTIONS)
            new_instruction = new_instruction_template % new_landmarks_strs
            symbolic_instruction = (new_landmarks, "between")

        elif die < 0.625:
            # swap landmark
            swap_candidates = [l for l in valid_landmarks if l != landmark]
            new_landmark = random.choice(swap_candidates)
            new_landmark_str = map_landmark_name_human(new_landmark)
            assert new_landmark_str != old_landmark_str
            assert old_instruction.count(old_landmark_str) == 1

            new_landmarks = (new_landmark,)
            new_instruction = old_instruction.replace(old_landmark_str,
                                                      new_landmark_str)
            symbolic_instruction = (new_landmark, direction)

        else:
            # swap direction
            if direction == "front":
                old_str = "the %s" % old_landmark_str
                new_str = "the back of the %s" % old_landmark_str
                new_direction = "back"
            elif direction == "right":
                old_str = "right of the %s" % old_landmark_str
                new_str = "left of the %s" % old_landmark_str
                new_direction = "left"
            elif direction == "left":
                old_str = "left of the %s" % old_landmark_str
                new_str = "right of the %s" % old_landmark_str
                new_direction = "right"
            elif direction == "back":
                old_str = "the back of the %s" % old_landmark_str
                new_str = "the %s" % old_landmark_str
                new_direction = "front"
            else:
                assert False
            assert old_instruction.count(old_str) == 1
            new_landmarks = (landmark,)
            new_instruction = old_instruction.replace(old_str, new_str)
            symbolic_instruction = (landmark, new_direction)

    else:
        # change to non-between instruction
        random.shuffle(valid_landmarks)
        new_landmark = valid_landmarks[0]
        new_instruction, new_direction = None, None
        while True:
            new_instruction, new_direction = sample_instruction(valid_landmarks)
            if new_direction != Direction.BETWEEN:
                break
        new_landmarks = (new_landmark,)
        new_direction_str = Direction.to_string(new_direction)
        symbolic_instruction = (new_landmark, new_direction_str)

    new_dir = Direction.to_dir(symbolic_instruction[1])
    end_x, end_z, end_rot = sample_goal(landmark_pos_dict, new_landmarks,
                                        new_dir, start_x, start_z, start_rot)
    new_item = {k: v for k, v in input_item.items()}
    new_item["instructions"] = [new_instruction]
    new_item["symbolic_instruction"] = symbolic_instruction
    new_item["end_x"] = [end_x]
    new_item["end_z"] = [end_z]
    new_item["end_rot"] = [end_rot]

    return new_item


if __name__ == "__main__":
    if DEBUG:
        debug()
    else:
        main()
