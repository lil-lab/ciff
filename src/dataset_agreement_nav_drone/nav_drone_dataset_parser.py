import json
import os
import random
from dataset_agreement.abstract_dataset_parser import AbstractDatasetParser
from dataset_agreement_nav_drone.nav_drone_datapoint import NavDroneDataPoint

action_code_map = {
    "F": "Forward",
    "R": "TurnRight",
    "L": "TurnLeft",
}


class DatasetParser(AbstractDatasetParser):

    def __init__(self):
        pass

    @staticmethod
    def parse(file_name, config):
        data_point_list = []
        with open(file_name) as f:
            raw_data_list = json.load(f)
        vocab = make_vocab_map(config["vocab_file"])
        for raw_data in raw_data_list:
            if not raw_data["valid"]:
                continue
            scene_config = parse_resources_json(raw_data["config_file"], config)
            if "lakeCoords" not in scene_config:
                scene_config[u"lakeCoords"] = []
            scene_path = parse_resources_json(raw_data["path_file"], config)
            # create segment data points
            if config["use_paragraphs"]:
                data_point = create_paragraph_data_point(
                    raw_data, scene_config, scene_path, config, vocab)
                data_point_list.append(data_point)
            else:
                data_points = create_segment_data_points(
                    raw_data, scene_config, scene_path, config, vocab)
                data_point_list.extend(data_points)

        return data_point_list


def create_segment_data_points(raw_data, scene_config, scene_path,
                               config, vocab):
    # build paragraph instruction, and indices
    paragraph_instruction = []
    instr_list = []
    indices_list = []
    num_instructions = len(raw_data["moves"])
    for i in range(num_instructions):
        instruction = process_instruction(
            raw_data["instructions"][i], vocab)
        i_start = len(paragraph_instruction)
        i_end = i_start + len(instruction)
        indices_list.append((i_start, i_end))
        paragraph_instruction.extend(instruction)
        instr_list.append(instruction)

    # build segment data points
    segment_data_points = []

    for i in range(num_instructions):
        instruction = instr_list[i]
        scene_name = "scene_%04d_%02d" % (int(raw_data["id"]), i)
        trajectory = make_trajectory(raw_data["moves"][i], config)
        start_pos = (raw_data["start_x"][i],
                     raw_data["start_z"][i],
                     raw_data["start_rot"][i])
        destination = (raw_data["end_x"][i],
                       raw_data["end_z"][i])
        data_point = NavDroneDataPoint(
            instruction=instruction,
            scene_name=scene_name,
            trajectory=trajectory,
            sub_trajectory_list=[trajectory],
            scene_config=json.dumps(scene_config),
            scene_path=json.dumps(scene_path),
            start_pos=start_pos,
            start_pos_list=[start_pos],
            destination_list=[destination],
            instruction_auto_segmented=[instruction],
            instruction_oracle_segmented=[instruction],
            landmark_pos_dict=get_landmark_pos_dict(scene_config),
            paragraph_instruction=paragraph_instruction,
            instruction_indices=indices_list[i],
            prev_instruction=instr_list[i - 1] if i > 0 else None,
            next_instruction=instr_list[i + 1] if i < num_instructions - 1 else None,
        )
        segment_data_points.append(data_point)
    return segment_data_points


def create_paragraph_data_point(raw_data, scene_config, scene_path,
                                config, vocab):
    num_segments = len(raw_data["moves"])
    instruction = []
    instruction_oracle_segmented = []
    trajectory = []
    sub_trajectory_list = []
    scene_name = "scene_%04d" % int(raw_data["id"])
    start_pos = (raw_data["start_x"][0],
                 raw_data["start_z"][0],
                 raw_data["start_rot"][0])
    start_pos_list = [(raw_data["start_x"][i],
                       raw_data["start_z"][i],
                       raw_data["start_rot"][i])
                      for i in range(num_segments)]
    destination_list = [(raw_data["end_x"][i], raw_data["end_z"][i])
                        for i in range(num_segments)]
    raw_text_list = []
    for i in range(len(raw_data["moves"])):
        tokens = process_instruction(raw_data["instructions"][i], vocab)
        raw_text_list.append(raw_data["instructions"][i])
        instruction.extend(tokens)
        instruction_oracle_segmented.append(tokens)
        actions = make_trajectory(raw_data["moves"][i], config)
        trajectory.extend(actions)
        sub_trajectory_list.append(actions)

    raw_text = " ".join(raw_text_list)
    auto_segments = raw_text.split(" . ")
    instruction_auto_segmented = []
    for seg in auto_segments:
        if len(seg.strip()) > 0:
            tokens = process_instruction(seg, vocab)
            instruction_auto_segmented.append(tokens)

    data_point = NavDroneDataPoint(
        instruction=instruction,
        scene_name=scene_name,
        trajectory=trajectory,
        sub_trajectory_list=sub_trajectory_list,
        scene_config=json.dumps(scene_config),
        scene_path=json.dumps(scene_path),
        start_pos=start_pos,
        start_pos_list=start_pos_list,
        destination_list=destination_list,
        instruction_auto_segmented=instruction_auto_segmented,
        instruction_oracle_segmented=instruction_oracle_segmented,
        landmark_pos_dict=get_landmark_pos_dict(scene_config),
        paragraph_instruction=instruction,
        instruction_indices=(0, len(instruction)))
    return data_point


def make_vocab_map(vocab_file):
    vocab = {}
    with open(vocab_file) as f:
        for line in f.readlines():
            try:
                token = line.strip().decode("utf-8")
            except Exception:
                token = line.strip()
            vocab[token] = len(vocab)
    return vocab


def process_instruction(instruction, vocab):
    return [vocab[token] for token in instruction.split()]


def make_trajectory(trajectory_string, config):
    action_names = config["action_names"]
    if len(action_names) == 4:
        return [action_names.index(action_code_map[a])
                for a in trajectory_string]
    else:
        return [random.choice(action_names[:-1])
                for _ in trajectory_string]


def parse_resources_json(file_name, config):
    resources_dir = config["resources_dir"]
    with open(os.path.join(resources_dir, file_name)) as f:
        return json.load(f)


def get_landmark_pos_dict(scene_config):
    landmarks = [str(name) for name in scene_config["landmarkName"]]
    x_pos_list = [225.0 + 0.05 * x_pos for x_pos in scene_config["xPos"]]
    z_pos_list = [225.0 + 0.05 * z_pos for z_pos in scene_config["zPos"]]
    pos_dict = {}
    for i, landmark in enumerate(landmarks):
        pos_dict[landmark] = (x_pos_list[i], z_pos_list[i])
    pos_dict["NECorner"] = (275.0, 275.0)
    pos_dict["NWCorner"] = (225.0, 275.0)
    pos_dict["SECorner"] = (275.0, 225.0)
    pos_dict["SWCorner"] = (225.0, 225.0)
    return pos_dict
