import json
from dataset_agreement.abstract_dataset_parser import AbstractDatasetParser
from .datapoint import DataPoint

actions = ["forward", "back", "slideleft", "slideright", "lookleft", "lookright", "stop"]


class DatasetParser(AbstractDatasetParser):

    def __init__(self):
        pass

    @staticmethod
    def _verify_trajectory(trajectory):
        for action in trajectory:
            if action not in actions:
                raise AssertionError("Unknown action. " + str(action))

    @staticmethod
    def _convert_to_indices(trajectory):
        trajectory_indices = []
        for action in trajectory:
            trajectory_indices.append(actions.index(action))
        return trajectory_indices

    @staticmethod
    def _convert_to_instruction_indices(instruction):
        return [int(index.strip()) for index in instruction[1:-1].split(",")]

    @staticmethod
    def parse(file_name, config, use_trajectory=False):

        ################################
        # traj_dict = {}
        # lines = open("./simulators/house/AssetsHouse/dataset_trajectory_segments.txt").readlines()
        # index = 0
        # curr = ""
        # id_ = 0
        # for line in lines:
        #     line = line.strip()
        #     if len(line) == 0:
        #         continue
        #     if line.startswith("$$$$$$$$$"):
        #         index += 1
        #         curr = ""
        #         continue
        #     if line.startswith("=========="):
        #         index = 0
        #         curr = ""
        #         id_ += 1
        #         continue
        #     curr = curr + line
        #     if index == 8:
        #         trajectory = curr
        #         jobj = json.loads(trajectory)
        #         dios = jobj["dios"]
        #         traj_dict[id_] = len(dios)
        #     else:
        #         curr = ""
        ################################

        data = json.load(open(file_name))
        datapoints_json = data["Dataset"]

        dataset = []
        for datapoint_json in datapoints_json:
            # instruction = DatasetParser._convert_to_instruction_indices(datapoint_json["Instruction"])
            instruction = datapoint_json["Instruction"]
            if len(instruction) < 3:
                continue
            house = datapoint_json["House"]
            trajectory_merged = datapoint_json["Trajectory"]
            datapoint_id = int(datapoint_json["ID"])
            if use_trajectory:
                trajectory = trajectory_merged.split("#")
                DatasetParser._verify_trajectory(trajectory)
                trajectory_indices = DatasetParser._convert_to_indices(trajectory)
            else:
                trajectory_indices = None

            #################################
            # traj_len = traj_dict[datapoint_id]
            # trajectory_indices = [1] * traj_len
            #################################

            datapoint = DataPoint(instruction, house, trajectory_indices, datapoint_id)
            dataset.append(datapoint)

        return dataset
