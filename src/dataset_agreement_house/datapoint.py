from dataset_agreement.abstract_datapoint import AbstractDataPoint


class DataPoint(AbstractDataPoint):

    def __init__(self, instruction, house_name, trajectory, datapoint_id):
        self.instruction = instruction
        self.house_name = house_name
        self.trajectory = trajectory
        self.datapoint_id = datapoint_id

    def get_instruction(self):
        return self.instruction

    def get_scene_name(self):
        return self.house_name

    def get_trajectory(self):
        return self.trajectory

    def get_sub_trajectory_list(self):
        raise NotImplementedError()

    def get_id(self):
        return self.datapoint_id

    def get_instruction_auto_segmented(self):
        raise NotImplementedError()

    def get_instruction_oracle_segmented(self):
        raise NotImplementedError()
