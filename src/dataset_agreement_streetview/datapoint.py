from dataset_agreement.abstract_datapoint import AbstractDataPoint


class DataPoint(AbstractDataPoint):

    def __init__(self, datapoint_id, instruction_token_seq, instruction_string, trajectory,
                 scene_name, start_orientation, end_orientation, pre_static_center_exists, post_static_center_exists,
                 pre_pano, post_pano):
        self.datapoint_id = datapoint_id
        self.instruction = instruction_token_seq
        self.instruction_string = instruction_string
        self.trajectory = trajectory
        self.scene_name = scene_name
        self.start_orientation = start_orientation
        self.end_orientation = end_orientation
        self.pre_static_center_exists = pre_static_center_exists
        self.post_static_center_exists = post_static_center_exists
        self.pre_pano = pre_pano
        self.post_pano = post_pano

    def get_datapoint_id(self):
        return self.datapoint_id

    def get_start_orientation(self):
        return self.start_orientation

    def get_end_orientation(self):
        return self.end_orientation

    def get_instruction_auto_segmented(self):
        raise NotImplementedError()

    def get_instruction_oracle_segmented(self):
        raise NotImplementedError()

    def get_sub_trajectory_list(self):
        raise NotImplementedError()

    def get_instruction(self):
        return self.instruction

    def get_trajectory(self):
        return self.trajectory

    def get_scene_name(self):
        return self.scene_name

    def get_target_pano(self):
        return self.trajectory[-1]

    def get_pre_static_center_exists(self):
        return self.pre_static_center_exists

    def get_post_static_center_exists(self):
        return self.post_static_center_exists

    def get_pre_pano(self):
        return self.pre_pano

    def get_post_pano(self):
        return self.post_pano
