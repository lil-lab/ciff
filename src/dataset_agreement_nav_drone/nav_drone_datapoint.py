from dataset_agreement.abstract_datapoint import AbstractDataPoint


class NavDroneDataPoint(AbstractDataPoint):
    def __init__(self, instruction, scene_name, trajectory, sub_trajectory_list,
                 scene_config, scene_path, start_pos, start_pos_list,
                 destination_list, instruction_auto_segmented=None,
                 instruction_oracle_segmented=None,
                 landmark_pos_dict=None, paragraph_instruction=None,
                 instruction_indices=None, prev_instruction=None,
                 next_instruction=None):
        self.instruction = instruction
        self.scene_name = scene_name
        self.trajectory = trajectory
        self.sub_trajectory_list = sub_trajectory_list
        self.scene_config = scene_config
        self.scene_path = scene_path
        self.start_pos = start_pos
        self.backup_pos = start_pos
        self.start_pos_list = start_pos_list
        self.destination_list = destination_list
        self.instruction_auto_segmented = instruction_auto_segmented
        self.instruction_oracle_segmented = instruction_oracle_segmented
        self.landmark_pos_dict = landmark_pos_dict
        self.paragraph_instruction = paragraph_instruction
        self.instruction_indices = instruction_indices
        self.prev_instruction = prev_instruction
        self.next_instruction = next_instruction
        self.six_start_images = None

    def get_instruction(self):
        return self.instruction

    def get_paragraph_instruction(self):
        return self.paragraph_instruction

    def get_instruction_indices(self):
        return self.instruction_indices

    def get_scene_name(self):
        return self.scene_name

    def get_trajectory(self):
        return self.trajectory

    def get_sub_trajectory_list(self):
        return self.sub_trajectory_list

    def get_scene_config(self):
        return self.scene_config

    def get_scene_path(self):
        return self.scene_path

    def get_start_pos(self):
        return self.start_pos

    #####################################
    def restore_pos(self):
        self.start_pos = self.backup_pos

    def change_pos(self, new_pos):
        self.start_pos = new_pos
    #####################################

    def get_start_pos_list(self):
        return self.start_pos_list

    def get_destination_list(self):
        return self.destination_list

    def get_instruction_auto_segmented(self):
        return self.instruction_auto_segmented

    def get_instruction_oracle_segmented(self):
        return self.instruction_oracle_segmented

    def get_landmark_pos_dict(self):
        return self.landmark_pos_dict

    def get_prev_instruction(self):
        return self.prev_instruction

    def get_next_instruction(self):
        return self.next_instruction

    def set_six_start_images(self, image):
        self.six_start_images = image

    def get_six_start_images(self):
        return self.six_start_images
