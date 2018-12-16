from dataset_agreement.abstract_datapoint import AbstractDataPoint


class DataPoint(AbstractDataPoint):

    def __init__(self, point_id):
        self.point_id = point_id
        self.instruction = None
        self.instruction_string = None

        # Label information
        self.start_image = None
        self.block_id = None
        self.start_location = None
        self.goal_location = None
        self.goal_pixel = None

    def get_id(self):
        return self.point_id

    def get_instruction(self):
        raise NotImplementedError()

    def get_scene_name(self):
        raise NotImplementedError()

    def get_trajectory(self):
        raise NotImplementedError()

    def get_sub_trajectory_list(self):
        raise NotImplementedError()

    def get_instruction_auto_segmented(self):
        raise NotImplementedError()

    def get_instruction_oracle_segmented(self):
        raise NotImplementedError()

    def set_instruction(self, instruction, instruction_string):
        self.instruction = instruction
        self.instruction_string = instruction_string

    def set_start_image(self, start_image):
        self.start_image = start_image

    def set_block_id(self, block_id):
        self.block_id = block_id

    def set_start_location(self, start_location):
        self.start_location = start_location

    def set_goal_location(self, goal_location):
        self.goal_location = goal_location

    def set_goal_pixel(self, goal_pixel):
        self.goal_pixel = goal_pixel
