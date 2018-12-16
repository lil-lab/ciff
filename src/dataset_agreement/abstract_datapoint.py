class AbstractDataPoint:

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
