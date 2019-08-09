import collections
import numpy as np


class AgentObservedState:
    """ Observed state used by the agent to make decisions """

    def __init__(self, instruction, config, constants,
                 start_image=None, previous_action=None,
                 pose=None, position_orientation=None, data_point=None, goal_image=None):

        self.instruction = instruction
        self.config = config
        self.constants = constants
        self.previous_action = previous_action
        self.goal_image = goal_image
        self.max_num_images = constants["max_num_images"]

        empty_image = np.zeros((3, config["image_height"], config["image_width"]))
        padding_images = [empty_image] * constants["max_num_images"]

        self.image_memory = collections.deque(padding_images, constants["max_num_images"])
        self.num_images = 0

        # Read pointer points to the next token to be read
        self.pose = pose
        self.position_orientation = position_orientation
        self.data_point = data_point
        self.symbolic_instruction = None
        self.time_step = 0
        ##################################
        self.goal = None
        ##################################
        if start_image is not None:
            self.image_memory.append(start_image)
            self.num_images += 1

    def update(self, new_image, new_action, pose=None,
               position_orientation=None, data_point=None):
        cloned_state = AgentObservedState(self.instruction,
                                          self.config,
                                          self.constants)
        cloned_state.image_memory = collections.deque(list(self.image_memory),
                                                      self.max_num_images)
        cloned_state.image_memory.append(new_image)

        cloned_state.previous_action = new_action
        cloned_state.num_images = min(self.num_images + 1, self.max_num_images)
        cloned_state.pose = pose
        cloned_state.position_orientation = position_orientation
        cloned_state.data_point = data_point
        cloned_state.goal_image = self.goal_image
        cloned_state.time_step = self.time_step + 1

        return cloned_state

    def update_on_read(self):
        cloned_state = AgentObservedState(self.instruction,
                                          self.config,
                                          self.constants)
        cloned_state.image_memory = collections.deque(list(self.image_memory),
                                                      self.max_num_images)
        cloned_state.previous_action = self.previous_action
        cloned_state.num_images = min(self.num_images + 1, self.max_num_images)
        cloned_state.pose = self.pose
        cloned_state.position_orientation = self.position_orientation
        cloned_state.data_point = self.data_point
        cloned_state.goal_image = self.goal_image
        cloned_state.time_step = self.time_step + 1

        return cloned_state

    def update_on_act_halt(self):
        cloned_state = AgentObservedState(self.instruction,
                                          self.config,
                                          self.constants)
        cloned_state.image_memory = collections.deque(list(self.image_memory),
                                                      self.max_num_images)
        cloned_state.previous_action = self.previous_action
        cloned_state.num_images = min(self.num_images + 1, self.max_num_images)
        cloned_state.pose = self.pose
        cloned_state.position_orientation = self.position_orientation
        cloned_state.data_point = self.data_point
        cloned_state.goal_image = self.goal_image
        cloned_state.time_step = self.time_step + 1

        return cloned_state

    def get_instruction(self):
        return self.instruction

    def get_previous_action(self):
        return self.previous_action

    def get_image(self):
        num_zeros = self.max_num_images - self.num_images
        image_list = list(self.image_memory)[-self.num_images:]
        image_list.extend(list(self.image_memory)[:num_zeros])
        return np.array(image_list)

    def get_last_image(self):
        return self.image_memory[-1]

    def get_num_images(self):
        return self.num_images

    def get_pose(self):
        return self.pose

    def get_goal_image(self):
        return self.goal_image

    def get_position_orientation(self):
        return self.position_orientation

    def get_landmark_pos_dict(self):
        try:
            return self.data_point.get_landmark_pos_dict()
        except:
            return None

    def get_final_goal_position(self):
        return self.data_point.get_destination_list()[-1]
