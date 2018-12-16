class AbstractServer:

    def __init__(self, config, action_space):
        self.config = config
        self.action_space = action_space

    def receive_feedback_nonblocking(self):
        raise NotImplementedError()

    def send_action_receive_feedback(self, action):
        raise NotImplementedError()

    def send_action_nonblocking(self, action):
        raise NotImplementedError()

    def halt_and_receive_feedback(self):
        raise NotImplementedError()

    def halt_nonblocking(self):
        raise NotImplementedError()

    def reset_receive_feedback(self, next_data_point):
        raise NotImplementedError()

    def reset_nonblocking(self, next_data_point):
        raise NotImplementedError()

    def clear_metadata(self):
        raise NotImplementedError()

    def force_goal_update(self):
        raise NotImplementedError()
