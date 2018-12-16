class Config:

    def __init__(self, height, width, step_pos, step_angle, enable_collision, hold):
        self.height = height
        self.width = width
        self.step_pos = step_pos
        self.step_angle = step_angle
        self.enable_collision = enable_collision
        self.hold = hold

    @staticmethod
    def parse_config(config_filename):
        lines = open(config_filename).readlines()
        height, width = None, None
        step_pos, step_angle = None, None
        enable_collision, hold = None, None
        for line in lines:
            words = line.strip().split(":")
            assert len(words) == 2
            key, value = words
            if key == "height":
                height = int(value)
            elif key == "width":
                width = int(value)
            elif key == "stepPos":
                step_pos = float(value)
            elif key == "stepAngle":
                step_angle = float(value)
            elif key == "enableCollision":
                enable_collision = (value == "True" or value == "true")
            elif key == "hold":
                hold = (value == "True" or value == "true")

        return Config(height, width, step_pos, step_angle, enable_collision, hold)