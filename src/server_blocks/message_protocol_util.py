class MessageProtocolUtil:
    """ This class deals with message protocol between agent and simulator.
    Receives following type of messages:
        1. Status_Code#Bisk Metric#File_Name#Instruction#Trajectory
           when receiving a new example
        2. Status_Code#Reward#File_Name#Reset_Flag
           when receiving feedback for an ongoing example
    """
    def __init__(self, num_direction, num_actions, use_stop):
        self.num_direction = num_direction
        self.num_actions = num_actions
        self.use_stop = use_stop

    @staticmethod
    def is_reset_message(msg):
        return msg.endswith("reset")

    @staticmethod
    def decode_message(message):
        """ Decodes the message: Status_Code#Reward#File_Name#Reset_Flag
         into individual items."""

        words = message.split("#")

        if not (len(words) == 4):
            raise AssertionError("Message must contain exactly 4 words Status_Code#Reward#File_Name#Reset_Flag")

        status_code = words[0]
        reward = float(words[1])
        file_name = words[2]
        reset_file_name = words[3]

        return status_code, reward, file_name, reset_file_name

    @staticmethod
    def decode_reset_message(message):
        """ Decodes the message: Status_Code#Bisk Metric#File_Name#Instruction#Trajectory into
        individual items. """

        words = message.split("#")
        if not (len(words) == 5):
            raise AssertionError(
                "Message must contain exactly 5 words Status_Code#Bisk Metric#File_Name#Instruction#Trajectory")

        status_code = words[0]
        bisk_metric = float(words[1])
        file_name = words[2]
        instruction = words[3]

        trajectory = [int(v) for v in words[4].split(",")[:-1]]

        return status_code, bisk_metric, file_name, instruction, trajectory
