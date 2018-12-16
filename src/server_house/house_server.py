import numpy as np
import scipy.misc

from .reliable_connect import ReliableConnect
from server.abstract_server import AbstractServer


class HouseServer(AbstractServer):

    def __init__(self, config, action_space, port):
        # Connect to simulator
        self.unity_ip = "0.0.0.0"

        self.connection = ReliableConnect(
            self.unity_ip, port, image_row=config["image_height"], image_col=config["image_width"])

        # Meta data related information
        self.sum_navigation_error = 0
        self.sum_manipulation_accuracy = 0
        self.num_examples_seen = 0

        AbstractServer.__init__(self, config, action_space)

    def connect(self):
        self.connection.connect()

    def initialize_server(self):
        self.connect()

    def send_action_receive_feedback(self, action):
        action_string = self.action_space.get_action_name(action)
        self.connection.send_message(HouseServer.to_byte_arr(action_string))
        image = self.connection.receive_image()
        message = self.connection.receive_message().lower()
        message = message.decode("utf-8")   # Python 2--> Python 3, P3 notices diff between byte string and str
        words = message[len("unity manager: "):].split("#")

        # Message format is: reward, scene-name, action-execution, bot-position, bot-angles,
        # track-position, next-goal-screen, distance-to-final-goal, distance-to-next-goal,
        # goal-type and manipulation accuracy.
        reward = float(words[0])
        meta_data = {"type": "action-" + str(action), "scene-name": words[1], "action-success": words[2],
                     "bot-position": words[3], "bot-angles": words[4], "tracking": self._read_vector_(words[5]),
                     "goal-screen": self._read_vector_(words[6]), "navigation-error": float(words[7]),
                     "distance-to-next-goal": float(words[8]), "goal-type": words[9],
                     "manipulation-accuracy": float(words[10])}

        return image, reward, meta_data

    def halt_and_receive_feedback(self):
        stop_action = self.action_space.get_stop_action_index()
        action_string = self.action_space.get_action_name(stop_action)
        self.connection.send_message(HouseServer.to_byte_arr(action_string))
        image = self.connection.receive_image()
        message = self.connection.receive_message().lower()
        message = message.decode("utf-8")  # Python 2--> Python 3, P3 notices diff between byte string and str
        words = message[len("unity manager: "):].split("#")

        # Message format is: reward, scene-name, action-execution, bot-position, bot-angles,
        # track-position, next-goal-screen, distance-to-final-goal, distance-to-next-goal,
        # goal-type and manipulation accuracy.
        reward = float(words[0])
        navigation_error = float(words[7])
        manipulation_accuracy = float(words[10]) * 100.0  # convert to percentage

        self.sum_navigation_error += navigation_error
        self.sum_manipulation_accuracy += manipulation_accuracy
        self.num_examples_seen += 1
        mean_navigation_error = self.sum_navigation_error / float(self.num_examples_seen)
        mean_manipulation_accuracy = self.sum_manipulation_accuracy / float(self.num_examples_seen)

        meta_data = {"type": "action-" + str(stop_action), "scene-name": words[1], "action-success": words[2],
                     "bot-position": words[3], "bot-angles": words[4], "tracking": self._read_vector_(words[5]),
                     "goal-screen": self._read_vector_(words[6]), "distance-to-final-goal": float(words[7]),
                     "distance-to-next-goal": float(words[8]), "goal-type": words[9],
                     "manipulation-accuracy": manipulation_accuracy, "navigation-error": navigation_error,
                     "mean-navigation-error": mean_navigation_error,
                     "mean-manipulation-accuracy": mean_manipulation_accuracy}

        return image, reward, meta_data

    def reset_receive_feedback(self, next_datapoint):
        self.connection.send_message(HouseServer.to_byte_arr("ok-reset " + str(next_datapoint.get_id())))
        image = self.connection.receive_image()
        message = self.connection.receive_message().lower()
        message = message.decode("utf-8")  # Python 2--> Python 3, P3 notices diff between byte string and str
        words = message[len("unity manager: "):].split("#")

        # Message format is: reward, scene-name, action-execution, bot-position, bot-angles,
        # track-position, next-goal-screen, distance-to-final-goal, distance-to-next-goal,
        # goal-type and manipulation accuracy.
        meta_data = {"type": "reset", "scene-name": words[1], "action-success": words[2], "bot-position": words[3],
                     "bot-angles": words[4], "tracking": self._read_vector_(words[5]),
                     "goal-screen": self._read_vector_(words[6]), "navigation-error": float(words[7]),
                     "distance-to-next-goal": float(words[8]), "goal-type": words[9],
                     "manipulation-accuracy": float(words[10])}

        return image, meta_data

    def explore(self):
        """ Get the panorama at the given position """

        self.connection.send_message(self.to_byte_arr("panorama"))
        images = []
        for i in range(0, 6):
            # Get the panoramic images
            image = self.connection.receive_image()
            scaled_image = scipy.misc.bytescale(image)
            images.append(scaled_image)

        image_order = [3, 4, 5, 0, 1, 2]

        ordered_images = []

        for ix in image_order:  # panaroma consists of 6 images stitched together
            ordered_images.append(images[ix].swapaxes(0, 1).swapaxes(1, 2))
        panoramic_image = np.hstack(ordered_images).swapaxes(1, 2).swapaxes(0, 1)

        message = self.connection.receive_message()

        return panoramic_image, message

    def set_tracking(self, camera_ix, row_value, col_value):
        """ Start tracking of whichever object is present at the given row and col value
        in the current agent POV. The row_value is a value in [0, 1] with 0 at the top of the image
        and 1 at the bottom and col_value is a value in [0, 1] with 0 at the left of the image
        and 1 at the right. """
        return self.connection.send_and_receive_message(self.to_byte_arr("track %d %f %f" % (camera_ix, row_value, col_value)))

    @staticmethod
    def _read_vector_(value):

        if value == "none":
            return None
        else:
            vector = [float(w) for w in value[1:-1].split(",")]
            assert len(vector) == 3, "Vector should have 3 values. found " + str(value)
            return vector

    def clear_metadata(self):
        self.sum_navigation_error = 0
        self.sum_manipulation_accuracy = 0
        self.num_examples_seen = 0

    def kill(self):
        self.connection.close()

    @staticmethod
    def to_byte_arr(message):
        b = bytearray()
        b.extend(message.encode())
        return b
