from collections import defaultdict
import logging
import struct
from multiprocessing import Queue, Process, Manager, Lock
import math
import numpy as np
import time
import json
from scipy import misc
from dataset_agreement_nav_drone.action_space import ActionSpace

from dataset_agreement_nav_drone.nav_drone_datapoint import NavDroneDataPoint
from server.abstract_server import AbstractServer
from server_nav_drone.core_socket_server import CoreSocketServer
from server_nav_drone.trajectory_evaluation import stop_distance, \
    norm_edit_distance, mean_closest_distance
from utils.debug_nav_drone_instruction import instruction_to_string

MESSAGE_TYPES = ["Init", "Request", "Screen", "StopFlag", "Feedback"]
REQUEST_TYPES = ["JsonConfig", "JsonPath", "Instructions", "TokenSpan",
                 "StartPos", "NextDest", "Move", "NextJob", "GoldTrajectory"]
STOP = "Stop"
FORCE_GOAL_UPDATE = "ForceGoalUpdate"
FLAG_ACT_HALT = "FlagActHalt"
EXPLORE = "Explore"
SERVER_MOVE_RESPONSES = ["Forward", "TurnRight", "TurnLeft",
                         "MoveUp", "MoveRight", "MoveLeft", "MoveDown",
                         STOP, FORCE_GOAL_UPDATE, FLAG_ACT_HALT, EXPLORE]
SERVER_JOB_RESPONSES = ["NewSegment", "NewConfig", "Terminate"]


class NavDroneServer(AbstractServer):
    def __init__(self, config, action_space):
        AbstractServer.__init__(self, config, action_space)
        self.unity_server_controller = UnityControllerServer(config)
        self.config = config
        self.action_space = action_space
        port = int(config["port"])
        hostname = config["hostname"]
        core_server = CoreSocketServer(hostname, port,
                                       self.unity_server_controller)
        # core_server.serve_forever()
        self.p = Process(target=core_server.serve_forever, args=())
        self.p.start()

    def send_action_receive_feedback(self, action):
        assert isinstance(action, int)
        move = self.action_space.get_action_name(action)
        self.unity_server_controller.make_move(move)
        return self.unity_server_controller.get_feedback()

    def send_action_nonblocking(self, action):
        assert isinstance(action, int)
        move = self.action_space.get_action_name(action)
        self.unity_server_controller.make_move(move)

    def receive_feedback_nonblocking(self):
        return self.unity_server_controller.get_feedback_nonblocking()

    def halt_and_receive_feedback(self):
        self.unity_server_controller.make_move(STOP)
        return self.unity_server_controller.get_feedback()

    def halt_nonblocking(self):
        self.unity_server_controller.make_move(STOP)

    def reset_receive_feedback(self, data_point):
        self.unity_server_controller.reset(data_point, self.action_space,
                                           self.config)
        return self.unity_server_controller.get_initial_image()

    def reset_nonblocking(self, data_point):
        self.unity_server_controller.reset(data_point, self.action_space,
                                           self.config)

    def receive_reset_feedback_nonblocking(self):
        return self.unity_server_controller.receive_reset_feedback_nonblocking()

    def force_goal_update(self):
        self.unity_server_controller.make_move(FORCE_GOAL_UPDATE)
        return self.unity_server_controller.get_feedback()

    def force_goal_update_nonblocking(self):
        self.unity_server_controller.make_move(FORCE_GOAL_UPDATE)

    def flag_act_halt(self):
        self.unity_server_controller.make_move(FLAG_ACT_HALT)
        return self.unity_server_controller.get_feedback()

    def flag_act_halt_nonblocking(self):
        self.unity_server_controller.make_move(FLAG_ACT_HALT)

    def explore(self):
        self.unity_server_controller.make_move(EXPLORE)
        return self.unity_server_controller.get_feedback()

    def explore_nonblocking(self):
        self.unity_server_controller.make_move(EXPLORE)

    def kill(self):
        self.p.terminate()

    def clear_metadata(self):
        self.unity_server_controller.clear_metadata()


class UnityControllerServer(object):
    def __init__(self, config):
        self.config = config
        self.manager = Manager()
        self.shared_data = self.manager.dict()
        self.shared_data["received_feedback"] = False
        self.shared_data["dist_lists"] = defaultdict(list)
        self.shared_data["end_x"] = float("NaN")
        self.shared_data["end_z"] = float("NaN")
        self.shared_data["dest_list"] = None
        self.shared_data["pos_list"] = None
        self.shared_data_lock = Lock()
        self.move_list = []
        self.move_queue_full = False

        # queues of data requested from unity
        self.scene_config_queue = Queue()
        self.path_queue = Queue()
        self.instructions_queue = Queue()
        self.start_pos_queue = Queue()
        self.next_dest_queue = Queue()
        self.move_queue = Queue()

        # queues of data sent from unity
        self.image_queue = Queue()
        self.rewards_queue = Queue()
        self.feedback_queue = Queue()
        self.goal_dist_queue = Queue()
        self.pos_queue = Queue()

    def process_message(self, message, response_writer):
        # get message type
        data = str(message)
        message_type = MESSAGE_TYPES[ord(data[0])]
        message_data = data[1:]
        # print "message received:", message_type
        if message_type == "Init":
            self.do_init()
        elif message_type == "Request":
            self.fetch_request(message_data, response_writer)
        elif message_type == "Screen":
            self.save_next_screen(message_data)
        elif message_type == "StopFlag":
            self.flag_stop(message_data)
        elif message_type == "Feedback":
            self.log_human_feedback(message_data)
        else:
            print ("invalid message type:", message_type)

    def get_initial_image(self):
        # clear dummy reward from queue first
        self.rewards_queue.get()
        x_pos, z_pos, y_angle = self.pos_queue.get()
        goal_dist = self.goal_dist_queue.get()
        with self.shared_data_lock:
            pos_list = [(x_pos, z_pos)]
            self.shared_data["pos_list"] = pos_list
            dest_list = self.shared_data["dest_list"]
            stop_dist = stop_distance(pos_list, dest_list)
            edit_dist = norm_edit_distance(pos_list, dest_list)
            closest_dist = mean_closest_distance(pos_list, dest_list)
        metadata = {
            "x_pos": x_pos,
            "z_pos": z_pos,
            "y_angle": y_angle,
            "goal_dist": goal_dist,
            "stop_dist_error": stop_dist,
            "edit_dist_error": edit_dist,
            "closest_dist_error": closest_dist,
            "error": edit_dist,
            "reward_dict": {move: 0.0 for move in SERVER_MOVE_RESPONSES},
            "feedback": "",
        }
        self.move_queue_full = False
        return self.image_queue.get(), metadata

    def make_move(self, move):
        """
        while True:
            m = raw_input("make a move: ")
            if m == "f":
                move = "Forward"
            elif m == "r":
                move = "TurnRight"
            elif m == "l":
                move = "TurnLeft"
            elif m == "s":
                move = "Stop"
            else:
                print "bad move: " + m
                continue
            break
        """
        # assert self.move_queue_full is False, "made second server move without obtaining feedback from first"
        self.move_queue.put(move)
        self.move_list.append(move)
        self.move_queue_full = True

    def get_feedback(self):
        if len(self.move_list) > 0:
            last_move = self.move_list[-1]
            reward_list = self.rewards_queue.get()
            reward = reward_list[SERVER_MOVE_RESPONSES.index(last_move)]
        else:
            reward = None
        image = self.image_queue.get()
        goal_dist = self.goal_dist_queue.get()
        x_pos, z_pos, y_angle = self.pos_queue.get()
        with self.shared_data_lock:
            pos_list = self.shared_data["pos_list"]
            pos_list.append((x_pos, z_pos))
            self.shared_data["pos_list"] = pos_list
            dest_list = self.shared_data["dest_list"]
            stop_dist = stop_distance(pos_list, dest_list)
            edit_dist = norm_edit_distance(pos_list, dest_list)
            closest_dist = mean_closest_distance(pos_list, dest_list)
        metadata = {
            "x_pos": x_pos,
            "z_pos": z_pos,
            "y_angle": y_angle,
            "goal_dist": goal_dist,
            "stop_dist_error": stop_dist,
            "edit_dist_error": edit_dist,
            "closest_dist_error": closest_dist,
            "error": edit_dist,
            "reward_dict": make_reward_dict(reward_list),
            "feedback": "",
        }
        with self.shared_data_lock:
            if self.shared_data["received_feedback"]:
                metadata["feedback"] = self.feedback_queue.get()
                self.shared_data["received_feedback"] = False
        self.move_queue_full = False
        return image, reward, metadata

    def get_feedback_nonblocking(self):
        if self.rewards_queue.empty():
            return None
        else:
            return self.get_feedback()

    def receive_reset_feedback_nonblocking(self):
        if self.image_queue.empty():
            return None
        else:
            return self.get_initial_image()

    def reset(self, data_point, action_space, config):
        assert isinstance(data_point, NavDroneDataPoint)
        assert isinstance(action_space, ActionSpace)
        self.move_list = []
        with self.shared_data_lock:
            self.shared_data["scene_name"] = data_point.get_scene_name()
            end_x, end_z = data_point.get_destination_list()[-1]
            self.shared_data["end_x"], self.shared_data["end_z"] = end_x, end_z
            self.shared_data["dest_list"] = data_point.get_destination_list()
            gold_moves = []
            for seg in data_point.get_sub_trajectory_list():
                moves = [action_space.get_action_name(a) for a in seg]
                gold_moves.extend(moves)
                gold_moves.append(STOP)
            self.shared_data["trajectory"] = [SERVER_MOVE_RESPONSES.index(m)
                                              for m in gold_moves]
        instruction_segments = data_point.get_instruction_oracle_segmented()
        instruction_string = ""
        for i, instruction_seg in enumerate(instruction_segments):
            if i % 2 == 0:
                color = "yellow"
            else:
                color = "magenta"
            instruction_string += "<color=%s>" % color
            instruction_string += instruction_to_string(instruction_seg, config)
            instruction_string += "</color> "
        self.scene_config_queue.put(data_point.get_scene_config())
        self.path_queue.put(data_point.get_scene_path())
        self.instructions_queue.put(instruction_string.strip())
        self.start_pos_queue.put(data_point.get_start_pos())
        self.next_dest_queue.put(data_point.get_destination_list())
        self.move_queue_full = True

    def do_init(self):
        # placeholder for future possible functionality
        pass

    def fetch_request(self, request_data, response_writer):
        request_type = REQUEST_TYPES[ord(request_data[0])]
        # print "request type:", request_type
        payload = ""
        if request_type == "JsonConfig":
            payload = self.scene_config_queue.get()
        elif request_type == "JsonPath":
            payload = self.path_queue.get()
        elif request_type == "Instructions":
            payload = self.instructions_queue.get().encode("utf-8")
        elif request_type == "TokenSpan":
            payload = struct.pack("<i", 0)
            payload += struct.pack("<i", 0)
        elif request_type == "StartPos":
            start_x, start_z, start_rot = self.start_pos_queue.get()
            payload = struct.pack("<f", start_x)
            payload += struct.pack("<f", start_z)
            payload += struct.pack("<f", start_rot)
        elif request_type == "NextDest":
            destination_list = self.next_dest_queue.get()
            payload = ""
            for end_x, end_z in destination_list:
                payload += struct.pack("<f", end_x)
                payload += struct.pack("<f", end_z)
        elif request_type == "Move":
            move = self.move_queue.get()
            move_id = SERVER_MOVE_RESPONSES.index(move)
            payload = struct.pack("<b", move_id)
        elif request_type == "NextJob":
            next_job = SERVER_JOB_RESPONSES.index("NewConfig")
            payload = struct.pack("<b", next_job)
        elif request_type == "GoldTrajectory":
            trajectory = self.shared_data["trajectory"]
            payload = struct.pack("<" + len(trajectory) * "b", *trajectory)

        # send pay.oad back to response writer
        response_writer(request_data[0] + str(payload))

    def save_next_screen(self, screen_reward_array):
        num_rewards = len(SERVER_MOVE_RESPONSES) - 2
        rewards = struct.unpack(
            "<" + "f" * num_rewards, screen_reward_array[:4*num_rewards])
        x_pos = struct.unpack(
            "<f", screen_reward_array[4*num_rewards:4*num_rewards+4])[0]
        z_pos = struct.unpack(
            "<f", screen_reward_array[4*num_rewards+4:4*num_rewards+8])[0]
        y_angle = struct.unpack(
            "<f", screen_reward_array[4*num_rewards+8:4*num_rewards+12])[0]
        next_dest_x = struct.unpack(
            "<f", screen_reward_array[4*num_rewards+12:4*num_rewards+16])[0]
        next_dest_z = struct.unpack(
            "<f", screen_reward_array[4*num_rewards+16:4*num_rewards+20])[0]
        self.rewards_queue.put(list(rewards) + [0.0, 0.0])
        image = process_unity_image(
            screen_reward_array[4*num_rewards+20:],
            height=self.config["image_height"],
            width=self.config["image_width"])
        self.image_queue.put(image)
        self.pos_queue.put((x_pos, z_pos, y_angle))
        dist_to_goal = math.sqrt((x_pos - next_dest_x) ** 2 + (z_pos - next_dest_z) ** 2)
        self.goal_dist_queue.put(dist_to_goal)

    def flag_stop(self, message_data):
        stop_successful = bool(struct.unpack("<b", message_data[:1])[0])
        stop_x = struct.unpack("<f", message_data[1:5])[0]
        stop_z = struct.unpack("<f", message_data[5:9])[0]
        with self.shared_data_lock:
            pos_list = self.shared_data["pos_list"] + [(stop_x, stop_z)]
            dest_list = self.shared_data["dest_list"]
            dists = {
                "stop_dist": stop_distance(pos_list, dest_list),
                "edit_dist": norm_edit_distance(pos_list, dest_list),
                "closest_dist": mean_closest_distance(pos_list, dest_list),
            }
            dist_lists = self.shared_data["dist_lists"]
            for k, dist in dists.iteritems():
                dist_lists[k].append(dist)
            self.shared_data["dist_lists"] = dist_lists

            scene_name = self.shared_data["scene_name"]
            feedback_template = "scene-name=%s --- stop-successful=%s"
            feedback = feedback_template % (scene_name, str(stop_successful))
            for k, dist_list in sorted(dist_lists.iteritems()):
                dists_template = " --- " + k + "=%f --- cum-mean-" + k + "=%f"
                dist = dist_list[-1]
                mean_dist = float(np.mean(dist_list))
                feedback += dists_template % (dist, mean_dist)

            self.feedback_queue.put(feedback)
            self.shared_data["received_feedback"] = True

    def log_human_feedback(self, feedback_data):
        pass

    def clear_metadata(self):
        with self.shared_data_lock:
            self.shared_data["dist_lists"] = defaultdict(list)


num_images = [0]

def process_unity_image(byte_str, height, width):
    """
    num_bytes = height * width * 4 * 4
    assert len(byte_str) == num_bytes, len(byte_str)

    s_format = "<" + "f" * 4 * height * width
    unity_shape = (height, width, 4)
    x = np.array(struct.unpack(s_format, byte_str)).reshape(unity_shape)
    # remove alpha channel
    x = x[:, :, :3]
    # flip so image is right-way up
    x = np.flip(x, 0)
    # swap axes to order expected by PyTorch
    x = x.swapaxes(1, 2)
    x = x.swapaxes(0, 1)
    return x
    """
    num_bytes_per_im = height * width * 4 * 4
    assert (len(byte_str) % num_bytes_per_im) == 0, "incorrect image bytes"
    num_cam = len(byte_str) // num_bytes_per_im

    s_format = "<" + "f" * 4 * height * width * num_cam
    multi_image_shape = (num_cam, height, width, 4)
    x = np.array(struct.unpack(s_format, byte_str)).reshape(multi_image_shape)

    # remove alpha channel
    x = x[:, :, :, :3]
    # flip so image is right-way up
    x = np.flip(x, 1)
    """
    for i, im in enumerate(x):
        print "hello"
        misc.imsave("multi_image_%d.png" % i, im)
    """
    """
    if num_cam == 6:
        x = x.swapaxes(0, 1)
        final_shape = (height, num_cam * width, 3)
    else:
        x = x.swapaxes(0, 1)
        x = x.swapaxes(1, 2)
        final_shape = (height, width, num_cam * 3)
    """
    x = x.swapaxes(0, 1)
    x = x.swapaxes(1, 2)
    final_shape = (height, width, num_cam * 3)

    x = x.reshape(final_shape)
    num_images[0] += 1
    # swap axes to order expected by PyTorch
    x = x.swapaxes(1, 2)
    x = x.swapaxes(0, 1)

    return x[:, :, :width]


def make_reward_dict(reward_list):
    reward_dict = {}
    for i, reward in enumerate(reward_list):
        reward_name = SERVER_MOVE_RESPONSES[i]
        reward_dict[reward_name] = reward
    return reward_dict
