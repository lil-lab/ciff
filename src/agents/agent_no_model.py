import random
import logging

import time

from server_house.house_server import HouseServer
from utils.launch_unity import launch_k_unity_builds


class Agent:

    STOP, ORACLE, RANDOM_WALK, MOST_FREQUENT = range(4)

    def __init__(self, agent_type, server, action_space, meta_data_util, constants):
        self.agent_type = agent_type
        self.server = server
        self.action_space = action_space
        self.meta_data_util = meta_data_util
        self.constants = constants

    def test(self, test_dataset):

        max_num_actions = self.constants["horizon"]

        metadata = {}
        for data_point in test_dataset:
            image, metadata = self.server.reset_receive_feedback(data_point)
            action_seq = data_point.get_trajectory()
            act_idx = 0
            num_actions = 0

            while True:

                if self.agent_type == Agent.STOP:
                    action = self.action_space.get_stop_action_index()
                elif self.agent_type == Agent.RANDOM_WALK:
                    actions = list(range(0, self.action_space.num_actions()))
                    # actions.remove(self.action_space.get_stop_action_index())
                    action = random.choice(actions)
                elif self.agent_type == Agent.ORACLE:
                    if act_idx == len(action_seq):
                        action = self.action_space.get_stop_action_index()
                    else:
                        action = action_seq[act_idx]
                        act_idx += 1
                elif self.agent_type == Agent.MOST_FREQUENT:
                    action = 0  # Assumes that most frequent action is the first action
                else:
                    raise AssertionError("Unknown type " + self.agent_type)

                if action == self.action_space.get_stop_action_index() or num_actions >= max_num_actions:
                    # Send the action and get feedback
                    image, reward, metadata = self.server.halt_and_receive_feedback()

                    # Update the scores based on meta_data
                    self.meta_data_util.log_results(metadata)
                    break
                else:
                    # Send the action and get feedback
                    image, reward, metadata = self.server.send_action_receive_feedback(action)
                    num_actions += 1

        logging.info("Final Result on %d test examples " % (len(test_dataset)))
        logging.info("Final Metadata: ", metadata)

    @staticmethod
    def test_multiprocess(house_id, test_dataset, config, action_space, port, agent_type,
                          meta_data_util, constants, vocab, logger, pushover_logger=None):

        # start the python client
        logger.log("In Testing...")
        launch_k_unity_builds([config["port"]], "./house_" + str(house_id) + "_elmer.x86_64",
                              arg_str="--config ./AssetsHouse/config" + str(house_id) + ".json",
                              cwd="./simulators/house/")
        logger.log("Launched Builds")

        # start the server
        server = HouseServer(config, action_space, port)
        server.initialize_server()
        server.clear_metadata()
        logger.log("Server Initialized...")

        max_num_actions = constants["horizon"]
        task_completion_accuracy = 0
        metadata = {"feedback": ""}
        action_counts = [0] * action_space.num_actions()

        for data_point in test_dataset:
            image, metadata = server.reset_receive_feedback(data_point)
            action_seq = data_point.get_trajectory()
            act_idx = 0
            num_actions = 0
            instruction_string = " ".join([vocab[token_id] for token_id in data_point.instruction])
            Agent.log("Instruction is %r " % instruction_string, logger)
            while True:

                if agent_type == Agent.STOP:
                    action = action_space.get_stop_action_index()
                elif agent_type == Agent.RANDOM_WALK:
                    actions = list(range(0, action_space.num_actions()))
                    # actions.remove(action_space.get_stop_action_index())
                    action = random.choice(actions)
                elif agent_type == Agent.ORACLE:
                    if act_idx == len(action_seq):
                        action = action_space.get_stop_action_index()
                    else:
                        action = action_seq[act_idx]
                        act_idx += 1
                elif agent_type == Agent.MOST_FREQUENT:
                    action = 0  # Assumes that most frequent action is the first action
                else:
                    raise AssertionError("Unknown type " + agent_type)

                if action == action_space.get_stop_action_index() or num_actions >= max_num_actions:
                    # Send the action and get feedback
                    image, reward, metadata = server.halt_and_receive_feedback()
                    action_counts[action_space.get_stop_action_index()] += 1

                    if metadata["navigation-error"] <= 1.0:
                        task_completion_accuracy += 1

                    # Update the scores based on meta_data
                    meta_data_util.log_results(metadata)
                    Agent.log(metadata, logger)
                    break
                else:
                    # Send the action and get feedback
                    image, reward, metadata = server.send_action_receive_feedback(action)
                    action_counts[action] += 1
                    num_actions += 1

        task_completion_accuracy = (task_completion_accuracy * 100.0) / float(max(len(test_dataset), 1))
        Agent.log("House %r Overall test results:" % house_id, logger)
        Agent.log("House %r Test Data Size %r:" % (house_id, len(test_dataset)), logger)
        Agent.log("House %r Overall mean navigation error %r:" % (house_id, metadata["mean-navigation-error"]), logger)
        Agent.log("House %r Testing: Final Metadata: %r" % (house_id, metadata), logger)
        Agent.log("House %r Testing: Action Distribution: %r" % (house_id, action_counts), logger)
        Agent.log("House %r Testing: Manipulation Accuracy: %r " %
                  (house_id, metadata["mean-manipulation-accuracy"]), logger)
        Agent.log("House %r Testing: Navigation Accuracy: %r " % (house_id, task_completion_accuracy), logger)
        # self.meta_data_util.log_results(metadata, logger)
        Agent.log("House %r Testing data action counts %r" % (house_id, action_counts), logger)
        if pushover_logger is not None:
            pushover_feedback = str(metadata["feedback"])
            pushover_logger.log(pushover_feedback)

    @staticmethod
    def log(message, logger=None):
        if logger is not None:
            logger.log(message)
        else:
            logging.info(message)

