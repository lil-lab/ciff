import json
import logging
import os
import sys
import traceback

from dataset_agreement_streetview.action_space import ActionSpace
from dataset_agreement_streetview.dataset_parser import DatasetParser
from server_streetview.streetview_server import StreetViewServer
from setup_agreement_streetview.validate_setup_streetview import StreetViewSetupValidator
from utils.multiprocess_logger import MultiprocessingLoggerManager


def main():

    experiment_name = "debug_oracle_trajectory"
    experiment = "./results/" + experiment_name
    print("EXPERIMENT NAME: ", experiment_name)

    # Create the experiment folder
    if not os.path.exists(experiment):
        os.makedirs(experiment)

    # Define log settings
    log_path = experiment + '/train_baseline.log'
    multiprocess_logging_manager = MultiprocessingLoggerManager(
        file_path=log_path, logging_level=logging.INFO)
    master_logger = multiprocess_logging_manager.get_logger("Master")
    master_logger.log("----------------------------------------------------------------")
    master_logger.log("                    STARING NEW EXPERIMENT                      ")
    master_logger.log("----------------------------------------------------------------")

    with open("data/streetview/config.json") as f:
        config = json.load(f)
    with open("data/shared/contextual_bandit_constants.json") as f:
        constants = json.load(f)
    print(json.dumps(config,indent=2))
    setup_validator = StreetViewSetupValidator()
    setup_validator.validate(config, constants)

    # log core experiment details
    master_logger.log("CONFIG DETAILS")
    for k, v in sorted(config.items()):
        master_logger.log("    %s --- %r" % (k, v))
    master_logger.log("CONSTANTS DETAILS")
    for k, v in sorted(constants.items()):
        master_logger.log("    %s --- %r" % (k, v))
    master_logger.log("START SCRIPT CONTENTS")
    with open(__file__) as f:
        for line in f.readlines():
            master_logger.log(">>> " + line.strip())
    master_logger.log("END SCRIPT CONTENTS")

    action_space = ActionSpace(config["action_names"], config["stop_action"])

    try:

        # Read the dataset
        train_split = DatasetParser.parse("data/streetview/navigation_train.json", config)
        server = StreetViewServer(config, action_space, forward_setting_strict=False)

        for data_point_ix, data_point in enumerate(train_split):

            _, metadata = server.reset_receive_feedback(data_point)
            trajectory = server.get_trajectory_exact(data_point.trajectory)
            trajectory = trajectory[:min(len(trajectory), constants["horizon"])]
            traj_node_ids = [server.fsa.panorama_to_node_dict[pano_id] for pano_id in data_point.trajectory]
            total_reward = 0

            master_logger.log("Route ID: %r " % traj_node_ids)
            node_ix = 0

            for action in trajectory:
                route_id = traj_node_ids[node_ix]
                _, reward, metadata = server.send_action_receive_feedback(action)
                total_reward += reward
                master_logger.log("Reward %r, Action %r, Metadata %r" % (reward, action, metadata))

                # current node id should be either same or next
                if route_id != metadata["panorama_id"]:  # hopefully updated
                    if node_ix >= len(traj_node_ids) - 1:
                        master_logger.log("Failed. Went to a node beyond the trajectory")
                        raise AssertionError()
                    elif traj_node_ids[node_ix + 1] != metadata["panorama_id"]:
                        master_logger.log("Supposed to go to %r but went to %r " %
                                          (traj_node_ids[node_ix + 1], metadata["panorama_id"]))
                        raise AssertionError()
                    else:
                        node_ix += 1

            _, reward, metadata = server.halt_and_receive_feedback()
            total_reward += reward
            master_logger.log("Reward %r, Action stop, Metadata %r" % (reward, metadata))
            master_logger.log("Total reward %r, Nav Error %r " % (total_reward, metadata["navigation_error"]))

    except Exception:
        exc_info = sys.exc_info()
        traceback.print_exception(*exc_info)
        # raise e


if __name__ == "__main__":
    print("SETTING THE START METHOD ")
    main()
