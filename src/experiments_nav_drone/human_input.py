import json
import logging
import sys
import traceback
import os
import utils.generic_policy as gp

from agents.keyboard_agent import KeyboardAgent
from dataset_agreement_nav_drone.nav_drone_dataset_parser import DatasetParser
from dataset_agreement_nav_drone.action_space import ActionSpace
from dataset_agreement_nav_drone.metadata_util import MetaDataUtil
from server_nav_drone.nav_drone_server_py3 import NavDroneServerPy3
from setup_agreement_nav_drone.validate_setup_nav_drone import \
    NavDroneSetupValidator
from utils.check_port import find_k_ports
from utils.launch_unity import launch_k_unity_builds

experiment = "./results/human_performance"

# Create the experiment folder
if not os.path.exists(experiment):
    os.makedirs(experiment)

# Define log settings
which_partition = int(sys.argv[1])
logging.basicConfig(filename=experiment + '/test_human_performance_partition_' + str(which_partition) + '.log',
                    level=logging.INFO)
logging.info("----------------------------------------------------------------")
logging.info("                    STARING NEW EXPERIMENT                      ")
logging.info("----------------------------------------------------------------")

# Test policy
test_policy = gp.get_argmax_action

with open("data/nav_drone/config_localmoves_4000_human_user.json") as f:
    config = json.load(f)
with open("data/shared/contextual_bandit_constants.json") as f:
    constants = json.load(f)

# Find a free port
ports = find_k_ports(1)
config["port"] = ports[0]

# Validate the configuration and constant
setup_validator = NavDroneSetupValidator()
setup_validator.validate(config, constants)

# log core experiment details
logging.info("CONFIG DETAILS")
for k, v in sorted(config.items()):
    logging.info("    %s --- %r" % (k, v))
logging.info("CONSTANTS DETAILS")
for k, v in sorted(constants.items()):
    logging.info("    %s --- %r" % (k, v))
logging.info("START SCRIPT CONTENTS")
with open(__file__) as f:
    for line in f.readlines():
        logging.info(">>> " + line.strip())
logging.info("END SCRIPT CONTENTS")

action_space = ActionSpace(config["action_names"], config["stop_action"])
meta_data_util = MetaDataUtil()

# Create the server
logging.log(logging.DEBUG, "STARTING SERVER")
server = NavDroneServerPy3(config, action_space)
logging.log(logging.DEBUG, "STARTED SERVER")

try:
    # Create the agent
    logging.log(logging.DEBUG, "STARTING AGENT")
    agent = KeyboardAgent(server=server,
                          action_space=action_space,
                          meta_data_util=meta_data_util,
                          config=config,
                          constants=constants)

    # Read the dataset
    all_train_data = DatasetParser.parse("data/nav_drone/dev_annotations_4000.json", config)

    random_indices_string = open("./random_indices.txt").readlines()
    random_indices = [int(v) for v in random_indices_string]
    assert len(random_indices) == 100

    sampled = []
    for random_index in random_indices:
        sampled.append(all_train_data[random_index])

    num_partition = 4
    partition_size = 100//4
    partitions = []
    for i in range(0, num_partition):
        partitions.append(sampled[i * partition_size: (i + 1) * partition_size])

    logging.info("Created sampled dataset of size %d ", len(sampled))

    logging.info("Start Human Study on partition %r of size %r",
                 which_partition, len(partitions[which_partition]))

    launch_k_unity_builds([config["port"]], "./simulators/NavDroneLinuxBuild.x86_64")
    agent.test(partitions[which_partition])

    server.kill()

except Exception:
    server.kill()
    exc_info = sys.exc_info()
    traceback.print_exception(*exc_info)
    # raise e
