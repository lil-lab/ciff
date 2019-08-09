import json
import logging
import argparse
import os
import sys
import traceback
import utils.generic_policy as gp

from agents.agent_no_model import Agent
from dataset_agreement_nav_drone.action_space import ActionSpace
from dataset_agreement_nav_drone.metadata_util import MetaDataUtil
from dataset_agreement_nav_drone.nav_drone_dataset_parser import DatasetParser
from server_nav_drone.nav_drone_server_py3 import NavDroneServerPy3
from setup_agreement_nav_drone.validate_setup_nav_drone import NavDroneSetupValidator
from utils.check_port import find_k_ports
from utils.launch_unity import launch_k_unity_builds


# Environment arguments
parser = argparse.ArgumentParser(description='LANI Baselines')
parser.add_argument('--baseline',type=str, default="random", help="Three options: stop, random, frequent.")
parser.add_argument('--split', type=int, default="dev", help="Three options: train, dev and test.")
args = parser.parse_args()

experiment = "./results/test_lani_baseline"

# Create the experiment folder
if not os.path.exists(experiment):
    os.makedirs(experiment)

# Define log settings
logging.basicConfig(filename=experiment + '/test_policy.log',
                    level=logging.INFO)
logging.info("----------------------------------------------------------------")
logging.info("                    STARING NEW EXPERIMENT                      ")
logging.info("----------------------------------------------------------------")

# Test policy
test_policy = gp.get_argmax_action

with open("data/nav_drone/config_localmoves_6000.json") as f:
    config = json.load(f)
with open("data/shared/contextual_bandit_constants.json") as f:
    constants = json.load(f)

# Validate the setup
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

# Find a free port
ports = find_k_ports(1)
config["port"] = ports[0]

# Create the server
logging.log(logging.DEBUG, "STARTING SERVER")
server = NavDroneServerPy3(config, action_space)
logging.log(logging.DEBUG, "STARTED SERVER")

try:
    # Create the agent
    logging.log(logging.DEBUG, "STARTING AGENT")

    if args.baseline == "stop":
        agent_type = Agent.STOP
    elif args.baseline == "random":
        agent_type = Agent.RANDOM_WALK
    elif args.baseline == "frequent":
        agent_type = Agent.MOST_FREQUENT
        # TODO compute most frequent action from the dataset
    else:
        raise AssertionError("Unhandled agent type %r. Only support stop, random and frequent." % args.baseline)

    agent = Agent(agent_type=agent_type,
                  server=server,
                  action_space=action_space,
                  meta_data_util=meta_data_util,
                  constants=constants)

    if args.split == "train":
        test_data = DatasetParser.parse("data/nav_drone/train_annotations_6000.json", config)
    elif args.split == "dev":
        test_data = DatasetParser.parse("data/nav_drone/dev_annotations_6000.json", config)
    elif args.split == "test":
        test_data = DatasetParser.parse("data/nav_drone/test_annotations_6000.json", config)
    else:
        raise AssertionError("Unhandled dataset split %r. Only support train, dev and test." % args.split)

    # Launch Unity Build
    launch_k_unity_builds([config["port"]], "./simulators/NavDroneLinuxBuild.x86_64")

    # Test on the dataset
    agent.test(test_data)

    server.kill()

except Exception:
    server.kill()
    exc_info = sys.exc_info()
    traceback.print_exception(*exc_info)
    # raise e
