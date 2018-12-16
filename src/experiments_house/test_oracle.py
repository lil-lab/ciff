import json
import logging
import os
import sys

import utils.generic_policy as gp
from agents.agent_no_model import Agent
from dataset_agreement_house.action_space import ActionSpace
from dataset_agreement_house.dataset_parser import DatasetParser
from dataset_agreement_house.metadata_util import MetaDataUtil
from server_house.house_server import HouseServer
from utils.check_port import find_k_ports
from utils.launch_unity import launch_k_unity_builds
# from utils.tensorboard import Tensorboard

experiment = "./results/house_test_oracle"

# Create the experiment folder
if not os.path.exists(experiment):
    os.makedirs(experiment)

# Define log settings
logging.basicConfig(filename=experiment + '/test_oracle.log',
                    level=logging.INFO)
logging.info("----------------------------------------------------------------")
logging.info("                    STARING NEW EXPERIMENT                      ")
logging.info("----------------------------------------------------------------")

# Test policy
test_policy = gp.get_argmax_action

with open("data/house/config.json") as f:
    config = json.load(f)
with open("data/shared/contextual_bandit_constants.json") as f:
    constants = json.load(f)

num_processes = 1
ports = find_k_ports(num_processes)

# log core experiment details
logging.info("CONFIG DETAILS")
for k, v in sorted(list(config.items())):
    logging.info("    %s --- %r" % (k, v))
logging.info("CONSTANTS DETAILS")
for k, v in sorted(list(constants.items())):
    logging.info("    %s --- %r" % (k, v))
logging.info("START SCRIPT CONTENTS")
with open(__file__) as f:
    for line in f.readlines():
        logging.info(">>> " + line.strip())
logging.info("END SCRIPT CONTENTS")

act_space = ActionSpace(config["action_names"], config["stop_action"])
meta_data_util = MetaDataUtil()

# Create the server
logging.log(logging.DEBUG, "STARTING SERVER")
server = HouseServer(config, act_space, ports[0])
logging.log(logging.DEBUG, "STARTED SERVER")

# Launch the build
launch_k_unity_builds([ports[0]], "./simulators/house_3_elmer.x86_64")
# Launched the build
server.connect()

# Create the agent
logging.log(logging.DEBUG, "STARTING AGENT")
agent = Agent(Agent.ORACLE, server, act_space, meta_data_util)

# Read the house dataset
dev_dataset = DatasetParser.parse("data/house/dataset/house_3_dev.json", config)
logging.info("Created test dataset of size %d ", len(dev_dataset))

# Test on this dataset
agent.test(dev_dataset)

server.kill()
