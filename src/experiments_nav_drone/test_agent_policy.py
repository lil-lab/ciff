import json
import logging
import os
import sys
import traceback

from models.incremental_model.incremental_ensemble_of_models import IncrementalEnsembleOfModels
from models.incremental_model.incremental_model_attention_chaplot_resnet import IncrementalModelAttentionChaplotResNet
from models.incremental_model.incremental_model_oracle_gold_prob import IncrementalModelOracleGoldProb
from server_nav_drone.nav_drone_server_py3 import NavDroneServerPy3

import utils.generic_policy as gp
from agents.agent import Agent
from dataset_agreement_nav_drone.action_space import ActionSpace
from dataset_agreement_nav_drone.metadata_util import MetaDataUtil
from dataset_agreement_nav_drone.nav_drone_dataset_parser import DatasetParser
from models.incremental_model.incremental_model_recurrent_policy_network_only_symbolic_text import\
    IncrementalModelRecurrentPolicyNetworkSymbolicTextResnet
from models.incremental_model.incremental_model_recurrent_policy_network_resnet import \
    IncrementalModelRecurrentPolicyNetworkResnet
from server_nav_drone.nav_drone_server import NavDroneServer
from setup_agreement_nav_drone.validate_setup_nav_drone import \
    NavDroneSetupValidator
from utils.check_port import find_k_ports
from utils.launch_unity import launch_k_unity_builds
from utils.tensorboard import Tensorboard

experiment = "./results/test_dataset_oracle_gold_predictions"

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
# if len(sys.argv) > 1:
#     config["port"] = int(sys.argv[1])
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
    # Create the model
    logging.log(logging.DEBUG, "CREATING MODEL")

    model = IncrementalModelOracleGoldProb(config, constants, use_image=False)
    model.load_saved_model("./results/oracle_gold_prob_cb_6000/contextual_bandit_5_epoch_17")

    logging.log(logging.DEBUG, "MODEL CREATED")

    # Create the agent
    logging.log(logging.DEBUG, "STARTING AGENT")
    agent = Agent(server=server,
                  model=model,
                  test_policy=test_policy,
                  action_space=action_space,
                  meta_data_util=meta_data_util,
                  config=config,
                  constants=constants)

    # create tensorboard
    tensorboard = Tensorboard("dummy")

    # Launch Unity Build
    launch_k_unity_builds([config["port"]], "./simulators/NavDroneLinuxBuild.x86_64")

    test_data = DatasetParser.parse("data/nav_drone/dev_annotations_6000.json", config)
    agent.test(test_data, tensorboard)

    server.kill()

except Exception:
    server.kill()
    exc_info = sys.exc_info()
    traceback.print_exception(*exc_info)
    # raise e
