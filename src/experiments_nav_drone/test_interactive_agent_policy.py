import json
import logging
import os
import sys
import traceback
import utils.generic_policy as gp

from agents.human_controlled_agent import HumanControlledAgent
from dataset_agreement_nav_drone.action_space import ActionSpace
from dataset_agreement_nav_drone.metadata_util import MetaDataUtil
from dataset_agreement_nav_drone.nav_drone_dataset_parser import DatasetParser
from models.incremental_model.incremental_model_chaplot import IncrementalModelChaplot
from models.incremental_model.incremental_model_recurrent_policy_network_only_symbolic_text import\
    IncrementalModelRecurrentPolicyNetworkSymbolicTextResnet
from models.incremental_model.incremental_model_recurrent_policy_network_resnet import \
    IncrementalModelRecurrentPolicyNetworkResnet
from server_nav_drone.nav_drone_server_py3 import NavDroneServerPy3
from setup_agreement_nav_drone.validate_setup_nav_drone import \
    NavDroneSetupValidator
from utils.tensorboard import Tensorboard

experiment = "./results/interactive_agent_policy"

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

with open("data/nav_drone/config_localmoves_4000.json") as f:
    config = json.load(f)
with open("data/shared/contextual_bandit_constants.json") as f:
    constants = json.load(f)
if len(sys.argv) > 1:
    config["port"] = int(sys.argv[1])
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
    # Create the model
    logging.log(logging.DEBUG, "CREATING MODEL")
    model = IncrementalModelChaplot(config, constants)
    model.load_saved_model(
        "./results/asynchronous_contextual_bandit_chaplot_model_horizon45_entropy0.1/contextual_bandit_5_epoch_4")
    logging.log(logging.DEBUG, "MODEL CREATED")

    # Create the agent
    logging.log(logging.DEBUG, "STARTING AGENT")
    agent = HumanControlledAgent(server=server,
                  model=model,
                  test_policy=test_policy,
                  action_space=action_space,
                  meta_data_util=meta_data_util,
                  config=config,
                  constants=constants)

    # create tensorboard
    tensorboard = Tensorboard("dummy")

    # Read the dataset
    # test_dataset = DatasetParser.parse("data/nav_drone/synthetic_dev_annotations_4000.json", config)
    # logging.info("Created test dataset of size %d ", len(test_dataset))

    ############################
    all_train_data = DatasetParser.parse("data/nav_drone/train_annotations_4000.json", config)
    num_train = (len(all_train_data) * 19) // 20
    while all_train_data[num_train].get_scene_name().split("_")[1] == \
            all_train_data[num_train - 1].get_scene_name().split("_")[1]:
        num_train += 1
    train_split = all_train_data[:num_train]
    tune_split = all_train_data[num_train:]
    agent.test(tune_split, tensorboard)
    ############################

    # test on this dataset
    # agent.test(test_dataset, tensorboard)

    server.kill()

except Exception:
    server.kill()
    exc_info = sys.exc_info()
    traceback.print_exception(*exc_info)
    # raise e
