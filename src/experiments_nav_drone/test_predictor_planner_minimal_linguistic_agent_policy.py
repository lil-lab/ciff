import json
import logging
import os
import sys
import traceback

from agents.predicter_planner_agent import PredictorPlannerAgent
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

experiment_name = "tmp_test_predictor_planner_minimal_linguistic_pairs_evaluation"
experiment = "./results/" + experiment_name

# Create the experiment folder
if not os.path.exists(experiment):
    os.makedirs(experiment)

# Define log settings
logging.basicConfig(filename=experiment + '/test_minimal_linguistic_policy.log',
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

    predictor_model = IncrementalModelAttentionChaplotResNet(config, constants,
                                                             final_model_type="unet-positional-encoding",
                                                             final_dimension=(64, 32, 32 * 6))
    predictor_model.load_saved_model(
        "./results/postbugfix_goal_prediction_360_6000_unet_logprob/goal_prediction_single_supervised_epoch_6")

    planner_model = IncrementalModelOracleGoldProb(config, constants)
    planner_model.load_saved_model("./results/oracle_gold_prob_cb_6000/contextual_bandit_5_epoch_17")

    logging.log(logging.DEBUG, "MODEL CREATED")

    # Create the agent
    logging.log(logging.DEBUG, "STARTING AGENT")
    agent = PredictorPlannerAgent(server=server,
                                  predictor_model=predictor_model,
                                  model=planner_model,
                                  test_policy=test_policy,
                                  action_space=action_space,
                                  meta_data_util=meta_data_util,
                                  config=config,
                                  constants=constants)

    # create tensorboard
    tensorboard = Tensorboard(experiment_name)

    # Launch Unity Build
    launch_k_unity_builds([config["port"]], "./simulators/NavDroneLinuxBuild.x86_64")

    # Read the minimally linguistic pair dataset
    pair1 = DatasetParser.parse("data/nav_drone/synthetic_v2_10k_dev_vocab_fix.json", config)
    pair2 = DatasetParser.parse("data/nav_drone/synthetic_v2_10k_minlin_dev_vocab_fix.json", config)

    assert len(pair1) == len(pair2), "Lingusitic pairs dataset should be of same size"

    print("Read dataset of size ", len(pair1))
    zipped_pairs = zip(pair1, pair2)

    # Test
    agent.test_minimal_linguistic_pairs(zipped_pairs, tensorboard)

    server.kill()

except Exception:
    server.kill()
    exc_info = sys.exc_info()
    traceback.print_exception(*exc_info)
    # raise e
