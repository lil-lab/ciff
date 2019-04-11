import json
import logging
import os
import sys
import time
import traceback
import argparse

import utils.generic_policy as gp
from agents.agent import Agent
from baselines.chaplot_model_concat_gavector import a3c_lstm_ga_concat_gavector
from baselines.chaplot_model_default import a3c_lstm_ga_default
from dataset_agreement_nav_drone.action_space import ActionSpace
from dataset_agreement_nav_drone.metadata_util import MetaDataUtil
from dataset_agreement_nav_drone.nav_drone_dataset_parser import DatasetParser
from server_nav_drone.nav_drone_server_py3 import NavDroneServerPy3
from setup_agreement_nav_drone.validate_setup_nav_drone import \
    NavDroneSetupValidator

from baselines.chaplot_baseline import ChaplotBaseline
from utils.check_port import find_k_ports
from utils.launch_unity import launch_k_unity_builds

experiment_name = "2000examples_horizon30_entropy0.11_epoch10"
experiment = "./results/" + experiment_name
print("EXPERIMENT NAME: ", experiment_name)

supervised = False

# Create the experiment folder
if not os.path.exists(experiment):
    os.makedirs(experiment)

# Define log settings
logging.basicConfig(filename=experiment + '/test_chaplot_baseline.log',
                    level=logging.INFO)
logging.info("----------------------------------------------------------------")
logging.info("                    STARING NEW EXPERIMENT                      ")
logging.info("----------------------------------------------------------------")

# Test policy
test_policy = gp.get_argmax_action


##########################################################################################################
##########################################################################################################
##########################################################################################################
parser = argparse.ArgumentParser(description='Gated-Attention for Grounding')

# Environment arguments
parser.add_argument('port',type=int,default=12345)
parser.add_argument('-l', '--max-episode-length', type=int, default=40,
                    help='maximum length of an episode (default: 40)')
parser.add_argument('-d', '--difficulty', type=str, default="hard",
                    help="""Difficulty of the environment,
                    "easy", "medium" or "hard" (default: hard)""")
parser.add_argument('--living-reward', type=float, default=0,
                    help="""Default reward at each time step (default: 0,
                    change to -0.005 to encourage shorter paths)""")
parser.add_argument('--frame-width', type=int, default=300,
                    help='Frame width (default: 300)')
parser.add_argument('--frame-height', type=int, default=168,
                    help='Frame height (default: 168)')
parser.add_argument('-v', '--visualize', type=int, default=0,
                    help="""Visualize the envrionment (default: 0,
                    use 0 for faster training)""")
parser.add_argument('--sleep', type=float, default=0,
                    help="""Sleep between frames for better
                    visualization (default: 0)""")
parser.add_argument('--scenario-path', type=str, default="maps/room.wad",
                    help="""Doom scenario file to load
                    (default: maps/room.wad)""")
parser.add_argument('--interactive', type=int, default=0,
                    help="""Interactive mode enables human to play
                    (default: 0)""")
parser.add_argument('--all-instr-file', type=str,
                    default="data/instructions_all.json",
                    help="""All instructions file
                    (default: data/instructions_all.json)""")
parser.add_argument('--train-instr-file', type=str,
                    default="data/instructions_train.json",
                    help="""Train instructions file
                    (default: data/instructions_train.json)""")
parser.add_argument('--test-instr-file', type=str,
                    default="data/instructions_test.json",
                    help="""Test instructions file
                    (default: data/instructions_test.json)""")
parser.add_argument('--object-size-file', type=str,
                    default="data/object_sizes.txt",
                    help='Object size file (default: data/object_sizes.txt)')

# A3C arguments
parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                    help='learning rate (default: 0.001)')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor for rewards (default: 0.99)')
parser.add_argument('--tau', type=float, default=1.00, metavar='T',
                    help='parameter for GAE (default: 1.00)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('-n', '--num-processes', type=int, default=4, metavar='N',
                    help='how many training processes to use (default: 4)')
parser.add_argument('--num-steps', type=int, default=20, metavar='NS',
                    help='number of forward steps in A3C (default: 20)')
parser.add_argument('--load', type=str, default="0",
                    help='model path to load, 0 to not reload (default: 0)')
parser.add_argument('-e', '--evaluate', type=int, default=0,
                    help="""0:Train, 1:Evaluate MultiTask Generalization
                    2:Evaluate Zero-shot Generalization (default: 0)""")
parser.add_argument('--dump-location', type=str, default="./saved/",
                    help='path to dump models and log (default: ./saved/)')

args = parser.parse_args()

print(args)
##########################################################################################################
##########################################################################################################
##########################################################################################################

with open("data/nav_drone/config_localmoves_6000.json") as f:
    config = json.load(f)
with open("data/shared/contextual_bandit_constants.json") as f:
    constants = json.load(f)
print(json.dumps(config, indent=2))
setup_validator = NavDroneSetupValidator()
setup_validator.validate(config, constants)

args.input_size = config['vocab_size'] + 2
config['num_client'] = 1

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

# Find a free port
ports = find_k_ports(1)
config["port"] = ports[0]

action_space = ActionSpace(config["action_names"], config["stop_action"])
meta_data_util = MetaDataUtil()

# Create the server
logging.log(logging.DEBUG, "STARTING SERVER")
server = NavDroneServerPy3(config, action_space)
logging.log(logging.DEBUG, "STARTED SERVER")
print("Launched Server...")

try:
    # create tensorboard
    tensorboard = None  # Tensorboard(experiment_name)

    # Create the model
    logging.log(logging.DEBUG, "CREATING MODEL")
    shared_model = a3c_lstm_ga_default(args, config=config)
    lstm_size = 256
    if isinstance(shared_model, a3c_lstm_ga_concat_gavector):
        lstm_size *= 3
    model = ChaplotBaseline(args, shared_model, config, constants, tensorboard, use_contextual_bandit=False,
                            lstm_size=lstm_size)
    model.load_saved_model("./results/folder-name/chaplot_model_client_0_epoch_160")
    logging.log(logging.DEBUG, "MODEL CREATED")
    print("Created Model...")

    # Create the agent
    logging.log(logging.DEBUG, "STARTING AGENT")
    agent = Agent(server=server,
                  model=model,
                  test_policy=test_policy,
                  action_space=action_space,
                  meta_data_util=meta_data_util,
                  config=config,
                  constants=constants)
    print("Created Agent...")

    # Read the dataset
    test_dataset = DatasetParser.parse("data/nav_drone/dev_annotations_6000.json", config)
    # num_train = (len(all_train_data) * 19) // 20
    # while all_train_data[num_train].get_scene_name().split("_")[1] == all_train_data[num_train - 1].get_scene_name().split("_")[1]:
    #    num_train += 1
    # train_split = all_train_data[:num_train]
    # tune_split = all_train_data[num_train:]

    # tune_split = tune_split[0:20]
    #
    # logging.info("Created train dataset of size %d ", len(train_split))
    # logging.info("Created tuning dataset of size %d ", len(tune_split))
    logging.info("Created tuning dataset of size %d ", len(test_dataset))

    # Launch the server
    launch_k_unity_builds([config["port"]], "./simulators/NavDroneLinuxBuild.x86_64")

    # Test agent
    print("Testing Agent...")
    start = time.time()
    agent.test(test_dataset, tensorboard)
    end = time.time()
    print ("Tested in time(s): ", end - start)

    server.kill()

except Exception:
    server.kill()
    exc_info = sys.exc_info()
    traceback.print_exception(*exc_info)
    # raise e
