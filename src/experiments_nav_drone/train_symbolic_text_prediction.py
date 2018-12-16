import json
import logging
import os
import sys
import traceback

import utils.generic_policy as gp
from agents.agent import Agent
from dataset_agreement_nav_drone.action_space import ActionSpace
from dataset_agreement_nav_drone.metadata_util import MetaDataUtil
from dataset_agreement_nav_drone.nav_drone_dataset_parser import DatasetParser
from learning.single_client.symbolic_text_prediction_train_test import SymbolicTextPredictionTrainTest
from models.model.model_symbolic_text_prediction_network import ModelSymbolicTextPrediction
from server_nav_drone.nav_drone_server import NavDroneServer
from setup_agreement_nav_drone.validate_setup_nav_drone import \
    NavDroneSetupValidator
from utils.tensorboard import Tensorboard

experiment = "./results/symbolic_text_prediction_squared_loss_theta"

# Create the experiment folder
if not os.path.exists(experiment):
    os.makedirs(experiment)

# Define log settings
logging.basicConfig(filename=experiment + '/train_symbolic_text_prediction_bucket_30.log',
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
    constants["learning_rate"] = 0.001
    constants["max_epochs"] = 100
if len(sys.argv) > 1:
    config["port"] = int(sys.argv[1])
setup_validator = NavDroneSetupValidator()
setup_validator.validate(config, constants)

# log core experiment details
logging.info("CONFIG DETAILS")
for k, v in sorted(config.iteritems()):
    logging.info("    %s --- %r" % (k, v))
logging.info("CONSTANTS DETAILS")
for k, v in sorted(constants.iteritems()):
    logging.info("    %s --- %r" % (k, v))
logging.info("START SCRIPT CONTENTS")
with open(__file__) as f:
    for line in f.xreadlines():
        logging.info(">>> " + line.strip())
logging.info("END SCRIPT CONTENTS")

action_space = ActionSpace(config["action_names"], config["stop_action"])
meta_data_util = MetaDataUtil()

# Create the server
logging.log(logging.DEBUG, "STARTING SERVER")
server = NavDroneServer(config, action_space)
logging.log(logging.DEBUG, "STARTED SERVER")

try:
    # Create the model
    logging.log(logging.DEBUG, "CREATING MODEL")
    model = ModelSymbolicTextPrediction(config, constants)
    # model.load_saved_model("./results/train_symbolic_text_prediction_1clock/ml_learning_symbolic_text_prediction_epoch_3")
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
    tensorboard = Tensorboard("synthetic_easy_text_prediction")

    # Read the dataset
    all_train_data = DatasetParser.parse("data/nav_drone/train_annotations_4000.json", config)
    num_train = (len(all_train_data) * 19) // 20
    while all_train_data[num_train].get_scene_name().split("_")[1] == all_train_data[num_train - 1].get_scene_name().split("_")[1]:
       num_train += 1
    train_split = all_train_data[:num_train]
    tune_split = all_train_data[num_train:]

    logging.info("Created train dataset of size %d ", len(train_split))
    logging.info("Created tuning dataset of size %d ", len(tune_split))

    # Train on this dataset
    learning_alg = SymbolicTextPredictionTrainTest(model=model,
                                            action_space=action_space,
                                            meta_data_util=meta_data_util,
                                            config=config,
                                            constants=constants,
                                            tensorboard=tensorboard)

    # learning_alg.dump_data(train_split, tune_split)
    learning_alg.do_train(agent, train_split, tune_split, experiment)
    # learning_alg.test_classifier(agent, tune_split)
    # model.save_model("results/nav_drone/debug_supervised_model")
    server.kill()

except Exception:
    server.kill()
    exc_info = sys.exc_info()
    traceback.print_exception(*exc_info)
    # raise e
