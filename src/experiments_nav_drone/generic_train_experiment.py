import argparse
import json
import logging
import os
import sys

from learning.contextual_bandit_learning import ContextualBandit
from learning.reinforce_learning import ReinforceLearning
from models.model_policy_network import ModelPolicyNetwork

import utils.generic_policy as gp
from dataset_agreement_nav_drone.action_space import ActionSpace
from dataset_agreement_nav_drone.metadata_util import MetaDataUtil
from dataset_agreement_nav_drone.nav_drone_dataset_parser import DatasetParser
from emnlp_model.policy_network import PolicyNetwork as EMNLPPolicyNetwork
from learning.single_client.ml_estimation import MaximumLikelihoodEstimation
from models.model.model_policy_network_resnet import ModelPolicyNetworkResnet
from server_nav_drone.nav_drone_server import NavDroneServer
from setup_agreement_nav_drone.validate_setup_nav_drone import \
    NavDroneSetupValidator
from utils.tensorboard import Tensorboard


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_name", type=str, required=True,
                        help="name of experiment")
    parser.add_argument("--config_file", type=str, required=True,
                        help="path to config file")
    parser.add_argument("--constants_file", type=str, required=True,
                        help="path to constants file")
    parser.add_argument("--model", type=str, required=True,
                        help="which model to run")
    parser.add_argument("--learning", type=str, requied=True,
                        choices=["supervised", "contextual_bandit",
                                 "reinforce"],
                        help="which learning algorithm to run")
    parser.add_argument("--use_emnlp", action="store-true",
                        help="use old EMNLP code")
    parser.add_argument("--paragraphs", action="store_true",
                        help="run on paragraphs instead of segments")
    parser.add_argument("--auto_segment_testing", type=str,
                        choices=["auto", "oracle"],
                        help="do paragraph testing sequentially on segments")
    parser.add_argument("--num_train", type=int,
                        help="restrict number of data points to train on")
    parser.add_argument("--num_test", type=int,
                        help="restrict number of data points to test on")
    parser.add_argument("--test_on_train", action="store_true",
                        help="do testing on train partition of data")
    return parser.parse_args()

def get_model_class(model_name, use_emnlp):
    if model_name == "policy_network" and use_emnlp:
        return EMNLPPolicyNetwork
    elif model_name == "policy_network":
        return ModelPolicyNetwork
    elif model_name == "resnet_policy_network" and not use_emnlp:
        return ModelPolicyNetworkResnet
    else:
        raise ValueError("invalid model %s with use-emnlp=%r"
                         % (model_name, use_emnlp))


def get_learner_class(learner_name):
    if learner_name == "supervised":
        return MaximumLikelihoodEstimation
    elif learner_name == "contextual_bandit":
        return ContextualBandit
    elif learner_name == "reinforce":
        return ReinforceLearning
    else:
        raise ValueError("invalid learner name %s" % learner_name)


args = parse_args()



experiment = "./results/%s" % args.experiment_name

# Create the experiment folder
if not os.path.exists(experiment):
    os.makedirs(experiment)

# Define log settings
logging.basicConfig(filename=experiment + '/train_contextual_bandit.log',
                    level=logging.INFO)
logging.info("----------------------------------------------------------------")
logging.info("                    STARING NEW EXPERIMENT                      ")
logging.info("----------------------------------------------------------------")

# Test policy
test_policy = gp.get_argmax_action

with open("data/nav_drone/test_config.json") as f:
    config = json.load(f)
with open("data/shared/contextual_bandit_constants.json") as f:
    constants = json.load(f)
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
    model = PolicyNetwork(128, 4)
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
    tensorboard = Tensorboard()

    # Read the dataset
    train_dataset = DatasetParser.parse("data/nav_drone/train_annotations.json", config)
    # train_dataset = train_dataset[0:10]
    logging.info("Created train dataset of size %d ", len(train_dataset))
    test_dataset = DatasetParser.parse("data/nav_drone/test_annotations.json", config)
    tune_dataset = test_dataset[0: int(0.05 * len(test_dataset))]
    # tune_dataset = test_dataset[0:10]
    logging.info("Created tuning dataset of size %d ", len(tune_dataset))

    # Train on this dataset
    learning_alg = ContextualBandit(model=model,
                                    action_space=action_space,
                                    meta_data_util=meta_data_util,
                                    config=config,
                                    constants=constants,
                                    tensorboard=tensorboard)

    # Create the session
    logging.log(logging.DEBUG, "CREATING SESSION")
    agent.init_session("./saved_supervised_emnlp_tf/ml_learning_epoch_4.ckpt")
    logging.log(logging.DEBUG, "MODEL SESSION")

    print "START BANDIT"
    learning_alg.do_train(agent, train_dataset, tune_dataset, experiment, agent.sess, agent.train_writer)

    # model.save_model("results/nav_drone/debug_supervised_model")
    server.kill()

except Exception:
    server.kill()
    exc_info = sys.exc_info()
    traceback.print_exception(*exc_info)
    # raise e
