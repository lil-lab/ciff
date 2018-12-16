import json
import logging
import os
import sys
import traceback

import utils.generic_policy as gp
from agents.two_stage_agent import TwoStageAgent
from dataset_agreement_nav_drone.action_space import ActionSpace
from dataset_agreement_nav_drone.metadata_util import MetaDataUtil
from dataset_agreement_nav_drone.nav_drone_dataset_parser import DatasetParser
from learning.single_client.supervised_learning_detect_turning_angle import SupervisedLearningDetectTurningAngle
from learning.single_client.symbolic_text_prediction_with_images_train_test import \
    SymbolicTextPredictionWithImagesTrainTest
from models.incremental_model.incremental_model_recurrent_policy_network_only_symbolic_text import\
    IncrementalModelRecurrentPolicyNetworkSymbolicTextResnet
from models.incremental_model.incremental_model_recurrent_policy_network_resnet import \
    IncrementalModelRecurrentPolicyNetworkResnet
from models.model.model_symbolic_text_prediction_network import ModelSymbolicTextPrediction
from models.model.model_symbolic_text_prediction_with_images_network import ModelSymbolicTextPredictionWithImages
from models.resnet_image_detection import ResnetImageDetection
from server_nav_drone.nav_drone_server import NavDroneServer
from setup_agreement_nav_drone.validate_setup_nav_drone import \
    NavDroneSetupValidator
from utils.tensorboard import Tensorboard

experiment = "./results/two_stage_model_theta_prediction_angle30_from_turning_angle_model"

# Create the experiment folder
if not os.path.exists(experiment):
    os.makedirs(experiment)

# Define log settings
logging.basicConfig(filename=experiment + '/test_two_stage_policy.log',
                    level=logging.INFO)
logging.info("----------------------------------------------------------------")
logging.info("                    STARING NEW EXPERIMENT                      ")
logging.info("----------------------------------------------------------------")

# Test policy
test_policy = gp.get_argmax_action

with open("data/nav_drone/config_localmoves_4000.json") as f:
    config = json.load(f)
with open("data/shared/full_recurrence_contextual_bandit_constants.json") as f:
    constants = json.load(f)
    constants["learning_rate"] = .0001
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
    angle_prediction_model = ResnetImageDetection(config, constants, None)
    angle_prediction_model.load_saved_model(
        "./results/supervised_turning_angle_ryan_style_v2_and_bigger_resnet_contd/symbolic_image_detection_resnet_epoch_73")
    symbolic_text_model = ModelSymbolicTextPredictionWithImages(config, constants)
    # symbolic_text_model.load_saved_model(
    #     "./results/max_symbolic_text_classification_with_images_theta_7.5_lr_.0001/ml_learning_symbolic_text_prediction_epoch_4")
    model = IncrementalModelRecurrentPolicyNetworkSymbolicTextResnet(config, constants)
    model.load_saved_model(
        "./results/symbolic_language_real_image_goal_expected_reward_discrete_30/expected_reward_epoch_9")
    logging.log(logging.DEBUG, "MODEL CREATED")

    # Create the agent
    logging.log(logging.DEBUG, "STARTING AGENT")
    agent = TwoStageAgent(server=server,
                          model=model,
                          symbolic_text_model=symbolic_text_model,
                          angle_prediction_model=angle_prediction_model,
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

    # Read the dataset
    train_images = SupervisedLearningDetectTurningAngle.parse("../logs/start_images/train_images")
    tune_images = SupervisedLearningDetectTurningAngle.parse("../logs/start_images/tune_images")

    ############################
    all_train_data = DatasetParser.parse("data/nav_drone/train_annotations_4000.json", config)
    num_train = (len(all_train_data) * 19) // 20
    while all_train_data[num_train].get_scene_name().split("_")[1] == \
            all_train_data[num_train - 1].get_scene_name().split("_")[1]:
        num_train += 1
    train_split = all_train_data[:num_train]
    tune_split = all_train_data[num_train:]
    agent.test(tune_split, tune_images, tensorboard)
    ############################

    # test on this dataset
    # agent.test(test_dataset, tensorboard)

    server.kill()

except Exception:
    server.kill()
    exc_info = sys.exc_info()
    traceback.print_exception(*exc_info)
    # raise e
