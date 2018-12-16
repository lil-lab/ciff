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
from learning.single_client.symbolic_text_prediction_with_images_train_test import \
    SymbolicTextPredictionWithImagesTrainTest
from models.model.model_symbolic_text_prediction_network import ModelSymbolicTextPrediction
from models.model.model_symbolic_text_prediction_with_images_network import ModelSymbolicTextPredictionWithImages
from models.resnet_image_detection import ResnetImageDetection
from server_nav_drone.nav_drone_server import NavDroneServer
from setup_agreement_nav_drone.validate_setup_nav_drone import \
    NavDroneSetupValidator
from utils.tensorboard import Tensorboard

experiment_name = "landmark_prediction_from_segment_with_log"
experiment = "./results/" + experiment_name

# Create the experiment folder
if not os.path.exists(experiment):
    os.makedirs(experiment)

# Define log settings
logging.basicConfig(filename=experiment + '/train_symbolic_text_prediction.log',
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
    constants["learning_rate"] = 0.0005
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
    model = ModelSymbolicTextPredictionWithImages(config, constants)
    # model.load_saved_model("./results/train_symbolic_text_prediction_1clock/ml_learning_symbolic_text_prediction_epoch_3")
    # model.load_resnet_model(
    #     "./results/supervised_visible_object_detection_localization_from_drive/symbolic_image_detection_resnet_epoch_84")
    resnet_detection_model = None
    # resnet_detection_model = ResnetImageDetection(config, constants, None)
    # resnet_detection_model.load_saved_model(
    #     "./results/supervised_visible_object_detection_localization_from_drive/symbolic_image_detection_resnet_epoch_84")
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
    tensorboard = Tensorboard(experiment_name)

    # Read the dataset
    all_train_data = DatasetParser.parse("data/nav_drone/train_annotations_4000.json", config)
    num_train = (len(all_train_data) * 19) // 20
    while all_train_data[num_train].get_scene_name().split("_")[1] == all_train_data[num_train - 1].get_scene_name().split("_")[1]:
       num_train += 1
    train_split = all_train_data[:num_train]
    tune_split = all_train_data[num_train:]

    # Read the dataset
    train_images = SymbolicTextPredictionWithImagesTrainTest.parse("../logs/oracle_images/train_images")
    tune_images = SymbolicTextPredictionWithImagesTrainTest.parse("../logs/oracle_images/tune_images")

    assert len(train_images) == len(train_split)
    assert len(tune_images) == len(tune_split)

    logging.info("Created train dataset of size %d ", len(train_split))
    logging.info("Created tuning dataset of size %d ", len(tune_split))

    # Train on this dataset
    learning_alg = SymbolicTextPredictionWithImagesTrainTest(model=model,
                                                             action_space=action_space,
                                                             meta_data_util=meta_data_util,
                                                             config=config,
                                                             constants=constants,
                                                             tensorboard=tensorboard,
                                                             resnet_detection_model=resnet_detection_model)

    learning_alg.do_train(agent, train_split, tune_split, train_images, tune_images, experiment)
    server.kill()

except Exception:
    server.kill()
    exc_info = sys.exc_info()
    traceback.print_exception(*exc_info)
    # raise e
