import json
import logging
import os
import time
import sys
import traceback
import utils.generic_policy as gp

from dataset_agreement_nav_drone.action_space import ActionSpace
from dataset_agreement_nav_drone.metadata_util import MetaDataUtil
from dataset_agreement_nav_drone.nav_drone_dataset_parser import DatasetParser
from learning.single_client.unet_goal_prediction_supervised_from_disk import \
    UnetGoalPredictionSupervisedLearningFromDisk
from models.model.unet_resnet import UnetResnet
from setup_agreement_nav_drone.validate_setup_nav_drone import \
    NavDroneSetupValidator
from utils.tensorboard import Tensorboard

experiment_name = "unet_goal_prediction_resnet_real_data_repeat"
experiment = "./results/" + experiment_name

# Create the experiment folder
if not os.path.exists(experiment):
    os.makedirs(experiment)

# Define log settings
logging.basicConfig(filename=experiment + '/train_goal_prediction.log',
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

try:
    # Create the model
    logging.log(logging.DEBUG, "CREATING MODEL")
    model = UnetResnet(config, constants)
    model.init_weights()
    # model.load_saved_model(
    #     "./results/unet_goal_prediction_resnet_synthetic_repeat/unet_goal_prediction_supervised_epoch_3")
    logging.log(logging.DEBUG, "MODEL CREATED")

    # Tensorboard
    tensorboard = Tensorboard(experiment_name)

    # Read the dataset
    all_train_data = DatasetParser.parse("data/nav_drone/train_annotations_4000.json", config)
    num_train = (len(all_train_data) * 19) // 20
    while all_train_data[num_train].get_scene_name().split("_")[1] == all_train_data[num_train - 1].get_scene_name().split("_")[1]:
       num_train += 1
    train_split = all_train_data[:num_train]
    tune_split = all_train_data[num_train:]

    logging.info("Created train dataset of size %d ", len(train_split))
    logging.info("Created tuning dataset of size %d ", len(tune_split))

    # Read the dataset
    # train_split = tune_split
    train_images, train_goal = UnetGoalPredictionSupervisedLearningFromDisk.parse(
        "../logs/oracle_images/train_images", train_split)
    tune_images, tune_goal = UnetGoalPredictionSupervisedLearningFromDisk.parse(
        "../logs/oracle_images/tune_images", tune_split)

    # Train on this dataset
    learning_alg = UnetGoalPredictionSupervisedLearningFromDisk(model=model,
                                                                action_space=action_space,
                                                                meta_data_util=meta_data_util,
                                                                config=config,
                                                                constants=constants,
                                                                tensorboard=tensorboard)

    print("STARTED LEARNING")
    start = time.time()
    # learning_alg.interactive_shell(tune_split, tune_images)
    learning_alg.do_train(train_split, train_images, train_goal, tune_split, tune_images, tune_goal, experiment)
    end = time.time()
    print("Time taken is " + str(end - start) + " seconds. ")
    logging.info("Taken taken is %r seconds", end - start)

except Exception:
    exc_info = sys.exc_info()
    traceback.print_exception(*exc_info)
    # raise e
