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
from learning.single_client.goal_prediction_single_360_image_supervised_from_disk import \
    GoalPredictionSingle360ImageSupervisedLearningFromDisk
from models.incremental_model.incremental_model_attention_chaplot_resnet import IncrementalModelAttentionChaplotResNet
from setup_agreement_nav_drone.validate_setup_nav_drone import \
    NavDroneSetupValidator
from utils.tensorboard import Tensorboard

# experiment_name = "emnlp-camera-ready-center-baseline-goal-prediction-standard-task-completion"
experiment_name = "emnlp-camera-ready-m4jksum1-goal-prediction-standard-task-completion"
# experiment_name = "emnlp-camera-ready-unet-goal-prediction-standard-task-completion"
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

with open("data/nav_drone/config_localmoves_6000.json") as f:
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
    model = IncrementalModelAttentionChaplotResNet(config, constants,
                                                   final_model_type="m4jksum1", #"unet-positional-encoding",
                                                   final_dimension=(64, 32, 32 * 6))
    # model.load_saved_model(
    #     "./results/postbugfix_goal_prediction_360_6000_unet_logprob/goal_prediction_single_supervised_epoch_6")
    model.load_saved_model(
        "./results/goal_prediction_m4jksum1-deadline-emnlp/goal_prediction_single_supervised_epoch_12")
    logging.log(logging.DEBUG, "MODEL CREATED")

    # Tensorboard
    tensorboard = Tensorboard(experiment_name)

    # Read the dataset
    # all_train_data = DatasetParser.parse("data/nav_drone/train_annotations_6000.json", config)
    # num_train = (len(all_train_data) * 19) // 20
    # while all_train_data[num_train].get_scene_name().split("_")[1] == \
    #         all_train_data[num_train - 1].get_scene_name().split("_")[1]:
    #     num_train += 1
    # train_split = all_train_data[:num_train]
    # tune_split = all_train_data[num_train:]
    # logging.info("Created train dataset of size %d ", len(train_split))
    # logging.info("Created tuning dataset of size %d ", len(tune_split))

    dev_data = DatasetParser.parse("data/nav_drone/dev_annotations_6000.json", config)

    # Read the dataset
    # train_images, train_goal = GoalPredictionSingle360ImageSupervisedLearningFromDisk.parse(
    #     "./6000_train_split_start_images", train_split, model, config, format_type="png")
    # tune_images, tune_goal = GoalPredictionSingle360ImageSupervisedLearningFromDisk.parse(
    #     "./6000_tune_split_start_images", tune_split, model, config, format_type="png")
    dev_images, dev_goal = GoalPredictionSingle360ImageSupervisedLearningFromDisk.parse(
        "./6000_dev_split_start_images", dev_data, model, config, format_type="png")

    # Train on this dataset
    learning_alg = GoalPredictionSingle360ImageSupervisedLearningFromDisk(model=model,
                                                                          action_space=action_space,
                                                                          meta_data_util=meta_data_util,
                                                                          config=config,
                                                                          constants=constants,
                                                                          tensorboard=tensorboard)

    print("STARTED LEARNING")
    start = time.time()
    # learning_alg.interactive_shell(tune_split, tune_images)
    # tune_images, tune_goal = GoalPredictionSingle360ImageSupervisedLearningFromDisk.parse(
    #     "./6000_tune_split_start_images", tune_split, model, config, format_type="png")
    learning_alg.test(dev_data, dev_images, dev_goal, tensorboard)
    # learning_alg.do_train(train_split, train_images, train_goal, tune_split, tune_images,
    #                       tune_goal, experiment, save_best_model=True)
    end = time.time()
    print("Time taken is " + str(end - start) + " seconds. ")
    logging.info("Taken taken is %r seconds", end - start)

except Exception:
    exc_info = sys.exc_info()
    traceback.print_exception(*exc_info)
    # raise e
