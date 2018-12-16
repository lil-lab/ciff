import json
import logging
import os
import sys
import traceback

import utils.generic_policy as gp
from dataset_agreement_nav_drone.action_space import ActionSpace
from dataset_agreement_nav_drone.metadata_util import MetaDataUtil
from dataset_agreement_nav_drone.nav_drone_dataset_parser import DatasetParser
from learning.single_client.supervised_learning_detect_turning_angle import\
    SupervisedLearningDetectTurningAngle
from models.resnet_image_detection import ResnetImageDetection
from setup_agreement_nav_drone.validate_setup_nav_drone import \
    NavDroneSetupValidator
from utils.tensorboard import Tensorboard

experiment_name = "supervised_turning_angle_ryan_style_v2_and_bigger_resnet_angular_correction"
experiment = "./results/" + experiment_name

# Create the experiment folder
if not os.path.exists(experiment):
    os.makedirs(experiment)

# Define log settings
logging.basicConfig(filename=experiment + '/train_supervised_visible_object_detection.log',
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

try:
    # Create the model
    logging.log(logging.DEBUG, "CREATING MODEL")
    model = ResnetImageDetection(config, constants, None)
    # model.load_saved_model(
    #     "./results/supervised_turning_angle_ryan_style_v2_and_bigger_resnet_contd/symbolic_image_detection_resnet_epoch_100")
    logging.log(logging.DEBUG, "MODEL CREATED")

    # create tensorboard
    tensorboard = Tensorboard(experiment_name)

    # Read the dataset
    train_split = SupervisedLearningDetectTurningAngle.parse("../logs/start_images/train_images")
    tune_split = SupervisedLearningDetectTurningAngle.parse("../logs/start_images/tune_images")

    logging.info("Created train dataset of size %d ", len(train_split))
    logging.info("Created tuning dataset of size %d ", len(tune_split))

    '''all_train_data = DatasetParser.parse("data/nav_drone/train_annotations_4000.json", config)
    num_train = (len(all_train_data) * 19) // 20
    while all_train_data[num_train].get_scene_name().split("_")[1] == \
            all_train_data[num_train - 1].get_scene_name().split("_")[1]:
        num_train += 1
    train_split_real = all_train_data[:num_train]
    tune_split_real = all_train_data[num_train:]'''

    # Train on this dataset
    learning_alg = SupervisedLearningDetectTurningAngle(model=model,
                                                        action_space=action_space,
                                                        meta_data_util=meta_data_util,
                                                        config=config,
                                                        constants=constants,
                                                        tensorboard=tensorboard)
    print "START Image Detection Experiment"
    #learning_alg.test1(tune_split, tune_split_real)
    learning_alg.do_train(train_split, tune_split, experiment)

except Exception:
    exc_info = sys.exc_info()
    traceback.print_exception(*exc_info)
    # raise e
