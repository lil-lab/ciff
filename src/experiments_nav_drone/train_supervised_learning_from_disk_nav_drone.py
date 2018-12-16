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
from models.incremental_model.incremental_model_recurrent_policy_network_resnet import \
    IncrementalModelRecurrentPolicyNetworkResnet
from learning.single_client.supervised_learning_from_disk import SupervisedLearningFromDisk
from setup_agreement_nav_drone.validate_setup_nav_drone import \
    NavDroneSetupValidator
from utils.tensorboard import Tensorboard

experiment_name = "supervised_learning_from_disk_fit_tune_batch8"
experiment = "./results/" + experiment_name

# Create the experiment folder
if not os.path.exists(experiment):
    os.makedirs(experiment)

# Define log settings
logging.basicConfig(filename=experiment + '/train_supervised_learning.log',
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

try:
    # Create the model
    logging.log(logging.DEBUG, "CREATING MODEL")
    model = IncrementalModelRecurrentPolicyNetworkResnet(config, constants)
    model.init_weights()
    model.load_resnet_model(
        "./results/supervised_visible_object_detection_localization_from_drive/symbolic_image_detection_resnet_epoch_84")
    # model.load_lstm_model("./results/train_symbolic_text_prediction_1clock/ml_learning_symbolic_text_prediction_epoch_2")
    logging.log(logging.DEBUG, "MODEL CREATED")

    # create tensorboard
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
    train_split = tune_split
    train_images = SupervisedLearningFromDisk.parse("../logs/oracle_images/tune_images")
    tune_images = SupervisedLearningFromDisk.parse("../logs/oracle_images/tune_images")

    assert len(train_images) == len(train_split)
    assert len(tune_images) == len(tune_split)

    # Train on this dataset
    learning_alg = SupervisedLearningFromDisk(model=model,
                                              action_space=action_space,
                                              meta_data_util=meta_data_util,
                                              config=config,
                                              constants=constants,
                                              tensorboard=tensorboard)

    print "STARTED LEARNING"
    start = time.time()
    learning_alg.do_train(train_split, train_images, tune_split, tune_images, experiment)
    end = time.time()
    print "Time taken is " + str(end - start) + " seconds. "
    logging.info("Taken taken is %r seconds", end - start)

except Exception:
    exc_info = sys.exc_info()
    traceback.print_exception(*exc_info)
    # raise e
