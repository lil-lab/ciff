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
from learning.single_client.supervised_learning_detect_visible_object import SupervisedLearningDetectVisibleObject
from models.resnet_image_detection import ResnetImageDetection
from models.standard_resnet_image_detection import StandardResnetImageDetection
from server_nav_drone.nav_drone_server import NavDroneServer
from setup_agreement_nav_drone.validate_setup_nav_drone import \
    NavDroneSetupValidator
from utils.tensorboard import Tensorboard

experiment = "./results/supervised_visible_object_detection_train_all_param"
experiment = "./results/dummy_visible"

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
    # model = ResnetImageDetection(config, constants,
    #                              "./results/distorangle_segmentlevel_symbolic_text_real_image_cb_cross_entropy/contextual_bandit_resnet_epoch_4/image_module_state.bin")
    # model = ResnetImageDetection(config, constants, None)
    model = StandardResnetImageDetection(config, constants, None)
    # model.load_saved_model("./results/supervised_visible_object_detection_train_all_param/object_detection_resnet_epoch_4")
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
    all_train_data = DatasetParser.parse("data/nav_drone/train_annotations_4000.json", config)
    num_train = (len(all_train_data) * 19) // 20
    while all_train_data[num_train].get_scene_name().split("_")[1] == all_train_data[num_train - 1].get_scene_name().split("_")[1]:
       num_train += 1
    train_split = all_train_data[:num_train]
    tune_split = all_train_data[num_train:]

    train_split = train_split[0:10]
    tune_split = train_split

    logging.info("Created train dataset of size %d ", len(train_split))
    logging.info("Created tuning dataset of size %d ", len(tune_split))

    # Train on this dataset
    learning_alg = SupervisedLearningDetectVisibleObject(model=model,
                                                         action_space=action_space,
                                                         meta_data_util=meta_data_util,
                                                         config=config,
                                                         constants=constants,
                                                         tensorboard=tensorboard)
    print "START Image Detection Experiment"
    learning_alg.do_train(agent, train_split, tune_split, experiment)
    server.kill()

except Exception:
    server.kill()
    exc_info = sys.exc_info()
    traceback.print_exception(*exc_info)
    # raise e
