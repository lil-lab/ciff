import json
import logging
import os
import time
import sys
import traceback

import utils.generic_policy as gp

from dataset_agreement_blocks.action_space import ActionSpace
from dataset_agreement_blocks.metadata_util import MetaDataUtil
from dataset_agreement_blocks.dataset_parser import DatasetParser
from learning.single_client.blocks_goal_prediction_supervised_from_disk import \
    BlockGoalPredictionSupervisedLearningFromDisk
from models.incremental_model.incremental_model_attention_chaplot_resnet import IncrementalModelAttentionChaplotResNet
from setup_agreement_blocks.validate_setup_blocks import BlocksSetupValidator
from utils.tensorboard import Tensorboard

experiment_name = "emnlp-rebuttal-unet-blocks"
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

with open("data/blocks/config.json") as f:
    config = json.load(f)
with open("data/shared/contextual_bandit_constants.json") as f:
    constants = json.load(f)
print(json.dumps(config, indent=2))
setup_validator = BlocksSetupValidator()
setup_validator.validate(config, constants)

# Create vocabulary
vocab = dict()
vocab_list = open("./Assets/vocab_both").readlines()
for i, tk in enumerate(vocab_list):
    token = tk.strip().lower()
    vocab[token] = i
vocab["$UNK$"] = len(vocab_list)
config["vocab_size"] = len(vocab_list) + 1

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

action_space = ActionSpace(config)
meta_data_util = MetaDataUtil()

try:
    # Create the model
    logging.log(logging.DEBUG, "CREATING MODEL")
    model = IncrementalModelAttentionChaplotResNet(config, constants,
                                                   final_model_type="unet-positional-encoding",
                                                   final_dimension=(64, 32, 32))
    # model.load_saved_model(
    #     "./results/postbugfix_goal_prediction_360_6000_unet_logprob/goal_prediction_single_supervised_epoch_6")
    logging.log(logging.DEBUG, "MODEL CREATED")

    # Tensorboard
    tensorboard = Tensorboard(experiment_name)

    # Read the dataset
    train_split = DatasetParser.parse("trainset.json", config)
    BlockGoalPredictionSupervisedLearningFromDisk.parse("./block_world_train_image_data", train_split, vocab)

    tune_split = DatasetParser.parse("devset.json", config)
    BlockGoalPredictionSupervisedLearningFromDisk.parse("./block_world_dev_image_data", tune_split, vocab)

    logging.info("Created train dataset of size %d ", len(train_split))
    logging.info("Created tuning dataset of size %d ", len(tune_split))

    # train_split = tune_split = train_split[0:10]

    # Train on this dataset
    learning_alg = BlockGoalPredictionSupervisedLearningFromDisk(model=model,
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
    # learning_alg.test(dev_data, dev_images, dev_goal, tensorboard)
    learning_alg.do_train(train_split, tune_split, experiment, save_best_model=True)
    end = time.time()
    print("Time taken is " + str(end - start) + " seconds. ")
    logging.info("Taken taken is %r seconds", end - start)

except Exception:
    exc_info = sys.exc_info()
    traceback.print_exception(*exc_info)
    # raise e
