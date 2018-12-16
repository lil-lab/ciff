import json
import logging
import os
import random
import time
import sys
import traceback

import torch

import utils.generic_policy as gp
from dataset_agreement_house.action_space import ActionSpace
from dataset_agreement_house.dataset_parser import DatasetParser
from dataset_agreement_house.metadata_util import MetaDataUtil

from learning.single_client.train_action_types import TrainActionTypes
from models.incremental_model.incremental_model_attention_chaplot_resnet import IncrementalModelAttentionChaplotResNet
from models.module.action_type_module import ActionTypeModule
from setup_agreement_house.validate_setup_house import HouseSetupValidator
from utils.tensorboard import Tensorboard

data_filename = "simulators/house/AssetsHouse"
experiment_name = "train_house_action_types"
experiment = "./results/" + experiment_name

# Create the experiment folder
if not os.path.exists(experiment):
    os.makedirs(experiment)

# Define log settings
logging.basicConfig(filename=experiment + '/test_goal_prediction.log',
                    level=logging.INFO)
logging.info("----------------------------------------------------------------")
logging.info("                    STARING NEW EXPERIMENT                      ")
logging.info("----------------------------------------------------------------")

# Test policy
test_policy = gp.get_argmax_action

with open("data/house/config.json") as f:
    config = json.load(f)
with open("data/shared/contextual_bandit_constants.json") as f:
    constants = json.load(f)
constants['horizon'] = 40  # TODO HACK!!
print(json.dumps(config, indent=2))

# Validate the setting
setup_validator = HouseSetupValidator()
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

action_space = ActionSpace(config["action_names"], config["stop_action"], config["use_manipulation"],
                           config["num_manipulation_row"], config["num_manipulation_col"])
meta_data_util = MetaDataUtil()

# TODO: Create vocabulary
vocab = dict()
vocab_list = open(data_filename + "/house_all_vocab.txt").readlines()
for i, tk in enumerate(vocab_list):
    token = tk.strip().lower()
    vocab[token] = i
    # vocab[i] = token
vocab["$UNK$"] = len(vocab_list)
# vocab[len(vocab_list)] = "$UNK$"
config["vocab_size"] = len(vocab_list) + 1

# Number of processes
house_ids = [1, 2, 3, 4, 5]

try:
    # Create the model
    logging.log(logging.DEBUG, "CREATING MODEL")

    shared_goal_prediction_model = IncrementalModelAttentionChaplotResNet(
        config, constants, final_model_type="unet-positional-encoding", final_dimension=(64, 32, 32 * 6))
    shared_goal_prediction_model.load_saved_model(
        "./results/house_goal_prediction/goal_prediction_single_supervised_epoch_4")

    text_embedding_model = shared_goal_prediction_model.text_module
    model = ActionTypeModule()
    if torch.cuda.is_available():
        model.cuda()

    logging.log(logging.DEBUG, "MODEL CREATED")

    # Tensorboard
    tensorboard = Tensorboard(experiment_name)

    # Read the dataset
    train_dataset, tune_dataset = [], []

    dataset = {}

    for house_id in house_ids:

        house_dataset = TrainActionTypes.parse(house_id=house_id)

        all_train_data = DatasetParser.parse(
            data_filename + "/tokenized_house" + str(house_id) + "_discrete_train.json", config)
        all_dev_data = DatasetParser.parse(
            data_filename + "/tokenized_house" + str(house_id) + "_discrete_dev.json", config)

        train_ids = [dp.datapoint_id for dp in all_train_data]
        dev_ids = [dp.datapoint_id for dp in all_dev_data]
        real_to_type_dict = {}

        for dp in all_train_data:
            real_to_type_dict[dp.datapoint_id] = dp
        for dp in all_dev_data:
            real_to_type_dict[dp.datapoint_id] = dp

        house_dataset_dict = {}
        for datapoint in house_dataset:
            task_id = datapoint.task_id

            if task_id in real_to_type_dict:
                datapoint.instruction = real_to_type_dict[task_id].instruction
                # datapoint.instruction_string = real_to_type_dict[task_id]

            if task_id in house_dataset_dict:
                house_dataset_dict[task_id].append(datapoint)
            else:
                house_dataset_dict[task_id] = [datapoint]

        for task_id in train_ids:
            if task_id in house_dataset_dict:
                train_dataset.extend(house_dataset_dict[task_id])

        for task_id in dev_ids:
            if task_id in house_dataset_dict:
                tune_dataset.extend(house_dataset_dict[task_id])

        dataset[house_id] = train_dataset[0:50]

    # Train on this dataset
    learning_alg = TrainActionTypes(model=model, text_embedding_model=text_embedding_model)

    # for house_id in range(1, 6):
    #     for dp in dataset[house_id]:
    #         learning_alg.save_datapoint(dp)

    print("Size of train dataset is %r and dev set is %r" % (len(train_dataset), len(tune_dataset)))
    random.shuffle(train_dataset)
    # train_dataset = tune_dataset = train_dataset[0:10]

    print("STARTED LEARNING")
    start = time.time()

    # learning_alg.interactive_shell(tune_split, tune_images)
    # tune_images, tune_goal = GoalPredictionSingle360ImageSupervisedLearningFromDisk.parse(
    #     "./6000_tune_split_start_images", tune_split, model, config, format_type="png")
    # learning_alg.test(tune_dataset, tensorboard)
    learning_alg.do_train(train_dataset, tune_dataset, experiment, save_best_model=False)
    end = time.time()
    print("Time taken is " + str(end - start) + " seconds. ")
    logging.info("Taken taken is %r seconds", end - start)

except Exception:
    exc_info = sys.exc_info()
    traceback.print_exception(*exc_info)
    # raise e
