import json
import logging
import os
import random
import time
import sys
import traceback
import utils.generic_policy as gp
from dataset_agreement_house.action_space import ActionSpace
from dataset_agreement_house.dataset_parser import DatasetParser
from dataset_agreement_house.metadata_util import MetaDataUtil

from learning.single_client.goal_prediction_house_single_360_image_supervised_from_disk import \
    GoalPredictionHouseSingle360ImageSupervisedLearningFromDisk
from models.incremental_model.incremental_model_attention_chaplot_resnet import IncrementalModelAttentionChaplotResNet
from setup_agreement_house.validate_setup_house import HouseSetupValidator
from utils.tensorboard import Tensorboard

data_filename = "simulators/house/AssetsHouse"
experiment_name = "train_house_goal_prediction_m4jksum1_repeat"
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

config["do_goal_prediction"] = True  # force the goal prediction to happen

# Number of processes
house_ids = [1, 2, 3, 4, 5]

try:
    # Create the model
    logging.log(logging.DEBUG, "CREATING MODEL")
    model = IncrementalModelAttentionChaplotResNet(config, constants,
                                                   final_model_type="m4jksum1",#"unet-positional-encoding",
                                                   final_dimension=(64, 32, 32 * 6))
    model.load_saved_model("./results/train_house_goal_prediction_m4jksum1_repeat/goal_prediction_single_supervised_epoch_2")
    logging.log(logging.DEBUG, "MODEL CREATED")

    # Tensorboard
    tensorboard = Tensorboard(experiment_name)

    # Read the dataset
    train_dataset, tune_dataset = [], []

    dataset = {}

    for house_id in house_ids:

        house_dataset = GoalPredictionHouseSingle360ImageSupervisedLearningFromDisk.parse(house_id=house_id,
                                                                                          vocab=vocab,
                                                                                          size=128)

        all_train_data = DatasetParser.parse(
            data_filename + "/tokenized_house" + str(house_id) + "_discrete_train.json", config)
        all_dev_data = DatasetParser.parse(
            data_filename + "/tokenized_house" + str(house_id) + "_discrete_dev.json", config)

        train_ids = [dp.datapoint_id for dp in all_train_data]
        dev_ids = [dp.datapoint_id for dp in all_dev_data]

        house_dataset_dict = {}
        for datapoint in house_dataset:
            task_id = datapoint.task_id
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
    learning_alg = GoalPredictionHouseSingle360ImageSupervisedLearningFromDisk(model=model,
                                                                               action_space=action_space,
                                                                               meta_data_util=meta_data_util,
                                                                               config=config,
                                                                               constants=constants,
                                                                               tensorboard=tensorboard)

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
