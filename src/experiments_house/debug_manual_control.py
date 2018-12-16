import json
import logging
import os
import sys
import traceback
import random
import torch.multiprocessing as mp

from random import shuffle

from agents.tmp_house_agent import TmpHouseAgent
from dataset_agreement_house.action_space import ActionSpace
from dataset_agreement_house.metadata_util import MetaDataUtil
from dataset_agreement_house.dataset_parser import DatasetParser
from learning.asynchronous.tmp_house_asynchronous_contextual_bandit_learning import TmpAsynchronousContextualBandit

from learning.asynchronous.tmp_house_supervised_learning import TmpSupervisedLearning
from models.incremental_model.tmp_house_incremental_model_chaplot import TmpHouseIncrementalModelChaplot
from server_house.house_server import HouseServer
# from setup_agreement_blocks.validate_setup_blocks import \
#     BlocksSetupValidator

from utils.check_port import find_k_ports
from utils.launch_unity import launch_k_unity_builds
from utils.multiprocess_logger import MultiprocessingLoggerManager


def main():

    data_filename = "./simulators/house/AssetsHouse"
    experiment_name = "tmp_house_1_debug_manual_control"
    experiment = "./results/" + experiment_name
    print("EXPERIMENT NAME: ", experiment_name)

    # Create the experiment folder
    if not os.path.exists(experiment):
        os.makedirs(experiment)

    # Define log settings
    log_path = experiment + '/train_baseline.log'
    multiprocess_logging_manager = MultiprocessingLoggerManager(
        file_path=log_path, logging_level=logging.INFO)
    master_logger = multiprocess_logging_manager.get_logger("Master")
    master_logger.log("----------------------------------------------------------------")
    master_logger.log("                    STARING NEW EXPERIMENT                      ")
    master_logger.log("----------------------------------------------------------------")

    with open("data/house/config.json") as f:
        config = json.load(f)
    with open("data/shared/contextual_bandit_constants.json") as f:
        constants = json.load(f)
    constants['horizon'] = 40  # TODO HACK!!
    print(json.dumps(config, indent=2))

    # TODO: HouseSetupValidator()
    # setup_validator = BlocksSetupValidator()
    # setup_validator.validate(config, constants)

    # log core experiment details
    master_logger.log("CONFIG DETAILS")
    for k, v in sorted(config.items()):
        master_logger.log("    %s --- %r" % (k, v))
    master_logger.log("CONSTANTS DETAILS")
    for k, v in sorted(constants.items()):
        master_logger.log("    %s --- %r" % (k, v))
    master_logger.log("START SCRIPT CONTENTS")
    with open(__file__) as f:
        for line in f.readlines():
            master_logger.log(">>> " + line.strip())
    master_logger.log("END SCRIPT CONTENTS")

    config["use_manipulation"] = True   # debug manipulation
    action_space = ActionSpace(config["action_names"], config["stop_action"], config["use_manipulation"],
                               config["num_manipulation_row"], config["num_manipulation_col"])
    meta_data_util = MetaDataUtil()

    # TODO: Create vocabulary
    vocab = dict()
    vocab_list = open(data_filename + "/house_all_vocab.txt").readlines()
    for i, tk in enumerate(vocab_list):
        token = tk.strip().lower()
        # vocab[token] = i
        vocab[i] = token
    # vocab["$UNK$"] = len(vocab_list)
    vocab[len(vocab_list)] = "$UNK$"
    config["vocab_size"] = len(vocab_list) + 1

    # Number of processes
    house_ids = [1]  # [1,2,3]
    num_processes = len(house_ids)

    try:
        # Create the model
        master_logger.log("CREATING MODEL")
        model_type = TmpHouseIncrementalModelChaplot
        shared_model = model_type(config, constants)
        # model.load_saved_model("./results/paragraph_chaplot_attention/chaplot_model_epoch_3")

        # make the shared model use share memory
        shared_model.share_memory()

        master_logger.log("MODEL CREATED")
        print("Created Model...")

        # Read the dataset
        tune_split, train_split = [], []
        for hid in house_ids:
            all_train_data = DatasetParser.parse(
                data_filename + "/tokenized_house" + str(hid) + "_discrete.json", config)
            all_dev_data = DatasetParser.parse(
                data_filename + "/tokenized_house" + str(hid) + "_discrete_dev.json", config)
            # num_tune = int(len(all_train_data) * 0.1)
            # train_split.append(list(all_train_data[num_tune:]))
            # tune_split.append(list(all_train_data[:num_tune]))

            # Extract type of the dataset
            # lines = open("./simulators/house/datapoint_type_house" + str(hid) + "_v5_110.txt").readlines()
            # datapoint_id_type = {}
            # for line in lines:
            #     datapoint_id, datapoint_type = line.split()
            #     datapoint_id_type[int(datapoint_id)] = datapoint_type.strip()
            #
            # # Filter manipulation type
            # all_train_data = list(
            #     filter(lambda datapoint: datapoint_id_type[datapoint.get_id()] == "navigation", all_train_data))

            all_train_data = all_train_data[0:50]
            train_split.append(all_train_data)
            tune_split.append(all_train_data)
            # train_split.append(all_train_data)
            # tune_split.append(all_dev_data)

        # Launch the server
        ports = find_k_ports(1)
        port = ports[0]
        tmp_config = {k: v for k, v in config.items()}
        tmp_config["port"] = port
        tmp_tune_split = tune_split[0]
        print("Client " + str(0) + " getting a validation set of size ", len(tmp_tune_split))
        server = HouseServer(tmp_config, action_space, port)

        house_id = house_ids[0]
        launch_k_unity_builds([tmp_config["port"]], "./house_" + str(house_id) + "_elmer.x86_64",
                              arg_str="--config ./AssetsHouse/config" + str(house_id) + ".json",
                              cwd="./simulators/house/")

        server.initialize_server()

        # Create a local model for rollouts
        local_model = model_type(tmp_config, constants)
        # local_model.train()

        # Create the Agent
        print("STARTING AGENT")
        tmp_agent = TmpHouseAgent(server=server,
                                  model=local_model,
                                  test_policy=None,
                                  action_space=action_space,
                                  meta_data_util=meta_data_util,
                                  config=tmp_config,
                                  constants=constants)
        print("Created Agent...")
        index = 0
        while True:
            print("Giving another data %r ", len(train_split[0]))
            # index = random.randint(0, len(train_split[0]) - 1)
            index = (index + 1) % len(train_split[0])
            print("Dataset id is " + str(train_split[0][index].get_id()))
            tmp_agent.debug_manual_control(train_split[0][index], vocab)
            # tmp_agent.debug_tracking(train_split[0][index], vocab)

    except Exception:
        exc_info = sys.exc_info()
        traceback.print_exception(*exc_info)
        # raise e


if __name__ == "__main__":
    print("SETTING THE START METHOD ")
    mp.freeze_support()
    mp.set_start_method('spawn', force=True)
    main()
