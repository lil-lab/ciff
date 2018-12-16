import json
import logging
import os
import sys
import traceback
import torch.multiprocessing as mp

from random import shuffle
from dataset_agreement_house.action_space import ActionSpace
from dataset_agreement_house.metadata_util import MetaDataUtil
from dataset_agreement_house.dataset_parser import DatasetParser
from learning.asynchronous.tmp_house_asynchronous_contextual_bandit_learning import TmpAsynchronousContextualBandit
from models.incremental_model.tmp_house_incremental_model_chaplot import TmpHouseIncrementalModelChaplot
from models.incremental_model.tmp_house_incremental_model_oracle_gold_prob import TmpHouseIncrementalModelOracleGoldProb
from models.incremental_model.tmp_house_misra_baseline import TmpHouseMisraBaseline
from server_house.house_server import HouseServer
from setup_agreement_house.validate_setup_house import HouseSetupValidator
from utils.check_port import find_k_ports
from utils.multiprocess_logger import MultiprocessingLoggerManager


def main():

    data_filename = "simulators/house/AssetsHouse"
    experiment_name = "house_unet_cb_navigation_gold_goal_no_RNN"
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

    # Validate the setting
    setup_validator = HouseSetupValidator()
    setup_validator.validate(config, constants)

    # Log core experiment details
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
    house_ids = [1, 2, 3, 4, 5]
    num_processes = len(house_ids)

    try:
        # Create the model
        master_logger.log("CREATING MODEL")
        model_type = TmpHouseMisraBaseline #TmpHouseIncrementalModelOracleGoldProb
        shared_model = model_type(config, constants, use_image=False)
        # model.load_saved_model("./results/paragraph_chaplot_attention/chaplot_model_epoch_3")

        # make the shared model use share memory
        shared_model.share_memory()

        master_logger.log("MODEL CREATED")
        print("Created Model...")

        # Read the dataset
        tune_split, train_split = [], []
        for hid in house_ids:
            all_train_data = DatasetParser.parse(
                data_filename + "/tokenized_house" + str(hid) + "_discrete_train.json", config)
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

            # Filter manipulation type
            # all_train_data = list(filter(lambda datapoint: datapoint_id_type[datapoint.get_id()] == "navigation", all_train_data))

            train_split.append(all_train_data)
            tune_split.append(all_dev_data)
            # train_split.append(all_train_data)
            # tune_split.append(all_dev_data)

        processes = []

        # Start the training thread(s)
        ports = find_k_ports(num_processes)
        master_logger.log("Found K Ports")
        for i, port in enumerate(ports):
            train_chunk = train_split[i]
            tmp_config = {k: v for k, v in config.items()}
            tmp_config["port"] = port
            tmp_tune_split = tune_split[i]
            print("Client " + str(i) + " getting a validation set of size ", len(tmp_tune_split))
            server = HouseServer(tmp_config, action_space, port)
            master_logger.log("Server Initialized")
            client_logger = multiprocess_logging_manager.get_logger(i)
            p = mp.Process(target=TmpAsynchronousContextualBandit.do_train, args=(house_ids[i], shared_model,
                                                                                  tmp_config, action_space,
                                                                                  meta_data_util, constants,
                                                                                  train_chunk, tmp_tune_split,
                                                                                  experiment, experiment_name, i,
                                                                                  server, client_logger,
                                                                                  model_type, vocab))
            p.daemon = False
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

    except Exception:
        exc_info = sys.exc_info()
        traceback.print_exception(*exc_info)
        # raise e


if __name__ == "__main__":
    print("SETTING THE START METHOD ")
    mp.freeze_support()
    mp.set_start_method('spawn', force=True)
    main()
