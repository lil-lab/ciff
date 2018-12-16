import json
import logging
import os
import sys
import traceback
import torch.multiprocessing as mp

from agents.agent_no_model import Agent
from dataset_agreement_house.action_space import ActionSpace
from dataset_agreement_house.metadata_util import MetaDataUtil
from dataset_agreement_house.dataset_parser import DatasetParser
from setup_agreement_house.validate_setup_house import HouseSetupValidator
from utils.check_port import find_k_ports
from utils.multiprocess_logger import MultiprocessingLoggerManager


def main():

    data_filename = "./simulators/house/AssetsHouse"
    experiment_name = "house_test_stop_dev_dataset_navigation_only"
    experiment = "./results/" + experiment_name
    print("EXPERIMENT NAME: ", experiment_name)

    # Create the experiment folder
    if not os.path.exists(experiment):
        os.makedirs(experiment)

    # Define log settings
    log_path = experiment + '/test_baseline.log'
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

    setup_validator = HouseSetupValidator()
    setup_validator.validate(config, constants)

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
        # Read the dataset
        test_data = []
        for hid in house_ids:
            all_dev_data = DatasetParser.parse(
                data_filename + "/tokenized_house" + str(hid) + "_discrete_dev.json", config)

            # Extract type of the dataset
            lines = open("./simulators/house/datapoint_type_house" + str(hid) + ".txt").readlines()
            datapoint_id_type = {}
            for line in lines:
                words = line.split()
                datapoint_id, datapoint_type = words[0], words[1:]
                datapoint_id_type[int(datapoint_id)] = datapoint_type#.strip()

            # Filter manipulation type
            all_dev_data = filter(lambda datapoint: "manipulation" not in datapoint_id_type[datapoint.get_id()], all_dev_data)
            test_data.append(list(all_dev_data))

        processes = []

        # Start the testing thread(s)
        ports = find_k_ports(num_processes)
        master_logger.log("Found K Ports")
        for i, port in enumerate(ports):
            test_chunk = test_data[i]  # Simulator i runs house i and uses the dataset for house i
            tmp_config = {k: v for k, v in config.items()}
            tmp_config["port"] = port
            print("Client " + str(i) + " getting test set of size ", len(test_chunk))
            client_logger = multiprocess_logging_manager.get_logger(i)
            p = mp.Process(target=Agent.test_multiprocess, args=(house_ids[i], test_chunk, tmp_config, action_space, port,
                                                                 Agent.STOP, meta_data_util, constants, vocab,
                                                                 client_logger, None))
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
