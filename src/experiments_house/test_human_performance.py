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
    experiment_name = "emnlp_camera_ready_test_human_performance"
    experiment = "./results/" + experiment_name
    print("EXPERIMENT NAME: ", experiment_name)

    # Create the experiment folder
    if not os.path.exists(experiment):
        os.makedirs(experiment)

    # Number of processes
    house_id = 3

    # Define log settings
    log_path = experiment + '/test_baseline_%d.log' % house_id
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
        vocab[i] = token
    vocab[len(vocab_list)] = "$UNK$"
    config["vocab_size"] = len(vocab_list) + 1

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
        test_split = DatasetParser.parse(data_filename + "/tokenized_house" + str(house_id) + "_discrete_dev.json", config)
        test_split = test_split[2:20]

        # Launch the server
        ports = find_k_ports(1)
        port = ports[0]
        tmp_config = {k: v for k, v in config.items()}
        tmp_config["port"] = port
        print("Client " + str(0) + " getting a validation set of size ", len(test_split))
        server = HouseServer(tmp_config, action_space, port)

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
        tmp_agent.test_human_performance(test_split, vocab, master_logger)

    except Exception:
        exc_info = sys.exc_info()
        traceback.print_exception(*exc_info)
        # raise e


if __name__ == "__main__":
    print("SETTING THE START METHOD ")
    mp.freeze_support()
    mp.set_start_method('spawn', force=True)
    main()
