import json
import logging
import os
import sys
import traceback
import torch.multiprocessing as mp
import utils.generic_policy as gp
from agents.agent import Agent

from dataset_agreement_blocks.action_space import ActionSpace
from dataset_agreement_blocks.metadata_util import MetaDataUtil
from dataset_agreement_blocks.dataset_parser import DatasetParser
from models.incremental_model.incremental_model_emnlp import IncrementalModelEmnlp
from server_blocks.blocks_server import BlocksServer
from setup_agreement_blocks.validate_setup_blocks import BlocksSetupValidator
from utils.check_port import find_k_ports
from utils.launch_unity import launch_k_unity_builds
from utils.multiprocess_logger import MultiprocessingLoggerManager
from utils.tensorboard import Tensorboard


def main():

    experiment_name = "blocks_experiments"
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

    with open("data/blocks/config.json") as f:
        config = json.load(f)
    with open("data/shared/contextual_bandit_constants.json") as f:
        constants = json.load(f)
    print(json.dumps(config,indent=2))
    setup_validator = BlocksSetupValidator()
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

    action_space = ActionSpace(config)
    meta_data_util = MetaDataUtil()

    # Create vocabulary
    vocab = dict()
    vocab_list = open("./Assets/vocab_both").readlines()
    for i, tk in enumerate(vocab_list):
        token = tk.strip().lower()
        vocab[token] = i
    vocab["$UNK$"] = len(vocab_list)
    config["vocab_size"] = len(vocab_list) + 1

    # Test policy
    test_policy = gp.get_argmax_action

    # Create tensorboard
    tensorboard = Tensorboard("Agent Test")

    try:
        # Create the model
        master_logger.log("CREATING MODEL")
        model_type = IncrementalModelEmnlp
        shared_model = model_type(config, constants)
        shared_model.load_saved_model("./results/model-folder-name/model-file-name")

        # Read the dataset
        test_data = DatasetParser.parse("devset.json", config)
        master_logger.log("Created test dataset of size %d " % len(test_data))

        # Create server and launch a client
        simulator_file = "./simulators/blocks/retro_linux_build.x86_64"
        config["port"] = find_k_ports(1)[0]
        server = BlocksServer(config, action_space, vocab=vocab)

        # Launch unity
        launch_k_unity_builds([config["port"]], simulator_file)
        server.initialize_server()

        # Create the agent
        master_logger.log("CREATING AGENT")
        agent = Agent(server=server,
                      model=shared_model,
                      test_policy=test_policy,
                      action_space=action_space,
                      meta_data_util=meta_data_util,
                      config=config,
                      constants=constants)

        agent.test(test_data, tensorboard)

    except Exception:
        exc_info = sys.exc_info()
        traceback.print_exception(*exc_info)
        # raise e


if __name__ == "__main__":
    print("SETTING THE START METHOD ")
    mp.freeze_support()
    mp.set_start_method('spawn')
    main()
