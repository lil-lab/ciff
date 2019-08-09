import json
import logging
import os
import sys
import argparse
import traceback
import torch.multiprocessing as mp

from agents.agent_no_model import Agent
from dataset_agreement_blocks.action_space import ActionSpace
from dataset_agreement_blocks.metadata_util import MetaDataUtil
from dataset_agreement_blocks.dataset_parser import DatasetParser
from server_blocks.blocks_server import BlocksServer
from setup_agreement_blocks.validate_setup_blocks import BlocksSetupValidator
from utils.check_port import find_k_ports
from utils.launch_unity import launch_k_unity_builds
from utils.multiprocess_logger import MultiprocessingLoggerManager


# Environment arguments
parser = argparse.ArgumentParser(description='Blocks Baselines')
parser.add_argument('--baseline',type=str, default="random", help="Three options: stop, random, frequent.")
parser.add_argument('--split', type=int, default="dev", help="Three options: train, dev and test.")
args = parser.parse_args()


def main():

    experiment_name = "test_block_baselines"
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

    try:
        # Read the dataset
        if args.split == "train":
            test_data = DatasetParser.parse("trainset.json", config)
        elif args.split == "dev":
            test_data = DatasetParser.parse("devset.json", config)
        elif args.split == "test":
            test_data = DatasetParser.parse("testset.json", config)
        else:
            raise AssertionError("Unhandled dataset split %r. Only support train, dev and test." % args.split)
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
        if args.baseline == "stop":
            agent_type = Agent.STOP
        elif args.baseline == "random":
            agent_type = Agent.RANDOM_WALK
        elif args.baseline == "frequent":
            agent_type = Agent.MOST_FREQUENT
            # TODO compute most frequent action from the dataset
        else:
            raise AssertionError("Unhandled agent type %r. Only support stop, random and frequent." % args.baseline)

        agent = Agent(agent_type=agent_type,
                      server=server,
                      action_space=action_space,
                      meta_data_util=meta_data_util,
                      constants=constants)

        agent.test(test_data)

    except Exception:
        exc_info = sys.exc_info()
        traceback.print_exception(*exc_info)
        # raise e


if __name__ == "__main__":
    print("SETTING THE START METHOD ")
    mp.freeze_support()
    mp.set_start_method('spawn')
    main()
