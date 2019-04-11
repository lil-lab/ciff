import json
import logging
import os
import sys
import traceback
import torch.multiprocessing as mp

from random import shuffle
from dataset_agreement_blocks.action_space import ActionSpace
from dataset_agreement_blocks.metadata_util import MetaDataUtil
from dataset_agreement_blocks.dataset_parser import DatasetParser
from learning.asynchronous.tmp_blocks_asynchronous_contextual_bandit_learning import TmpAsynchronousContextualBandit
from learning.asynchronous.tmp_blocks_supervised_learning import TmpSupervisedLearning
from models.incremental_model.incremental_model_emnlp import IncrementalModelEmnlp
from models.incremental_model.tmp_blocks_incremental_model_chaplot import TmpBlocksIncrementalModelChaplot
from server_blocks.blocks_server import BlocksServer
from setup_agreement_blocks.validate_setup_blocks import \
    BlocksSetupValidator
from utils.check_port import find_k_ports
from utils.multiprocess_logger import MultiprocessingLoggerManager


def main():

    experiment_name = "blocks_world_experiments"
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

    # Number of processes
    num_processes = 6

    try:
        # Create the model
        master_logger.log("CREATING MODEL")
        model_type = IncrementalModelEmnlp
        shared_model = model_type(config, constants)

        # make the shared model use share memory
        shared_model.share_memory()

        master_logger.log("MODEL CREATED")
        print("Created Model...")

        # Read the dataset
        all_train_data = DatasetParser.parse("trainset.json", config)
        num_train = int(0.8 * len(all_train_data))
        train_split = all_train_data[:num_train]
        tune_split = list(all_train_data[num_train:])
        shuffle(train_split)  # shuffle the split to break ties

        master_logger.log("Created train dataset of size %d " % len(train_split))
        master_logger.log("Created tuning/validation dataset of size %d " % len(tune_split))

        processes = []

        # Split the train data between processes
        train_split_process_chunks = []
        chunk_size = int(len(train_split)/num_processes)
        pad = 0
        for i in range(0, num_processes):
            chunk = train_split[pad: pad + chunk_size]
            pad += chunk_size
            train_split_process_chunks.append(chunk)

        simulator_file = "./simulators/blocks/retro_linux_build.x86_64"

        # Start the training thread(s)
        ports = find_k_ports(num_processes)
        for i, port in enumerate(ports):
            train_chunk = train_split_process_chunks[i]
            tmp_config = {k: v for k, v in config.items()}
            tmp_config["port"] = port
            if i == num_processes - 1:
                # Master client which does testing. Don't want each client to do testing.
                tmp_tune_split = tune_split
            else:
                tmp_tune_split = []
            print("Client " + str(i) + " getting a validation set of size ", len(tmp_tune_split))
            server = BlocksServer(tmp_config, action_space)
            client_logger = multiprocess_logging_manager.get_logger(i)
            p = mp.Process(target=TmpAsynchronousContextualBandit.do_train, args=(simulator_file, shared_model,
                                                                                  tmp_config, action_space,
                                                                                  meta_data_util, constants, train_chunk,
                                                                                  tmp_tune_split, experiment,
                                                                                  experiment_name, i, server,
                                                                                  client_logger, model_type, vocab))
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
    mp.set_start_method('spawn')
    main()
