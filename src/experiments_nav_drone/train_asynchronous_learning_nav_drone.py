import json
import logging
import os
import random
import sys
import traceback
import torch.multiprocessing as mp

from dataset_agreement_nav_drone.action_space import ActionSpace
from dataset_agreement_nav_drone.metadata_util import MetaDataUtil
from dataset_agreement_nav_drone.nav_drone_dataset_parser import DatasetParser
from learning.asynchronous.asynchronous_actor_critic_learning import AsynchronousAdvantageActorGAECritic
from learning.asynchronous.asynchronous_contextual_bandit_learning import AsynchronousContextualBandit
from learning.asynchronous.asynchronous_supervised_learning import AsynchronousSupervisedLearning
from models.incremental_model.incremental_model_attention_chaplot_resnet import IncrementalModelAttentionChaplotResNet
from models.incremental_model.incremental_model_oracle_gold_prob import IncrementalModelOracleGoldProb, \
    IncrementalModelOracleGoldProbWithImage
from server_nav_drone.nav_drone_server_py3 import NavDroneServerPy3
from setup_agreement_nav_drone.validate_setup_nav_drone import \
    NavDroneSetupValidator
from utils.check_port import find_k_ports
from utils.multiprocess_logger import MultiprocessingLoggerManager


def main():

    experiment_name = "emnlp-rebuttal-oracle_gold_prob_cb_6000-no-LSTM-no-Image"
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

    with open("data/nav_drone/config_localmoves_6000.json") as f:
        config = json.load(f)
    with open("data/shared/contextual_bandit_constants.json") as f:
        constants = json.load(f)
    print(json.dumps(config,indent=2))
    setup_validator = NavDroneSetupValidator()
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

    action_space = ActionSpace(config["action_names"], config["stop_action"])
    meta_data_util = MetaDataUtil()

    # Number of processes
    num_processes = 6

    try:
        # Create the model
        master_logger.log("CREATING MODEL")
        model_type = IncrementalModelOracleGoldProb
        shared_model = model_type(config, constants)
        shared_model.init_weights()
        '''shared_model.load_saved_model(
            "./results/full_task_unet_chaplottextmodule_32_provide_prob/contextual_bandit_5_epoch_19")'''

        # make the shared model use share memory
        shared_model.share_memory()

        master_logger.log("MODEL CREATED")
        print("Created Model...")

        # Read the dataset
        all_train_data = DatasetParser.parse("data/nav_drone/train_annotations_6000.json", config)
        num_train = (len(all_train_data) * 19) // 20
        while all_train_data[num_train].get_scene_name().split("_")[1] \
                == all_train_data[num_train - 1].get_scene_name().split("_")[1]:
            num_train += 1
        train_split = all_train_data[:num_train]
        tune_split = all_train_data[num_train:]

        master_logger.log("Created train dataset of size %d " % len(train_split))
        master_logger.log("Created tuning dataset of size %d " % len(tune_split))

        processes = []

        simulator_file = "./simulators/NavDroneLinuxBuild.x86_64"

        # Split the train data between processes
        train_split_process_chunks = []
        chunk_size = int(len(train_split)/num_processes)
        pad = 0
        for i in range(0, num_processes):
            chunk = train_split[pad: pad + chunk_size]
            pad += chunk_size
            train_split_process_chunks.append(chunk)

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
            server = NavDroneServerPy3(tmp_config, action_space, multi_client=True)
            client_logger = multiprocess_logging_manager.get_logger(i)
            p = mp.Process(target=AsynchronousContextualBandit.do_train, args=(simulator_file, shared_model, tmp_config,
                                                                               action_space, meta_data_util,
                                                                               constants, train_chunk, tmp_tune_split,
                                                                               experiment, experiment_name, i, server,
                                                                               client_logger, model_type))
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
