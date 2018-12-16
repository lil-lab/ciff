import json
import logging
import os
import sys
import traceback
import argparse
import torch.multiprocessing as mp

from dataset_agreement_streetview.action_space import ActionSpace
from dataset_agreement_streetview.metadata_util import MetaDataUtil
from dataset_agreement_streetview.dataset_parser import DatasetParser
from learning.asynchronous.tmp_streetview_asynchronous_contextual_bandit_learning import \
    TmpStreetViewAsynchronousContextualBandit
from learning.asynchronous.tmp_streetview_asynchronous_supervised_learning import \
    TmpStreetViewAsynchronousSupervisedLearning
from models.incremental_model.tmp_streetview_incremental_model_deepmind_policy_network import \
    TmpStreetviewIncrementalModelDeepMindPolicyNetwork
from models.incremental_model.tmp_streetview_incremental_model_recurrent_policy_network import \
    TmpStreetviewIncrementalModelConcatRecurrentPolicyNetwork
from models.incremental_model.incremental_model_chaplot import IncrementalModelChaplot
from server_streetview.streetview_server import StreetViewServer
from setup_agreement_streetview.validate_setup_streetview import StreetViewSetupValidator
from utils.multiprocess_logger import MultiprocessingLoggerManager


def main():

    experiment_name = "train-streetview-deepmind-mixture-learning-repeat"
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

    with open("data/streetview/config.json") as f:
        config = json.load(f)
    with open("data/shared/contextual_bandit_constants.json") as f:
        constants = json.load(f)
    print(json.dumps(config,indent=2))
    setup_validator = StreetViewSetupValidator()
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

    # Learning algorithm
    parser = argparse.ArgumentParser(description='Parser Values')
    parser.add_argument('--num_processes', type=int, default=6, help='num of process')
    parser.add_argument('--learning_alg', type=str, default="cb", help='learning alg ("cb", "sup", "mix"')
    args = parser.parse_args()

    num_processes = args.num_processes
    learning_alg = args.learning_alg
    master_logger.log("Num processes %r, Learning Algorithm %r " % (num_processes, learning_alg))

    try:
        # Create the model
        master_logger.log("CREATING MODEL")
        model_type = TmpStreetviewIncrementalModelDeepMindPolicyNetwork
        # model_type = TmpStreetviewIncrementalModelConcatRecurrentPolicyNetwork
        # model_type = IncrementalModelChaplot
        shared_model = model_type(config, constants)
        shared_model.init_weights()

        # make the shared model use share memory
        shared_model.share_memory()

        master_logger.log("MODEL CREATED")
        print("Created Model...")

        # Read the dataset
        train_split = DatasetParser.parse("data/streetview/navigation_train.json", config)
        tune_split = DatasetParser.parse("data/streetview/navigation_dev.json", config)
        master_logger.log("Created train dataset of size %d " % len(train_split))
        master_logger.log("Created tuning dataset of size %d " % len(tune_split))

        processes = []

        # Split the train data between processes
        train_split_process_chunks = []
        tune_split_process_chunks = []
        train_chunk_size = int(len(train_split)/num_processes)
        tune_chunk_size = int(len(tune_split)/num_processes)
        train_pad = 0
        tune_pad = 0
        for i in range(0, num_processes):
            train_split_process_chunks.append(train_split[train_pad: train_pad + train_chunk_size])
            tune_split_process_chunks.append(tune_split[tune_pad: tune_pad + tune_chunk_size])
            train_pad += train_chunk_size
            tune_pad += tune_chunk_size

        # Start the training thread(s)
        for i in range(0, num_processes):
            train_chunk = train_split_process_chunks[i]
            if i == num_processes - 1:
                # Don't want each client to do testing.
                tmp_tune_split = tune_split_process_chunks[i]
            else:
                tmp_tune_split = tune_split_process_chunks[i]
            print("Client " + str(i) + " getting a validation set of size ", len(tmp_tune_split))
            server = StreetViewServer(config, action_space, forward_setting_strict=False)
            client_logger = multiprocess_logging_manager.get_logger(i)

            if learning_alg == "cb" or (learning_alg == "mix" and i < num_processes - 2):
                p = mp.Process(target=TmpStreetViewAsynchronousContextualBandit.do_train, args=(shared_model, config,
                                                                                                action_space,
                                                                                                meta_data_util,
                                                                                                constants, train_chunk,
                                                                                                tmp_tune_split,
                                                                                                experiment, experiment_name,
                                                                                                i, server,
                                                                                                client_logger, model_type))
            elif learning_alg == "sup" or (learning_alg == "mix" and i >= num_processes - 2):
                p = mp.Process(target=TmpStreetViewAsynchronousSupervisedLearning.do_train, args=(shared_model, config,
                                                                                                  action_space,
                                                                                                  meta_data_util,
                                                                                                  constants, train_chunk,
                                                                                                  tmp_tune_split,
                                                                                                  experiment,
                                                                                                  experiment_name,
                                                                                                  i, server,
                                                                                                  client_logger,
                                                                                                  model_type))
            else:
                raise NotImplementedError()
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
