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
from models.incremental_model.incremental_model_chaplot import IncrementalModelChaplot
from models.incremental_model.tmp_streetview_incremental_model_deepmind_policy_network import \
    TmpStreetviewIncrementalModelDeepMindPolicyNetwork
from models.incremental_model.tmp_streetview_incremental_model_recurrent_policy_network import \
    TmpStreetviewIncrementalModelConcatRecurrentPolicyNetwork
from server_streetview.streetview_server import StreetViewServer
from setup_agreement_streetview.validate_setup_streetview import StreetViewSetupValidator
from utils.multiprocess_logger import MultiprocessingLoggerManager


def main():

    # Learning algorithm
    parser = argparse.ArgumentParser(description='Parser Values')
    parser.add_argument('--name', type=str, help='name of the experiment')
    parser.add_argument('--num_processes', type=int, default=6, help='num of process')
    parser.add_argument('--split', type=str, help='learning alg ("train", "dev", "test")')
    parser.add_argument('--model', type=str, help='model ("chaplot", "concat")')
    parser.add_argument('--learning_alg', type=str, help='learning alg ("cb", "sup", "mix"')
    args = parser.parse_args()

    experiment_name = args.name
    experiment = "./results/" + experiment_name
    print("EXPERIMENT NAME: ", experiment_name)

    # Create the experiment folder
    if not os.path.exists(experiment):
        os.makedirs(experiment)

    # Define log settings
    log_path = experiment + '/test_baseline_%s.log' % args.split
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

    num_processes = args.num_processes
    model_name = args.model
    data_split = args.split
    learning_alg = args.learning_alg

    # Number of processes
    master_logger.log("Num processes %r, Model %r, Alg %r, Split %r " %
                      (num_processes, model_name, learning_alg, data_split))

    try:
        # Create the model
        master_logger.log("CREATING MODEL")
        if model_name == "concat":
            model_type = TmpStreetviewIncrementalModelDeepMindPolicyNetwork
        elif model_name == "chaplot":
            model_type = IncrementalModelChaplot
        else:
            raise AssertionError("Model name not known. %r " % model_name)

        shared_model = model_type(config, constants)
        shared_model.init_weights()

        if model_name == "concat":
            if learning_alg == "sup":
                shared_model.load_saved_model(
                    "./results/train-streetview-deepmind-supervised-learning/supervised_learning0_epoch_13")
            elif learning_alg == "cb":
                shared_model.load_saved_model("./results/train-streetview-deepmind-cb/contextual_bandit_0_epoch_38")
            elif learning_alg == "mix":
                shared_model.load_saved_model(
                    "./results/train-streetview-deepmind-mixture-algorithm/supervised_learning5_epoch_54")
            else:
                raise AssertionError("Unregistered learning algorithm %r " % learning_alg)
        elif model_name == "chaplot":
            if learning_alg == "sup":
                shared_model.load_saved_model(
                    "./results/train-streetview-chaplot-supervised-learning/supervised_learning0_epoch_36")
            elif learning_alg == "cb":
                shared_model.load_saved_model("./results/train-streetview-chaplot-cb/contextual_bandit_0_epoch_66")
            elif learning_alg == "mix":
                shared_model.load_saved_model(
                    "./results/train-streetview-chaplot-mixture-repeat2/contextual_bandit_0_epoch_34")
            else:
                raise AssertionError("Unregistered learning algorithm %r " % learning_alg)
        else:
            raise AssertionError("Unregistered model %r " % model_name)

        # make the shared model use share memory
        shared_model.share_memory()

        master_logger.log("MODEL CREATED")
        print("Created Model...")

        # Read the dataset
        test_split = DatasetParser.parse("data/streetview/navigation_%s.json" % data_split, config)

        master_logger.log("Created tuning dataset of size %d " % len(test_split))

        processes = []

        # Split the train data between processes
        test_split_process_chunks = []
        tune_chunk_size = int(len(test_split)/num_processes)
        tune_pad = 0
        for i in range(0, num_processes):
            if i < num_processes - 1:
                test_split_process_chunks.append(test_split[tune_pad: tune_pad + tune_chunk_size])
            else:
                test_split_process_chunks.append(test_split[tune_pad:])
            tune_pad += tune_chunk_size

        assert sum([len(chunk) for chunk in test_split_process_chunks]) == len(test_split), "Test dataset not properly partitioned." 

        # Start the training thread(s)
        for i in range(0, num_processes):
            test_chunk = test_split_process_chunks[i]
            print("Client " + str(i) + " getting a test set of size ", len(test_chunk))
            server = StreetViewServer(config, action_space, forward_setting_strict=False)
            client_logger = multiprocess_logging_manager.get_logger(i)
            p = mp.Process(target=TmpStreetViewAsynchronousContextualBandit.do_test, args=(shared_model, config,
                                                                                           action_space,
                                                                                           meta_data_util,
                                                                                           constants, test_chunk,
                                                                                           experiment_name, i, server,
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
