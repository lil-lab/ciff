import json
import logging
import os
import sys
import traceback
import argparse
import torch.multiprocessing as mp

from baselines.chaplot_baseline_multiprocess_streetview import ChaplotBaselineStreetView
from baselines.chaplot_model_concat_gavector import a3c_lstm_ga_concat_gavector
from baselines.chaplot_model_default import a3c_lstm_ga_default
from baselines.chaplot_model_default_auxiliary import a3c_lstm_ga_default_with_aux
from dataset_agreement_streetview.action_space import ActionSpace
from dataset_agreement_streetview.metadata_util import MetaDataUtil
from dataset_agreement_streetview.dataset_parser import DatasetParser
from server_streetview.streetview_server import StreetViewServer
from setup_agreement_streetview.validate_setup_streetview import StreetViewSetupValidator
from utils.multiprocess_logger import MultiprocessingLoggerManager


def main():

    experiment_name = "train_a3c_ga_chaplot_baseline_streetview"
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

    parser = argparse.ArgumentParser(description='Gated-Attention for Grounding')

    # Environment arguments
    parser.add_argument('-l', '--max-episode-length', type=int, default=50,
                        help='maximum length of an episode (default: 40)')
    parser.add_argument('-d', '--difficulty', type=str, default="hard",
                        help="""Difficulty of the environment,
                        "easy", "medium" or "hard" (default: hard)""")
    parser.add_argument('--living-reward', type=float, default=0,
                        help="""Default reward at each time step (default: 0,
                        change to -0.005 to encourage shorter paths)""")
    parser.add_argument('--frame-width', type=int, default=300,
                        help='Frame width (default: 300)')
    parser.add_argument('--frame-height', type=int, default=168,
                        help='Frame height (default: 168)')
    parser.add_argument('-v', '--visualize', type=int, default=0,
                        help="""Visualize the envrionment (default: 0,
                        use 0 for faster training)""")
    parser.add_argument('--sleep', type=float, default=0,
                        help="""Sleep between frames for better
                        visualization (default: 0)""")
    parser.add_argument('--scenario-path', type=str, default="maps/room.wad",
                        help="""Doom scenario file to load
                        (default: maps/room.wad)""")
    parser.add_argument('--interactive', type=int, default=0,
                        help="""Interactive mode enables human to play
                        (default: 0)""")
    parser.add_argument('--all-instr-file', type=str,
                        default="data/instructions_all.json",
                        help="""All instructions file
                        (default: data/instructions_all.json)""")
    parser.add_argument('--train-instr-file', type=str,
                        default="data/instructions_train.json",
                        help="""Train instructions file
                        (default: data/instructions_train.json)""")
    parser.add_argument('--test-instr-file', type=str,
                        default="data/instructions_test.json",
                        help="""Test instructions file
                        (default: data/instructions_test.json)""")
    parser.add_argument('--object-size-file', type=str,
                        default="data/object_sizes.txt",
                        help='Object size file (default: data/object_sizes.txt)')

    # A3C arguments
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                        help='discount factor for rewards (default: 0.99)')
    parser.add_argument('--tau', type=float, default=1.00, metavar='T',
                        help='parameter for GAE (default: 1.00)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('-n', '--num-processes', type=int, default=6, metavar='N',
                        help='how many training processes to use (default: 6)')
    parser.add_argument('--num-steps', type=int, default=20, metavar='NS',
                        help='number of forward steps in A3C (default: 20)')
    parser.add_argument('--load', type=str, default="0",
                        help='model path to load, 0 to not reload (default: 0)')
    parser.add_argument('-e', '--evaluate', type=int, default=0,
                        help="""0:Train, 1:Evaluate MultiTask Generalization
                        2:Evaluate Zero-shot Generalization (default: 0)""")
    parser.add_argument('--dump-location', type=str, default="./saved/",
                        help='path to dump models and log (default: ./saved/)')

    args = parser.parse_args()

    with open("data/streetview/config.json") as f:
        config = json.load(f)
    with open("data/shared/contextual_bandit_constants.json") as f:
        constants = json.load(f)
    print(json.dumps(config,indent=2))
    setup_validator = StreetViewSetupValidator()
    setup_validator.validate(config, constants)
    args.input_size = config['vocab_size'] + 2

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

    try:
        # create tensorboard
        tensorboard = None  # Tensorboard(experiment_name)

        # Create the model
        master_logger.log("CREATING MODEL")
        model_type = a3c_lstm_ga_default
        shared_model = model_type(args, config=config, final_image_height=3, final_image_width=3)

        # make the shared model use share memory
        shared_model.share_memory()

        lstm_size = 256
        if isinstance(shared_model, a3c_lstm_ga_concat_gavector):
            lstm_size *= 3
        contextual_bandit = False
        model = ChaplotBaselineStreetView(args, shared_model, config, constants, tensorboard,
                                          use_contextual_bandit=contextual_bandit, lstm_size=lstm_size)

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
        train_chunk_size = int(len(train_split) / args.num_processes)
        tune_chunk_size = int(len(tune_split) / args.num_processes)
        train_pad = 0
        tune_pad = 0
        for i in range(0, args.num_processes):
            train_split_process_chunks.append(train_split[train_pad: train_pad + train_chunk_size])
            tune_split_process_chunks.append(tune_split[tune_pad: tune_pad + tune_chunk_size])
            train_pad += train_chunk_size
            tune_pad += tune_chunk_size

        # Start the training thread(s)
        for i in range(args.num_processes):
            train_chunk = train_split_process_chunks[i]
            tune_chunk = tune_split_process_chunks[i]
            print ("Client " + str(i) + " receives train-split of size %d and tune-split of size %d " %
                   (len(train_chunk), len(tune_chunk)))
            server = StreetViewServer(config, action_space, forward_setting_strict=False)
            client_logger = multiprocess_logging_manager.get_logger(i)
            p = mp.Process(target=ChaplotBaselineStreetView.do_train, args=(model, shared_model, config,
                                                                            action_space, meta_data_util, args,
                                                                            constants, train_chunk, tune_chunk,
                                                                            experiment, experiment_name, i, server,
                                                                            client_logger, model_type,
                                                                            contextual_bandit))

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
