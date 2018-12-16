import json
import logging
import os
import sys
import traceback
import argparse
import torch.multiprocessing as mp

from baselines.chaplot_model_concat_gavector import a3c_lstm_ga_concat_gavector
from baselines.chaplot_model_default import a3c_lstm_ga_default
from baselines.chaplot_model_default_auxiliary import a3c_lstm_ga_default_with_aux
from dataset_agreement_nav_drone.action_space import ActionSpace
from dataset_agreement_nav_drone.metadata_util import MetaDataUtil
from dataset_agreement_nav_drone.nav_drone_dataset_parser import DatasetParser
from server_nav_drone.nav_drone_server_py3 import NavDroneServerPy3
from setup_agreement_nav_drone.validate_setup_nav_drone import NavDroneSetupValidator
from baselines.chaplot_baseline_multiprocess import ChaplotBaseline
from baselines.chaplot_baseline_multiprocess_with_auxiliary import ChaplotBaselineWithAuxiliary
from utils.check_port import find_k_ports
from utils.multiprocess_logger import MultiprocessingLoggerManager


def main():

    experiment_name = "a3c_ga_chaplot_baseline_6000paragraphs"
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

    print(args)
    with open("data/nav_drone/config_localmoves_6000.json") as f:
        config = json.load(f)
    with open("data/shared/contextual_bandit_constants.json") as f:
        constants = json.load(f)
    print(json.dumps(config,indent=2))
    setup_validator = NavDroneSetupValidator()
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
        shared_model = model_type(args, config=config)

        # make the shared model use share memory
        shared_model.share_memory()

        lstm_size = 256
        if isinstance(shared_model, a3c_lstm_ga_concat_gavector):
            lstm_size *= 3
        contextual_bandit = False
        model = ChaplotBaseline(args, shared_model, config, constants,
                                tensorboard, use_contextual_bandit=contextual_bandit, lstm_size=lstm_size)
        # model.load_image_text_model("./results/realdata_goal_prediction_supervised_trajectories"
        #                        "/chaplot_model_client_5_epoch_36")

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

        # Split the train data between processes
        train_split_process_chunks = []
        chunk_size = int(len(train_split)/args.num_processes)
        pad = 0
        for i in range(0, args.num_processes):
            chunk = train_split[pad: pad + chunk_size]
            pad += chunk_size
            train_split_process_chunks.append(chunk)

        # Start the training thread(s)
        ports = find_k_ports(args.num_processes)
        for i, port in enumerate(ports):
            train_chunk = train_split_process_chunks[i]
            tmp_config = {k: v for k, v in config.items()}
            tmp_config["port"] = port
            if i == args.num_processes - 1:
                # Master client which does testing. Don't want each client to do testing.
                tmp_tune_split = tune_split
            else:
                tmp_tune_split = []
            print ("Client " + str(i) + " getting a validation set of size ", len(tmp_tune_split))
            server = NavDroneServerPy3(tmp_config, action_space, multi_client=True)
            client_logger = multiprocess_logging_manager.get_logger(i)
            p = mp.Process(target=ChaplotBaseline.do_train, args=(model, shared_model, tmp_config,
                                                                               action_space, meta_data_util, args,
                                                                               constants, train_chunk, tmp_tune_split,
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
