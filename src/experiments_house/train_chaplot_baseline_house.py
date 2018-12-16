import json
import logging
import os
import sys
import traceback
import argparse
import torch.multiprocessing as mp
import utils.generic_policy as gp

from baselines.chaplot_model_concat_gavector import a3c_lstm_ga_concat_gavector
from baselines.chaplot_model_house_default import a3c_lstm_ga_default
from dataset_agreement_house.action_space import ActionSpace
from dataset_agreement_house.metadata_util import MetaDataUtil
from dataset_agreement_house.dataset_parser import DatasetParser
from server_house.house_server import HouseServer
from setup_agreement_house.validate_setup_house import HouseSetupValidator
from utils.tensorboard import Tensorboard
from utils.multiprocess_logger import MultiprocessingLoggerManager
from utils.check_port import find_k_ports
from baselines.chaplot_baseline_house import ChaplotBaselineHouse


def main(args):

    experiment_name = "train_house_chaplot_house_baseline_postbugfix"
    experiment = "./results/" + experiment_name
    print("EXPERIMENT NAME: ", experiment_name)

    data_filename = "simulators/house/AssetsHouse"

    supervised = False

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

    # Test policy
    test_policy = gp.get_argmax_action

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

    # Create vocabulary
    vocab = dict()
    vocab_list = open(data_filename + "/house_all_vocab.txt").readlines()
    for i, tk in enumerate(vocab_list):
        token = tk.strip().lower()
        # vocab[token] = i
        vocab[i] = token
    # vocab["$UNK$"] = len(vocab_list)
    vocab[len(vocab_list)] = "$UNK$"

    args.input_size = config['vocab_size'] + 2

    # Number of processes
    house_ids = [1, 2, 3, 4, 5]
    num_processes = len(house_ids)
    args.num_processes = num_processes

    try:
        # Create the model
        master_logger.log("CREATING MODEL")
        model_type = a3c_lstm_ga_default
        shared_model = model_type(args, action_space=action_space, config=config)
        # shared_model = model_type(config, constants)

        # make the shared model use share memory
        shared_model.share_memory()

        lstm_size = 256
        if isinstance(shared_model, a3c_lstm_ga_concat_gavector):
            lstm_size *= 3
        model = ChaplotBaselineHouse(args, shared_model, config, constants, tensorboard=None,
                                     use_contextual_bandit=False, lstm_size=lstm_size)

        master_logger.log("MODEL CREATED")
        print("Created Model...")

        # Read the dataset
        tune_split, train_split = [], []
        for hid in house_ids:
            all_train_data = DatasetParser.parse(
                data_filename + "/tokenized_house" + str(hid) + "_discrete_train.json", config)
            all_dev_data = DatasetParser.parse(
                data_filename + "/tokenized_house" + str(hid) + "_discrete_dev.json", config)

            train_split.append(all_train_data)
            tune_split.append(all_dev_data)

            master_logger.log("Created train dataset of size {} ".format(len(all_train_data)))
            master_logger.log("Created tuning dataset of size {} ".format(len(all_dev_data)))

        # Start the training thread(s)
        ports = find_k_ports(num_processes)
        master_logger.log("Found K Ports")
        processes = []
        for i, port in enumerate(ports):
            train_chunk = train_split[i]
            print("Size of training data: {}".format(len(train_chunk)))
            tmp_config = {k: v for k, v in config.items()}
            tmp_config["port"] = port
            tmp_tune_split = tune_split[i]
            print("Client " + str(house_ids[i]) + " getting a validation set of size ", len(tmp_tune_split))
            server = HouseServer(tmp_config, action_space, port)
            client_logger = multiprocess_logging_manager.get_logger(i)

            # Run the Training
            p = mp.Process(target=ChaplotBaselineHouse.do_train, args=(house_ids[i], model, shared_model, tmp_config,
                                                                       action_space, meta_data_util,
                                                                       constants, train_chunk, tmp_tune_split,
                                                                       experiment, experiment_name, i, server,
                                                                       client_logger, model_type, vocab, args,
                                                                       False, lstm_size))
            p.daemon = False
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

    except Exception:
        # server.kill()
        exc_info = sys.exc_info()
        traceback.print_exception(*exc_info)
        # raise e


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Gated-Attention for Grounding')

    # Environment arguments
    parser.add_argument('--port', type=int, default=12345)
    parser.add_argument('-l', '--max-episode-length', type=int, default=40,
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
    parser.add_argument('-n', '--num-processes', type=int, default=4, metavar='N',
                        help='how many training processes to use (default: 4)')
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

    print("SETTING THE START METHOD ")
    mp.freeze_support()
    mp.set_start_method('spawn', force=True)
    main(args)
