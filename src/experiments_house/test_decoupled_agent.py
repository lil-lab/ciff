import json
import logging
import os
import sys
import traceback

import torch
import torch.multiprocessing as mp

from random import shuffle

from agents.house_decoupled_predictor_navigator_model import HouseDecoupledPredictorNavigatorAgent
from dataset_agreement_house.action_space import ActionSpace
from dataset_agreement_house.metadata_util import MetaDataUtil
from dataset_agreement_house.dataset_parser import DatasetParser
from learning.asynchronous.tmp_house_asynchronous_contextual_bandit_learning import TmpAsynchronousContextualBandit
from models.incremental_model.incremental_model_attention_chaplot_resnet import IncrementalModelAttentionChaplotResNet
from models.incremental_model.tmp_house_incremental_model_chaplot import TmpHouseIncrementalModelChaplot
from models.incremental_model.tmp_house_incremental_model_oracle_gold_prob import TmpHouseIncrementalModelOracleGoldProb
from models.module.action_type_module import ActionTypeModule
from server_house.house_server import HouseServer
from setup_agreement_house.validate_setup_house import HouseSetupValidator
from utils.check_port import find_k_ports
from utils.multiprocess_logger import MultiprocessingLoggerManager


def main():

    data_filename = "simulators/house/AssetsHouse"
    experiment_name = "emnlp-camera-ready-figure-plot"
    experiment = "./results/" + experiment_name
    print("EXPERIMENT NAME: ", experiment_name)

    # Create the experiment folder
    if not os.path.exists(experiment):
        os.makedirs(experiment)

    # Define log settings
    log_path = experiment + '/test_baseline_inferred_oos.log'
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

    # Validate the setting
    setup_validator = HouseSetupValidator()
    setup_validator.validate(config, constants)

    # Log core experiment details
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

    # TODO: Create vocabulary
    vocab = dict()
    vocab_list = open(data_filename + "/house_all_vocab.txt").readlines()
    for i, tk in enumerate(vocab_list):
        token = tk.strip().lower()
        # vocab[token] = i
        vocab[i] = token
    # vocab["$UNK$"] = len(vocab_list)
    vocab[len(vocab_list)] = "$UNK$"
    config["vocab_size"] = len(vocab_list) + 1

    # Number of processes
    house_ids = [1, 2, 3, 4, 5]
    num_processes = len(house_ids)

    try:
        master_logger.log("CREATING MODEL")

        # Create the goal prediction model
        # shared_goal_prediction_model = IncrementalModelAttentionChaplotResNet(
        #     config, constants, final_model_type="m4jksum1", final_dimension=(64, 32, 32 * 6))
        shared_goal_prediction_model = IncrementalModelAttentionChaplotResNet(
            config, constants, final_model_type="unet-positional-encoding", final_dimension=(64, 32, 32 * 6))
        shared_goal_prediction_model.load_saved_model(
            "./results/house_goal_prediction/goal_prediction_single_supervised_epoch_4")
        # shared_goal_prediction_model.load_saved_model(
        #     "./results/train_house_goal_prediction_m4jksum1_repeat/goal_prediction_single_supervised_epoch_4")
        # shared_goal_prediction_model.load_saved_model(
        #     "./results/train_house_two_stage_model/predictor_contextual_bandit_2_epoch_2")
        # shared_goal_prediction_model.load_saved_model(
        #     "./results/train_house_goal_prediction_dummy_token/goal_prediction_single_supervised_epoch_9")

        # Create the navigation model
        model_type = TmpHouseIncrementalModelOracleGoldProb  # TmpHouseIncrementalModelChaplot
        shared_navigator_model = model_type(config, constants, use_image=False)
        # shared_navigator_model.load_saved_model(
        #     "./results/train_house_two_stage_model/navigator_contextual_bandit_2_epoch_2")
        shared_navigator_model.load_saved_model(
            "./results/house_unet_cb_navigation_gold_goal/contextual_bandit_0_epoch_5")
        # shared_navigator_model.load_saved_model(
        #     "./results/house_unet_cb_navigation_gold_goal_no_RNN/contextual_bandit_0_epoch_10")

        # Create the action type model
        shared_action_type_model = ActionTypeModule()
        shared_action_type_model.cuda()
        shared_action_type_model.load_state_dict(
            torch.load("./results/train_house_action_types/goal_prediction_single_supervised_epoch_7"))

        # make the shared models use share memory
        shared_goal_prediction_model.share_memory()
        shared_navigator_model.share_memory()
        shared_action_type_model.share_memory()

        master_logger.log("MODEL CREATED")
        print("Created Model...")

        # Read the dataset
        test_split = []
        for hid in house_ids:
            all_test_data = DatasetParser.parse(
                data_filename + "/tokenized_house" + str(hid) + "_discrete_dev.json", config)

            # # Extract type of the dataset
            # lines = open("./simulators/house/datapoint_type_house" + str(hid) + ".txt").readlines()
            # datapoint_id_type = {}
            # for line in lines:
            #     words = line.split()
            #     datapoint_id, datapoint_type = words[0], words[1:]
            #     datapoint_id_type[int(datapoint_id)] = datapoint_type  # .strip()
            #
            # # Filter manipulation type
            # all_test_data = list(filter(lambda datapoint: "manipulation" not in datapoint_id_type[datapoint.get_id()],
            #                       all_test_data))

            test_split.append(all_test_data)

        processes = []

        # Start the training thread(s)
        ports = find_k_ports(num_processes)
        master_logger.log("Found K Ports")
        for i, port in enumerate(ports):
            test_chunk = test_split[i]
            tmp_config = {k: v for k, v in config.items()}
            tmp_config["port"] = port
            print("Client " + str(i) + " getting a test set of size ", len(test_chunk))
            server = HouseServer(tmp_config, action_space, port)
            master_logger.log("Server Initialized")
            client_logger = multiprocess_logging_manager.get_logger(i)
            p = mp.Process(target=HouseDecoupledPredictorNavigatorAgent.do_test, args=(house_ids[i],
                                                                                       shared_goal_prediction_model,
                                                                                       shared_navigator_model,
                                                                                       shared_action_type_model,
                                                                                       tmp_config, action_space,
                                                                                       meta_data_util, constants,
                                                                                       test_chunk, experiment_name,
                                                                                       i, server,
                                                                                       client_logger, vocab, "inferred"))
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
    mp.set_start_method('spawn', force=True)
    main()
