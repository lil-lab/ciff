import json
import logging
import os

from agents.agent_no_model import Agent
from dataset_agreement_streetview.action_space import ActionSpace
from dataset_agreement_streetview.metadata_util import MetaDataUtil
from dataset_agreement_streetview.dataset_parser import DatasetParser
from server_streetview.streetview_server import StreetViewServer
from setup_agreement_streetview.validate_setup_streetview import StreetViewSetupValidator


def main():

    experiment_name = "streetview_test_stop_dev"
    experiment = "./results/" + experiment_name
    print("EXPERIMENT NAME: ", experiment_name)

    # Create the experiment folder
    if not os.path.exists(experiment):
        os.makedirs(experiment)

    # Define log settings
    logging.basicConfig(filename=experiment + '/baseline_info.log',
                        level=logging.INFO)
    logging.info("----------------------------------------------------------------")
    logging.info("                    STARING NEW EXPERIMENT                      ")
    logging.info("----------------------------------------------------------------")

    with open("data/streetview/config.json") as f:
        config = json.load(f)
    with open("data/shared/contextual_bandit_constants.json") as f:
        constants = json.load(f)
    print(json.dumps(config, indent=2))

    setup_validator = StreetViewSetupValidator()
    setup_validator.validate(config, constants)

    # log core experiment details
    logging.info("CONFIG DETAILS")
    for k, v in sorted(config.items()):
        logging.info("    %s --- %r" % (k, v))
        logging.info("CONSTANTS DETAILS")
    for k, v in sorted(constants.items()):
        logging.info("    %s --- %r" % (k, v))
        logging.info("START SCRIPT CONTENTS")
    with open(__file__) as f:
        for line in f.readlines():
            logging.info(">>> " + line.strip())
    logging.info("END SCRIPT CONTENTS")

    action_space = ActionSpace(config["action_names"], config["stop_action"])
    meta_data_util = MetaDataUtil()

    vocab = dict()
    vocab_list = open(config["vocab_file"]).readlines()
    for i, tk in enumerate(vocab_list):
        token = tk.strip().lower()
        vocab[i] = token
    vocab[len(vocab_list)] = "$UNK$"
    config["vocab_size"] = len(vocab_list) + 1

    # Create the server
    server = StreetViewServer(config, action_space, forward_setting_strict=False)

    # Read the dataset
    test_data = DatasetParser.parse("data/streetview/navigation_dev.json", config)

    agent = Agent(Agent.STOP, server, action_space, meta_data_util, constants)
    agent.test(test_data)


if __name__ == "__main__":
    print("SETTING THE START METHOD ")
    main()
