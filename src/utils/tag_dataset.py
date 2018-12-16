import nltk
import json
import sys

from utils.debug_nav_drone_instruction import instruction_to_string
from dataset_agreement_nav_drone.nav_drone_dataset_parser import DatasetParser
from setup_agreement_nav_drone.validate_setup_nav_drone import \
    NavDroneSetupValidator


def tag_dataset(dataset, config):
    noun_set = dict([])

    for data_point in dataset:
        instruction = instruction_to_string(
            data_point.get_instruction(), config)

        token_seq = nltk.tokenize.word_tokenize(instruction)
        tagger = nltk.pos_tag(token_seq)
        for tag in tagger:
            if tag[1] == "NN" or tag[1] == "NNP":
                noun = tag[0].lower()
                if noun in noun_set:
                    noun_set[noun] += 1
                else:
                    noun_set[noun] = 1

    sorted_nouns = sorted(noun_set.items(), key=lambda x: -x[1])
    print "Noun set is " + str(sorted_nouns)


with open("data/nav_drone/config_localmoves_4000.json") as f:
    config = json.load(f)
with open("data/shared/full_recurrence_contextual_bandit_constants.json") as f:
    constants = json.load(f)
if len(sys.argv) > 1:
    config["port"] = int(sys.argv[1])
setup_validator = NavDroneSetupValidator()
setup_validator.validate(config, constants)

# Read the dataset
all_train_data = DatasetParser.parse("data/nav_drone/train_annotations_4000.json", config)
num_train = (len(all_train_data) * 19) // 20
while all_train_data[num_train].get_scene_name().split("_")[1] == all_train_data[num_train - 1].get_scene_name().split("_")[1]:
   num_train += 1
train_split = all_train_data[:num_train]
tune_split = all_train_data[num_train:]

tag_dataset(train_split, config)
