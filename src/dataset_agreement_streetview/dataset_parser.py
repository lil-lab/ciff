import json
import nltk
import ast

from dataset_agreement.abstract_dataset_parser import AbstractDatasetParser
from dataset_agreement_streetview.datapoint import DataPoint


class DatasetParser(AbstractDatasetParser):

    @staticmethod
    def parse(file_name, config):

        # Read the vocabulary
        vocab = [token.strip() for token in open(config["vocab_file"]).readlines()]
        vocab_dict = dict()
        for i, tk in enumerate(vocab):
            vocab_dict[tk] = i

        lines = open(file_name).readlines()
        dataset = []

        for line in lines:
            jobj = json.loads(line)

            datapoint_id = jobj["route_id"]
            instruction_string = jobj["navigation_text"].strip().lower()
            instruction_token_seq = DatasetParser.tokenize(instruction_string, vocab_dict)
            route_panoids = jobj["route_panoids"]
            start_orientation = jobj["start_heading"]
            end_orientation = jobj["end_heading"]

            pre_static_center_ast = ast.literal_eval(jobj["pre_static_center"])
            post_static_center_ast = ast.literal_eval(jobj["post_static_center"])

            pre_static_center = (int(pre_static_center_ast["x"]), int(pre_static_center_ast["y"]))
            post_static_center = (int(post_static_center_ast["x"]), int(post_static_center_ast["y"]))
            pre_pano = jobj["pre_pano"]
            post_pano = jobj["post_pano"]

            pre_static_center_exists = True
            post_static_center_exists = True

            if pre_static_center == (-1, -1):
                pre_static_center_exists = False
            if post_static_center == (-1, -1):
                post_static_center_exists = False

            datapoint = DataPoint(datapoint_id, instruction_token_seq, instruction_string,
                                  route_panoids, datapoint_id, start_orientation, end_orientation,
                                  pre_static_center_exists, post_static_center_exists, pre_pano, post_pano)
            dataset.append(datapoint)

        return dataset

    @staticmethod
    def tokenize(instruction, vocab_dict):

        tokens = nltk.tokenize.word_tokenize(instruction)
        return [vocab_dict[tk] for tk in tokens]
