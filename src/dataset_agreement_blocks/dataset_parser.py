from dataset_agreement.abstract_dataset_parser import AbstractDatasetParser
from dataset_agreement_blocks.datapoint import DataPoint


class DatasetParser(AbstractDatasetParser):

    @staticmethod
    def parse(file_name, config):

        if file_name == "trainset.json":
            dataset_size = 11871
        elif file_name == "devset.json":
            dataset_size = 1719
        elif file_name == "testset.json":
            dataset_size = 3177
        else:
            raise AssertionError("Unknown train file.")

        dataset = []
        for i in range(0, dataset_size):
            datapoint = DataPoint(i)
            dataset.append(datapoint)

        return dataset
