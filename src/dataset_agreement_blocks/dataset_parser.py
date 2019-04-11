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
            # Dataset for the block world is read on the simulator side. The simulator resets to an example
            # based on the shared config file containing the name of the split and an id of the example.
            # For this reason, the block world dataset contains only an ID.

            # TODO add other information for logging by also parsing the data on client side.
            datapoint = DataPoint(i)
            dataset.append(datapoint)

        return dataset
