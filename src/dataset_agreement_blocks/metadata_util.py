import logging

from dataset_agreement.abstract_metadata_util import AbstractMetaDataUtil


class MetaDataUtil(AbstractMetaDataUtil):

    def __init__(self):
        pass

    def log_results(self, metadata, logger=None):
        if logger is not None:
            logger.log(metadata)
        else:
            logging.info(metadata)

    def state_update_metadata(self, state, metadata):
        pass

    def tensorboard_using_metadata(self, tensorboard, metadata):
        pass