import logging

from dataset_agreement.abstract_metadata_util import AbstractMetaDataUtil


class MetaDataUtil(AbstractMetaDataUtil):

    def __init__(self):
        AbstractMetaDataUtil.__init__(self)

    def log_results(self, metadata, logger=None):
        if logger is not None:
            logger.log(metadata)
        else:
            logging.info(metadata)

    def start_state_update_metadata(self, state, metadata):
        # In block world the instruction are provided by the server using metadata
        state.instruction = metadata["instruction"]

    def state_update_metadata(self, state, metadata):
        pass

    def tensorboard_using_metadata(self, tensorboard, metadata):
        pass