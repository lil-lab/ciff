import logging

from dataset_agreement.abstract_metadata_util import AbstractMetaDataUtil


class MetaDataUtil(AbstractMetaDataUtil):

    def __init__(self):
        AbstractMetaDataUtil.__init__(self)

    def log_results(self, metadata, logger=None):
        if logger is not None:
            logger.log("StreetView Metadata: " + str(metadata))
        else:
            logging.info("StreetView Metadata: " + str(metadata))
