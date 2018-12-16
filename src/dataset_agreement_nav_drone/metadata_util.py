import logging

from dataset_agreement.abstract_metadata_util import AbstractMetaDataUtil


class MetaDataUtil(AbstractMetaDataUtil):

    def log_results(self, metadata, logger=None):
        if logger is not None:
            logger.log("Meta Data Feedback: " + str(metadata["feedback"]))
        else:
            logging.info("Meta Data Feedback: " + str(metadata["feedback"]))
