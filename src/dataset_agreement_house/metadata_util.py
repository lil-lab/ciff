import logging

from dataset_agreement.abstract_metadata_util import AbstractMetaDataUtil


class MetaDataUtil(AbstractMetaDataUtil):

    def log_results(self, metadata):
        logging.info("Meta Data: %r" % metadata)
