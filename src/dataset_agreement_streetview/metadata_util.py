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


    def state_update_metadata(self, state, metadata):

        # For LANI, the metadata contains pose, position and orientation information
        pose = int(metadata["y_angle"] / 15.0)
        position_orientation = (metadata["x_pos"], metadata["z_pos"], metadata["y_angle"])

        state.pose = pose
        state.position_orientation = position_orientation

    def tensorboard_using_metadata(self, tensorboard, metadata):

        tensorboard.log_all_train_errors(
            metadata["edit_dist_error"], metadata["closest_dist_error"], metadata["stop_dist_error"])
