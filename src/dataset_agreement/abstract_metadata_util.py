class AbstractMetaDataUtil:

    def __init__(self):
        pass

    def log_results(self, metadata):
        """ Log the metadata """
        raise NotImplementedError()

    def start_state_update_metadata(self, state, metadata):
        """ Given a starting agent observed state and a received metadata dictionary, perform
                updates to the state."""
        raise NotImplementedError()

    def state_update_metadata(self, state, metadata):
        """ Given an agent observed state (not a start) and a received metadata dictionary, perform
        updates to the state."""
        raise NotImplementedError()

    def tensorboard_using_metadata(self, tensorboard, metadata):
        """ Update the tensorboard using result in the metada """
        raise NotImplementedError()