import os
import numpy as np
import torch

from agents.agent_observed_state import AgentObservedState
from models.incremental_model.abstract_incremental_model import AbstractIncrementalModel
from models.incremental_module.incremental_multimodal_attention_chaplot_module import \
    IncrementalMultimodalAttentionChaplotModule, IncrementalMultimodalAttentionChaplotModuleM4JKSUM1
from models.incremental_module.incremental_multimodal_chaplot_module import IncrementalMultimodalChaplotModule
from models.incremental_module.incremental_multimodal_dense_valts_recurrent_simple_module import \
    IncrementalMultimodalDenseValtsRecurrentSimpleModule
from models.incremental_module.incremental_multimodal_valts_recurrent_simple_module import \
    IncrementalMultimodalValtsRecurrentSimpleModule
from models.incremental_module.incremental_recurrence_chaplot_module import IncrementalRecurrenceChaplotModule
from models.incremental_module.incremental_unet_attention_module import IncrementalUnetAttentionModule, \
    IncrementalUnetAttentionModuleJustProb
from models.incremental_module.tmp_incremental_multimodal_dense_valts_recurrent_simple_module import \
    TmpIncrementalMultimodalDenseValtsRecurrentSimpleModule
from models.module.action_simple_module import ActionSimpleModule
from models.module.chaplot_bilstm_text_module import ChaplotBiLSTMTextModule
from models.module.chaplot_image_module import ChaplotImageModule
from models.module.chaplot_text_module import ChaplotTextModule
from models.module.goal_prediction_module import GoalPredictionModule
from models.module.image_chaplot_resnet_module import ImageChaplotResnetModule
from models.module.image_resnet_module import ImageResnetModule
from models.incremental_module.incremental_multimodal_recurrent_simple_module \
    import IncrementalMultimodalRecurrentSimpleModule
from models.module.object_detection_module import ObjectDetectionModule
from models.module.pixel_identification_module import PixelIdentificationModule
from models.module.symbolic_language_prediction_module import SymbolicLanguagePredictionModule
from models.module.text_bilstm_module import TextBiLSTMModule
from models.module.text_pointer_module import TextPointerModule
from models.module.text_simple_module import TextSimpleModule
from models.module.action_prediction_module import ActionPredictionModule
from models.module.temporal_autoencoder_module import TemporalAutoencoderModule
from models.incremental_module.incremental_recurrence_simple_module import IncrementalRecurrenceSimpleModule
from models.module.unet_image_module import UnetImageModule
from models.resnet_image_detection import ImageDetectionModule
from utils.cuda import cuda_var, cuda_tensor
from utils.nav_drone_landmarks import get_all_landmark_names
from utils.debug_nav_drone_instruction import instruction_to_string


class IncrementalEnsembleOfModels(AbstractIncrementalModel):
    """ A generic class that is used for doing inference with ensemble of models """

    def __init__(self, base_models, config, constants):
        AbstractIncrementalModel.__init__(self, config, constants)
        self.base_models = base_models
        self.config = config
        self.constants = constants

    def get_probs_batch(self, agent_observed_state_list, mode=None):
        raise NotImplementedError()

    def get_probs(self, agent_observed_state, model_state, mode=None, volatile=False):

        new_ensemble_model_state = []
        base_probs_batch = []
        for base_model_ix, base_model in enumerate(self.base_models):
            if model_state is None:
                base_model_state = None
            else:
                base_model_state = model_state[base_model_ix]
            probs_batch, new_model_state, image_emb_seq, state_feature = base_model.get_probs(
                agent_observed_state, base_model_state, mode, volatile)
            new_ensemble_model_state.append(new_model_state)
            base_probs_batch.append(probs_batch)

        mixing_coeff = [0.5, 0.5]
        overall_prob_batch = None
        for ix, prob_batch in enumerate(base_probs_batch):
            if overall_prob_batch is None:
                overall_prob_batch = mixing_coeff[ix] * prob_batch
            else:
                overall_prob_batch += mixing_coeff[ix] * prob_batch

        return overall_prob_batch, new_ensemble_model_state, None, None

    def init_weights(self):
        for base_model in self.base_models:
            base_model.init_weights()

    def share_memory(self):
        for base_model in self.base_models:
            base_model.share_memory()

    def load_saved_model(self, load_dirs):
        for base_model_ix, base_model in enumerate(self.base_models):
            base_model.load_saved_model(load_dirs[base_model_ix])
