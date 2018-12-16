import os
import numpy as np
import torch

from agents.agent_observed_state import AgentObservedState
from models.incremental_model.abstract_incremental_model import AbstractIncrementalModel
from models.incremental_module.incremental_multimodal_emnlp import IncrementalMultimodalEmnlp
from models.incremental_module.incremental_multimodal_emnlp_single_action_space import \
    IncrementalMultimodalEmnlpSingleActionSpace
from models.incremental_module.incremental_recurrence_chaplot_module import IncrementalRecurrenceChaplotModule
from models.incremental_module.tmp_blocks_incremental_multimodal_chaplot_module import \
    TmpBlocksIncrementalMultimodalChaplotModule
from models.module.action_simple_module import ActionSimpleModule
from models.module.chaplot_image_module import ChaplotImageModule
from models.module.chaplot_text_module import ChaplotTextModule
from models.module.goal_prediction_module import GoalPredictionModule
from models.module.image_cnn_emnlp import ImageCnnEmnlp
from models.module.image_simple_module import ImageSimpleModule
from models.module.object_detection_module import ObjectDetectionModule
from models.module.symbolic_language_prediction_module import SymbolicLanguagePredictionModule
from models.module.action_prediction_module import ActionPredictionModule
from models.module.temporal_autoencoder_module import TemporalAutoencoderModule
from models.module.text_simple_module import TextSimpleModule
from utils.cuda import cuda_var
from utils.nav_drone_landmarks import get_all_landmark_names


class IncrementalModelEmnlpSingleActionSpace(AbstractIncrementalModel):
    def __init__(self, config, constants):
        AbstractIncrementalModel.__init__(self, config, constants)
        self.none_action = config["num_actions"]

        self.config = config
        self.constants = constants

        # CNN over images - using what is essentially SimpleImage currently
        self.image_module = ImageCnnEmnlp(
            image_emb_size=constants["image_emb_dim"],
            input_num_channels=3*5, # 3 channels per image - 5 images in history
            image_height=config["image_height"],
            image_width=config["image_width"])

        # LSTM to embed text
        self.text_module = TextSimpleModule(
            emb_dim=constants["word_emb_dim"],
            hidden_dim=constants["lstm_emb_dim"],
            vocab_size=config["vocab_size"])

        # Action module to embed previous action+block
        self.action_module = ActionSimpleModule(
            num_actions=config["num_actions"],
            action_emb_size=constants["action_emb_dim"]
        )

        # Put it all together
        self.final_module = IncrementalMultimodalEmnlpSingleActionSpace(
            image_module=self.image_module,
            text_module=self.text_module,
            action_module=self.action_module,
            input_embedding_size=constants["lstm_emb_dim"] + constants["image_emb_dim"] + constants["action_emb_dim"],
            output_hidden_size=config["h1_hidden_dim"],
            blocks_hidden_size=config["blocks_hidden_dim"],
            directions_hidden_size=config["num_actions"], #this includes stop
            max_episode_length = (constants["horizon"] + 5)
        )

        if torch.cuda.is_available():
            self.image_module.cuda()
            self.text_module.cuda()
            self.action_module.cuda()
            self.final_module.cuda()


    def get_probs_batch(self, agent_observed_state_list, mode=None):
        raise NotImplementedError()

    def get_probs(self, agent_observed_state, model_state, mode=None, volatile=False):

        assert isinstance(agent_observed_state, AgentObservedState)

        # Image list is already padded with zero-images if <5 images are available
        images = agent_observed_state.get_image()[-5:]
        image_batch = cuda_var(torch.from_numpy(np.array(images)).float(), volatile)

        # Flatten them? TODO: maybe don't hardcode this later on? batch size is 1 ;)
        image_batch = image_batch.view(1, 3*5, self.config["image_height"], self.config["image_width"])

        # List of instructions. False is there because it expects a second argument. TODO: figure out what this is
        instructions_batch = ([agent_observed_state.get_instruction()], False)

        # Previous action
        prev_actions_raw = [agent_observed_state.get_previous_action()]

        # If previous action is non-existant then encode that as a stop?
        prev_actions = [self.none_action if a is None else a
                        for a in prev_actions_raw]
        prev_actions_batch = cuda_var(torch.from_numpy(np.array(prev_actions)))

        # Get probabilities
        probs_batch, new_model_state = self.final_module(
            image_batch, instructions_batch, prev_actions_batch, model_state
        )

        # last two we don't really need...
        return probs_batch, new_model_state, None, None

    def init_weights(self):
        self.text_module.init_weights()
        self.image_module.init_weights()
        self.action_module.init_weights()
        self.final_module.init_weights()

    def share_memory(self):
        self.image_module.share_memory()
        self.text_module.share_memory()
        self.action_module.share_memory()
        self.final_module.share_memory()

    def get_state_dict(self):
        nested_state_dict = dict()
        nested_state_dict["image_module"] = self.image_module.state_dict()
        nested_state_dict["text_module"] = self.text_module.state_dict()
        nested_state_dict["action_module"] = self.action_module.state_dict()
        nested_state_dict["final_module"] = self.final_module.state_dict()

        return nested_state_dict

    def load_from_state_dict(self, nested_state_dict):
        self.image_module.load_state_dict(nested_state_dict["image_module"])
        self.text_module.load_state_dict(nested_state_dict["text_module"])
        self.action_module.load_state_dict(nested_state_dict["action_module"])
        self.final_module.load_state_dict(nested_state_dict["final_module"])

    def load_resnet_model(self, load_dir):
        if torch.cuda.is_available():
            torch_load = torch.load
        else:
            torch_load = lambda f_: torch.load(f_, map_location=lambda s_, l_: s_)
        image_module_path = os.path.join(load_dir, "image_module_state.bin")
        self.image_module.load_state_dict(torch_load(image_module_path))

    def load_lstm_model(self, load_dir):
        if torch.cuda.is_available():
            torch_load = torch.load
        else:
            torch_load = lambda f_: torch.load(f_, map_location=lambda s_, l_: s_)
        text_module_path = os.path.join(load_dir, "text_module_state.bin")
        self.text_module.load_state_dict(torch_load(text_module_path))

    def load_saved_model(self, load_dir):
        if torch.cuda.is_available():
            torch_load = torch.load
        else:
            torch_load = lambda f_: torch.load(f_, map_location=lambda s_, l_: s_)
        image_module_path = os.path.join(load_dir, "image_module_state.bin")
        self.image_module.load_state_dict(torch_load(image_module_path))
        action_module_path = os.path.join(load_dir, "action_module_state.bin")
        self.action_module.load_state_dict(
            torch_load(action_module_path))
        text_module_path = os.path.join(load_dir, "text_module_state.bin")
        self.text_module.load_state_dict(torch_load(text_module_path))
        # action_module_path = os.path.join(load_dir, "action_module_state.bin")
        # self.action_module.load_state_dict(torch_load(action_module_path))
        final_module_path = os.path.join(load_dir, "final_module_state.bin")
        self.final_module.load_state_dict(torch_load(final_module_path))

    def save_model(self, save_dir):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # save state file for image nn
        image_module_path = os.path.join(save_dir, "image_module_state.bin")
        torch.save(self.image_module.state_dict(), image_module_path)
        # save state file for image recurrence nn
        action_module_path = os.path.join(
            save_dir, "action_module_state.bin")
        #torch.save(self.action_module.state_dict(),
        #           action_module_path)
        torch.save(self.action_module.state_dict(),
                   action_module_path)
        # save state file for text nn
        text_module_path = os.path.join(save_dir, "text_module_state.bin")
        torch.save(self.text_module.state_dict(), text_module_path)
        # save state file for action emb
        # action_module_path = os.path.join(save_dir, "action_module_state.bin")
        # torch.save(self.action_module.state_dict(), action_module_path)
        # save state file for final nn
        final_module_path = os.path.join(save_dir, "final_module_state.bin")
        torch.save(self.final_module.state_dict(), final_module_path)

    def get_parameters(self):
        # parameters = list(self.image_module.parameters())
        # parameters += list(self.action_module.parameters())
        # parameters += list(self.text_module.parameters())
        parameters = list(self.final_module.parameters())

        return parameters

    def get_named_parameters(self):
        # named_parameters = list(self.image_module.named_parameters())
        # named_parameters += list(self.action_module.named_parameters())
        # named_parameters += list(self.text_module.named_parameters())
        named_parameters = list(self.final_module.named_parameters())
        return named_parameters
