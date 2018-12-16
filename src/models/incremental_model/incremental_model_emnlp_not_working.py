import os
import numpy as np
import torch

from agents.agent_observed_state import AgentObservedState
from models.incremental_model.abstract_incremental_model import AbstractIncrementalModel
from models.incremental_module.incremental_multimodal_emnlp import IncrementalMultimodalEmnlp
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


class IncrementalModelEmnlp(AbstractIncrementalModel):
    def __init__(self, config, constants):
        AbstractIncrementalModel.__init__(self, config, constants)
        self.none_action = config["num_actions"]

        self.config = config
        self.constants = constants

        # CNN over images - using SimpleImage for testing for now!
        self.image_module = ImageCnnEmnlp(
            image_emb_size=config["image_emb_dim"],
            input_num_channels=3*5, #3 channels per image - 5 images in history
            image_height=config["image_height"],
            image_width=config["image_width"])

        # this is somewhat counter intuitivie - emb_dim is the word size
        # hidden_size is the output size
        self.text_module = TextSimpleModule(
            emb_dim=config["word_emb_dim"],
            hidden_dim=config["lstm_emb_dim"],
            vocab_size=config["vocab_size"])

        self.previous_action_module = ActionSimpleModule(
            num_actions=config["no_actions"],
            action_emb_size=config["previous_action_embedding_dim"]
        )

        self.previous_block_module = ActionSimpleModule(
            num_actions=config["no_blocks"],
            action_emb_size=config["previous_block_embedding_dim"]
        )

        self.final_module = IncrementalMultimodalEmnlp(
            image_module=self.image_module,
            text_module=self.text_module,
            previous_action_module=self.previous_action_module,
            previous_block_module=self.previous_block_module,
            input_embedding_size=config["lstm_emb_dim"] + config["image_emb_dim"] + config["previous_action_embedding_dim"] + config["previous_block_embedding_dim"],
            output_hidden_size=config["h1_hidden_dim"],
            blocks_hidden_size=config["no_blocks"],
            directions_hidden_size=config["no_actions"],
            max_episode_length = (constants["horizon"] + 5)
        )

        if torch.cuda.is_available():
            self.image_module.cuda()
            self.text_module.cuda()
            self.previous_action_module.cuda()
            self.previous_block_module.cuda()
            self.final_module.cuda()


    def get_probs_batch(self, agent_observed_state_list, mode=None):
        raise NotImplementedError()

    def get_probs(self, agent_observed_state, model_state, mode=None, volatile=False):

        assert isinstance(agent_observed_state, AgentObservedState)

        #supposedly this is already padded with zeros, but i need to double check that code
        images = agent_observed_state.get_image()[-5:]

        # image_seqs = [[aos.get_last_image()]
        #               for aos in agent_observed_state_list]
        image_batch = cuda_var(torch.from_numpy(np.array(images)).float(), volatile)

        #flatten them? TODO: maybe don't hardcode this later on? batch size is 1 ;)
        image_batch = image_batch.view(1, 15, 128, 128)

        # list of list :)
        instructions_batch = ([agent_observed_state.get_instruction()], False)
        #instructions_batch = (cuda_var(torch.from_numpy(np.array(instructions)).long()), False)

        #print("instructions", instructions)
        #print("instructins_batch", instructions_batch)

        prev_actions_raw = agent_observed_state.get_previous_action()
        prev_actions_raw = self.none_action if prev_actions_raw is None else prev_actions_raw


        if prev_actions_raw == 81:
            previous_direction_id = [4]
        else:
            previous_direction_id = [prev_actions_raw % 4]
        #this input is is over the space 81 things :)
        previous_block_id = [int(prev_actions_raw / 4)]

        prev_block_id_batch = cuda_var(torch.from_numpy(np.array(previous_block_id)))
        prev_direction_id_batch = cuda_var(torch.from_numpy(np.array(previous_direction_id)))

        # prev_actions = [self.none_action if a is None else a
        #                 for a in prev_actions_raw]
        #prev_actions_batch = cuda_var(torch.from_numpy(np.array(prev_actions)))

        probs_batch, new_model_state = self.final_module(
            image_batch, instructions_batch, prev_block_id_batch, prev_direction_id_batch, model_state
        )

        # last two we don't really need...
        return probs_batch, new_model_state, None, None

    def init_weights(self):
        self.text_module.init_weights()
        self.image_module.init_weights()
        self.previous_action_module.init_weights()
        self.previous_block_module.init_weights()
        self.final_module.init_weights()

    def share_memory(self):
        self.image_module.share_memory()
        self.text_module.share_memory()
        self.previous_action_module.share_memory()
        self.previous_block_module.share_memory()
        self.final_module.share_memory()

    def get_state_dict(self):
        nested_state_dict = dict()
        nested_state_dict["image_module"] = self.image_module.state_dict()
        nested_state_dict["text_module"] = self.text_module.state_dict()
        nested_state_dict["previous_action_module"] = self.previous_action_module.state_dict()
        nested_state_dict["previous_block_module"] = self.previous_block_module.state_dict()
        nested_state_dict["final_module"] = self.final_module.state_dict()

        return nested_state_dict

    def load_from_state_dict(self, nested_state_dict):
        self.image_module.load_state_dict(nested_state_dict["image_module"])
        self.text_module.load_state_dict(nested_state_dict["text_module"])
        self.previous_action_module.load_state_dict(nested_state_dict["previous_action_module"])
        self.previous_block_module.load_state_dict(nested_state_dict["previous_block_module"])
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

        previous_action_module_path = os.path.join(load_dir, "previous_action_module_state.bin")
        self.previous_action_module.load_state_dict(
            torch_load(previous_action_module_path))

        previous_block_module_path = os.path.join(load_dir, "previous_block_module_state.bin")
        self.previous_block_module.load_state_dict(
            torch_load(previous_block_module_path))

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


        previous_action_module_path = os.path.join(
            save_dir, "previous_action_module_state.bin")
        torch.save(self.previous_action_module.state_dict(),
                   previous_action_module_path)

        previous_block_module_path = os.path.join(
            save_dir, "previous_block_module_state.bin")
        torch.save(self.previous_block_module.state_dict(),
                   previous_block_module_path)

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
