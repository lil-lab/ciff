import os

import numpy as np
import torch
from models.module.action_simple_module import ActionSimpleModule
from models.module.image_ga_resnet_module import ImageGAResnetModule
from models.module.multimodal_attention_module import \
    MultimodalAttentionModule
from models.module.text_pointer_module import TextPointerModule

from agents.agent_observed_state import AgentObservedState
from models.model.abstract_model import AbstractModel
from models.module.text_simple_module import TextSimpleModule
from utils.cuda import cuda_var


class ModelGAResnet(AbstractModel):
    def __init__(self, config, constants):
        AbstractModel.__init__(self, config, constants)
        self.none_action = config["num_actions"]
        self.image_module = ImageGAResnetModule(
            image_emb_size=constants["image_emb_dim"],
            input_num_channels=3*constants["max_num_images"],
            image_height=config["image_height"],
            image_width=config["image_width"],
            text_emb_size=constants["lstm_emb_dim"])
        if config["use_pointer_model"]:
            self.text_module = TextPointerModule(
                emb_dim=constants["word_emb_dim"],
                hidden_dim=constants["lstm_emb_dim"],
                vocab_size=config["vocab_size"])
        else:
            self.text_module = TextSimpleModule(
                emb_dim=constants["word_emb_dim"],
                hidden_dim=constants["lstm_emb_dim"],
                vocab_size=config["vocab_size"])
        self.action_module = ActionSimpleModule(
            num_actions=config["num_actions"],
            action_emb_size=constants["action_emb_dim"])
        total_emb_size = (constants["image_emb_dim"]
                          + constants["action_emb_dim"])
        final_module = MultimodalAttentionModule(
            image_module=self.image_module,
            text_module=self.text_module,
            action_module=self.action_module,
            total_emb_size=total_emb_size,
            num_actions=config["num_actions"])
        self.final_module = final_module
        if torch.cuda.is_available():
            self.image_module.cuda()
            self.text_module.cuda()
            self.action_module.cuda()
            self.final_module.cuda()

    def get_probs_batch(self, agent_observed_state_list, mode=None):
        for aos in agent_observed_state_list:
            assert isinstance(aos, AgentObservedState)
        # print "batch size:", len(agent_observed_state_list)

        # sort list by instruction length
        agent_observed_state_list = sorted(
            agent_observed_state_list,
            key=lambda aos_: len(aos_.get_instruction()),
            reverse=True
        )

        images = [aos.get_image() for aos in agent_observed_state_list]
        image_batch = cuda_var(torch.from_numpy(np.array(images)).float())

        instructions = [aos.get_instruction()
                        for aos in agent_observed_state_list]
        read_pointers = [aos.get_read_pointers()
                         for aos in agent_observed_state_list]
        instructions_batch = (instructions, read_pointers)

        prev_actions_raw = [aos.get_previous_action()
                            for aos in agent_observed_state_list]
        prev_actions = [self.none_action if a is None else a
                        for a in prev_actions_raw]
        prev_actions_batch = cuda_var(torch.from_numpy(np.array(prev_actions)))

        probs_batch = self.final_module(image_batch, instructions_batch,
                                        prev_actions_batch, mode)
        return probs_batch

    def load_saved_model(self, load_dir):
        if torch.cuda.is_available():
            torch_load = torch.load
        else:
            torch_load = lambda f_: torch.load(f_, map_location=lambda s_, l_: s_)
        image_module_path = os.path.join(load_dir, "image_module_state.bin")
        self.image_module.load_state_dict(torch_load(image_module_path))
        text_module_path = os.path.join(load_dir, "text_module_state.bin")
        self.text_module.load_state_dict(torch_load(text_module_path))
        action_module_path = os.path.join(load_dir, "action_module_state.bin")
        self.action_module.load_state_dict(torch_load(action_module_path))
        final_module_path = os.path.join(load_dir, "final_module_state.bin")
        self.final_module.load_state_dict(torch_load(final_module_path))

    def save_model(self, save_dir):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # save state file for image nn
        image_module_path = os.path.join(save_dir, "image_module_state.bin")
        torch.save(self.image_module.state_dict(), image_module_path)
        # save state file for text nn
        text_module_path = os.path.join(save_dir, "text_module_state.bin")
        torch.save(self.text_module.state_dict(), text_module_path)
        # save state file for action emb
        action_module_path = os.path.join(save_dir, "action_module_state.bin")
        torch.save(self.action_module.state_dict(), action_module_path)
        # save state file for final nn
        final_module_path = os.path.join(save_dir, "final_module_state.bin")
        torch.save(self.final_module.state_dict(), final_module_path)

    def get_parameters(self):
        parameters = list(self.image_module.parameters())
        parameters += list(self.text_module.parameters())
        parameters += list(self.action_module.parameters())
        parameters += list(self.final_module.parameters())
        return parameters
