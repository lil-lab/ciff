import os

import torch

from agents.agent_observed_state import AgentObservedState
from models.module.segmentation_final_module import SegmentationFinalModule
from models.module.text_pointer_module import TextPointerModule
from utils.nav_drone_landmarks import get_all_landmark_names


class TextSegmentationModel(object):
    def __init__(self, config, constants):
        self.none_action = config["num_actions"]
        self.landmark_names = get_all_landmark_names()

        self.text_module = TextPointerModule(
            emb_dim=constants["word_emb_dim"],
            hidden_dim=constants["lstm_emb_dim"],
            vocab_size=config["vocab_size"]
        )
        self.final_module = SegmentationFinalModule(
            text_module=self.text_module,
            text_emb_size=4*constants["lstm_emb_dim"]
        )
        if torch.cuda.is_available():
            self.text_module.cuda()
            self.final_module.cuda()

    def get_segmentation_probs(self, agent_observed_state_list):
        for aos in agent_observed_state_list:
            assert isinstance(aos, AgentObservedState)
        # print "batch size:", len(agent_observed_state_list)

        # sort list by instruction length
        agent_observed_state_list = sorted(
            agent_observed_state_list,
            key=lambda aos_: len(aos_.get_instruction()),
            reverse=True
        )

        instructions = [aos.get_instruction()
                        for aos in agent_observed_state_list]
        read_pointers = [aos.get_read_pointers()
                         for aos in agent_observed_state_list]
        instructions_batch = (instructions, read_pointers)
        probs_batch = self.final_module(instructions_batch)

        return probs_batch

    def load_saved_model(self, load_dir):
        if torch.cuda.is_available():
            torch_load = torch.load
        else:
            torch_load = lambda f_: torch.load(f_, map_location=lambda s_, l_: s_)
        text_module_path = os.path.join(load_dir, "text_module_state.bin")
        self.text_module.load_state_dict(torch_load(text_module_path))
        final_module_path = os.path.join(load_dir, "final_module_state.bin")
        self.final_module.load_state_dict(torch_load(final_module_path))

    def save_model(self, save_dir):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # save state file for image nn
        text_module_path = os.path.join(save_dir, "text_module_state.bin")
        torch.save(self.text_module.state_dict(), text_module_path)
        # save state file for final nn
        final_module_path = os.path.join(save_dir, "final_module_state.bin")
        torch.save(self.final_module.state_dict(), final_module_path)

    def get_parameters(self):
        parameters = list(self.text_module.parameters())
        parameters += list(self.final_module.parameters())
        return parameters

    def get_named_parameters(self):
        named_parameters = list(self.text_module.named_parameters())
        named_parameters += list(self.final_module.named_parameters())
        return named_parameters
