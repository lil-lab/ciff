import os

import torch
from models.module.text_classification_module import TextClassificationModule
from models.module.text_simple_module import TextSimpleModule
from models.module.text_bilstm_module import TextBiLSTMModule

from agents.agent_observed_state import AgentObservedState
from models.model.abstract_model import AbstractModel


class ModelSymbolicTextPrediction(AbstractModel):
    def __init__(self, config, constants):
        AbstractModel.__init__(self, config, constants)
        self.none_action = config["num_actions"]

        self.text_module = TextSimpleModule(
            emb_dim=constants["word_emb_dim"],
            hidden_dim=constants["lstm_emb_dim"],
            vocab_size=config["vocab_size"])

        total_emb_size = constants["lstm_emb_dim"]
        final_module = TextClassificationModule(
            text_module=self.text_module,
            total_emb_size=total_emb_size)
        self.final_module = final_module
        if torch.cuda.is_available():
            self.text_module.cuda()
            self.final_module.cuda()

    def get_probs_batch(self, agent_observed_state_list, mode=None):
        raise AssertionError("Not a policy")

    def get_symbolic_text_batch(self, agent_observed_state_list):

        for aos in agent_observed_state_list:
            assert isinstance(aos, AgentObservedState)

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

        return self.final_module(instructions_batch)

    def load_saved_model(self, load_dir):
        if torch.cuda.is_available():
            torch_load = torch.load
        else:
            torch_load = lambda f_: torch.load(f_, map_location=lambda s_, l_: s_)
        # image_module_path = os.path.join(load_dir, "image_module_state.bin")
        # self.image_module.load_state_dict(torch_load(image_module_path))
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
        # image_module_path = os.path.join(save_dir, "image_module_state.bin")
        # torch.save(self.image_module.state_dict(), image_module_path)
        # save state file for text nn
        text_module_path = os.path.join(save_dir, "text_module_state.bin")
        torch.save(self.text_module.state_dict(), text_module_path)
        # # save state file for action emb
        # action_module_path = os.path.join(save_dir, "action_module_state.bin")
        # torch.save(self.action_module.state_dict(), action_module_path)
        # save state file for final nn
        final_module_path = os.path.join(save_dir, "final_module_state.bin")
        torch.save(self.final_module.state_dict(), final_module_path)

    def get_parameters(self):
        # parameters = list(self.image_module.parameters())
        parameters = list(self.text_module.parameters())
        # parameters += list(self.action_module.parameters())
        parameters += list(self.final_module.parameters())
        return parameters

    def get_named_parameters(self):
        parameters = list(self.text_module.named_parameters())
        parameters += list(self.final_module.named_parameters())
        return parameters

