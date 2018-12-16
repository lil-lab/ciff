import math
import os

import numpy as np
import torch
from models.module.action_simple_module import ActionSimpleModule
from models.module.multimodal_simple_module import MultimodalSimpleModule
from models.module.text_pointer_module import TextPointerModule
from models.module.text_simple_module import TextSimpleModule

from agents.agent_observed_state import AgentObservedState
from models.model.abstract_model import AbstractModel
from models.module.symbolic_embddings import AngleModule
from models.module.symbolic_embddings import RadiusModule, LandmarkModule
from models.module.symbolic_image_module import SymbolicImageModule
from utils.cuda import cuda_var
from utils.nav_drone_landmarks import get_all_landmark_names


class ModelPolicyNetworkSymbolic360(AbstractModel):
    def __init__(self, config, constants):
        AbstractModel.__init__(self, config, constants)
        self.none_action = config["num_actions"]
        landmark_names = get_all_landmark_names()
        self.radius_module = RadiusModule(15)
        self.angle_module = AngleModule(48)
        self.landmark_module = LandmarkModule(63)
        self.image_module = SymbolicImageModule(
            landmark_names=landmark_names,
            radius_module=self.radius_module,
            angle_module=self.angle_module,
            landmark_module=self.landmark_module)
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
        total_emb_size = (32 * 3 * 63
                          + constants["lstm_emb_dim"]
                          + constants["action_emb_dim"])
        final_module = MultimodalSimpleModule(
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
            self.radius_module.cuda()
            self.angle_module.cuda()
            self.landmark_module.cuda()

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

        symbolic_image_list = []
        for aos in agent_observed_state_list:
            x_pos, z_pos, y_angle = aos.get_position_orientation()
            landmark_pos_dict = aos.get_landmark_pos_dict()
            symbolic_image = get_visible_landmark_r_theta(
                x_pos, z_pos, y_angle, landmark_pos_dict)
            symbolic_image_list.append(symbolic_image)
        image_batch = symbolic_image_list

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
        radius_module_path = os.path.join(load_dir, "radius_module_state.bin")
        self.radius_module.load_state_dict(torch_load(radius_module_path))
        angle_module_path = os.path.join(load_dir, "angle_module_state.bin")
        self.angle_module.load_state_dict(torch_load(angle_module_path))
        landmark_module_path = os.path.join(load_dir, "landmark_module_state.bin")
        self.landmark_module.load_state_dict(torch_load(landmark_module_path))

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
        # save state file for radius nn
        radius_module_path = os.path.join(save_dir, "radius_module_state.bin")
        torch.save(self.radius_module.state_dict(), radius_module_path)
        # save state file for angle nn
        angle_module_path = os.path.join(save_dir, "angle_module_state.bin")
        torch.save(self.angle_module.state_dict(), angle_module_path)
        # save state file for landmark nn
        landmark_module_path = os.path.join(save_dir, "landmark_module_state.bin")
        torch.save(self.landmark_module.state_dict(), landmark_module_path)
        # save state file for radius nn
        radius_module_path = os.path.join(save_dir, "radius_module_state.bin")
        torch.save(self.radius_module.state_dict(), radius_module_path)
        # save state file for angle nn
        angle_module_path = os.path.join(save_dir, "angle_module_state.bin")
        torch.save(self.angle_module.state_dict(), angle_module_path)
        # save state file for landmark nn
        landmark_module_path = os.path.join(save_dir, "landmark_module_state.bin")
        torch.save(self.landmark_module.state_dict(), landmark_module_path)

    def get_parameters(self):
        parameters = list(self.image_module.parameters())
        parameters += list(self.text_module.parameters())
        parameters += list(self.action_module.parameters())
        parameters += list(self.final_module.parameters())
        parameters += list(self.radius_module.parameters())
        parameters += list(self.angle_module.parameters())
        parameters += list(self.landmark_module.parameters())
        parameters += list(self.radius_module.parameters())
        parameters += list(self.angle_module.parameters())
        parameters += list(self.landmark_module.parameters())
        return parameters


def get_visible_landmark_r_theta(x_pos, z_pos, y_angle, landmark_pos_dict):
    landmark_r_theta_dict = {}
    for landmark, (landmark_x, landmark_z) in landmark_pos_dict.iteritems():
        # get angle between drone's current orientation and landmark
        landmark_angle = 90.0 - np.arctan2(landmark_z - z_pos, landmark_x - x_pos) * 180.0 / math.pi
        angle_diff = landmark_angle - y_angle
        while angle_diff > 180.0:
            angle_diff -= 360.0
        while angle_diff < -180.0:
            angle_diff += 360.0
        angle_discrete = int((angle_diff + 180.0) / 7.5)

        # get discretized radius
        radius = ((landmark_x - x_pos) ** 2 + (landmark_z - z_pos) ** 2) ** 0.5
        radius_discrete = int(radius / 5.0)

        landmark_r_theta_dict[landmark] = (radius_discrete, angle_discrete)
    return landmark_r_theta_dict
