import os
import numpy as np
import torch

from agents.agent_observed_state import AgentObservedState
from models.incremental_model.abstract_incremental_model import AbstractIncrementalModel
from models.module.chaplot_text_module import ChaplotTextModule
from models.module.pixel_identification_module import PixelIdentificationModule
from models.module.unet_image_module import UnetImageModule
from models.module.unet_module import Unet5Contextual
from utils.cuda import cuda_var
from utils.nav_drone_landmarks import get_all_landmark_names
from utils.debug_nav_drone_instruction import instruction_to_string


class UnetResnet(AbstractIncrementalModel):
    def __init__(self, config, constants):
        AbstractIncrementalModel.__init__(self, config, constants)
        self.none_action = config["num_actions"]
        self.image_module = UnetImageModule(
            image_emb_size=constants["image_emb_dim"],
            input_num_channels=3,
            image_height=config["image_height"],
            image_width=config["image_width"],
            using_recurrence=True)
        num_channels, image_height, image_width = 64, 32, 32
        self.num_cameras = 1

        if config["use_pointer_model"]:
            raise NotImplementedError()
        else:
            self.text_module = ChaplotTextModule(
                emb_dim=32,
                hidden_dim=256,
                vocab_size=config["vocab_size"],
                image_height=image_height, image_width=image_width)

        if config["do_object_detection"]:
            self.landmark_names = get_all_landmark_names()
            self.object_detection_module = PixelIdentificationModule(
                num_channels=num_channels, num_objects=67)
        else:
            self.object_detection_module = None

        if config["do_goal_prediction"]:
            self.goal_prediction_module = None  # GoalPredictionModule(total_emb_size=32)
        else:
            self.goal_prediction_module = None

        self.final_module = Unet5Contextual(
            in_channels=num_channels,
            out_channels=1,
            embedding_size=256)

        if torch.cuda.is_available():
            self.image_module.cuda()
            self.text_module.cuda()
            self.final_module.cuda()
            if self.object_detection_module is not None:
                self.object_detection_module.cuda()
            if self.goal_prediction_module is not None:
                self.goal_prediction_module.cuda()

    def get_probs_batch(self, agent_observed_state_list, mode=None):
        raise NotImplementedError()

    def get_unet_output(self, agent_observed_state, model_state, mode=None, volatile=False):

        assert isinstance(agent_observed_state, AgentObservedState)
        agent_observed_state_list = [agent_observed_state]

        image_seqs = [[aos.get_last_image()]
                      for aos in agent_observed_state_list]
        image_batch = cuda_var(torch.from_numpy(np.array(image_seqs)).float(), volatile)

        instructions = [aos.get_instruction()
                        for aos in agent_observed_state_list]
        instructions_batch = cuda_var(torch.from_numpy(np.array(instructions)).long())

        time = agent_observed_state.time_step
        time = cuda_var(torch.from_numpy(np.array([time])).long())

        instruction_string = instruction_to_string(agent_observed_state.instruction, self.config)

        # Embed the text
        _, text_emb_raw = self.text_module(instructions_batch)

        # Embed the image
        image_emb_seq = self.image_module(image_batch)
        image_embedding = image_emb_seq[:, 0, :, :, :]  # 1 x num_channels x height x width

        unet_output = self.final_module(image_embedding, text_emb_raw)
        return unet_output

    def get_probs_and_visible_objects(self, agent_observed_state_list, batch_image_feature):
        raise NotImplementedError()
        assert self.object_detection_module is not None, "Object detection module not created. Check config."
        landmarks_visible = []
        for aos in agent_observed_state_list:
            x_pos, z_pos, y_angle = aos.get_position_orientation()
            landmark_pos_dict = aos.get_landmark_pos_dict()
            visible_landmarks_dict = self.object_detection_module.get_visible_landmark_r_theta(
                x_pos, z_pos, y_angle, landmark_pos_dict)
            landmarks_visible.append(visible_landmarks_dict)

        # shape is BATCH_SIZE x num objects x 2
        landmark_log_prob, distance_log_prob, theta_log_prob = self.object_detection_module(batch_image_feature)

        # landmarks_visible is list of length BATCH_SIZE, each item is a set containing landmark indices
        return landmark_log_prob, distance_log_prob, theta_log_prob, landmarks_visible

    def get_pixel_level_object_prob(self, agent_observed_state_list, batch_image_feature):
        raise NotImplementedError()
        assert self.object_detection_module is not None, "Object detection module not created. Check config."
        landmarks_visible = []
        for aos in agent_observed_state_list:
            x_pos, z_pos, y_angle = aos.get_position_orientation()
            landmark_pos_dict = aos.get_landmark_pos_dict()
            visible_landmarks_dict = self.object_detection_module.get_visible_landmark_r_theta(
                x_pos, z_pos, y_angle, landmark_pos_dict)
            landmarks_visible.append(visible_landmarks_dict)

        # shape is BATCH_SIZE x num objects x 2
        log_prob = self.object_detection_module(batch_image_feature)

        # landmarks_visible is list of length BATCH_SIZE, each item is a set containing landmark indices
        return log_prob, landmarks_visible

    def init_weights(self):
        self.text_module.init_weights()
        self.image_module.init_weights()
        self.final_module.init_weights()

    def share_memory(self):
        self.image_module.share_memory()
        self.text_module.share_memory()
        self.final_module.share_memory()
        if self.object_detection_module is not None:
            self.object_detection_module.share_memory()
        if self.goal_prediction_module is not None:
            self.goal_prediction_module.share_memory()

    def get_state_dict(self):
        nested_state_dict = dict()
        nested_state_dict["image_module"] = self.image_module.state_dict()
        nested_state_dict["text_module"] = self.text_module.state_dict()
        nested_state_dict["final_module"] = self.final_module.state_dict()
        if self.object_detection_module is not None:
            nested_state_dict["od_module"] = self.object_detection_module.state_dict()
        if self.goal_prediction_module is not None:
            nested_state_dict["goal_pred_module"] = self.goal_prediction_module.state_dict()
        return nested_state_dict

    def load_from_state_dict(self, nested_state_dict):
        self.image_module.load_state_dict(nested_state_dict["image_module"])
        self.text_module.load_state_dict(nested_state_dict["text_module"])
        self.final_module.load_state_dict(nested_state_dict["final_module"])

        if self.object_detection_module is not None:
            self.object_detection_module.load_state_dict(nested_state_dict["od_module"])
        if self.goal_prediction_module is not None:
            self.goal_prediction_module.load_state_dict(nested_state_dict["goal_pred_module"])

    def load_resnet_model(self, load_dir):
        if torch.cuda.is_available():
            torch_load = torch.load
        else:
            torch_load = lambda f_: torch.load(f_, map_location=lambda s_, l_: s_)
        image_module_path = os.path.join(load_dir, "image_module_state.bin")
        self.image_module.load_state_dict(torch_load(image_module_path))

    def fix_resnet(self):
        self.image_module.fix_resnet()

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
        text_module_path = os.path.join(load_dir, "text_module_state.bin")
        self.text_module.load_state_dict(torch_load(text_module_path))
        final_module_path = os.path.join(load_dir, "final_module_state.bin")
        self.final_module.load_state_dict(torch_load(final_module_path), strict=False)
        if self.object_detection_module is not None:
            auxiliary_object_detection_path = os.path.join(load_dir, "auxiliary_object_detection.bin")
            self.object_detection_module.load_state_dict(torch_load(auxiliary_object_detection_path))
        if self.goal_prediction_module is not None:
            auxiliary_goal_prediction_path = os.path.join(load_dir, "auxiliary_goal_prediction.bin")
            self.goal_prediction_module.load_state_dict(torch_load(auxiliary_goal_prediction_path))

    def save_model(self, save_dir):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # save state file for image nn
        image_module_path = os.path.join(save_dir, "image_module_state.bin")
        torch.save(self.image_module.state_dict(), image_module_path)
        # save state file for image recurrence nn
        # save state file for text nn
        text_module_path = os.path.join(save_dir, "text_module_state.bin")
        torch.save(self.text_module.state_dict(), text_module_path)
        # save state file for action emb
        # action_module_path = os.path.join(save_dir, "action_module_state.bin")
        # torch.save(self.action_module.state_dict(), action_module_path)
        # save state file for final nn
        final_module_path = os.path.join(save_dir, "final_module_state.bin")
        torch.save(self.final_module.state_dict(), final_module_path)
        # save the auxiliary models
        if self.object_detection_module is not None:
            auxiliary_object_detection_path = os.path.join(save_dir, "auxiliary_object_detection.bin")
            torch.save(self.object_detection_module.state_dict(), auxiliary_object_detection_path)
        if self.goal_prediction_module is not None:
            auxiliary_goal_prediction_path = os.path.join(save_dir, "auxiliary_goal_prediction.bin")
            torch.save(self.goal_prediction_module.state_dict(), auxiliary_goal_prediction_path)

    def get_parameters(self):
        parameters = list(self.image_module.parameters())
        # parameters += list(self.image_recurrence_module.parameters())
        parameters += list(self.text_module.parameters())
        parameters += list(self.final_module.parameters())
        if self.object_detection_module is not None:
            parameters += list(self.object_detection_module.parameters())
        if self.goal_prediction_module is not None:
            parameters += list(self.goal_prediction_module.parameters())

        return parameters

    def get_named_parameters(self):
        named_parameters = list(self.image_module.named_parameters())
        # named_parameters += list(self.image_recurrence_module.named_parameters())
        named_parameters += list(self.text_module.named_parameters())
        named_parameters += list(self.final_module.named_parameters())
        if self.object_detection_module is not None:
            named_parameters += list(self.object_detection_module.named_parameters())
        if self.goal_prediction_module is not None:
            named_parameters += list(self.goal_prediction_module.named_parameters())
        return named_parameters
