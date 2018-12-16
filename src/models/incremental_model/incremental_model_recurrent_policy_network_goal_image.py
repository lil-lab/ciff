import os
import numpy as np
import torch
import torchvision.models.resnet as resnet
import scipy.misc

from agents.agent_observed_state import AgentObservedState
from abstract_incremental_model import AbstractIncrementalModel
from models.incremental_module.incremental_multimodal_recurrent_simple_goal_image_module import \
    IncrementalMultimodalRecurrentSimpleGoalImageModule
from models.module.action_simple_module import ActionSimpleModule
from models.module.image_resnet_module import ImageResnetModule
from models.module.object_detection_module import ObjectDetectionModule
from models.module.action_prediction_module import ActionPredictionModule
from models.module.temporal_autoencoder_module import TemporalAutoencoderModule
from models.incremental_module.incremental_recurrence_simple_module import IncrementalRecurrenceSimpleModule
from utils.cuda import cuda_var, cuda_tensor
from utils.nav_drone_landmarks import get_all_landmark_names


class IncrementalModelRecurrentPolicyNetworkGoalImageResnet(AbstractIncrementalModel):
    def __init__(self, config, constants):
        AbstractIncrementalModel.__init__(self, config, constants)
        self.none_action = config["num_actions"]

        self.image_module = ImageResnetModule(
            image_emb_size=constants["image_emb_dim"],
            input_num_channels=3,
            image_height=config["image_height"],
            image_width=config["image_width"],
            using_recurrence=True)
        # self.image_module = resnet.resnet18(pretrained=True)
        # constants["image_emb_dim"] = 1000
        self.image_recurrence_module = IncrementalRecurrenceSimpleModule(
            input_emb_dim=constants["image_emb_dim"],
            output_emb_dim=constants["image_emb_dim"])
        self.action_module = ActionSimpleModule(
            num_actions=config["num_actions"],
            action_emb_size=constants["action_emb_dim"])
        total_emb_size = (2 * constants["image_emb_dim"]
                          + constants["action_emb_dim"])

        if config["do_action_prediction"]:
            self.action_prediction_module = ActionPredictionModule(
                2 * constants["image_emb_dim"], constants["image_emb_dim"], config["num_actions"])
        else:
            self.action_prediction_module = None

        if config["do_temporal_autoencoding"]:
            self.temporal_autoencoder_module = TemporalAutoencoderModule(
                self.action_module, constants["image_emb_dim"], constants["action_emb_dim"], constants["image_emb_dim"])
        else:
            self.temporal_autoencoder_module = None

        if config["do_object_detection"]:
            self.landmark_names = get_all_landmark_names()
            self.object_detection_module = ObjectDetectionModule(
                image_module=self.image_module, image_emb_size=constants["image_emb_dim"], num_objects=63)
        else:
            self.object_detection_module = None

        final_module = IncrementalMultimodalRecurrentSimpleGoalImageModule(
            image_module=self.image_module,
            image_recurrence_module=self.image_recurrence_module,
            action_module=self.action_module,
            total_emb_size=total_emb_size,
            num_actions=config["num_actions"])
        self.final_module = final_module
        if torch.cuda.is_available():
            self.image_module.cuda()
            self.image_recurrence_module.cuda()
            self.action_module.cuda()
            self.final_module.cuda()
            if self.action_prediction_module is not None:
                self.action_prediction_module.cuda()
            if self.temporal_autoencoder_module is not None:
                self.temporal_autoencoder_module.cuda()
            if self.object_detection_module is not None:
                self.object_detection_module.cuda()

    def get_probs_batch(self, agent_observed_state_list, mode=None):
        raise NotImplementedError()
        for aos in agent_observed_state_list:
            assert isinstance(aos, AgentObservedState)
        # print "batch size:", len(agent_observed_state_list)

        # sort list by instruction length
        agent_observed_state_list = sorted(
            agent_observed_state_list,
            key=lambda aos_: len(aos_.get_instruction()),
            reverse=True
        )

        image_seq_lens = [aos.get_num_images()
                          for aos in agent_observed_state_list]
        image_seq_lens_batch = cuda_tensor(
            torch.from_numpy(np.array(image_seq_lens)))
        max_len = max(image_seq_lens)
        image_seqs = [aos.get_image()[:max_len]
                      for aos in agent_observed_state_list]
        image_batch = cuda_var(torch.from_numpy(np.array(image_seqs)).float())

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

        probs_batch, _ = self.final_module(image_batch, image_seq_lens_batch,
                                           instructions_batch, prev_actions_batch,
                                           mode, model_state=None)
        return probs_batch

    # def resize(self, img):
    #     img = img.swapaxes(0, 1).swapaxes(1, 2)
    #     resized_img = scipy.misc.imresize(img, (224, 224))
    #     return resized_img.swapaxes(1, 2).swapaxes(0, 1)
    #
    # def get_probs(self, agent_observed_state, model_state, mode=None):
    #
    #     assert isinstance(agent_observed_state, AgentObservedState)
    #     agent_observed_state_list = [agent_observed_state]
    #
    #     image_seq_lens = [1]
    #     image_seq_lens_batch = cuda_tensor(
    #         torch.from_numpy(np.array(image_seq_lens)))
    #     image_seqs = [self.resize(aos.get_last_image())
    #                   for aos in agent_observed_state_list]
    #     image_batch = cuda_var(torch.from_numpy(np.array(image_seqs)).float())
    #
    #     goal_image_seqs = [self.resize(aos.get_goal_image()) for aos in agent_observed_state_list]
    #     goal_image_batch = cuda_var(torch.from_numpy(np.array(goal_image_seqs)).float())
    #
    #     prev_actions_raw = [aos.get_previous_action()
    #                         for aos in agent_observed_state_list]
    #     prev_actions = [self.none_action if a is None else a
    #                     for a in prev_actions_raw]
    #     prev_actions_batch = cuda_var(torch.from_numpy(np.array(prev_actions)))
    #
    #     probs_batch, new_model_state, image_emb_seq = self.final_module(image_batch, image_seq_lens_batch,
    #                                                                     goal_image_batch, prev_actions_batch,
    #                                                                     mode, model_state)
    #     return probs_batch, new_model_state, image_emb_seq

    def get_probs(self, agent_observed_state, model_state, mode=None):

        assert isinstance(agent_observed_state, AgentObservedState)
        agent_observed_state_list = [agent_observed_state]

        image_seq_lens = [1]
        image_seq_lens_batch = cuda_tensor(
            torch.from_numpy(np.array(image_seq_lens)))
        image_seqs = [[aos.get_last_image()]
                      for aos in agent_observed_state_list]
        image_batch = cuda_var(torch.from_numpy(np.array(image_seqs)).float())

        goal_image_seqs = [[aos.get_goal_image()] for aos in agent_observed_state_list]
        goal_image_batch = cuda_var(torch.from_numpy(np.array(goal_image_seqs)).float())

        prev_actions_raw = [aos.get_previous_action()
                            for aos in agent_observed_state_list]
        prev_actions = [self.none_action if a is None else a
                        for a in prev_actions_raw]
        prev_actions_batch = cuda_var(torch.from_numpy(np.array(prev_actions)))

        probs_batch, new_model_state, image_emb_seq = self.final_module(image_batch, image_seq_lens_batch,
                                                                        goal_image_batch, prev_actions_batch,
                                                                        mode, model_state)
        return probs_batch, new_model_state, image_emb_seq

    def action_prediction_log_prob(self, batch_input):
        assert self.action_prediction_module is not None, "Action prediction module not created. Check config."
        return self.action_prediction_module(batch_input)

    def predict_action_result(self, batch_image_feature, action_batch):
        assert self.temporal_autoencoder_module is not None, "Temporal action module not created. Check config."
        return self.temporal_autoencoder_module(batch_image_feature, action_batch)

    def get_probs_and_visible_objects(self, agent_observed_state_list, batch_image_feature):
        assert self.object_detection_module is not None, "Object detection module not created. Check config."
        landmarks_visible = []
        for aos in agent_observed_state_list:
            x_pos, z_pos, y_angle = aos.get_position_orientation()
            landmark_pos_dict = aos.get_landmark_pos_dict()
            visible_landmarks = self.object_detection_module.get_visible_landmark_r_theta(
                x_pos, z_pos, y_angle, landmark_pos_dict, self.landmark_names)
            landmarks_visible.append(visible_landmarks)

        # shape is BATCH_SIZE x 63 x 2
        probs_batch = self.object_detection_module(batch_image_feature)

        # landmarks_visible is list of length BATCH_SIZE, each item is a set containing landmark indices
        return probs_batch, landmarks_visible

    def load_resnet_model(self, load_dir):
        if torch.cuda.is_available():
            torch_load = torch.load
        else:
            torch_load = lambda f_: torch.load(f_, map_location=lambda s_, l_: s_)
        image_module_path = os.path.join(load_dir, "image_module_state.bin")
        self.image_module.load_state_dict(torch_load(image_module_path))

    def load_saved_model(self, load_dir):
        if torch.cuda.is_available():
            torch_load = torch.load
        else:
            torch_load = lambda f_: torch.load(f_, map_location=lambda s_, l_: s_)
        image_module_path = os.path.join(load_dir, "image_module_state.bin")
        self.image_module.load_state_dict(torch_load(image_module_path))
        image_recurrence_module_path = os.path.join(
            load_dir, "image_recurrence_module_state.bin")
        self.image_recurrence_module.load_state_dict(
            torch_load(image_recurrence_module_path))
        action_module_path = os.path.join(load_dir, "action_module_state.bin")
        self.action_module.load_state_dict(torch_load(action_module_path))
        final_module_path = os.path.join(load_dir, "final_module_state.bin")
        self.final_module.load_state_dict(torch_load(final_module_path))
        if self.action_prediction_module is not None:
            auxiliary_action_prediction_path = os.path.join(load_dir, "auxiliary_action_prediction.bin")
            self.action_prediction_module.load_state_dict(torch_load(auxiliary_action_prediction_path))
        if self.temporal_autoencoder_module is not None:
            auxiliary_temporal_autoencoder_path = os.path.join(load_dir, "auxiliary_temporal_autoencoder.bin")
            self.temporal_autoencoder_module.load_state_dict(torch_load(auxiliary_temporal_autoencoder_path))
        if self.object_detection_module is not None:
            auxiliary_object_detection_path = os.path.join(load_dir, "auxiliary_object_detection.bin")
            self.object_detection_module.load_state_dict(torch_load(auxiliary_object_detection_path))

    def save_model(self, save_dir):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # save state file for image nn
        image_module_path = os.path.join(save_dir, "image_module_state.bin")
        torch.save(self.image_module.state_dict(), image_module_path)
        # save state file for image recurrence nn
        image_recurrence_module_path = os.path.join(
            save_dir, "image_recurrence_module_state.bin")
        torch.save(self.image_recurrence_module.state_dict(),
                   image_recurrence_module_path)
        # save state file for action emb
        action_module_path = os.path.join(save_dir, "action_module_state.bin")
        torch.save(self.action_module.state_dict(), action_module_path)
        # save state file for final nn
        final_module_path = os.path.join(save_dir, "final_module_state.bin")
        torch.save(self.final_module.state_dict(), final_module_path)
        # save the auxiliary models
        if self.action_prediction_module is not None:
            auxiliary_action_prediction_path = os.path.join(save_dir, "auxiliary_action_prediction.bin")
            torch.save(self.action_prediction_module.state_dict(), auxiliary_action_prediction_path)
        if self.temporal_autoencoder_module is not None:
            auxiliary_temporal_autoencoder_path = os.path.join(save_dir, "auxiliary_temporal_autoencoder.bin")
            torch.save(self.temporal_autoencoder_module.state_dict(), auxiliary_temporal_autoencoder_path)
        if self.object_detection_module is not None:
            auxiliary_object_detection_path = os.path.join(save_dir, "auxiliary_object_detection.bin")
            torch.save(self.object_detection_module.state_dict(), auxiliary_object_detection_path)

    def get_parameters(self):
        parameters = list(self.image_module.parameters())
        parameters += list(self.image_recurrence_module.parameters())
        parameters += list(self.action_module.parameters())
        parameters += list(self.final_module.parameters())
        if self.action_prediction_module is not None:
            parameters += list(self.action_prediction_module.parameters())
        if self.temporal_autoencoder_module is not None:
            parameters += list(self.temporal_autoencoder_module.parameters())
        if self.object_detection_module is not None:
            parameters += list(self.object_detection_module.parameters())
        return parameters
