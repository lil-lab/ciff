import os
import numpy as np
import torch

from agents.agent_observed_state import AgentObservedState
from models.incremental_model.abstract_incremental_model import AbstractIncrementalModel
from models.incremental_module.incremental_multimodal_valts_recurrent_simple_module import \
    IncrementalMultimodalValtsRecurrentSimpleModule
from models.module.action_simple_module import ActionSimpleModule
from models.module.image_resnet_module import ImageResnetModule
from models.incremental_module.incremental_multimodal_recurrent_simple_module \
    import IncrementalMultimodalRecurrentSimpleModule
from models.incremental_module.incremental_multimodal_dense_valts_recurrent_simple_module import \
    IncrementalMultimodalDenseValtsRecurrentSimpleModule
from models.module.image_ryan_resnet_module import ImageRyanResnetModule
from models.module.symbolic_instruction_module import SymbolicInstructionModule
from models.module.object_detection_module import ObjectDetectionModule
from models.module.action_prediction_module import ActionPredictionModule
from models.module.symbolic_embddings import *
from models.module.temporal_autoencoder_module import TemporalAutoencoderModule
from models.incremental_module.incremental_recurrence_simple_module import IncrementalRecurrenceSimpleModule
from utils.cuda import cuda_var, cuda_tensor
from utils.nav_drone_landmarks import get_all_landmark_names


class IncrementalModelRecurrentPolicyNetworkSymbolicTextResnet(AbstractIncrementalModel):
    def __init__(self, config, constants):
        AbstractIncrementalModel.__init__(self, config, constants)
        self.none_action = config["num_actions"]
        landmark_names = get_all_landmark_names()
        self.radius_module = RadiusModule(15)
        self.angle_module = AngleModule(12)  # (48)
        self.landmark_module = LandmarkModule(67)
        self.num_cameras = 1
        self.image_module = ImageRyanResnetModule(
            image_emb_size=constants["image_emb_dim"],
            input_num_channels=3,
            image_height=config["image_height"],
            image_width=config["image_width"],
            using_recurrence=True)
        self.image_recurrence_module = IncrementalRecurrenceSimpleModule(
            input_emb_dim=constants["image_emb_dim"] * self.num_cameras, # + constants["action_emb_dim"],
            output_emb_dim=constants["image_emb_dim"])
        self.text_module = SymbolicInstructionModule(
            radius_embedding=self.radius_module,
            theta_embedding=self.angle_module,
            landmark_embedding=self.landmark_module)
        self.action_module = ActionSimpleModule(
            num_actions=config["num_actions"],
            action_emb_size=constants["action_emb_dim"])
        total_emb_size = ((self.num_cameras) * constants["image_emb_dim"]
                          + 32 * 2
                          + constants["action_emb_dim"])

        if config["do_action_prediction"]:
            self.action_prediction_module = ActionPredictionModule(
                2 * self.num_cameras * constants["image_emb_dim"], constants["image_emb_dim"], config["num_actions"])
        else:
            self.action_prediction_module = None

        if config["do_temporal_autoencoding"]:
            self.temporal_autoencoder_module = TemporalAutoencoderModule(
                self.action_module, self.num_cameras * constants["image_emb_dim"],
                constants["action_emb_dim"], constants["image_emb_dim"])
        else:
            self.temporal_autoencoder_module = None

        if config["do_object_detection"]:
            self.landmark_names = get_all_landmark_names()
            self.object_detection_module = ObjectDetectionModule(
                image_module=self.image_module,
                image_emb_size=self.num_cameras * constants["image_emb_dim"], num_objects=67)
        else:
            self.object_detection_module = None

        final_module = IncrementalMultimodalRecurrentSimpleModule(
            image_module=self.image_module,
            image_recurrence_module=self.image_recurrence_module,
            text_module=self.text_module,
            action_module=self.action_module,
            total_emb_size=total_emb_size,
            num_actions=config["num_actions"])
        self.final_module = final_module
        if torch.cuda.is_available():
            self.image_module.cuda()
            self.image_recurrence_module.cuda()
            self.text_module.cuda()
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

    def get_probs(self, agent_observed_state, model_state, mode=None, volatile=False):

        assert isinstance(agent_observed_state, AgentObservedState)
        agent_observed_state_list = [agent_observed_state]

        image_seq_lens = [1]
        image_seq_lens_batch = cuda_tensor(
            torch.from_numpy(np.array(image_seq_lens)))
        image_seqs = [[aos.get_last_image()]
                      for aos in agent_observed_state_list]
        image_batch = cuda_var(torch.from_numpy(np.array(image_seqs)).float(), volatile)

        instructions_batch = [aos.get_symbolic_instruction()
                              for aos in agent_observed_state_list]

        prev_actions_raw = [aos.get_previous_action()
                            for aos in agent_observed_state_list]
        prev_actions = [self.none_action if a is None else a
                        for a in prev_actions_raw]
        prev_actions_batch = cuda_var(torch.from_numpy(np.array(prev_actions)), volatile)

        probs_batch, new_model_state, image_emb_seq, state_feature = self.final_module(image_batch, image_seq_lens_batch,
                                                                        instructions_batch, prev_actions_batch,
                                                                        mode, model_state)
        return probs_batch, new_model_state, image_emb_seq, state_feature

    def get_probs_symbolic_text(self, agent_observed_state, symbolic_text, model_state, mode=None, volatile=False):
        """ Same as get_probs instead forces the model to use the given symbolic text """

        assert isinstance(agent_observed_state, AgentObservedState)
        agent_observed_state_list = [agent_observed_state]

        image_seq_lens = [1]
        image_seq_lens_batch = cuda_tensor(
            torch.from_numpy(np.array(image_seq_lens)))
        image_seqs = [[aos.get_last_image()]
                      for aos in agent_observed_state_list]
        image_batch = cuda_var(torch.from_numpy(np.array(image_seqs)).float(), volatile)

        instructions_batch = [symbolic_text]

        prev_actions_raw = [aos.get_previous_action()
                            for aos in agent_observed_state_list]
        prev_actions = [self.none_action if a is None else a
                        for a in prev_actions_raw]
        prev_actions_batch = cuda_var(torch.from_numpy(np.array(prev_actions)), volatile)

        probs_batch, new_model_state, image_emb_seq, state_feature = self.final_module(image_batch, image_seq_lens_batch,
                                                                        instructions_batch, prev_actions_batch,
                                                                        mode, model_state)
        return probs_batch, new_model_state, image_emb_seq, state_feature

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
            visible_landmarks_dict = self.object_detection_module.get_visible_landmark_r_theta(
                x_pos, z_pos, y_angle, landmark_pos_dict)
            landmarks_visible.append(visible_landmarks_dict)

        # shape is BATCH_SIZE x num objects x 2
        landmark_log_prob, distance_log_prob, theta_log_prob = self.object_detection_module(batch_image_feature)

        # landmarks_visible is list of length BATCH_SIZE, each item is a set containing landmark indices
        return landmark_log_prob, distance_log_prob, theta_log_prob, landmarks_visible

    def init_weights(self):
        self.image_module.init_weights()
        self.image_recurrence_module.init_weights()

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
        text_module_path = os.path.join(load_dir, "text_module_state.bin")
        self.text_module.load_state_dict(torch_load(text_module_path))
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
        # save state file for text nn
        text_module_path = os.path.join(save_dir, "text_module_state.bin")
        torch.save(self.text_module.state_dict(), text_module_path)
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
        parameters += list(self.text_module.parameters())
        parameters += list(self.action_module.parameters())
        parameters += list(self.final_module.parameters())
        if self.action_prediction_module is not None:
            parameters += list(self.action_prediction_module.parameters())
        if self.temporal_autoencoder_module is not None:
            parameters += list(self.temporal_autoencoder_module.parameters())
        if self.object_detection_module is not None:
            parameters += list(self.object_detection_module.parameters())
        return parameters

    def get_named_parameters(self):
        named_parameters = list(self.image_module.named_parameters())
        named_parameters += list(self.image_recurrence_module.named_parameters())
        named_parameters += list(self.text_module.named_parameters())
        named_parameters += list(self.action_module.named_parameters())
        named_parameters += list(self.final_module.named_parameters())
        if self.action_prediction_module is not None:
            named_parameters += list(self.action_prediction_module.named_parameters())
        if self.temporal_autoencoder_module is not None:
            named_parameters += list(self.temporal_autoencoder_module.named_parameters())
        if self.object_detection_module is not None:
            named_parameters += list(self.object_detection_module.named_parameters())
        '''if self.symbolic_language_prediction_module is not None:
            named_parameters += list(self.symbolic_language_prediction_module.named_parameters())'''
        return named_parameters
