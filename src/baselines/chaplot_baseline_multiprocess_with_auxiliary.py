import os
import math
import sys

import time
import traceback

import torch
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
import utils.generic_policy as gp
import scipy.misc
import matplotlib.pyplot as plt
from agents.agent import Agent
from torch.autograd import Variable

from utils.geometry import get_turn_angle_from_metadata_datapoint, get_distance_from_metadata_datapoint
from utils.launch_unity import launch_k_unity_builds
from utils.oracle_policy import get_oracle_trajectory
from utils.pushover_logger import PushoverLogger
from utils.tensorboard import Tensorboard


class ChaplotBaselineWithAuxiliary(object):
    def __init__(self, args, shared_model, config, constants, tensorboard,
                 use_contextual_bandit=False, lstm_size=256):
        self.args = args
        self.shared_model = shared_model
        if torch.cuda.is_available():
            shared_model.cuda()

        self.config = config
        self.constants = constants
        self.tensorboard = tensorboard
        self.contextual_bandit = use_contextual_bandit
        self.lstm_size = lstm_size

    def get_probs(self, state, model_state):

        image = torch.from_numpy(state.get_last_image()).float()
        curr_instr = state.get_instruction()
        prev_instr = state.get_prev_instruction()
        if prev_instr is None:
            prev_instr = [self.config["vocab_size"] + 1]
        next_instr = state.get_next_instruction()
        if next_instr is None:
            next_instr = [self.config["vocab_size"] + 1]

        curr_instruction_idx = torch.from_numpy(np.array(curr_instr)).view(1, -1)
        prev_instruction_idx = torch.from_numpy(np.array(prev_instr)).view(1, -1)
        next_instruction_idx = torch.from_numpy(np.array(next_instr)).view(1, -1)

        if model_state is None:
            cx = Variable(torch.zeros(1, self.lstm_size).cuda(), volatile=True)
            hx = Variable(torch.zeros(1, self.lstm_size).cuda(), volatile=True)
            episode_length = 1
            cached_computation = None
        else:
            (hx, cx, episode_length, cached_computation) = model_state
            hx = Variable(hx.data.cuda(), volatile=True)
            cx = Variable(cx.data.cuda(), volatile=True)

        tx = Variable(torch.from_numpy(np.array([episode_length])).long().cuda(), volatile=True)

        value, logit, (hx, cx), cached_computation = self.shared_model(
            (Variable(image.unsqueeze(0).cuda(), volatile=True),
             Variable(curr_instruction_idx.cuda(), volatile=True),
             Variable(prev_instruction_idx.cuda(), volatile=True),
             Variable(next_instruction_idx.cuda(), volatile=True),
             (tx, hx, cx)), cached_computation)

        log_prob = F.log_softmax(logit, dim=1)[0]
        new_model_state = (hx, cx, episode_length + 1, cached_computation)
        return log_prob, new_model_state

    @staticmethod
    def get_goal_location(metadata, datapoint):

        angle_phi = get_turn_angle_from_metadata_datapoint(metadata, datapoint)  # angle is in -180 to 180 degree
        if angle_phi < -30 or angle_phi > 30:
            return None, None
        distance = get_distance_from_metadata_datapoint(metadata, datapoint)  # return distance
        tan_phi = math.tan(math.radians(angle_phi))
        cos_phi = math.cos(math.radians(angle_phi))
        tan_theta = math.tan(math.radians(30.0))  # camera width is 30.0
        height_drone = 2.5

        row = int(3 + (3 * height_drone) /(distance * cos_phi * tan_theta))
        col = int(3 + (3 * tan_phi)/tan_theta)

        if row < 0:
            row = 0
        elif row >= 6:
            row = 5

        if col < 0:
            col = 0
        elif col >= 6:
            col = 5

        return row, col

    @staticmethod
    def save_visualized_image(image, goal_location, global_id):

        if goal_location[0] is None:
            return

        print("Goal location is ", goal_location)
        row, col = goal_location
        image = image.cpu().numpy().swapaxes(0, 1).swapaxes(1, 2)

        kernel = np.zeros(shape=(128, 128))
        d = int(128/6.0)
        for i in range(0, d):
            for j in range(0, d):
                kernel[row * d + i][col * d + j] = 1.0

        plt.imshow(image)
        plt.imshow(kernel, cmap='jet', alpha=0.5)
        plt.savefig("./goal_coloring/" + str(global_id) + ".png")
        plt.clf()
        time.sleep(5)

    @staticmethod
    def do_train(chaplot_baseline, shared_model, config, action_space, meta_data_util,
                 args, constants, train_dataset, tune_dataset, experiment,
                 experiment_name, rank, server, logger, model_type, contextual_bandit=False, use_pushover=False):

        try:
            sys.stderr = sys.stdout
            server.initialize_server()
            # Local Config Variables
            lstm_size = 256

            # Test policy
            test_policy = gp.get_argmax_action
            # torch.manual_seed(args.seed + rank)

            if rank == 0:  # client 0 creates a tensorboard server
                tensorboard = Tensorboard(experiment_name)
            else:
                tensorboard = None

            # Create the Agent
            logger.log("STARTING AGENT")
            agent = Agent(server=server,
                          model=chaplot_baseline,
                          test_policy=test_policy,
                          action_space=action_space,
                          meta_data_util=meta_data_util,
                          config=config,
                          constants=constants)
            logger.log("Created Agent...")

            # Create a local model for rollouts
            local_model = model_type(args, config=config)
            if torch.cuda.is_available():
                local_model.cuda()
            chaplot_baseline.shared_model = local_model
            local_model.train()

            #  Our Environment Interface
            env = NavDroneServerInterface(agent, local_model, experiment,
                                          config, constants, None, train_dataset,
                                          tune_dataset, rank, logger, use_pushover)
            logger.log("Created NavDroneServerInterface")

            # optimizer = optim.SGD(self.shared_model.parameters(), lr=self.args.lr) --- changed Chaplot's optimizer
            optimizer = optim.Adam(shared_model.parameters(), lr=0.00025)
            p_losses = []
            v_losses = []

            launch_k_unity_builds([config["port"]],
                                  "./simulators/NavDroneLinuxBuild.x86_64")
            (image, instr), _, _, metadata, data_point = env.reset()
            curr_instr, prev_instr, next_instr = instr
            curr_instruction_idx = np.array(curr_instr)
            prev_instruction_idx = np.array(prev_instr)
            next_instruction_idx = np.array(next_instr)

            image = torch.from_numpy(image).float()
            curr_instruction_idx = torch.from_numpy(curr_instruction_idx).view(1, -1)
            prev_instruction_idx = torch.from_numpy(prev_instruction_idx).view(1, -1)
            next_instruction_idx = torch.from_numpy(next_instruction_idx).view(1, -1)

            done = True

            episode_length = 0
            num_iters = 0
            cx, hx = None, None
            global_id = 1

            while True:
                # Sync with the shared model
                local_model.load_state_dict(shared_model.state_dict())
                if done:
                    episode_length = 0
                    cx = Variable(torch.zeros(1, lstm_size).cuda())
                    hx = Variable(torch.zeros(1, lstm_size).cuda())

                else:
                    cx = Variable(cx.data.cuda())
                    hx = Variable(hx.data.cuda())

                values = []
                log_probs = []
                rewards = []
                entropies = []
                cached_information = None
                #############################
                lstm_rep = []
                image_rep = []
                actions = []
                goal_locations = []
                #############################

                for step in range(args.num_steps):
                    episode_length += 1
                    tx = Variable(torch.from_numpy(np.array([episode_length])).long().cuda())

                    value, logit, (hx, cx), cached_information = local_model((
                                                    Variable(image.unsqueeze(0).cuda()),
                                                    Variable(curr_instruction_idx.cuda()),
                                                    Variable(prev_instruction_idx.cuda()),
                                                    Variable(next_instruction_idx.cuda()),
                                                    (tx, hx, cx)), cached_information)

                    prob = F.softmax(logit, dim=1)
                    log_prob = F.log_softmax(logit, dim=1)
                    entropy = -(log_prob * prob).sum(1)
                    entropies.append(entropy)

                    action = prob.multinomial().data
                    ####################################
                    lstm_rep.append(cached_information["lstm_rep"])
                    image_rep.append(cached_information["image_rep"])
                    actions.append(action)
                    goal_location = ChaplotBaselineWithAuxiliary.get_goal_location(metadata, data_point)
                    goal_locations.append(goal_location)
                    # ChaplotBaselineWithAuxiliary.save_visualized_image(image, goal_location, global_id)
                    global_id += 1
                    ####################################
                    log_prob = log_prob.gather(1, Variable(action.cuda()))
                    action = action.cpu().numpy()[0, 0]

                    (image, _), reward, done, _, metadata = env.step(action)

                    # done = done or (episode_length >= self.args.max_episode_length)
                    if not done and (episode_length >= args.max_episode_length):
                        # If the agent has not taken
                        _, _, done, _, metadata = env.step(env.agent.action_space.get_stop_action_index())
                        done = True

                    if done:
                        (image, instr), _, _, metadata, data_point = env.reset()
                        curr_instr, prev_instr, next_instr = instr
                        curr_instruction_idx = np.array(curr_instr)
                        prev_instruction_idx = np.array(prev_instr)
                        next_instruction_idx = np.array(next_instr)
                        curr_instruction_idx = torch.from_numpy(curr_instruction_idx).view(1, -1)
                        prev_instruction_idx = torch.from_numpy(prev_instruction_idx).view(1, -1)
                        next_instruction_idx = torch.from_numpy(next_instruction_idx).view(1, -1)

                    image = torch.from_numpy(image).float()

                    values.append(value)
                    log_probs.append(log_prob)
                    rewards.append(reward)

                    if done:
                        break

                if rank == 0 and tensorboard is not None:
                    # Log total reward and entropy
                    tensorboard.log_scalar("Total_Reward", sum(rewards))
                    mean_entropy = sum(entropies).data[0]/float(max(episode_length, 1))
                    tensorboard.log_scalar("Chaplot_Baseline_Entropy", mean_entropy)

                R = torch.zeros(1, 1)
                if not done:
                    tx = Variable(torch.from_numpy(np.array([episode_length])).long().cuda())
                    value, _, _, _ = local_model((
                        Variable(image.unsqueeze(0).cuda()),
                        Variable(curr_instruction_idx.cuda()),
                        Variable(prev_instruction_idx.cuda()),
                        Variable(next_instruction_idx.cuda()),
                        (tx, hx, cx)))
                    R = value.data

                values.append(Variable(R.cuda()))
                policy_loss = 0
                value_loss = 0
                R = Variable(R.cuda())

                gae = torch.zeros(1, 1).cuda()
                entropy_coeff = max(0.0, 0.11 - env.num_epochs * 0.01)
                for i in reversed(range(len(rewards))):
                    R = args.gamma * R + rewards[i]
                    advantage = R - values[i]
                    value_loss = value_loss + 0.5 * advantage.pow(2)

                    if contextual_bandit:
                        # Just focus on immediate reward
                        gae = torch.from_numpy(np.array([[rewards[i]]])).float()
                    else:
                        # Generalized Advantage Estimataion
                        delta_t = rewards[i] + args.gamma * \
                                  values[i + 1].data - values[i].data
                        gae = gae * args.gamma * args.tau + delta_t

                    policy_loss = policy_loss - \
                                  log_probs[i] * Variable(gae.cuda()) - entropy_coeff * entropies[i]

                temporal_autoencoding_loss = None  # local_model.get_tae_loss(image_rep, actions)
                reward_prediction_loss = None  # local_model.get_reward_prediction_loss(lstm_rep, actions, rewards)
                alignment_loss, alignment_norm = None, None   # local_model.alignment_auxiliary(image_rep, cached_information["text_rep"])
                goal_prediction_loss = local_model.calc_goal_prediction_loss(
                    image_rep, cached_information["text_rep"], goal_locations)
                optimizer.zero_grad()

                p_losses.append(policy_loss.data[0, 0])
                v_losses.append(value_loss.data[0, 0])

                if len(p_losses) > 1000:
                    num_iters += 1
                    logger.log(" ".join([
                        # "Training thread: {}".format(rank),
                        "Num iters: {}K".format(num_iters),
                        "Avg policy loss: {}".format(np.mean(p_losses)),
                        "Avg value loss: {}".format(np.mean(v_losses))]))
                    p_losses = []
                    v_losses = []

                if rank == 0 and tensorboard is not None:
                    if done:
                        tensorboard.log_scalar("train_dist_error", metadata["stop_dist_error"])
                        task_completion = 0
                        if metadata["stop_dist_error"] < 5.0:
                            task_completion = 1
                        tensorboard.log_scalar("train_task_completion", task_completion)
                    # Log total reward and entropy
                    tensorboard.log_scalar("Value_Loss", float(value_loss.data))
                    if temporal_autoencoding_loss is not None:
                        tensorboard.log_scalar("TAE_Loss", float(temporal_autoencoding_loss.data))
                    if reward_prediction_loss is not None:
                        tensorboard.log_scalar("RP_Loss", float(reward_prediction_loss.data))
                    if alignment_loss is not None:
                        tensorboard.log_scalar("Mean_Current_Segment_Alignment_Loss", float(alignment_loss.data))
                        tensorboard.log_scalar("Alignment_Norm", float(alignment_norm.data))
                    if goal_prediction_loss is not None:
                        tensorboard.log_scalar("Goal_Prediction_Loss", float(goal_prediction_loss.data)/float(len(rewards)))

                loss = policy_loss + 0.5 * value_loss
                if temporal_autoencoding_loss is not None:
                    loss += 0.5 * temporal_autoencoding_loss
                if reward_prediction_loss is not None:
                    loss += 0.5 * reward_prediction_loss
                if alignment_loss is not None:
                    loss += 0.5 * alignment_loss
                if goal_prediction_loss is not None:
                    loss += 0.5 * goal_prediction_loss
                loss.backward()
                torch.nn.utils.clip_grad_norm(local_model.parameters(), 40)
                ChaplotBaselineWithAuxiliary.ensure_shared_grads(local_model, shared_model)
                optimizer.step()
        except Exception:
            print ("Exception")
            exc_info = sys.exc_info()
            traceback.print_exception(*exc_info)

    @staticmethod
    def do_supervised_train(chaplot_baseline, shared_model, config, action_space, meta_data_util, args,
                            constants, train_dataset, tune_dataset, experiment, experiment_name, rank,
                            server, logger, model_type, contextual_bandit=False, use_pushover=False):

        try:
            sys.stderr = sys.stdout
            server.initialize_server()
            # Local Config Variables
            lstm_size = 256

            # Test policy
            test_policy = gp.get_argmax_action
            # torch.manual_seed(args.seed + rank)

            if rank == 0:  # client 0 creates a tensorboard server
                tensorboard = Tensorboard(experiment_name)
            else:
                tensorboard = None

            # Create the Agent
            logger.log("STARTING AGENT")
            agent = Agent(server=server,
                          model=chaplot_baseline,
                          test_policy=test_policy,
                          action_space=action_space,
                          meta_data_util=meta_data_util,
                          config=config,
                          constants=constants)
            logger.log("Created Agent...")

            # Create a local model for rollouts
            local_model = model_type(args, config=config)
            if torch.cuda.is_available():
                local_model.cuda()
            chaplot_baseline.shared_model = local_model
            local_model.train()

            #  Our Environment Interface
            env = NavDroneServerInterface(agent, local_model, experiment,
                                          config, constants, None, train_dataset,
                                          tune_dataset, rank, logger, use_pushover)
            logger.log("Created NavDroneServerInterface")

            # optimizer = optim.SGD(self.shared_model.parameters(), lr=self.args.lr) --- changed Chaplot's optimizer
            optimizer = optim.Adam(shared_model.parameters(), lr=0.00025)
            p_losses = []
            v_losses = []

            launch_k_unity_builds([config["port"]],
                                  "/home/dipendra/Downloads/NavDroneLinuxBuild/NavDroneLinuxBuild.x86_64")

            done = True

            num_iters = 0
            global_id = 1

            while True:

                # Sync with the shared model
                local_model.load_state_dict(shared_model.state_dict())

                # Get a new datapoint
                (image, instr), _, _, metadata, data_point = env.reset()
                curr_instr, prev_instr, next_instr = instr
                curr_instruction_idx = np.array(curr_instr)
                prev_instruction_idx = np.array(prev_instr)
                next_instruction_idx = np.array(next_instr)

                image = torch.from_numpy(image).float()
                curr_instruction_idx = torch.from_numpy(curr_instruction_idx).view(1, -1)
                prev_instruction_idx = torch.from_numpy(prev_instruction_idx).view(1, -1)
                next_instruction_idx = torch.from_numpy(next_instruction_idx).view(1, -1)

                episode_length = 0
                cx = Variable(torch.zeros(1, lstm_size).cuda())
                hx = Variable(torch.zeros(1, lstm_size).cuda())

                goal_x, goal_z = data_point.get_destination_list()[-1]
                trajectory_str = get_oracle_trajectory(metadata, goal_x, goal_z, data_point)
                trajectory = [action_space.get_action_index(act_str) for act_str in trajectory_str]
                # trajectory = data_point.get_trajectory()
                num_steps = len(trajectory) + 1  # 1 for stopping

                values = []
                log_probs = []
                rewards = []
                entropies = []
                cached_information = None
                #############################
                lstm_rep = []
                image_rep = []
                actions = []
                goal_locations = []
                #############################

                for step in range(num_steps):
                    episode_length += 1
                    tx = Variable(torch.from_numpy(np.array([episode_length])).long().cuda())

                    value, logit, (hx, cx), cached_information = local_model((
                                                    Variable(image.unsqueeze(0).cuda()),
                                                    Variable(curr_instruction_idx.cuda()),
                                                    Variable(prev_instruction_idx.cuda()),
                                                    Variable(next_instruction_idx.cuda()),
                                                    (tx, hx, cx)), cached_information)

                    prob = F.softmax(logit, dim=1)
                    log_prob = F.log_softmax(logit, dim=1)
                    entropy = -(log_prob * prob).sum(1)
                    entropies.append(entropy)

                    if step == len(trajectory):
                        action = action_space.get_stop_action_index()
                    else:
                        action = trajectory[step]
                    action_var = torch.from_numpy(np.array([[action]]))

                    ####################################
                    lstm_rep.append(cached_information["lstm_rep"])
                    image_rep.append(cached_information["image_rep"])
                    actions.append(action_var)
                    goal_location = ChaplotBaselineWithAuxiliary.get_goal_location(metadata, data_point)
                    goal_locations.append(goal_location)
                    # ChaplotBaselineWithAuxiliary.save_visualized_image(image, goal_location, global_id)
                    global_id += 1
                    ####################################
                    log_prob = log_prob.gather(1, Variable(action_var.cuda()))

                    (image, _), reward, done, _, metadata = env.step(action)
                    image = torch.from_numpy(image).float()

                    values.append(value)
                    log_probs.append(log_prob)
                    rewards.append(reward)

                assert done, "Should be done as all trajectories are fully executed and stop with 'stop' action."

                if rank == 0 and tensorboard is not None:
                    # Log total reward and entropy
                    tensorboard.log_scalar("Total_Reward", sum(rewards))
                    mean_entropy = sum(entropies).data[0]/float(max(episode_length, 1))
                    tensorboard.log_scalar("Chaplot_Baseline_Entropy", mean_entropy)

                R = torch.zeros(1, 1)
                values.append(Variable(R.cuda()))
                policy_loss = 0
                value_loss = 0
                R = Variable(R.cuda())

                entropy_coeff = max(0.0, 0.11 - env.num_epochs * 0.01)
                for i in reversed(range(len(rewards))):
                    R = args.gamma * R + rewards[i]
                    advantage = R - values[i]
                    value_loss = value_loss + 0.5 * advantage.pow(2)
                    policy_loss = policy_loss - \
                                  log_probs[i] - entropy_coeff * entropies[i]

                temporal_autoencoding_loss = None  # local_model.get_tae_loss(image_rep, actions)
                reward_prediction_loss = None  # local_model.get_reward_prediction_loss(lstm_rep, actions, rewards)
                alignment_loss, alignment_norm = None, None   # local_model.alignment_auxiliary(image_rep, cached_information["text_rep"])
                goal_prediction_loss = local_model.calc_goal_prediction_loss(
                    image_rep, cached_information["text_rep"], goal_locations)
                optimizer.zero_grad()

                p_losses.append(policy_loss.data[0, 0])
                v_losses.append(value_loss.data[0, 0])

                if len(p_losses) > 1000:
                    num_iters += 1
                    logger.log(" ".join([
                        # "Training thread: {}".format(rank),
                        "Num iters: {}K".format(num_iters),
                        "Avg policy loss: {}".format(np.mean(p_losses)),
                        "Avg value loss: {}".format(np.mean(v_losses))]))
                    p_losses = []
                    v_losses = []

                if rank == 0 and tensorboard is not None:
                    # Log total reward and entropy
                    tensorboard.log_scalar("Value_Loss", float(value_loss.data))
                    if temporal_autoencoding_loss is not None:
                        tensorboard.log_scalar("TAE_Loss", float(temporal_autoencoding_loss.data))
                    if reward_prediction_loss is not None:
                        tensorboard.log_scalar("RP_Loss", float(reward_prediction_loss.data))
                    if alignment_loss is not None:
                        tensorboard.log_scalar("Mean_Current_Segment_Alignment_Loss", float(alignment_loss.data))
                        tensorboard.log_scalar("Alignment_Norm", float(alignment_norm.data))
                    if goal_prediction_loss is not None:
                        tensorboard.log_scalar("Goal_Prediction_Loss", float(goal_prediction_loss.data)/float(num_steps))

                loss = policy_loss + 0.5 * value_loss
                if temporal_autoencoding_loss is not None:
                    loss += 0.5 * temporal_autoencoding_loss
                if reward_prediction_loss is not None:
                    loss += 0.5 * reward_prediction_loss
                if alignment_loss is not None:
                    loss += 0.5 * alignment_loss
                if goal_prediction_loss is not None:
                    loss += 20.0 * goal_prediction_loss
                    loss = goal_prediction_loss

                loss.backward()
                torch.nn.utils.clip_grad_norm(local_model.parameters(), 40)
                ChaplotBaselineWithAuxiliary.ensure_shared_grads(local_model, shared_model)
                optimizer.step()
        except Exception:
            print ("Exception")
            exc_info = sys.exc_info()
            traceback.print_exception(*exc_info)

    @staticmethod
    def ensure_shared_grads(model, shared_model):
        for param, shared_param in zip(model.parameters(),
                                       shared_model.parameters()):
            if shared_param.grad is not None:
                return
            shared_param._grad = param.grad

    def load_saved_model(self, load_dir):
        if torch.cuda.is_available():
            torch_load = torch.load
        else:
            torch_load = lambda f_: torch.load(f_, map_location=lambda s_, l_: s_)
        chaplot_module_path = os.path.join(load_dir, "chaplot_model.bin")
        self.shared_model.load_state_dict(torch_load(chaplot_module_path))

    def load_image_text_model(self, load_dir):
        if torch.cuda.is_available():
            torch_load = torch.load
        else:
            torch_load = lambda f_: torch.load(f_, map_location=lambda s_, l_: s_)
        chaplot_module_path = os.path.join(load_dir, "chaplot_model.bin")
        # pairs = ['conv1.weight', 'conv1.bias', 'conv2.weight', 'conv2.bias', 'conv3.weight', 'conv3.bias',
        #          'embedding.weight', 'gru.weight_ih_l0', 'gru.weight_hh_l0', 'gru.bias_ih_l0', 'gru.bias_hh_l0',
        #          'attn_linear.weight', 'attn_linear.bias', 'time_emb_layer.weight', 'linear.weight', 'linear.bias',
        #          'lstm.weight_ih', 'lstm.weight_hh', 'lstm.bias_ih', 'lstm.bias_hh',
        #          'critic_linear.weight', 'critic_linear.bias', 'actor_linear.weight', 'actor_linear.bias',
        #          'action_embedding.weight', 'tae_linear_1.weight', 'tae_linear_1.bias', 'tae_linear_2.weight',
        #          'tae_linear_2.bias', 'rp_linear_1.weight', 'rp_linear_1.bias', 'rp_linear_2.weight',
        #          'rp_linear_2.bias', 'W_image_text_alignment.weight', 'W_image_text_alignment.bias',
        #          'conv4.weight', 'conv4.bias', 'goal_prediction_bilinear.weight', 'goal_prediction_bilinear.bias']

        not_included_list = ['linear.weight', 'linear.bias', 'lstm.weight_ih', 'lstm.weight_hh',
                             'lstm.bias_ih', 'lstm.bias_hh', 'critic_linear.weight', 'critic_linear.bias',
                             'actor_linear.weight', 'actor_linear.bias']
        loaded_dict = torch_load(chaplot_module_path)
        new_dict = dict()
        for key in loaded_dict:
            if key not in not_included_list:
                print ("Loading ", key)
                new_dict[key] = loaded_dict[key]

        self.shared_model.load_state_dict(new_dict, strict=False)

class DatasetIterator:

    def __init__(self, dataset, client_id, logger, log_per_ix=100):
        self.dataset = dataset
        self.dataset_size = len(dataset)
        self.datapoint_ix = 0
        self.client_id = client_id
        self.log_per_ix = log_per_ix
        self.logger = logger

    def get_next(self):
        if self.datapoint_ix == self.dataset_size:
            return None
        else:
            datapoint = self.dataset[self.datapoint_ix]
            self.datapoint_ix += 1
            if self.log_per_ix is not None and ((self.datapoint_ix + 1) % self.log_per_ix == 0):
                self.logger.log("Client: %r Done %d out of %d" % (self.client_id, self.datapoint_ix + 1, self.dataset_size))
            return datapoint

    def reset(self):
        self.datapoint_ix = 0


class NavDroneServerInterface:

    def __init__(self, agent, local_model, experiment_name, config, constants,
                 tensorboard, train_dataset, tune_dataset, client_id, logger,
                 use_pushover):
        self.dataset_iterator = DatasetIterator(train_dataset, client_id, logger)
        self.tune_dataset = tune_dataset
        self.tensorboard = tensorboard
        self.local_model = local_model
        self.experiment_name = experiment_name
        self.client_id = client_id
        self.agent = agent
        self.server = agent.server
        self.num_actions = 0
        self.num_epochs = 1
        self.current_instr = None
        self.config = config
        self.logger = logger
        if use_pushover:
            self.pushover_logger = PushoverLogger(experiment_name)
        else:
            self.pushover_logger = None

    def save_model(self, save_dir):
        self.logger.log("Saving model in: " + save_dir)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # save state file for image nn
        chaplot_module_path = os.path.join(save_dir, "chaplot_model.bin")
        torch.save(self.local_model.state_dict(), chaplot_module_path)

    def reset(self):

        # get instruction
        self.data_point = data_point = self.dataset_iterator.get_next()
        if data_point is None:
            self.logger.log("End of epoch %r" % self.num_epochs)
            self.logger.log("Client " + str(self.client_id) + " reporting end of epoch")
            self.save_model(self.experiment_name + "/chaplot_model_client_" + str(self.client_id)
                            + "_epoch_" + str(self.num_epochs))
            if len(self.tune_dataset) > 0:
                self.logger.log("Client " + str(self.client_id) + " going for testing.")
                try:
                        self.agent.test(self.tune_dataset, self.tensorboard, logger=self.logger, pushover_logger=self.pushover_logger)
                except Exception:
                        print ("Got exception while testing.")
            self.num_epochs += 1
            self.logger.log("Client " + str(self.client_id) + " starting epoch " + str(self.num_epochs))
            self.logger.log("Starting epoch %r" % self.num_epochs)

            self.dataset_iterator.reset()
            self.data_point = data_point = self.dataset_iterator.get_next()

        # get the instruction
        curr_instr = data_point.get_instruction()
        prev_instr = data_point.get_prev_instruction()
        if prev_instr is None:
            prev_instr = [self.config["vocab_size"] + 1]
        next_instr = data_point.get_next_instruction()
        if next_instr is None:
            next_instr = [self.config["vocab_size"] + 1]
        instr = (curr_instr, prev_instr, next_instr)

        self.current_instr = instr

        # get the image
        image, metadata = self.reset_datapoint_blocking(data_point)
        state = (image, instr)

        # is final
        is_final = 0

        # extra args?
        extra_args = None

        return state, is_final, extra_args, metadata, data_point

    def step(self, action):
        """ Interface for Chaplot's code and our code """

        image, reward, metadata, is_final = self.take_action_blocking(action)
        instr = self.current_instr
        state = (image, instr)

        # is final
        self.num_actions += 1
        if action == self.agent.action_space.get_stop_action_index():
            is_final = 1
        else:
            is_final = 0

        # extra args?
        extra_args = None

        return state, reward, is_final, extra_args, metadata

    def get_trajectory(self):
        return self.data_point.get_trajectory()

    def get_supervised_action(self):
        cached_trajectory = self.data_point.get_trajectory()
        cached_len = len(cached_trajectory)
        if self.num_actions == cached_len - 1:
            return 3
        else:
            return cached_trajectory[self.num_actions]

    def reset_datapoint_blocking(self, datapoint):
        """ Resets to the given datapoint and returns starting image """
        image, metadata = self.server.reset_receive_feedback(datapoint)
        return image, metadata

    def take_action_blocking(self, action):
        """ Takes an action and returns image, reward and metadata """

        if action == self.agent.action_space.get_stop_action_index():
            image, reward, metadata = self.server.halt_and_receive_feedback()
            done = True
        else:
            image, reward, metadata = self.server.send_action_receive_feedback(action)
            done = False

        return image, reward, metadata, done
