import os
import sys
import torch
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
import utils.generic_policy as gp

from agents.tmp_streetview_agent import Agent
from torch.autograd import Variable
from utils.pushover_logger import PushoverLogger
from utils.tensorboard import Tensorboard


class ChaplotBaselineStreetView(object):
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

        curr_instruction_idx = torch.from_numpy(np.array(curr_instr)).view(1,-1)

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
             Variable(curr_instruction_idx.cuda(), volatile=True), None, None,
             (tx, hx, cx)), cached_computation)

        log_prob = F.log_softmax(logit, dim=1)[0]
        new_model_state = (hx, cx, episode_length + 1, cached_computation)
        return log_prob, new_model_state

    @staticmethod
    def do_train(chaplot_baseline, shared_model, config, action_space, meta_data_util,
                 args, constants, train_dataset, tune_dataset, experiment,
                 experiment_name, rank, server, logger, model_type, contextual_bandit, use_pushover=False):

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
        local_model = model_type(args, config=config, final_image_height=3, final_image_width=3)
        if torch.cuda.is_available():
            local_model.cuda()
        chaplot_baseline.shared_model = local_model
        local_model.train()

        #  Our Environment Interface
        env = StreetViewServerInterface(agent, local_model, experiment,
                                        config, constants, None, train_dataset,
                                        tune_dataset, rank, logger, use_pushover)

        env.game_init()
        logger.log("Created StreetViewServerInterface")

        # optimizer = optim.SGD(self.shared_model.parameters(), lr=self.args.lr) --- changed Chaplot's optimizer
        optimizer = optim.Adam(shared_model.parameters(), lr=0.00025)
        p_losses = []
        v_losses = []

        (image, instr), _, _ = env.reset()
        curr_instruction_idx = np.array(instr)

        image = torch.from_numpy(image).float()
        curr_instruction_idx = torch.from_numpy(curr_instruction_idx).view(1, -1)

        done = True

        episode_length = 0
        num_iters = 0

        while True:
            # Sync with the shared model
            local_model.load_state_dict(shared_model.state_dict())
            if done:
                episode_length = 0
                cx = Variable(torch.zeros(1, lstm_size).cuda())
                hx = Variable(torch.zeros(1, lstm_size).cuda())

            else:
                # assert False, "Assertion put by Max and Dipendra. Code shouldn't reach here."
                cx = Variable(cx.data.cuda())
                hx = Variable(hx.data.cuda())

            values = []
            log_probs = []
            rewards = []
            entropies = []
            cached_information = None

            for step in range(args.num_steps):
                episode_length += 1
                tx = Variable(torch.from_numpy(np.array([episode_length])).long().cuda())

                value, logit, (hx, cx), cached_information = local_model((
                                                Variable(image.unsqueeze(0).cuda()),
                                                Variable(curr_instruction_idx.cuda()),
                                                None, None, (tx, hx, cx)), cached_information)

                prob = F.softmax(logit, dim=1)
                log_prob = F.log_softmax(logit, dim=1)
                entropy = -(log_prob * prob).sum(1)
                entropies.append(entropy)

                action = prob.multinomial().data
                log_prob = log_prob.gather(1, Variable(action.cuda()))
                action = action.cpu().numpy()[0, 0]

                (image, _), reward, done, _ = env.step(action)

                # done = done or (episode_length >= self.args.max_episode_length)
                if not done and (episode_length >= args.max_episode_length):
                    # If the agent has not taken
                    _, _, done, _ = env.step(env.agent.action_space.get_stop_action_index())
                    done = True

                if done:
                    (image, instr), _, _ = env.reset()
                    curr_instruction_idx = np.array(instr)
                    curr_instruction_idx = torch.from_numpy(curr_instruction_idx).view(1, -1)

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
                    None, None, (tx, hx, cx)))
                R = value.data

            values.append(Variable(R.cuda()))
            policy_loss = 0
            value_loss = 0
            R = Variable(R.cuda())

            gae = torch.zeros(1, 1).cuda()
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
                              log_probs[i] * Variable(gae.cuda()) - 0.02 * entropies[i]

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

            (policy_loss + 0.5 * value_loss).backward()
            torch.nn.utils.clip_grad_norm(local_model.parameters(), 40)

            ChaplotBaselineStreetView.ensure_shared_grads(local_model, shared_model)
            optimizer.step()

    @staticmethod
    def ensure_shared_grads(model, shared_model):
        for param, shared_param in zip(model.parameters(),
                                       shared_model.parameters()):
            if shared_param.grad is not None:
                return
            shared_param._grad = param.grad

    @staticmethod
    def do_supervsed_train(chaplot_baseline, shared_model, config, action_space, meta_data_util,
                           args, constants, train_dataset, tune_dataset, experiment,
                           experiment_name, rank, server, logger, model_type, use_pushover=False):
        raise NotImplementedError()
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

        env = StreetViewServerInterface(agent, local_model, experiment,
                                        config, constants, None, train_dataset,
                                        tune_dataset, rank, logger, use_pushover)

        env.game_init()

        shared_model.train()

        # optimizer = optim.SGD(self.shared_model.parameters(), lr=self.args.lr)
        optimizer = optim.Adam(shared_model.parameters(), lr=0.00025)

        p_losses = []
        v_losses = []
        num_iters = 0

        while True:

            # Get datapoint
            (image, instr), _, _ = env.reset()
            instruction_idx = np.array(instr)

            image = torch.from_numpy(image).float()
            instruction_idx = torch.from_numpy(instruction_idx).view(1, -1)

            # Sync with the shared model
            # model.load_state_dict(shared_model.state_dict())
            episode_length = 0
            cx = Variable(torch.zeros(1, lstm_size).cuda())
            hx = Variable(torch.zeros(1, lstm_size).cuda())

            log_probs = []
            rewards = []
            entropies = []
            trajectory = env.get_trajectory()
            min_length = min(len(trajectory), args.max_episode_length - 1)
            trajectory = trajectory[0:min_length]
            trajectory.append(agent.action_space.get_stop_action_index())

            for action in trajectory:
                episode_length += 1
                tx = Variable(torch.from_numpy(np.array([episode_length])).long().cuda())

                value, logit, (hx, cx) = shared_model((Variable(image.unsqueeze(0).cuda()),
                                                       Variable(instruction_idx.cuda()),
                                                       None, None, (tx, hx, cx)))

                prob = F.softmax(logit)
                log_prob = F.log_softmax(logit)
                entropy = -(log_prob * prob).sum(1)
                entropies.append(entropy)

                action_tensor = torch.from_numpy(np.array([[action]]))
                log_prob = log_prob.gather(1, Variable(action_tensor.cuda()))
                (image, _), reward, done, _ = env.step(action)
                image = torch.from_numpy(image).float()

                log_probs.append(log_prob)
                rewards.append(reward)

                if done:
                    break

            policy_loss = 0
            for i in range(0, len(rewards)):
                policy_loss = policy_loss - log_probs[i] - 0.01 * entropies[i]

            # Log total reward and entropy
            if tensorboard is not None:
                tensorboard.log_scalar("Total_Reward", sum(rewards))
                mean_entropy = sum(entropies) / float(max(episode_length, 1))
                tensorboard.log_scalar("Chaplot_Baseline_Entropy", mean_entropy)
                tensorboard.log_scalar("Policy_Loss", policy_loss)

            optimizer.zero_grad()
            p_losses.append(policy_loss.data[0, 0])

            if len(p_losses) > 1000:
                num_iters += 1
                logger.log(" ".join([
                    # "Training thread: {}".format(rank),
                    "Num iters: {}K".format(num_iters),
                    "Avg policy loss: {}".format(np.mean(p_losses)),
                    "Avg value loss: {}".format(np.mean(v_losses))]))
                p_losses = []
                v_losses = []

            policy_loss.backward()
            torch.nn.utils.clip_grad_norm(shared_model.parameters(), 40)

            # ensure_shared_grads(model, shared_model)
            optimizer.step()

    def load_saved_model(self, load_dir):
        if torch.cuda.is_available():
            torch_load = torch.load
        else:
            torch_load = lambda f_: torch.load(f_, map_location=lambda s_, l_: s_)
        chaplot_module_path = os.path.join(load_dir, "chaplot_model.bin")
        self.shared_model.load_state_dict(torch_load(chaplot_module_path))


class Client:

    def take_action_blocking(self, action):
        """ Takes an action and returns image, reward and metadata """

        if action == self.agent.action_space.get_stop_action_index():
            image, reward, metadata = self.agent.server.halt_and_receive_feedback()
            done = True
        else:
            image, reward, metadata = self.agent.server.send_action_receive_feedback(action)
            done = False

        return image, reward, metadata, done


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
                self.logger.log("Client: %r Done %d out of %d" % (self.client_id, self.datapoint_ix + 1,
                                                                  self.dataset_size))
            return datapoint

    def reset(self):
        self.datapoint_ix = 0


class StreetViewServerInterface:

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
        self.num_actions = 0
        self.num_epochs = 1
        self.current_instr = None
        self.config = config
        self.constants = constants
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

    def game_init(self):
        pass

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
                self.agent.test(self.tune_dataset, self.tensorboard,
                                logger=self.logger,
                                pushover_logger=self.pushover_logger)
            self.num_epochs += 1
            self.logger.log("Client " + str(self.client_id) + " starting epoch " + str(self.num_epochs))
            self.logger.log("Starting epoch %r" % self.num_epochs)

            self.dataset_iterator.reset()
            self.data_point = data_point = self.dataset_iterator.get_next()

        # get the instruction
        self.current_instr = data_point.get_instruction()

        # get the image
        image, metadata = self.agent.server.reset_receive_feedback(data_point)
        state = (image, self.current_instr)

        # is final
        is_final = 0

        # extra args?
        extra_args = None

        return state, is_final, extra_args

    def step(self, action):
        """ Interface for Chaplot's code and our code """

        if action == self.agent.action_space.get_stop_action_index():
            image, reward, metadata = self.agent.server.halt_and_receive_feedback()
            is_final = 1
        else:
            image, reward, metadata = self.agent.server.send_action_receive_feedback(action)
            is_final = 0

        instr = self.current_instr
        state = (image, instr)

        # is final
        self.num_actions += 1

        # extra args?
        extra_args = None

        return state, reward, is_final, extra_args

    def get_trajectory(self):

        trajectory = self.agent.server.get_trajectory_exact(self.data_point.trajectory)
        trajectory = trajectory[:min(len(trajectory), self.constants["horizon"])]

        return trajectory

    def get_supervised_action(self):
        cached_trajectory = self.data_point.get_trajectory()
        cached_len = len(cached_trajectory)
        if self.num_actions == cached_len - 1:
            return 3
        else:
            return cached_trajectory[self.num_actions]
