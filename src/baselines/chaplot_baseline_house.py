import os
import torch
import logging
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
import utils.generic_policy as gp

from utils.launch_unity import launch_k_unity_builds
from utils.tensorboard import Tensorboard
from agents.replay_memory_item import ReplayMemoryItem
from agents.agent_observed_state import AgentObservedState
from agents.tmp_house_agent import TmpHouseAgent
from torch.autograd import Variable


class ChaplotBaselineHouse(object):
    """" Chaplot et al. Gated Attention Model with GAE learning, adapted for the House corpus """

    def __init__(self, args, shared_model, config, constants, tensorboard, use_contextual_bandit=False, lstm_size=256):
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

        image = torch.from_numpy(state.get_last_image().copy()).float()
        curr_instr = state.get_instruction()
        # prev_instr = state.get_prev_instruction()
        # if prev_instr is None:
        #     prev_instr = [self.config["vocab_size"] + 1]
        # next_instr = state.get_next_instruction()
        # if next_instr is None:
        #     next_instr = [self.config["vocab_size"] + 1]

        curr_instruction_idx = torch.from_numpy(np.array(curr_instr)).view(1,-1)
        # prev_instruction_idx = torch.from_numpy(np.array(prev_instr)).view(1,-1)
        # next_instruction_idx = torch.from_numpy(np.array(next_instr)).view(1,-1)

        if model_state is None:
            cx = Variable(torch.zeros(1, self.lstm_size).cuda(), volatile=True)
            hx = Variable(torch.zeros(1, self.lstm_size).cuda(), volatile=True)
            cached_computation = None
            episode_length = 1
        else:
            (hx, cx, episode_length, cached_computation) = model_state
            hx = Variable(hx.data.cuda(), volatile=True)
            cx = Variable(cx.data.cuda(), volatile=True)

        tx = Variable(torch.from_numpy(np.array([episode_length])).long(),
                      volatile=True).cuda()

        value, logit, (hx, cx), cached_computation = self.shared_model(
            (Variable(image.unsqueeze(0).cuda(), volatile=True),
             Variable(curr_instruction_idx.cuda(), volatile=True),
             # Variable(prev_instruction_idx.cuda(), volatile=True),
             # Variable(next_instruction_idx.cuda(), volatile=True),
             (tx, hx, cx)), cached_computation)

        log_prob = F.log_softmax(logit)[0]
        new_model_state = (hx, cx, episode_length + 1, cached_computation)
        return log_prob, new_model_state

    @staticmethod
    def do_test(house_id, chaplot_baseline, config, action_space, meta_data_util, constants,
                test_dataset, experiment_name, rank, server, logger):

        # torch.manual_seed(args.seed + rank)

        # Launch the Unity Build
        launch_k_unity_builds([config["port"]], "./house_" + str(house_id) + "_elmer.x86_64",
                              arg_str="--config ./AssetsHouse/config" + str(house_id) + ".json",
                              cwd="./simulators/house/")

        # Initialize Server
        server.initialize_server()
        server.clear_metadata()
        logger.log("Server Initialized")

        # Test policy
        test_policy = gp.get_argmax_action

        # Create the Agent
        agent = TmpHouseAgent(server=server,
                              model=chaplot_baseline,
                              test_policy=test_policy,
                              action_space=action_space,
                              meta_data_util=meta_data_util,
                              config=config,
                              constants=constants)

        # Create tensorboard server
        if rank == 0:  # client 0 creates a tensorboard server
            tensorboard = Tensorboard(experiment_name)
            logger.log('Created Tensorboard Server...')
        else:
            tensorboard = None

        agent.test(test_dataset, vocab=None, tensorboard=tensorboard, logger=logger)

    @staticmethod
    def do_train(house_id, chaplot_baseline, shared_model, config, action_space, meta_data_util, constants,
                 train_dataset, tune_dataset, experiment, experiment_name, rank, server,
                 logger, model_type, vocab, args, contextual_bandit, lstm_size):

        # torch.manual_seed(args.seed + rank)

        # Launch the Unity Build
        launch_k_unity_builds([config["port"]], "./house_" + str(house_id) + "_elmer.x86_64",
                              arg_str="--config ./AssetsHouse/config" + str(house_id) + ".json",
                              cwd="./simulators/house/")

        # Initialize Server
        server.initialize_server()
        server.clear_metadata()
        logger.log("Server Initialized")

        # Create a local model for rollouts
        local_model = model_type(args, action_space=action_space, config=config)
        if torch.cuda.is_available():
            local_model.cuda()
        chaplot_baseline.shared_model = local_model
        local_model.train()

        # Test policy
        test_policy = gp.get_argmax_action

        # Create the Agent
        agent = TmpHouseAgent(server=server,
                              model=chaplot_baseline,
                              test_policy=test_policy,
                              action_space=action_space,
                              meta_data_util=meta_data_util,
                              config=config,
                              constants=constants)

        # Create tensorboard server
        if rank == 0:  # client 0 creates a tensorboard server
            tensorboard = Tensorboard(experiment_name)
            logger.log('Created Tensorboard Server...')
        else:
            tensorboard = None

        # Provide an interface for Chaplot's code
        env = HouseServerInterface(agent, local_model, experiment, config, constants,
                                   tensorboard, train_dataset, tune_dataset, logger)
        env.game_init()
        logger.log("Contextual bandit is %r and horizon is %r" % (contextual_bandit, args.max_episode_length))

        # optimizer = optim.SGD(self.shared_model.parameters(), lr=self.args.lr) --- changed Chaplot's optimizer
        optimizer = optim.Adam(shared_model.parameters(), lr=0.00025)

        p_losses = []
        v_losses = []

        (image, instr), _, _ = env.reset()
        # curr_instr, prev_instr, next_instr = instr
        curr_instr = instr
        curr_instruction_idx = np.array(curr_instr)
        # prev_instruction_idx = np.array(prev_instr)
        # next_instruction_idx = np.array(next_instr)

        image = torch.from_numpy(image.copy()).float().cuda()
        curr_instruction_idx = torch.from_numpy(curr_instruction_idx).view(1, -1)
        # prev_instruction_idx = torch.from_numpy(prev_instruction_idx).view(1, -1)
        # next_instruction_idx = torch.from_numpy(next_instruction_idx).view(1, -1)

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

            for step in range(args.num_steps):
                episode_length += 1
                tx = Variable(torch.from_numpy(np.array([episode_length])).long().cuda())

                value, logit, (hx, cx), _ = local_model((Variable(image.unsqueeze(0).cuda()),
                                                         Variable(curr_instruction_idx.cuda()),
                                                         # Variable(prev_instruction_idx.cuda()),
                                                         #  Variable(next_instruction_idx.cuda()),
                                                         (tx, hx, cx)))

                prob = F.softmax(logit)
                log_prob = F.log_softmax(logit)
                entropy = -(log_prob * prob).sum(1)
                entropies.append(entropy)

                action = prob.multinomial().data
                log_prob = log_prob.gather(1, Variable(action.cuda()))
                action = action.cpu().numpy()[0, 0]

                (image, _), reward, done, _ = env.step(action)
                # logger.log("Took action %r and got reward %r " % (action_space.get_action_name(action), reward))

                # done = done or (episode_length >= self.args.max_episode_length)
                if not done and (episode_length >= args.max_episode_length):
                    # If the agent has not taken
                    _, _, done, _ = env.step(env.client.agent.action_space.get_stop_action_index())
                    done = True

                if done:
                    (image, instr), _, _ = env.reset()
                    # curr_instr, prev_instr, next_instr = instr
                    curr_instr = instr
                    curr_instruction_idx = np.array(curr_instr)
                    # prev_instruction_idx = np.array(prev_instr)
                    # next_instruction_idx = np.array(next_instr)
                    curr_instruction_idx = torch.from_numpy(curr_instruction_idx).view(1, -1)
                    # prev_instruction_idx = torch.from_numpy(prev_instruction_idx).view(1, -1)
                    # next_instruction_idx = torch.from_numpy(next_instruction_idx).view(1, -1)

                image = torch.from_numpy(image.copy()).float()

                values.append(value)
                log_probs.append(log_prob)
                rewards.append(reward)

                if done:
                    break

            # Log total reward and entropy
            if tensorboard is not None:
                tensorboard.log_scalar("Total_Reward", sum(rewards))
                mean_entropy = sum(entropies).data[0] / float(max(episode_length, 1))
                tensorboard.log_scalar("Chaplot_Baseline_Entropy", mean_entropy)

            R = torch.zeros(1, 1)
            if not done:
                tx = Variable(torch.from_numpy(np.array([episode_length])).long().cuda())
                value, _, _, _ = local_model((
                    Variable(image.unsqueeze(0).cuda()),
                    Variable(curr_instruction_idx.cuda()),
                    # Variable(prev_instruction_idx.cuda()),
                    # Variable(next_instruction_idx.cuda()),
                    (tx, hx, cx)))
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
                              log_probs[i] * Variable(gae.cuda()) - 0.01 * entropies[i]

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
                logger.log(" ".join([
                    # "Training thread: {}".format(rank),
                    "Num iters: {}K".format(num_iters),
                    "Avg policy loss: {}".format(np.mean(p_losses)),
                    "Avg value loss: {}".format(np.mean(v_losses))]))
                p_losses = []
                v_losses = []

            (policy_loss + 0.5 * value_loss).backward()
            torch.nn.utils.clip_grad_norm(local_model.parameters(), 40)

            ChaplotBaselineHouse.ensure_shared_grads(local_model, shared_model)
            optimizer.step()

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


class Client:
    """ Client can be in one of the following state:
    1. Free and Waiting for new example
    2. Waiting to take the next action
    3. Waiting to receive the next image and message.

    Client operates in an automaton following the transitions below:
    Wait for a new example -> repeat [Take an action -> Wait to receive next image and message ] -> Go back to (1) """

    WAITING_FOR_EXAMPLE, WAITING_FOR_ACTION, WAITING_TO_RECEIVE = range(3)

    def __init__(self, agent, config, constants, tensorboard, client_ix, batch_replay_items):
        self.agent = agent
        self.config = config
        self.constants = constants
        self.tensorboard = tensorboard

        # Client specific information
        self.status = Client.WAITING_FOR_EXAMPLE
        self.client_ix = client_ix
        self.server = agent.server  # agent.servers[client_ix]
        self.metadata = None

        # Datapoint specific variable
        self.max_num_actions = None
        self.state = None
        self.model_state = None
        self.image_emb_seq = None
        self.current_data_point = None
        self.last_action = None
        self.last_log_prob = None
        self.factor_entropy = None
        self.num_action = 0
        self.total_reward = 0
        self.forced_stop = False
        self.batch_replay_items = batch_replay_items

    def get_state(self):
        return self.state

    def get_status(self):
        return self.status

    def get_model_state(self):
        return self.model_state

    def _get_all_rewards(self, metadata):
        rewards = []
        for i in range(0, self.config["num_actions"]):
            reward = metadata["reward_dict"][self.agent.action_space.get_action_name(i)]
            rewards.append(reward)
        return rewards

    def try_to_progress(self):

        # If in state (1) or (2) then return immediately
        if self.status == Client.WAITING_FOR_EXAMPLE or self.status == Client.WAITING_FOR_ACTION:
            return self.status

        assert self.status == Client.WAITING_TO_RECEIVE

        # If in state (3) then see if the message is available. If the message
        # is available then return to waiting for an action or a new example.
        if self.state is None:
            feedback = self.server.receive_reset_feedback_nonblocking()
        else:
            feedback = self.server.receive_feedback_nonblocking()

        if feedback is None:
            return self.status
        else:
            if self.state is None:
                # assert False, "state should not be none"
                # Feedback is in response to reset
                image, metadata = feedback

                pose = int(metadata["y_angle"] / 15.0)
                position_orientation = (metadata["x_pos"], metadata["z_pos"],
                                        metadata["y_angle"])
                self.state = AgentObservedState(instruction=self.current_data_point.instruction,
                                                config=self.config,
                                                constants=self.constants,
                                                start_image=image,
                                                previous_action=None,
                                                data_point=self.current_data_point)

                # Waiting for action
                self.status = Client.WAITING_FOR_ACTION
            else:
                # Feedback is in response to an action
                image, reward, metadata = feedback
                self.total_reward += reward

                # Create a replay item unless it is forced
                if not self.forced_stop:
                    all_rewards = self._get_all_rewards(metadata)
                    replay_item = ReplayMemoryItem(
                        self.state, self.last_action, reward, log_prob=self.last_log_prob,
                        image_emb_seq=self.image_emb_seq, factor_entropy=self.factor_entropy,
                        all_rewards=all_rewards)
                    self.batch_replay_items.append(replay_item)

                # Update the agent state
                self.state = self.state.update(
                    image, self.last_action, data_point=self.current_data_point)

                if self.last_action == self.agent.action_space.get_stop_action_index():
                    # Update the scores based on meta_data
                    # self.meta_data_util.log_results(metadata)

                    if self.tensorboard is not None:
                        self.tensorboard.log_all_train_errors(
                            metadata["edit_dist_error"], metadata["closest_dist_error"],
                            metadata["stop_dist_error"])
                    self.status = Client.WAITING_FOR_EXAMPLE
                else:

                    if self.num_action >= self.max_num_actions:
                        # Send forced stop action and wait to receive
                        self._take_forced_stop()
                        self.status = Client.WAITING_TO_RECEIVE
                    else:
                        # Wait to take another action
                        self.status = Client.WAITING_FOR_ACTION

            self.metadata = metadata
            return self.status

    def accept_new_example(self, data_point, max_num_actions):
        assert self.status == Client.WAITING_FOR_EXAMPLE
        self.state = None
        self.metadata = None
        self.model_state = None
        self.image_emb_seq = None
        self.factor_entropy = None
        self.max_num_actions = max_num_actions
        self.server.reset_nonblocking(data_point)
        self.current_data_point = data_point
        self.last_action = None
        self.last_log_prob = None
        self.num_action = 0
        self.total_reward = 0
        self.forced_stop = False
        self.status = Client.WAITING_TO_RECEIVE

    def take_action(self, log_probabilities, new_model_state, image_emb_seq, factor_entropy):
        assert self.status == Client.WAITING_FOR_ACTION

        probability = list(torch.exp(log_probabilities.data))[0]

        self.model_state = new_model_state
        self.last_log_prob = log_probabilities
        self.image_emb_seq = image_emb_seq
        self.factor_entropy = factor_entropy

        # Use test policy to get the action
        self.last_action = gp.sample_action_from_prob(probability)
        self.num_action += 1

        # if self.metadata["goal_dist"] < 5:
        #     # Add a forced stop action to replay items
        #     imp_weight = float(probability[3])
        #     reward = 1.0
        #     replay_item = ReplayMemoryItem(
        #         self.state, self.agent.action_space.get_stop_action_index(), reward * imp_weight,
        #         log_prob=self.last_log_prob, image_emb_seq=self.image_emb_seq, factor_entropy=self.factor_entropy)
        #     self.batch_replay_items.append(replay_item)

        if self.last_action == self.agent.action_space.get_stop_action_index():
            self.server.halt_nonblocking()
        else:
            self.server.send_action_nonblocking(self.last_action)

        self.status = Client.WAITING_TO_RECEIVE

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

    def _take_forced_stop(self):
        # Use test policy to get the action
        self.last_action = self.agent.action_space.get_stop_action_index()
        self.forced_stop = True
        self.server.halt_nonblocking()
        self.status = Client.WAITING_TO_RECEIVE


class DatasetIterator:

    def __init__(self, dataset, log_per_ix=100):
        self.dataset = dataset
        self.dataset_size = len(dataset)
        self.datapoint_ix = 0
        self.log_per_ix = log_per_ix

    def get_next(self):
        if self.datapoint_ix == self.dataset_size:
            return None
        else:
            datapoint = self.dataset[self.datapoint_ix]
            self.datapoint_ix += 1
            if self.log_per_ix is not None and ((self.datapoint_ix + 1) % self.log_per_ix == 0):
                logging.info("Done %d out of %d", self.datapoint_ix, self.dataset_size)
            return datapoint

    def reset(self):
        self.datapoint_ix = 0


class HouseServerInterface:

    def __init__(self, agent, shared_model, experiment, config, constants, tensorboard,
                 train_dataset, tune_dataset, logger):
        self.dataset_iterator = DatasetIterator(train_dataset)
        self.tune_dataset = tune_dataset
        self.tensorboard = tensorboard
        self.shared_model = shared_model
        self.experiment = experiment
        self.client = Client(agent, config, constants, tensorboard, 0, [])
        self.num_actions = 0
        self.num_epochs = 1
        self.current_instr = None
        self.config = config
        self.logger = logger

    def save_model(self, save_dir):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # save state file for image nn
        chaplot_module_path = os.path.join(save_dir, "chaplot_model.bin")
        torch.save(self.shared_model.state_dict(), chaplot_module_path)

    def game_init(self):
        pass

    def reset(self):

        # get instruction
        self.data_point = data_point = self.dataset_iterator.get_next()
        if data_point is None:
            self.logger.log("End of epoch %r" % self.num_epochs)
            self.save_model(self.experiment + "/chaplot_model_epoch_" + str(self.num_epochs))
            self.client.agent.test(self.tune_dataset, vocab=None, tensorboard=self.tensorboard, logger=self.logger)
            self.num_epochs += 1
            self.logger.log("Starting epoch %r" % self.num_epochs)
            self.dataset_iterator.reset()
            self.data_point = data_point = self.dataset_iterator.get_next()

        # get the instruction
        curr_instr = data_point.get_instruction()
        # prev_instr = data_point.get_prev_instruction()
        # if prev_instr is None:
        #     prev_instr = [self.config["vocab_size"] + 1]
        # next_instr = data_point.get_next_instruction()
        # if next_instr is None:
        #     next_instr = [self.config["vocab_size"] + 1]
        # instr = (curr_instr, prev_instr, next_instr)
        instr = curr_instr

        self.current_instr = instr

        # get the image
        image, metadata = self.client.reset_datapoint_blocking(data_point)

        state = (image, instr)

        # is final
        is_final = 0

        # extra args?
        extra_args = None

        return state, is_final, extra_args

    def step(self, action):
        """ Interface for Chaplot's code and our code """

        image, reward, metadata, is_final = self.client.take_action_blocking(action)
        # print("Action: ", str(action), " , Reward: ", str(reward),
        # " , Stop Dist Error: ", str(metadata['stop_dist_error']))
        instr = self.current_instr
        state = (image, instr)

        # is final
        self.num_actions += 1
        # print("get stop action is " + str(self.client.agent.action_space.get_stop_action_index()))
        if action == self.client.agent.action_space.get_stop_action_index():
            is_final = 1
        else:
            is_final = 0

        # extra args?
        extra_args = None

        return state, reward, is_final, extra_args

    def get_trajectory(self):
        return self.data_point.get_trajectory()

    def get_supervised_action(self):
        cached_trajectory = self.data_point.get_trajectory()
        cached_len = len(cached_trajectory)
        if self.num_actions == cached_len - 1:
            return 3
        else:
            return cached_trajectory[self.num_actions]



