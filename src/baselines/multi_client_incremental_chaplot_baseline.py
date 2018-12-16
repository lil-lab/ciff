import torch
import os
import logging
import torch.optim as optim
import numpy as np
import torch.nn.functional as F

from torch.autograd import Variable


class MultiClientIncrementalChaplotBaseline:
    """ Perform expected reward learning """

    def __init__(self, model, shared_model, action_space, meta_data_util, config, constants,
                 args, contextual_bandit, tensorboard, lstm_size):
        self.max_epoch = constants["max_epochs"]
        self.model = model
        self.shared_model = shared_model
        self.action_space = action_space
        self.meta_data_util = meta_data_util
        self.config = config
        self.constants = constants
        self.args = args
        self.contextual_bandit = contextual_bandit
        self.num_client = config["num_client"]
        self.tensorboard = tensorboard
        self.lstm_size = lstm_size

        self.entropy = None
        self.cross_entropy = None
        self.entropy_coef = constants["entropy_coefficient"]

        # Array for logs
        self.p_losses = []
        self.v_losses = []
        self.num_iter = 0

        # optimizer = optim.SGD(self.shared_model.parameters(), lr=self.args.lr) --- changed Chaplot's optimizer
        self.optimizer = optim.Adam(self.shared_model.parameters(), lr=constants["learning_rate"])
        logging.info("Contextual bandit is %r and horizon is %r", self.contextual_bandit, self.args.max_episode_length)

    def update(self, history):

        for log_probs, values, rewards, entropies in history:

            if len(log_probs) == 0:
                continue

            R = torch.zeros(1, 1)
            # if not done:
            #     tx = Variable(torch.from_numpy(np.array([episode_length])).long()).cuda()
            #     value, _, _ = self.shared_model((
            #         Variable(image.unsqueeze(0)).cuda(),
            #         Variable(curr_instruction_idx).cuda(),
            #         Variable(prev_instruction_idx).cuda(),
            #         Variable(next_instruction_idx).cuda(),
            #         (tx, hx, cx)))
            #     R = value.data

            values.append(Variable(R).cuda())
            policy_loss = 0
            value_loss = 0
            R = Variable(R).cuda()

            gae = torch.zeros(1, 1).cuda()
            for i in reversed(range(len(rewards))):
                R = self.args.gamma * R + rewards[i]
                advantage = R - values[i]
                value_loss = value_loss + 0.5 * advantage.pow(2)

                if self.contextual_bandit:
                    # Just focus on immediate reward
                    gae = torch.from_numpy(np.array([[rewards[i]]])).float()
                else:
                    # Generalized Advantage Estimataion
                    delta_t = rewards[i] + self.args.gamma * \
                                           values[i + 1].data - values[i].data
                    gae = gae * self.args.gamma * self.args.tau + delta_t

                policy_loss = policy_loss - \
                              log_probs[i] * Variable(gae).cuda() - 0.01 * entropies[i]

            self.optimizer.zero_grad()

            self.p_losses.append(policy_loss.data[0, 0])
            self.v_losses.append(value_loss.data[0, 0])
            self.num_iter += 1

            if len(self.p_losses) > 1000:
                mean_p_loss = np.mean(self.p_losses)
                mean_v_loss = np.mean(self.v_losses)
                self.tensorboard.log_scalar("Avg policy loss:", mean_p_loss)
                self.tensorboard.log_scalar("Avg policy loss:", mean_v_loss)
                logging.info("Avg policy loss %r and Avg value loss %r ",
                             mean_p_loss, mean_v_loss)
                self.p_losses = []
                self.v_losses = []

            (policy_loss + 0.5 * value_loss).backward()
            torch.nn.utils.clip_grad_norm(self.shared_model.parameters(), 40)

            # ensure_shared_grads(model, shared_model)
            self.optimizer.step()

    def save_model(self, save_dir):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # save state file for image nn
        chaplot_module_path = os.path.join(save_dir, "chaplot_model.bin")
        torch.save(self.shared_model.state_dict(), chaplot_module_path)

    def do_train(self, agent, train_dataset, tune_dataset, experiment_name):
        """ Perform training """

        clients = []
        for client_ix in range(0, self.num_client):
            client = Client(agent, self.config, self.constants, self.tensorboard, client_ix, self.lstm_size)
            clients.append(client)

        dataset_iterator = DatasetIterator(train_dataset)
        epoch = 1

        if epoch <= self.max_epoch:
            logging.info("Starting epoch %d", epoch)
            # Test on tuning data
            agent.test(tune_dataset, tensorboard=self.tensorboard)

        self.shared_model.train()
        history = []
        while True:

            for client_ix in range(0, self.num_client):

                client = clients[client_ix]

                # See if the client can progress
                client_status = client.try_to_progress()
                if client_status == Client.WAITING_FOR_EXAMPLE:
                    # Perform update
                    log_probs, values, rewards, entropies = client.get_history()
                    history.append((log_probs, values, rewards, entropies))
                    client.flush_history()

                    # Provide the next example
                    datapoint = dataset_iterator.get_next()
                    if datapoint is None:
                        continue
                    client.accept_new_example(datapoint, max_num_actions=self.args.max_episode_length)

                elif client_status == Client.WAITING_FOR_ACTION:

                    # Generate probabilities over actions and take action
                    value, logit, (hx, cx) = self.shared_model(client.get_state())
                    new_state = (hx, cx)
                    client.take_action(logit, value, new_state)

                elif client_status == Client.WAITING_TO_RECEIVE:
                    pass
                else:
                    raise AssertionError("Unknown status. Found " + str(client_status))

            if len(history) > 10:
                self.update(history)
                history = []

            # Check if an epoch is finished. An epoch is over if all clients are waiting
            # for an example (at which point the iterator also returns none)
            epoch_completed = all([client.get_status() == Client.WAITING_FOR_EXAMPLE for client in clients])
            if epoch_completed:
                assert dataset_iterator.get_next() is None

                # Reset the iterator
                dataset_iterator.reset()

                # Save the model
                self.save_model(experiment_name + "/chaplot_gated_attention_" + str(epoch))
                if epoch >= self.max_epoch:
                    break
                epoch += 1
                logging.info("Starting epoch %d", epoch)

                # Test on tuning data
                agent.test(tune_dataset, tensorboard=self.tensorboard)


class Client:
    """ Client can be in one of the following state:
    1. Free and Waiting for new example
    2. Waiting to take the next action
    3. Waiting to receive the next image and message.

    Client operates in an automaton following the transitions below:
    Wait for a new example -> repeat [Take an action -> Wait to receive next image and message ] -> Go back to (1) """

    WAITING_FOR_EXAMPLE, WAITING_FOR_ACTION, WAITING_TO_RECEIVE = range(3)

    def __init__(self, agent, config, constants, tensorboard, client_ix, lstm_size):
        self.agent = agent
        self.config = config
        self.constants = constants
        self.tensorboard = tensorboard
        self.lstm_size = lstm_size

        # Client specific information
        self.status = Client.WAITING_FOR_EXAMPLE
        self.client_ix = client_ix
        self.server = agent.servers[client_ix]
        self.metadata = None

        # Datapoint specific variable
        self.max_num_actions = None
        self.state = None
        self.model_state = None
        self.current_data_point = None
        self.last_action = None
        self.last_log_prob = None
        self.factor_entropy = None
        self.num_action = 0
        self.total_reward = 0
        self.forced_stop = False

        # Client's history containing rollout information
        self.values = []
        self.rewards = []
        self.log_probs = []
        self.entropies = []

    def get_state(self):

        (image, instr) = self.state
        image = torch.from_numpy(image).float()
        (tx, hx, cx) = self.model_state
        instr_tensor_curr, instr_tensor_prev, instr_tensor_next = instr

        return (Variable(image.unsqueeze(0)).cuda(), instr_tensor_curr,
                instr_tensor_prev, instr_tensor_next, (tx, hx, cx))

    def get_status(self):
        return self.status

    def get_history(self):
        return self.log_probs, self.values, self.rewards, self.entropies

    def flush_history(self):
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.entropies = []

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
                curr_instr = self.current_data_point.get_instruction()
                prev_instr = self.current_data_point.get_prev_instruction()
                if prev_instr is None:
                    prev_instr = [self.config["vocab_size"] + 1]
                next_instr = self.current_data_point.get_next_instruction()
                if next_instr is None:
                    next_instr = [self.config["vocab_size"] + 1]
                curr_instruction_idx = np.array(curr_instr)
                prev_instruction_idx = np.array(prev_instr)
                next_instruction_idx = np.array(next_instr)

                curr_instruction_idx = torch.from_numpy(curr_instruction_idx).view(1, -1)
                prev_instruction_idx = torch.from_numpy(prev_instruction_idx).view(1, -1)
                next_instruction_idx = torch.from_numpy(next_instruction_idx).view(1, -1)
                instr_cuda_tensor = (Variable(curr_instruction_idx).cuda(),
                                     Variable(prev_instruction_idx).cuda(),
                                     Variable(next_instruction_idx).cuda())

                self.state = (image, instr_cuda_tensor)
                cx = Variable(torch.zeros(1, self.lstm_size)).cuda()
                hx = Variable(torch.zeros(1, self.lstm_size)).cuda()
                tx = Variable(torch.from_numpy(np.array([self.num_action + 1])).long()).cuda()
                self.model_state = (tx, hx, cx)

                # Waiting for action
                self.status = Client.WAITING_FOR_ACTION
            else:
                # Feedback is in response to an action
                image, reward, metadata = feedback

                # Create a replay item unless it is forced
                if not self.forced_stop:

                    self.rewards.append(reward)
                    self.total_reward += reward

                    # Update the agent state
                    inst = self.state[1]
                    self.state = (image, inst)

                if self.last_action == self.agent.action_space.get_stop_action_index():
                    # Update the scores based on meta_data
                    # self.meta_data_util.log_results(metadata)

                    if self.tensorboard is not None:
                        self.tensorboard.log_all_train_errors(
                            metadata["edit_dist_error"], metadata["closest_dist_error"], metadata["stop_dist_error"])
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
        self.server.reset_nonblocking(data_point)
        self.current_data_point = data_point
        self.last_action = None
        self.last_log_prob = None
        self.num_action = 0
        self.max_num_actions = max_num_actions
        self.total_reward = 0
        self.forced_stop = False
        self.values = []
        self.rewards = []
        self.log_probs = []
        self.entropies = []
        self.status = Client.WAITING_TO_RECEIVE

    def take_action(self, logit, value, new_state):
        assert self.status == Client.WAITING_FOR_ACTION

        self.num_action += 1
        hx, cx = new_state
        prob = F.softmax(logit)
        log_prob = F.log_softmax(logit)
        entropy = -(log_prob * prob).sum(1)
        self.entropies.append(entropy)
        tx = Variable(torch.from_numpy(np.array([self.num_action + 1])).long()).cuda()
        self.model_state = (tx, hx, cx)

        action = prob.multinomial().data
        log_prob = log_prob.gather(1, Variable(action)).cuda()
        action = action.cpu().numpy()[0, 0]
        self.values.append(value)
        self.log_probs.append(log_prob)

        # Use test policy to get the action
        self.last_action = action

        if self.last_action == self.agent.action_space.get_stop_action_index():
            self.server.halt_nonblocking()
        else:
            self.server.send_action_nonblocking(self.last_action)

        self.status = Client.WAITING_TO_RECEIVE

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
