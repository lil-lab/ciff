import torch
import logging
import torch.optim as optim
import utils.generic_policy as gp

from agents.agent_observed_state import AgentObservedState
from agents.replay_memory_item import ReplayMemoryItem
from abstract_learning import AbstractLearning
from utils.nav_drone_landmarks import get_name_of_landmark


class SupervisedLearningDetectVisibleObject(AbstractLearning):
    """ Perform supervised learning to train a simple classifier on resnet features """

    def __init__(self, model, action_space, meta_data_util, config, constants, tensorboard):
        self.max_epoch = 100  # constants["max_epochs"]
        self.model = model
        self.action_space = action_space
        self.meta_data_util = meta_data_util
        self.config = config
        self.constants = constants
        self.tensorboard = tensorboard
        self.entropy = None
        self.cross_entropy = None
        self.entropy_coef = constants["entropy_coefficient"]
        self.optimizer = optim.Adam(model.get_parameters(),
                                    lr=constants["learning_rate"])

        ###################
        self.confusion_num_count = []
        self.confusion_denom_count = []

        for i in range(0, 63):
            self.confusion_num_count.append([0.0] * 63)
            self.confusion_denom_count.append([0.0] * 63)
        ###################

        AbstractLearning.__init__(self, self.model, self.calc_loss,
                                  self.optimizer, self.config, self.constants)

    def calc_loss(self, batch_replay_items):

        agent_observation_state_ls = []
        for replay_item in batch_replay_items:
            agent_observation_state_ls.append(replay_item.get_agent_observed_state())

        log_prob, visible_objects = self.model.get_probs_and_visible_objects(agent_observation_state_ls)
        num_states = int(log_prob.size()[0])

        # print "Log prob size is " + str(log_prob.size())

        objective = None
        for i in range(0, num_states):
            visible_objects_example = visible_objects[i]
            for landmark in range(0, 63):
                if landmark in visible_objects_example:
                    if objective is None:
                        objective = log_prob[i, landmark, 1]
                    else:
                        objective = objective + log_prob[i, landmark, 1]
                else:
                    if objective is None:
                        objective = log_prob[i, landmark, 0]
                    else:
                        objective = objective + log_prob[i, landmark, 0]

        objective = objective / (num_states * 63.0)
        loss = -objective

        return loss

    def update_confusion_matrix(self, gold_set, predicted_set):

        for i in range(0, 63):
            for j in range(0, 63):
                # If i is present and predicted but j is not present
                if i in gold_set and i in predicted_set and j not in gold_set:
                    self.confusion_denom_count[i][j] += 1
                    if j in predicted_set:
                        self.confusion_num_count[i][j] += 1

    def print_confusion_matrix(self):

        for i in range(0, 63):
            normalized_vals = []
            for j in range(0, 63):
                normalized_val = self.confusion_num_count[i][j]/max(self.confusion_denom_count[i][j], 1.0)
                if normalized_val > 0.1:
                    normalized_vals.append([(get_name_of_landmark(j), normalized_val)])
            logging.info("Row %r is %r", get_name_of_landmark(i), normalized_vals)

    @staticmethod
    def get_f1_score(gold_set, predicted_set):
        precision = 0
        recall = 0
        for val in predicted_set:
            if val in gold_set:
                precision += 1

        for val in gold_set:
            if val in predicted_set:
                recall += 1

        if len(gold_set) == 0 and len(predicted_set) == 0:
            return 1
        if len(gold_set) == 0 and len(predicted_set) != 0:
            return 0
        if len(gold_set) != 0 and len(predicted_set) == 0:
            return 0

        recall /= float(max(len(gold_set), 1))
        precision /= float(max(len(predicted_set), 1))

        if precision == 0 and recall == 0:
            # logging.info("Precision %r, recall %r, f1-score %r", precision, recall, 0)
            return 0
        else:
            # logging.info("Precision %r, recall %r, f1-score %r",
            #              precision, recall, (2 * precision * recall) / (precision + recall))
            return (2 * precision * recall) / (precision + recall)

    def test(self, agent, test_dataset):

        mean_f1_score = 0
        num_data_points = 0

        for data_point_ix, data_point in enumerate(test_dataset):

            image, metadata = agent.server.reset_receive_feedback(data_point)
            pose = int(metadata["y_angle"] / 15.0)
            position_orientation = (metadata["x_pos"], metadata["z_pos"],
                                    metadata["y_angle"])
            state = AgentObservedState(instruction=data_point.instruction,
                                       config=self.config,
                                       constants=self.constants,
                                       start_image=image,
                                       previous_action=None,
                                       pose=pose,
                                       position_orientation=position_orientation,
                                       data_point=data_point)

            trajectory = data_point.get_trajectory()

            for action in trajectory:

                # Compute probabilities over list of visible objects
                log_prob, visible_objects = self.model.get_probs_and_visible_objects([state])

                prob = list(torch.exp(log_prob.data)[0])
                predicted_set = set([])
                for i in range(0, 63):
                    argmax_val = gp.get_argmax_action(prob[i])
                    if argmax_val == 1:
                        predicted_set.add(i)

                f1_score = SupervisedLearningDetectVisibleObject.get_f1_score(visible_objects[0], predicted_set)
                self.update_confusion_matrix(visible_objects[0], predicted_set)
                mean_f1_score += f1_score
                num_data_points += 1

                # print "Visible objects " + str(visible_objects[0])
                # print "Predicted Set" + str(predicted_set)
                # print "F1 score " + str(f1_score)
                # raw_input("Enter to proceed")

                # Send the action and get feedback
                image, reward, metadata = agent.server.send_action_receive_feedback(action)

                # Update the agent state
                pose = int(metadata["y_angle"] / 15.0)
                position_orientation = (metadata["x_pos"],
                                        metadata["z_pos"],
                                        metadata["y_angle"])
                state = state.update(
                    image, action, pose=pose,
                    position_orientation=position_orientation,
                    data_point=data_point)

            # Send final STOP action and get feedback
            image, reward, metadata = agent.server.halt_and_receive_feedback()

        mean_f1_score /= float(max(num_data_points, 1))

        logging.info("Object detection accuracy on a dataset of size %r the mean f1 score is %r",
                     num_data_points, mean_f1_score)

    def do_train(self, agent, train_dataset, tune_dataset, experiment_name):
        """ Perform training """

        dataset_size = len(train_dataset)

        for epoch in range(1, self.max_epoch + 1):

            logging.info("Starting epoch %d", epoch)
            action_counts = [0] * self.action_space.num_actions()

            # Test on tuning data
            self.test(agent, tune_dataset)
            # self.print_confusion_matrix()

            batch_replay_items = []
            total_reward = 0
            episodes_in_batch = 0

            for data_point_ix, data_point in enumerate(train_dataset):

                if (data_point_ix + 1) % 100 == 0:
                    logging.info("Done %d out of %d", data_point_ix, dataset_size)
                    logging.info("Training data action counts %r", action_counts)

                num_actions = 0

                image, metadata = agent.server.reset_receive_feedback(data_point)
                pose = int(metadata["y_angle"] / 15.0)
                position_orientation = (metadata["x_pos"], metadata["z_pos"],
                                        metadata["y_angle"])
                state = AgentObservedState(instruction=data_point.instruction,
                                           config=self.config,
                                           constants=self.constants,
                                           start_image=image,
                                           previous_action=None,
                                           pose=pose,
                                           position_orientation=position_orientation,
                                           data_point=data_point)

                trajectory = data_point.get_trajectory()

                for action in trajectory:

                    action_counts[action] += 1

                    # Send the action and get feedback
                    image, reward, metadata = agent.server.send_action_receive_feedback(action)

                    # Store it in the replay memory list
                    replay_item = ReplayMemoryItem(state, action, reward)
                    batch_replay_items.append(replay_item)

                    # Update the agent state
                    pose = int(metadata["y_angle"] / 15.0)
                    position_orientation = (metadata["x_pos"],
                                            metadata["z_pos"],
                                            metadata["y_angle"])
                    state = state.update(
                        image, action, pose=pose,
                        position_orientation=position_orientation,
                        data_point=data_point)

                    num_actions += 1
                    total_reward += reward

                # Send final STOP action and get feedback
                image, reward, metadata = agent.server.halt_and_receive_feedback()
                total_reward += reward

                # Store it in the replay memory list
                replay_item = ReplayMemoryItem(state, agent.action_space.get_stop_action_index(), reward)
                batch_replay_items.append(replay_item)

                # Perform update
                episodes_in_batch += 1
                if episodes_in_batch == 1:
                    loss_val = self.do_update(batch_replay_items)
                    batch_replay_items = []
                    # cross_entropy = float(self.cross_entropy.data[0])
                    # self.tensorboard.log(cross_entropy, loss_val, total_reward)
                    total_reward = 0
                    episodes_in_batch = 0

                if self.tensorboard is not None:
                    self.tensorboard.log_all_train_errors(
                        metadata["edit_dist_error"], metadata["closest_dist_error"], metadata["stop_dist_error"])

                if data_point_ix == 2000:
                    self.test(agent, tune_dataset)

                if data_point_ix == 6000:
                    self.test(agent, tune_dataset)

            # Save the model
            self.model.save_model(experiment_name + "/object_detection_resnet_epoch_" + str(epoch))

            logging.info("Training data action counts %r", action_counts)
