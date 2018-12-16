import logging
import tensorflow as tf

from agent_observed_state import AgentObservedState


class Agent:

    def __init__(self, server, model, test_policy, action_space, meta_data_util,
                 config, constants):
        self.server = server
        self.model = model
        self.test_policy = test_policy
        self.action_space = action_space
        self.meta_data_util = meta_data_util
        self.config = config
        self.constants = constants
        self.sess = None
        self.train_writer = None

    def init_session(self, model_file=None, gpu_memory_fraction=0.5):
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        if model_file is None:
            self.sess.run(tf.initialize_all_variables())
            logging.info("Initialized all variables ")
            saver = tf.train.Saver()
            saver.save(self.sess, "./saved/init.ckpt")
        else:
            saver = tf.train.Saver()
            saver.restore(self.sess, model_file)
            logging.info("Loaded model from the file " + str(model_file))

        self.train_writer = tf.train.SummaryWriter('./train_summaries/', self.sess.graph)

    def test(self, test_dataset, tensorboard=None):

        self.server.clear_metadata()
        action_counts = [0] * self.action_space.num_actions()

        metadata = ""
        for data_point in test_dataset:
            image, metadata = self.server.reset_receive_feedback(data_point)
            state = AgentObservedState(instruction=data_point.instruction,
                                       config=self.config,
                                       constants=self.constants,
                                       start_image=image,
                                       previous_action=None)

            num_actions = 0
            max_num_actions = len(data_point.get_trajectory())
            max_num_actions += self.constants["max_extra_horizon_auto_segmented"]
            # self._save_agent_state(state, num_actions)

            gold_trajectory = data_point.get_trajectory()

            while True:

                # Generate probabilities over actions
                probabilities = self.model.get_probs(state, self.sess)
                # print "test probs:", probabilities

                # Use test policy to get the action
                action = self.test_policy(probabilities)
                action_counts[action] += 1

                # logging.info("Taking action-num=%d horizon=%d action=%s from %s",
                #              num_actions, max_num_actions, str(action), str(probabilities))

                if action == self.action_space.get_stop_action_index() or num_actions >= max_num_actions:
                    # Send the action and get feedback
                    image, reward, metadata = self.server.halt_and_receive_feedback()
                    if tensorboard is not None:
                        tensorboard.log_test_error(metadata["error"])

                    # Update the scores based on meta_data
                    self.meta_data_util.log_results(metadata)
                    break
                else:
                    # Send the action and get feedback
                    image, reward, metadata = self.server.send_action_receive_feedback(action)

                    # Update the agent state
                    state = state.update(image, action)
                    num_actions += 1

                # self._save_agent_state(state, num_actions)

        self.meta_data_util.log_results(metadata)
        logging.info("Testing data action counts %r", action_counts)

    def test_auto_segmented(self, test_dataset, tensorboard=None, segmenting_type="auto"):
        assert segmenting_type in ("auto", "oracle")
        self.server.clear_metadata()
        action_counts = [0] * self.action_space.num_actions()

        metadata = ""

        for data_point in test_dataset:

            if segmenting_type == "auto":
                segmented_instruction = data_point.get_instruction_auto_segmented()
            else:
                segmented_instruction = data_point.get_instruction_oracle_segmented()
            num_segments = len(segmented_instruction)
            gold_num_actions = len(data_point.get_trajectory())
            horizon = gold_num_actions // num_segments
            horizon += self.constants["max_extra_horizon_auto_segmented"]

            image, metadata = self.server.reset_receive_feedback(data_point)

            for instruction in segmented_instruction:

                state = AgentObservedState(instruction=instruction,
                                           config=self.config,
                                           constants=self.constants,
                                           start_image=image,
                                           previous_action=None)

                num_actions = 0
                # self._save_agent_state(state, num_actions)

                while True:

                    # Generate probabilities over actions
                    probabilities = self.model.get_probs(state, self.sess)
                    # print "test probs:", probabilities

                    # Use test policy to get the action
                    action = self.test_policy(probabilities)
                    action_counts[action] += 1

                    # logging.info("Taking action-num=%d horizon=%d action=%s from %s",
                    #              num_actions, max_num_actions, str(action), str(probabilities))

                    if action == self.action_space.get_stop_action_index() or num_actions >= horizon:
                        break

                    else:
                        # Send the action and get feedback
                        image, reward, metadata = self.server.send_action_receive_feedback(action)

                        # Update the agent state
                        state = state.update(image, action)
                        num_actions += 1

            _,  _, metadata = self.server.halt_and_receive_feedback()
            self.meta_data_util.log_results(metadata)
            if tensorboard is not None:
                tensorboard.log_test_error(metadata["error"])

        self.meta_data_util.log_results(metadata)
        logging.info("Testing data action counts %r", action_counts)
