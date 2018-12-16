from server.abstract_server import AbstractServer
from dataset_agreement_streetview.streetview_fsa import StreetViewFSA, StreetViewUtils


class StreetViewServer(AbstractServer):
    """ Server for StreetView dataset. Unlike servers using unity3d such as
    blocks, house, navdrone, this server is programmed on the python side. """

    def __init__(self, config, action_space, forward_setting_strict=False):

        self.config = config
        self.action_space = action_space

        # Create the Finite State Automaton
        self.fsa = StreetViewFSA(config["node_file"], config["link_file"], config["image_feature_folder"],
                                 forward_setting_strict=forward_setting_strict, mapping_type=config["mapping_type"])
        self.streetview_fsa_utils = StreetViewUtils(self.fsa, action_space)

        # Information stored across dataset until metadata is cleared
        self.list_all_states = []
        self.sum_navigation_error = 0
        self.sum_node_error = 0
        self.sum_task_completion_accuracy = 0
        self.sum_edit_distance = 0
        self.num_examples = 0

        # Information stored per dataset
        self.start_state = None
        self.current_datapoint = None
        self.current_state = None
        self.goal_state = None

        AbstractServer.__init__(self, config, action_space)

    def initialize_server(self):
        return

    def send_action_receive_feedback(self, action):

        act_name = self.action_space.get_action_name(action)
        new_state = self.fsa.take_action_return_new_state(self.current_state, act_name)

        # get image
        image = self.fsa.get_image_from_state(new_state)

        # get reward
        reward = self.streetview_fsa_utils.get_reward(self.current_state, act_name, new_state, self.goal_state)

        self.current_state = new_state
        self.list_all_states.append(new_state)

        metadata = {"panorama_id": new_state.get_node_id(),
                    "orientation": new_state.get_orientation()}

        return image, reward, metadata

    def halt_and_receive_feedback(self):

        action = self.action_space.get_stop_action_index()
        act_name = self.action_space.get_action_name(action)
        new_state = self.fsa.take_action_return_new_state(self.current_state, act_name)

        # get image
        image = self.fsa.get_image_from_state(new_state)

        # get reward
        reward = self.streetview_fsa_utils.get_reward(self.current_state, act_name, new_state, self.goal_state)

        self.current_state = new_state

        # get navigation error
        navigation_error, path, node_error = self.streetview_fsa_utils.path_to_goal(self.current_state, self.goal_state)
        start_nav_error, _, _ = self.streetview_fsa_utils.path_to_goal(self.start_state, self.goal_state)

        # Compute task completion accuracy
        task_completion = self.streetview_fsa_utils.get_task_completion_accuracy(
            self.current_state, self.current_datapoint)

        # Get edit distance error
        edit_distance = self.streetview_fsa_utils.edit_distance(self.list_all_states, self.current_datapoint)

        self.sum_navigation_error += navigation_error
        self.sum_node_error += node_error
        if task_completion:
            self.sum_task_completion_accuracy += 1.0
        self.sum_edit_distance += edit_distance
        self.num_examples += 1

        mean_navigation_error = self.sum_navigation_error / float(self.num_examples)
        mean_node_error = self.sum_node_error / float(self.num_examples)
        mean_task_completion_accuracy = (self.sum_task_completion_accuracy * 100.0) / float(self.num_examples)
        mean_edit_distance = self.sum_edit_distance / float(self.num_examples)

        metadata = {"panorama_id": new_state.get_node_id(),
                    "orientation": new_state.get_orientation(),
                    "navigation_error": navigation_error,
                    "num_tokens": len(self.current_datapoint.get_instruction()),
                    "start_nav_err": start_nav_error,
                    "node_error": node_error,
                    "task_completion_accuracy": task_completion,
                    "edit_distance": edit_distance,
                    "mean_navigation_error": mean_navigation_error,
                    "mean_node_error": mean_node_error,
                    "mean_task_completion_accuracy": mean_task_completion_accuracy,
                    "mean_edit_distance": mean_edit_distance}

        return image, reward, metadata

    def reset_receive_feedback(self, next_data_point):

        self.current_datapoint = next_data_point
        self.current_state, self.goal_state = self.fsa.reset_to_new_state(next_data_point)
        self.start_state = self.current_state
        metadata = {"panorama_id": self.current_state.get_node_id(),
                    "orientation": self.current_state.get_orientation()}

        # get image
        image = self.fsa.get_image_from_state(self.current_state)

        # Reset the utils
        self.streetview_fsa_utils.reset_utils(self.goal_state)

        # Reset the list of states
        self.list_all_states = [self.current_state]

        return image, metadata

    def get_trajectory_shortest(self):
        """ Returns the shortest path trajectory from one location to another """

        _, path, _ = self.streetview_fsa_utils.path_to_goal(self.current_state, self.goal_state, get_path=True)
        act_names = [step[1] for step in path[1:]]  # ignore the first action which is a dummy stop action
        return [self.action_space.get_action_index(act_name) for act_name in act_names]

    def get_trajectory_exact(self, route_ids):

        # Route-ids contain a list of node-ids from the start to the goal.
        prev_route_id = route_ids[0]
        prev_node_id = self.fsa.panorama_to_node_dict[prev_route_id]
        assert prev_node_id == self.current_state.get_node_id(), "Current Node-ID should match start of Route IDs"

        trajectory = []
        iter_state = self.current_state

        for route_id in route_ids[1:]:

            node_id = self.fsa.panorama_to_node_dict[route_id]

            # Add trajectory for reaching from prev_node_id to this node_id
            trajectory_segment, iter_state = self.streetview_fsa_utils.reach_neighbors(iter_state, node_id)
            trajectory = trajectory + trajectory_segment
            assert iter_state.get_node_id() == node_id, "Reach should reach the node-id %r. " % node_id

        return trajectory

    def explore(self):

        image = self.fsa.get_panorama(self.current_state)
        metadata = {"panorama_id": self.current_state.get_node_id(),
                    "orientation": self.current_state.get_orientation()}

        return image, metadata

    def clear_metadata(self):
        self.sum_navigation_error = 0
        self.sum_node_error = 0
        self.sum_task_completion_accuracy = 0
        self.sum_edit_distance = 0
        self.num_examples = 0

    def halt_nonblocking(self):
        raise NotImplementedError()

    def receive_feedback_nonblocking(self):
        raise NotImplementedError()

    def send_action_nonblocking(self, action):
        raise NotImplementedError()

    def force_goal_update(self):
        raise NotImplementedError()

    def reset_nonblocking(self, next_data_point):
        raise NotImplementedError()
