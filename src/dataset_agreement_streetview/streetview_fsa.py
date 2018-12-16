import os
import time
import operator
import numpy as np

from collections import defaultdict

from utils.edit_distance import levenshtein_distance
from utils.fibonacci_heap import Fibonacci_heap
from cachetools import LRUCache


class StreetViewState:
    """ Class denoting the state of the FSA """

    def __init__(self, node_id, orientation):
        """ ID of the node where the agent is and
        the orientation of the agent. """
        self.node_id = node_id
        self.orientation = orientation

    def get_node_id(self):
        return self.node_id

    def get_orientation(self):
        return self.orientation

    def __str__(self):
        return "(%r, %r)" % (self.node_id, self.orientation)


class StreetViewFSA:
    """ Finite State Automaton for Street-View """

    OLD_FEATURE, NEW_FEATURE, RGB = range(3)

    image_feature_cache = LRUCache(maxsize=100000)

    def __init__(self, node_file, link_file, image_feature_folder, forward_setting_strict, mapping_type):

        self.image_feature_folder = image_feature_folder
        self.forward_setting_strict = forward_setting_strict

        if mapping_type == "old":
            self.mapping = StreetViewFSA.OLD_FEATURE
        elif mapping_type == "new":
            self.mapping = StreetViewFSA.NEW_FEATURE
        elif mapping_type == "rgb":
            self.mapping = StreetViewFSA.RGB
        else:
            raise AssertionError("Unhandled Mapping type " % mapping_type)

        if self.mapping == StreetViewFSA.OLD_FEATURE:

            self.image_feature_mapping = defaultdict(list)
            for feature_file in os.listdir(self.image_feature_folder):
                # the file name is formated as {panorama_id}.{perspective_angle}.npy
                panorama_id, angle, ext = feature_file.split(".")
                self.image_feature_mapping[panorama_id].append((
                    int(angle), os.path.join(self.image_feature_folder, feature_file)))

            # sort angles for all perspective views in each node (panorama)
            for panorama_id in self.image_feature_mapping.keys():
                self.image_feature_mapping[panorama_id] = sorted(
                    self.image_feature_mapping[panorama_id], key=lambda x: x[0])
        else:
            self.image_feature_mapping = None

        self.node_to_panorama_dict = dict()
        self.panorama_to_node_dict = dict()

        node_lines = open(node_file).readlines()
        for line in node_lines:
            words = line.split(",")
            assert len(words) == 5

            # ignoring latitude and longititude information at 4th and 5th place
            node_id = words[0]
            panorama_id = words[1]
            panorama_orientation = int(words[2])
            self.node_to_panorama_dict[node_id] = (panorama_id, panorama_orientation)
            self.panorama_to_node_dict[panorama_id] = node_id

        link_lines = open(link_file).readlines()

        self.outgoing_edges = dict()

        for line in link_lines:
            words = line.split(",")
            assert len(words) == 4

            source_node_id = words[0]
            orientation = int(words[1])
            dest_node_id = words[2]
            
            if source_node_id in self.outgoing_edges:
                self.outgoing_edges[source_node_id].append((dest_node_id, orientation))
            else:
                self.outgoing_edges[source_node_id] = [(dest_node_id, orientation)]

        # self.fsa_states = set()
        # for source_id in self.outgoing_edges:
        #     for dest_node_id, orientation in self.outgoing_edges[source_id]:
        #         self.fsa_states.add((dest_node_id, orientation))  # one can enter dest_node_id with this orientation
        #         self.fsa_states.add((source_id, orientation))  # one can also turn in the original node to face this orientation

    def reset_to_new_state(self, datapoint):

        trajectory = datapoint.get_trajectory()
        start_panorama_id = trajectory[0]
        goal_panorama_id = trajectory[-1]

        start_node_id = self.panorama_to_node_dict[start_panorama_id]
        goal_node_id = self.panorama_to_node_dict[goal_panorama_id]

        start_orientation = datapoint.get_start_orientation()
        goal_orientation = datapoint.get_end_orientation()

        if self.mapping == StreetViewFSA.RGB:

            # For RGBs image we align the start orientation to face the road
            shortest_angle, shortest_angle_orientation = 1000, None
            for (_, orientation) in self.outgoing_edges[start_node_id]:
                if orientation == start_orientation:
                    shortest_angle_orientation = None
                    break
                angle_diff = self.get_turn_angle(start_orientation, orientation, "shortest")
                if angle_diff < shortest_angle:
                    shortest_angle = angle_diff
                    shortest_angle_orientation = orientation

            if shortest_angle_orientation is not None:
                start_orientation = shortest_angle_orientation

        return StreetViewState(start_node_id, start_orientation), StreetViewState(goal_node_id, goal_orientation)

    def take_action_return_new_state(self, state, act_name):

        node_id = state.get_node_id()
        orientation = state.get_orientation()
        outgoing_edges = self.outgoing_edges[node_id]

        if act_name == "forward":
            # Find the closest road where the agent can go to

            smallest_angle_diff = 1000  # set to a value more than 360 degree
            outgoing_edge = outgoing_edges[0]
            for (new_node_id, new_orientation) in outgoing_edges:
                angle_diff = self.get_turn_angle(orientation, new_orientation, direction="shortest")
                if angle_diff < smallest_angle_diff:
                    smallest_angle_diff = angle_diff
                    outgoing_edge = (new_node_id, new_orientation)

            new_node_id, new_orientation = outgoing_edge

            next_smallest_angle_diff = 1000
            closest_next_orientation = new_orientation
            new_outgoing_edges = self.outgoing_edges[new_node_id]
            for (_, next_orientation) in new_outgoing_edges:
                angle_diff = self.get_turn_angle(next_orientation, new_orientation, direction="shortest")
                if angle_diff < next_smallest_angle_diff:
                    next_smallest_angle_diff = angle_diff
                    closest_next_orientation = next_orientation

            if self.forward_setting_strict:
                if smallest_angle_diff == 0.0:
                    return StreetViewState(new_node_id, closest_next_orientation)
                else:
                    return StreetViewState(node_id, orientation)
            else:
                return StreetViewState(new_node_id, closest_next_orientation)

        elif act_name == "turnleft":

            smallest_angle_diff = 1000  # set to a value more than 360 degree
            outgoing_edge = outgoing_edges[0]
            for (new_node_id, new_orientation) in outgoing_edges:
                if new_orientation == orientation:
                    continue
                angle_diff = self.get_turn_angle(orientation, new_orientation, direction="left")
                if 0 < angle_diff < smallest_angle_diff:
                    smallest_angle_diff = angle_diff
                    outgoing_edge = (new_node_id, new_orientation)

            new_node_id, new_orientation = outgoing_edge

            return StreetViewState(node_id, new_orientation)

        elif act_name == "turnright":

            smallest_angle_diff = 1000  # set to a value more than 360 degree
            outgoing_edge = outgoing_edges[0]
            for (new_node_id, new_orientation) in outgoing_edges:
                if new_orientation == orientation:
                    continue
                angle_diff = self.get_turn_angle(orientation, new_orientation, direction="right")
                if 0 < angle_diff < smallest_angle_diff:
                    smallest_angle_diff = angle_diff
                    outgoing_edge = (new_node_id, new_orientation)

            new_node_id, new_orientation = outgoing_edge

            return StreetViewState(node_id, new_orientation)

        elif act_name == "stop":

            return StreetViewState(state.get_node_id(), state.get_orientation())

        else:
            raise AssertionError("Unhandled action name %s " % act_name)

    def get_image_from_state(self, state):

        if self.mapping == StreetViewFSA.OLD_FEATURE:
            return self.get_image_from_state_old(state)
        elif self.mapping == StreetViewFSA.NEW_FEATURE:
            return self.get_image_from_state_condensed_features(state)
        elif self.mapping == StreetViewFSA.RGB:
            return self.get_image_rgb(state)
        else:
            raise NotImplementedError()

    def get_image_from_state_condensed_features(self, state):

        panorama_id, _ = self.node_to_panorama_dict[state.get_node_id()]
        state_orientation = state.get_orientation()

        key = (panorama_id, state_orientation)

        if key in self.image_feature_cache:
            return self.image_feature_cache[key]
        else:
            _, panorama_orientation = self.node_to_panorama_dict[state.get_node_id()]

            path = os.path.join(self.image_feature_folder, '{}.npy'.format(panorama_id))
            image = np.load(path)
            image = np.expand_dims(image, axis=0)
            pano_width = image.shape[1]

            shift_angle = 157.5 + panorama_orientation - state_orientation
            shift = int(pano_width * shift_angle / 360)
            image = np.roll(image, shift, axis=1)
            image = image.swapaxes(1, 2).swapaxes(0, 1)
            self.image_feature_cache[key] = image

            return image

    def get_image_from_state_old(self, state):

        panorama_id, _ = self.node_to_panorama_dict[state.get_node_id()]
        state_orientation = state.get_orientation()

        key = (panorama_id, state_orientation)

        if key in self.image_feature_cache:
            return self.image_feature_cache[key]
        else:

            # OLD FEATURES
            # find the closest angle from the agent orientation
            index, _ = min(enumerate(self.image_feature_mapping[panorama_id]), key=lambda x: (x[1][0] - state_orientation) % 360)
            feature_paths = [feature_path for _, feature_path in self.image_feature_mapping[panorama_id]]

            # rotate the list to the right so the closest feature is in the middle
            feature_paths = feature_paths[index - 3:] + feature_paths[:index - 3]

            # load perspective image features and transpose so the depth is at the last dimension
            perspective_features = [np.load(feature_path).transpose(1, 2, 0) for feature_path in feature_paths]

            # for each perspective image, the shape is (height, width, depth) = (100, 58, 128)
            height, width, depth = perspective_features[0].shape

            # concatenate all perspective images along the width axis
            # so the full pano has shape (height, width, depth) = (100, 58 * 8, 128)
            image = np.concatenate(perspective_features, axis=1)
            # rotate to the right so half of the last feature is cut off and get appended to the left most

            image = np.roll(image, int(width // 2), axis=1)
            image = image.swapaxes(1, 2).swapaxes(0, 1)
            self.image_feature_cache[key] = image

            return image

    def get_image_rgb(self, state):

        panorama_id, _ = self.node_to_panorama_dict[state.get_node_id()]
        state_orientation = int(state.get_orientation())
        key = (panorama_id, state_orientation)

        if key in self.image_feature_cache:
            return self.image_feature_cache[key]
        else:
            path = os.path.join(self.image_feature_folder, '{}.{}.npy'.format(panorama_id, state_orientation))
            image = np.load(path)
            image = image / 255.0  # important to keep values in 0-1
            image = image.swapaxes(1, 2).swapaxes(0, 1)
            self.image_feature_cache[key] = image

            return image

    def get_panorama(self, state):
        raise NotImplementedError()

    @staticmethod
    def get_turn_angle(angle1, angle2, direction="shortest"):
        """ Turn angle from angle1 to angle2 """
        if direction == "shortest":
            diff = (angle1 - angle2) % 360
            return min(360 - diff, diff)
        elif direction == "left":
            # turning left reduces the orientation
            return (angle1 - angle2) % 360
        elif direction == "right":
            # turning right increases the orientation
            return (angle2 - angle1) % 360
        else:
            raise AssertionError("Dir has to be either shortest, left or right.")


class StreetViewUtils:
    """ Util function for streetview """

    MAX_DISTANCE = 10000.0

    def __init__(self, fsa, action_space):

        self.fsa = fsa
        self.forward_setting_strict = fsa.forward_setting_strict
        self.action_space = action_space
        self.dist = dict()  # dictionary of node_ids to real values
        self.prev = dict()  # dictionary of node_ids to node_ids
        self.fibonacci_node_map = dict()
        self.fib_heap = Fibonacci_heap()
        self.closed_set = set()  # set of closed states
        # self.goal_node = None

        self.goal_node = None
        self.distance_cache = dict()

    def reset_utils(self, goal_state):

        self.goal_node = (goal_state.get_node_id(), goal_state.get_orientation())
        self.distance_cache = dict()

    def path_to_goal(self, start_state, destination_state, get_path=True):
        """ Returns the shortest path to the goal state from the current state using Dijskta's algorithm """

        source_node = (start_state.get_node_id(), start_state.get_orientation())
        destination_node = (destination_state.get_node_id(), destination_state.get_orientation())

        if destination_node == self.goal_node and source_node in self.distance_cache:
            return self.distance_cache[source_node]

        self.dist = dict()  # dictionary of node_ids to real values
        self.prev = dict()  # dictionary of node_ids to node_ids
        self.fibonacci_node_map = dict()
        self.fib_heap = Fibonacci_heap()
        self.closed_set = set()

        self.dist[source_node] = 0.0
        self.prev[source_node] = (None, "stop")
        fib_node = self.fib_heap.enqueue(source_node, self.dist[source_node])
        self.fibonacci_node_map[source_node] = fib_node

        do_no_break_loop = True
        distance = StreetViewUtils.MAX_DISTANCE
        final_node = None

        while do_no_break_loop:

            min_data = self.fib_heap.dequeue_min()

            if min_data is None:
                break

            dist_u = min_data.get_priority()
            u = min_data.get_value()

            self.closed_set.add(u)

            # Consider the following break condition. In each condition, we let the
            # children of the breaking nodes be added to the heap before terminating.
            # this way, we do not loose the children when continuing the next time.
            if u[0] == destination_node[0]:
                distance = dist_u
                final_node = u
                do_no_break_loop = False

            if dist_u == StreetViewUtils.MAX_DISTANCE:
                # The shortest node that is not visited is at infinite distance
                # so relaxation does not makes sense.
                do_no_break_loop = False

            neighbors = StreetViewUtils.get_neighbors(u, self.fsa)
            for v, edge_weight, name in neighbors:

                if v in self.closed_set:
                    continue

                if v in self.dist:
                    fib_node = self.fibonacci_node_map[v]
                    dist_v = self.dist[v]
                else:
                    # Insert to Fib Heap
                    dist_v = StreetViewUtils.MAX_DISTANCE
                    self.dist[v] = dist_v
                    self.prev[v] = (None, "stop")
                    fib_node = self.fib_heap.enqueue(v, dist_v)
                    self.fibonacci_node_map[v] = fib_node

                alt = dist_u + edge_weight
                if alt < dist_v:
                    self.dist[v] = alt
                    self.prev[v] = (u, name)  # name is the action required to go from u to this node

                    # Update the fibonacci heap in O(1)
                    self.fib_heap.decrease_key(fib_node, alt)

        path = []
        if get_path:
            iter_node = final_node
            while iter_node is not None:
                iter_node_pairs = self.prev[iter_node]
                path.append(iter_node_pairs)
                iter_node = iter_node_pairs[0]
            path = list(reversed(path))

        iter_node = final_node
        dist_ix = 0
        node_error = 0
        while iter_node is not None:
            self.distance_cache[iter_node] = (dist_ix, None, node_error)
            dist_ix += 1
            iter_node_pairs = self.prev[iter_node]
            iter_node = iter_node_pairs[0]
            act_name = iter_node_pairs[1]
            if act_name == "forward":
                node_error += 1

        assert (dist_ix - 1) == distance, "distance should match"

        if distance >= StreetViewUtils.MAX_DISTANCE:
            raise AssertionError("Shortest Path Distance exceeds max limit")

        self.distance_cache[source_node] = (distance, path, node_error)

        return distance, path, node_error

    def get_task_completion_accuracy(self, state, datapoint):

        node_id = state.get_node_id()
        pano_id, _ = self.fsa.node_to_panorama_dict[node_id]

        if pano_id == datapoint.get_target_pano() or \
                (datapoint.get_pre_static_center_exists() and pano_id == datapoint.get_pre_pano()) or \
                (datapoint.get_post_static_center_exists() and pano_id == datapoint.get_post_pano()):
            return True
        else:
            return False

    def edit_distance(self, list_all_states, datapoint):

        trajectory = datapoint.get_trajectory()
        node_seq1 = self.__generate_flattened_node_seq__([state.get_node_id() for state in list_all_states])
        node_seq2 = self.__generate_flattened_node_seq__(
            [self.fsa.panorama_to_node_dict[pano_id] for pano_id in trajectory])

        return levenshtein_distance(node_seq1, node_seq2)

    @staticmethod
    def __generate_flattened_node_seq__(node_ids):

        flattened_node_ids = []
        last_node_id = None
        for i, node_id in enumerate(node_ids):
            if i == 0:
                flattened_node_ids.append(node_id)
            elif node_id != last_node_id:
                flattened_node_ids.append(node_id)
            last_node_id = node_id

        return flattened_node_ids

    def reach_neighbors(self, state, neigbhor_node_id):
        """ A fast-implementation for reaching from the state to the neigbhoring node-id """

        node_id = state.get_node_id()
        orientation = state.get_orientation()
        outgoing_edges = self.fsa.outgoing_edges[node_id]

        left_angle = []
        right_angle = []

        neigbhor_node_orientation = None

        for (new_node_id, new_orientation) in outgoing_edges:
            if new_node_id == neigbhor_node_id:
                neigbhor_node_orientation = new_orientation
            left_angle.append((new_node_id, self.fsa.get_turn_angle(orientation, new_orientation, direction="left")))
            right_angle.append((new_node_id, self.fsa.get_turn_angle(orientation, new_orientation, direction="right")))

        assert neigbhor_node_orientation is not None, "Route IDs successive nodes are not neigbhors. %r and %r " % \
                                                      (node_id, neigbhor_node_id)

        # Sort the node-ids by turn angle
        left_angle.sort(key=operator.itemgetter(1))
        right_angle.sort(key=operator.itemgetter(1))

        left_rank_dict = dict()
        left_rank = 0

        for tup in left_angle:
            if tup[1] == 0:
                left_rank_dict[tup[0]] = 0
            else:
                left_rank += 1
                left_rank_dict[tup[0]] = left_rank

        right_rank_dict = dict()
        right_rank = 0

        for tup in right_angle:
            if tup[1] == 0:
                right_rank_dict[tup[0]] = 0
            else:
                right_rank += 1
                right_rank_dict[tup[0]] = right_rank

        # once it is in the new state. Find the closest angle node
        next_smallest_angle_diff = 1000
        new_outgoing_edges = self.fsa.outgoing_edges[neigbhor_node_id]
        closest_next_orientation = neigbhor_node_orientation
        for (_, next_orientation) in new_outgoing_edges:
            angle_diff = self.fsa.get_turn_angle(next_orientation, neigbhor_node_orientation, direction="shortest")
            if angle_diff < next_smallest_angle_diff:
                next_smallest_angle_diff = angle_diff
                closest_next_orientation = next_orientation

        new_state = StreetViewState(node_id=neigbhor_node_id, orientation=closest_next_orientation)

        if left_rank_dict[neigbhor_node_id] < right_rank_dict[neigbhor_node_id]:
            return [self.action_space.get_action_index("turnleft")] * left_rank_dict[neigbhor_node_id] + \
                   [self.action_space.get_action_index("forward")], new_state
        else:
            return [self.action_space.get_action_index("turnright")] * right_rank_dict[neigbhor_node_id] + \
                   [self.action_space.get_action_index("forward")], new_state

    @staticmethod
    def get_neighbors(node, fsa):
        """ Neighbors of a node consist of the left and right node achieved on turning and if the node can
            move forward then move forward."""

        orientation = node[1]
        outgoing_edges = fsa.outgoing_edges[node[0]]

        left_most_node = None
        shortest_left_most_angle = 360

        right_most_node = None
        shortest_right_most_angle = 360

        shortest_angle_node = None
        shortest_turn_angle = 360

        for (new_node_id, new_orientation) in outgoing_edges:

            left_angle = fsa.get_turn_angle(orientation, new_orientation, direction="left")
            right_angle = fsa.get_turn_angle(orientation, new_orientation, direction="right")
            turn_angle = fsa.get_turn_angle(orientation, new_orientation, direction="shortest")

            if 0 < left_angle < shortest_left_most_angle:
                shortest_left_most_angle = left_angle
                left_most_node = (node[0], new_orientation)

            if 0 < right_angle < shortest_right_most_angle:
                shortest_right_most_angle = right_angle
                right_most_node = (node[0], new_orientation)

            if turn_angle < shortest_turn_angle:
                shortest_turn_angle = turn_angle
                shortest_angle_node = (new_node_id, new_orientation)

        neighbors = set()

        if (not fsa.forward_setting_strict) or shortest_turn_angle == 0.0:

            if shortest_angle_node is not None:

                next_smallest_angle_diff = 1000
                new_outgoing_edges = fsa.outgoing_edges[shortest_angle_node[0]]
                closest_next_orientation = shortest_angle_node[1]
                for (_, next_orientation) in new_outgoing_edges:
                    angle_diff = fsa.get_turn_angle(next_orientation, shortest_angle_node[1],
                                                    direction="shortest")
                    if angle_diff < next_smallest_angle_diff:
                        next_smallest_angle_diff = angle_diff
                        closest_next_orientation = next_orientation

                forward_node = (shortest_angle_node[0], closest_next_orientation)
                neighbors.add((forward_node, 1.0, "forward"))

        if left_most_node is not None:
            neighbors.add((left_most_node, 1.0, "turnleft"))

        if right_most_node is not None:
            neighbors.add((right_most_node, 1.0, "turnright"))

        return neighbors

    def get_reward(self, source_state, act_name, new_state, goal_state):

        dist1, path1, _ = self.path_to_goal(source_state, goal_state)
        dist2, path2, _ = self.path_to_goal(new_state, goal_state)

        problem_reward = -0.2

        if act_name == "stop":
            if dist2 == 0:
                problem_reward = 1.0  # stopping and have reached the goal
            else:
                problem_reward = -1.0  # stopping and failed to reach the goal
        elif dist1 == dist2 and dist1 > 0:  # made no progress at all and still haven't reached the goal
            problem_reward = -1.0

        # Goal based reward shaping
        reward = problem_reward + dist1 - dist2
        # print("Reward: %s -> %s | %s is %f dist 1: %f, dist2: %f, problem-reward %f" %
        #       (str(source_state), str(new_state), str(goal_state), reward, dist1, dist2, problem_reward))

        return reward
