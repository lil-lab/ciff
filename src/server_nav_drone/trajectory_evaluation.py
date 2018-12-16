import numpy as np


def mean_closest_distance(trajectory, destination_list):
    pos_array = np.array(trajectory)
    dest_array = np.array(destination_list)
    dists = np.array([min(((pos_array - dest) ** 2).sum(1) ** 0.5)
                      for dest in dest_array])
    # logging.info(str(dists))
    return float(dists.mean())


def norm_edit_distance(trajectory, destination_list):
    def get_edit_distance(i_, j_, cache):
        # calculates the edit distance starting from index i_ in trajectory
        # and index j_ in destination_list
        key = i_, j_
        if j_ >= len(destination_list):
            return 0.0
        elif key in cache:
            return cache[key]
        elif i_ == len(trajectory) - 1:
            final_pos = np.array(trajectory[-1])
            dest_array = np.array(destination_list[j_:])
            dists_from_final = ((dest_array - final_pos) ** 2).sum(1) ** 0.5
            total_dist = float(sum(dists_from_final))
            cache[key] = total_dist
            return total_dist
        else:
            assert i_ < len(trajectory) - 1 and j_ < len(destination_list)
            start_pos = np.array(trajectory[i_])
            start_goal = np.array(destination_list[j_])
            dist_i_j = float(sum((start_goal - start_pos) ** 2)) ** 0.5
            dist_1 = dist_i_j + get_edit_distance(i_ + 1, j_ + 1, cache)
            dist_2 = get_edit_distance(i_ + 1, j_, cache)
            total_dist = min(dist_1, dist_2)
            cache[key] = total_dist
            return total_dist

    cache = {}
    return get_edit_distance(0, 0, cache) / len(destination_list)


def stop_distance(trajectory, destination_list):
    goal = np.array(destination_list[-1])
    end_pos = np.array(trajectory[-1])
    return float(sum((goal - end_pos) ** 2)) ** 0.5
