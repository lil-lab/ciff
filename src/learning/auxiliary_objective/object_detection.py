import torch


class ObjectDetection:

    def __init__(self, model, num_objects):
        self.num_objects = num_objects
        self.model = model

    def calc_loss(self, batch_replay_items):

        batch_image_feature = []
        agent_observed_state_list = []
        for i, replay_item in enumerate(batch_replay_items):
            agent_observed_state_list.append(replay_item.get_agent_observed_state())
            batch_image_feature.append(replay_item.get_image_emb())

        batch_image_feature = torch.cat(batch_image_feature)
        landmark_log_prob, distance_log_prob, theta_log_prob, visible_objects = self.model.get_probs_and_visible_objects(
            agent_observed_state_list, batch_image_feature)

        num_states = int(landmark_log_prob.size()[0])
        landmark_objective = None
        distance_objective = None
        theta_objective = None
        num_visible = 0.0
        for i in range(0, num_states):
            visible_objects_example = visible_objects[i]
            for landmark in range(0, self.num_objects):
                # See if the landmark is present and visible in the agent's field of view
                if landmark in visible_objects_example and visible_objects_example[landmark][1] != -1:
                    r, theta = visible_objects_example[landmark]
                    if landmark_objective is None:
                        landmark_objective = landmark_log_prob[i, landmark, 1]
                    else:
                        landmark_objective = landmark_objective + landmark_log_prob[i, landmark, 1]

                    if distance_objective is None:
                        distance_objective = distance_log_prob[i, landmark, r]
                    else:
                        distance_objective = distance_objective + distance_log_prob[i, landmark, r]

                    if theta_objective is None:
                        theta_objective = theta_log_prob[i, landmark, theta]
                    else:
                        theta_objective = theta_objective + theta_log_prob[i, landmark, theta]

                    num_visible += 1.0
                else:
                    if landmark_objective is None:
                        landmark_objective = landmark_log_prob[i, landmark, 0]
                    else:
                        landmark_objective = landmark_objective + landmark_log_prob[i, landmark, 0]

        landmark_objective = landmark_objective / float(num_states * self.num_objects)
        loss = - landmark_objective
        if num_visible != 0:
            distance_objective = distance_objective / num_visible
            theta_objective = theta_objective / num_visible
            loss = loss - distance_objective - theta_objective

        return loss
