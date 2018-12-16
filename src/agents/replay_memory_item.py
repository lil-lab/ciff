class ReplayMemoryItem:
    """ Contains the state, the action that was taken in that state and the feedback that the agent gave """

    def __init__(self, agent_observed_state, action, reward,
                 mode=None, distance=None, all_rewards=None,
                 log_prob=None, image_emb_seq=None, factor_entropy=None, text_emb=None,
                 symbolic_text=None, next_image_emb_seq=None, goal=None, state_feature=None, volatile=None):
        self.agent_observed_state = agent_observed_state
        self.action = action
        self.reward = reward
        self.q_val = None
        self.mode = mode
        self.distance = distance
        self.all_rewards = all_rewards
        self.log_prob = log_prob
        self.image_emb_seq = image_emb_seq
        self.next_image_emb_seq = next_image_emb_seq
        self.factor_entropy = factor_entropy
        self.text_emb = text_emb
        self.symbolic_text = symbolic_text
        self.goal = goal
        self.state_feature = state_feature
        self.volatile_features = volatile

    def get_agent_observed_state(self):
        return self.agent_observed_state

    def get_action(self):
        return self.action

    def get_reward(self):
        return self.reward

    def set_q_val(self, q_val):
        self.q_val = q_val

    def get_q_val(self):
        return self.q_val

    def get_mode(self):
        return self.mode

    def get_distance(self):
        return self.distance

    def get_all_rewards(self):
        return self.all_rewards

    def get_log_prob(self):
        return self.log_prob

    def get_image_emb(self):
        return self.image_emb_seq

    def get_next_image_emb(self):
        return self.next_image_emb_seq

    def set_next_image_emb(self, next_image_emb_seq):
        self.next_image_emb_seq = next_image_emb_seq

    def get_factor_entropy(self):
        return self.factor_entropy

    def get_text_emb(self):
        return self.text_emb

    def get_goal(self):
        return self.goal

    def get_state_feature(self):
        return self.state_feature

    def get_volatile_features(self):
        return self.volatile_features
