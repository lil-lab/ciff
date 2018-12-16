class SymbolicTextReplayMemoryItem:
    """ Contains the state and the position you want to predict """

    def __init__(self, agent_observed_state, symbolic_text):
        self.agent_observed_state = agent_observed_state
        self.symbolic_text = symbolic_text

    def get_agent_observed_state(self):
        return self.agent_observed_state

    def get_symbolic_text(self):
        return self.symbolic_text
