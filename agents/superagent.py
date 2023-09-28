from agents.dqn_agent import DQNAgent
from agents.fnn_agent import FNNAgent
from agents.nlnn import NLNNAgent


class SuperAgent:
    def __init__(self, dqn_config, fnn_config, nlp_rnn_config, environment, data_file_path):
        self.dqn_agent = DQNAgent(**dqn_config, environment=environment, data_file_path=data_file_path)
        self.fnn_agent = FNNAgent(**fnn_config, environment=environment, data_file_path=data_file_path)
        self.nlp_rnn_agent = NLNNAgent(**nlp_rnn_config, environment=environment, data_file_path=data_file_path)

    def act(self, state):
        dqn_action = self.dqn_agent.act(state)
        fnn_action = self.fnn_agent.act(state)
        nlp_rnn_action = self.nlp_rnn_agent.act(state)

        # Combine the actions in a meaningful way
        combined_action = self.combine_actions(dqn_action, fnn_action, nlp_rnn_action)
        return combined_action

    def learn(self, batch_size):
        self.dqn_agent.learn(batch_size)
        self.fnn_agent.learn(batch_size)
        self.nlp_rnn_agent.learn(batch_size)

    def remember(self, state, action, reward, next_state, done):
        self.dqn_agent.remember(state, action, reward, next_state, done)
        self.fnn_agent.remember(state, action, reward, next_state, done)
        self.nlp_rnn_agent.remember(state, action, reward, next_state, done)

    def combine_actions(self, dqn_action, fnn_action, nlp_rnn_action):
        # Define a strategy to combine the actions of the individual agents
        # For example, you might average them, choose one based on some criteria, or create a composite action
        # This will depend on the nature of the actions and the requirements of your environment
        pass
