from agents.nlp_agent import NLPAgent
from agents.rnn_agent import RNNModelAgent


class NLNNAgent:
    def __init__(self, nlp_model_config, rnn_model_config, environment, data_file_path):
        # Initialize both NLP and RNN agents
        self.nlp_agent = NLPAgent(model_config=nlp_model_config, environment=environment, data_file_path=data_file_path)
        self.rnn_agent = RNNModelAgent(model_config=rnn_model_config, environment=environment,
                                       data_file_path=data_file_path)

    def act(self, state):
        # Process the state using both agents and decide the action
        nlp_action = self.nlp_agent.act(state)
        rnn_action = self.rnn_agent.act(state)

        # Combine, compare or evaluate both actions and decide the final action
        # This is where you decide how to merge the results of both agents to make a final decision
        final_action = self.evaluate_actions(nlp_action, rnn_action)
        return final_action

    def evaluate_actions(self, nlp_action, rnn_action):
        # Define the logic to evaluate and combine the actions suggested by both agents
        # For example, you might average them, choose the one with higher confidence, or any other logic you find suitable.
        # ...
        combined_action = ...
        return combined_action

    def learn(self, state, action, reward, next_state):
        # Let both agents learn from the experience
        self.nlp_agent.learn(state, action, reward, next_state)
        self.rnn_agent.learn(state, action, reward, next_state)
