from agents.miniAgents.nlp_agent import NLPAgent
from agents.miniAgents.rnn_agent import RNNModelAgent


class NLNNAgent:
    def __init__(self, nlp_model_config, rnn_model_config, environment, data_file_path):
        # Initialize both NLP and RNN agents
        self.nlp_agent = NLPAgent(model_config=nlp_model_config, environment=environment, data_file_path=data_file_path)
        self.rnn_agent = RNNModelAgent(model_config=rnn_model_config, environment=environment,
                                       data_file_path=data_file_path)

    def act(self, state):
        # Process the state using both agents and decide the action
        nlp_action, nlp_confidence = self.nlp_agent.act(state)
        rnn_action, rnn_confidence = self.rnn_agent.act(state)

        # Combine, compare, or evaluate both actions and decide the final action
        final_action = self.evaluate_actions(nlp_action, nlp_confidence, rnn_action, rnn_confidence)
        return final_action

    def evaluate_actions(self, nlp_action, nlp_confidence, rnn_action, rnn_confidence):
        # Compare the confidence levels of the two actions and return the one with higher confidence
        if nlp_confidence >= rnn_confidence:
            return nlp_action
        else:
            return rnn_action

    def learn(self, state, action, reward, next_state):
        # Let both agents learn from the experience
        self.nlp_agent.learn(state, action, reward, next_state)
        self.rnn_agent.learn(state, action, reward, next_state)

