from agents.miniAgents.fnn_agent import FNNAgent
from agents.combinedAgents.nlnn_agent import NLNNAgent


class BigAgent:
    def __init__(self, fnn_model_config, nlp_model_config, rnn_model_config, environment, data_file_path):
        # Initialize both FNNAgent and NLNNAgent
        self.fnn_agent = FNNAgent(model_config=fnn_model_config, environment=environment, data_file_path=data_file_path)
        self.nlnn_agent = NLNNAgent(nlp_model_config=nlp_model_config, rnn_model_config=rnn_model_config,
                                    environment=environment, data_file_path=data_file_path)

    def act(self, state):
        # Get actions and confidences from both agents
        fnn_action, fnn_confidence = self.fnn_agent.act(state)
        nlnn_action, nlnn_confidence = self.nlnn_agent.act(state)

        # Evaluate actions and choose the one with higher confidence
        final_action = self.evaluate_actions(fnn_action, fnn_confidence, nlnn_action, nlnn_confidence)
        return final_action

    def evaluate_actions(self, fnn_action, fnn_confidence, nlnn_action, nlnn_confidence):
        # Compare the confidence levels and return the action with higher confidence
        if fnn_confidence >= nlnn_confidence:
            return fnn_action
        else:
            return nlnn_action

    def learn(self, state, action, reward, next_state):
        # Let both agents learn from the experience
        self.fnn_agent.learn(state, action, reward, next_state)
        self.nlnn_agent.learn(state, action, reward, next_state)
