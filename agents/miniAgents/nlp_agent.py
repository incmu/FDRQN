# nlp_agent.py

from models.nlp.nlp_model import NLPModel
from preprocessing.nlp_data import load_and_preprocess_data
from environments import db_environment


class NLPAgent:
    def __init__(self, model_config, environment, data_file_path):
        self.model = NLPModel(model_config=model_config)
        self.environment = environment
        self.preprocessed_data = load_and_preprocess_data(data_file_path)

    def act(self, state, other_agent_action=None):
        # Modify this method as necessary to handle actions from other agents
        processed_state = self.preprocess_state(state)

        # If actions from other agents are involved in deciding the action,
        # incorporate them here
        # e.g., action = self.model.predict(processed_state, other_agent_action)

        action, confidence = self.model.predict(processed_state)
        return action, confidence

    def learn(self, state, action, reward, next_state):
        processed_state = self.preprocess_state(state)
        processed_next_state = self.preprocess_state(next_state)

        # Adjust the model training to take into account the reward and next_state
        # Possibly you might need a target value calculated using the reward and next_state
        # e.g., target = reward + gamma * self.model.predict(processed_next_state)

        # Train the model and include the CustomRewardCallback in the training process
        self.model.train(processed_state, action, reward, processed_next_state)

    def preprocess_state(self, state):
        # Modify if state preprocessing is needed
        return state

    def on_epoch_end(self, epoch, logs=None):
        # If there are any additional tasks or logging you would like to perform at the end of each epoch,
        # you can add them here.
        pass
