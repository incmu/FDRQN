# nlp_agent.py

from models.nlp.nlp_model import NLPModel
from preprocessing.nlp_data import load_and_preprocess_data
from environment import environment

class NLPAgent:
    def __init__(self, model_config, environment, data_file_path):
        self.model = NLPModel(model_config=model_config)
        self.environment = environment
        self.preprocessed_data = load_and_preprocess_data(data_file_path)


    def act(self, state):
        processed_state = self.preprocess_state(state)
        action = self.model.predict(processed_state)
        return action

    def learn(self, state, action, reward, next_state):
        processed_state = self.preprocess_state(state)
        processed_next_state = self.preprocess_state(next_state)

        # Train the model and include the CustomRewardCallback in the training process
        self.model.train(processed_state, action, processed_next_state)

    def preprocess_state(self, state):
        return state  # Modify if state preprocessing is needed

    def on_epoch_end(self, epoch, logs=None):
        # If there are any additional tasks or logging you would like to perform at the end of each epoch,
        # you can add them here.
        pass
