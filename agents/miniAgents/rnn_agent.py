import numpy as np
from models.rnn.rnn_model import RNNModel
from preprocessing.rnn_data import load_and_preprocess_data
from environments import db_environment  # Assuming Environment is the class you have defined in environments.py

class RNNModelAgent:
    def __init__(self, model_config, environment, data_file_path):
        # Assuming model_config is a dictionary containing the 'input_shape' and 'num_classes' keys
        self.model = RNNModel(input_shape=model_config['input_shape'], num_classes=model_config['num_classes'])
        self.environment = environment
        self.preprocessed_data = load_and_preprocess_data(data_file_path, 100)

    def act(self, state):
        processed_state = self.preprocess_state(state)
        action, confidence = self.model.predict(processed_state)
        return action, confidence

    def learn(self, state, action, reward, next_state):
        processed_state = self.preprocess_state(state)
        processed_next_state = self.preprocess_state(next_state)
        # Adjust the model training to take into account the reward and next_state
        self.model.train(processed_state, action, reward, processed_next_state)

    def preprocess_state(self, state):
        # Modify if state preprocessing is needed
        return state

    def get_preprocessed_data_for_state(self, state):
        # Modify as needed based on how the preprocessed_data is structured
        return self.preprocessed_data[state]
