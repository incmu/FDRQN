import numpy as np
from models.rnn.rnn_model import RNNModel
from preprocessing.rnn_data import load_and_preprocess_data
from environment import reward

class RNNModelAgent:
    def __init__(self, model_config, environment, data_file_path, reward_config):
        self.model = RNNModel(model_config=model_config)
        self.environment = environment
        self.preprocessed_data = load_and_preprocess_data(data_file_path)
        self.reward_callback = reward(validation_data=self.preprocessed_data, **reward_config)

    def act(self, state):
        """
        Uses the model to make a decision based on the current state.

        :param state: The current state of the environment.
        :return: The action selected by the agent.
        """
        # Retrieve the preprocessed data corresponding to the state
        processed_state = self.get_preprocessed_data_for_state(state)

        # Use the RNN model to make a prediction
        action = self.model.predict(processed_state)

        return action

    def learn(self, state, action, reward, next_state):
        processed_state = self.preprocess_state(state)
        processed_next_state = self.preprocess_state(next_state)

        # Train the model and include the CustomRewardCallback in the training process
        self.model.train(processed_state, action, reward, processed_next_state, callbacks=[self.reward_callback])

    def get_preprocessed_data_for_state(self, state):
        """
        Retrieves the preprocessed data corresponding to the given state.

        :param state: The original state from the environment.
        :return: The preprocessed data for the state.
        """
        # Assuming state can be used as an index to retrieve the corresponding preprocessed data
        # Modify as needed based on how the preprocessed data is structured
        return self.preprocessed_data[state]
