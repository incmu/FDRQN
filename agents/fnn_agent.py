import numpy as np
from models.fnn.fnn_model import FNNModel
from preprocessing.fnn_data import load_and_preprocess_data
from environment import environment


class FNNAgent:
    def __init__(self, model_config, environment, data_file_path):
        FNNModel(input_dim=model_config['input_dim'], num_classes=model_config['num_classes'])
        self.environment = environment
        self.preprocessed_data = load_and_preprocess_data(data_file_path)


    def act(self, state):
        """
        Uses the model to make a decision based on the current state.

        :param state: The current state of the environment.
        :return: The action selected by the agent.
        """
        # Retrieve the preprocessed data corresponding to the state
        processed_state = self.get_preprocessed_data_for_state(state)

        # Use the FNN model to make a prediction
        action = self.model.predict(processed_state)
        return action

    def learn(self, state, action, reward, next_state):
        """
        Updates the model based on the received reward and the transition dynamics.

        :param state: The current state of the environment.
        :param action: The action taken by the agent.
        :param reward: The reward received for taking the action.
        :param next_state: The next state of the environment after taking the action.
        """
        # Retrieve the preprocessed data corresponding to the states
        processed_state = self.get_preprocessed_data_for_state(state)
        processed_next_state = self.get_preprocessed_data_for_state(next_state)



        # Perform learning/updating of the model with the reward callback
        self.model.train(processed_state, action, processed_next_state)

    def get_preprocessed_data_for_state(self, state):
        """
        Retrieves the preprocessed data corresponding to the given state.

        :param state: The original state from the environment.
        :return: The preprocessed data for the state.
        """
        # Assuming state can be used as an index to retrieve the corresponding preprocessed data
        # Modify as needed based on how the preprocessed data is structured
        return self.preprocessed_data[state]
