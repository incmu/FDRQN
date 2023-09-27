import numpy as np
from models.dqn.dqn_model import DQNModel
from preprocessing.dqn_data import load_and_preprocess_data


class DQNAgent:
    def __init__(self, state_size, action_size, environment, data_file_path):
        self.model = DQNModel(state_size, action_size)
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

        # Use the DQN model to make a decision
        return self.model.act(processed_state)

    def learn(self, batch_size):
        """
        Updates the model based on the experiences in memory.

        :param batch_size: The size of the batch to sample from memory.
        """
        # Replay experiences and learn from them
        self.model.replay(batch_size)

    def remember(self, state, action, reward, next_state, done):
        """
        Stores the experience in memory.

        :param state: The current state of the environment.
        :param action: The action taken by the agent.
        :param reward: The reward received for taking the action.
        :param next_state: The next state of the environment after taking the action.
        :param done: Whether the episode has ended.
        """
        # Retrieve the preprocessed data corresponding to the states
        processed_state = self.get_preprocessed_data_for_state(state)
        processed_next_state = self.get_preprocessed_data_for_state(next_state)

        # Store the experience in memory
        self.model.remember(processed_state, action, reward, processed_next_state, done)

    def get_preprocessed_data_for_state(self, state):
        """
        Retrieves the preprocessed data corresponding to the given state.

        :param state: The original state from the environment.
        :return: The preprocessed data for the state.
        """
        # Assuming state can be used as an index to retrieve the corresponding preprocessed data
        # Modify as needed based on how the preprocessed data is structured
        return self.preprocessed_data[state]
