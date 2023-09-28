import numpy as np
from models.ppo.ppo_model import PPOModel
from preprocessing.ppo_data import load_and_preprocess_data
from environments import db_environment

class PPOAgent:
    def __init__(self, state_size, action_size, environment, data_file_path):
        self.model = PPOModel(state_size, action_size)
        self.environment = db_environment
        self.preprocessed_data = load_and_preprocess_data(data_file_path)

    def act(self, state):
        # You might need to modify the state based on your environment's state
        processed_state = self.get_preprocessed_data_for_state(state)
        action = self.model.act(processed_state)

        # Execute the action in the environment and get the new state, reward, and done status
        next_state, reward, done = self.environment.step(action)
        return action, reward, next_state, done

    def learn(self, states, actions, rewards, next_states, dones):
        processed_states = np.array([self.get_preprocessed_data_for_state(state) for state in states])
        processed_next_states = np.array([self.get_preprocessed_data_for_state(state) for state in next_states])
        self.model.learn(processed_states, actions, rewards, processed_next_states, dones)

    def get_preprocessed_data_for_state(self, state):
        return self.preprocessed_data[state]
