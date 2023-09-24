import numpy as np
import tensorflow as tf
from keras.layers import Dense
from keras import Model
from keras.optimizers import Adam
import matplotlib.pyplot as plt
from keras.models import load_model

from enivornment_gym import BankEnv
from data_preprocessing import preprocess_data
from RNN import train_rnn

X_train, X_test, y_train, y_test, X_val, y_val = preprocess_data()


class QNetwork(Model):
    def __init__(self, action_size, input_shape):
        super(QNetwork, self).__init__()
        self.dense1 = Dense(24, activation='relu', input_shape=input_shape)
        self.dense2 = Dense(24, activation='relu')
        self.dense3 = Dense(action_size, activation='linear')

    def call(self, state):
        x = self.dense1(state)
        x = self.dense2(x)
        return self.dense3(x)


class DQNAgent:
    def __init__(self, action_size, state_shape, rnn_model):
        self.rnn_model = rnn_model
        self.q_network = QNetwork(action_size, state_shape)
        self.optimizer = Adam(learning_rate=1e-3)
        self.criterion = tf.keras.losses.MeanSquaredError()

    def learn(self, state, action, reward, next_state, done, gamma=0.99):
        print("Preprocessing state and next_state with RNN model...")
        state = state.reshape((1, 1, -1))
        state = self.rnn_model.predict(state)
        next_state = next_state.reshape((1, 1, -1))
        next_state = self.rnn_model.predict(next_state)

        print("Calculating loss and applying gradients...")
        with tf.GradientTape() as tape:
            q_values = self.q_network(state)
            next_q_values = self.q_network(next_state)
            q_value = tf.gather(q_values, action, batch_dims=1)
            next_q_value = tf.reduce_max(next_q_values, axis=1)
            target_q_value = reward + (1 - done) * gamma * next_q_value
            loss = self.criterion(q_value, target_q_value)

        grads = tape.gradient(loss, self.q_network.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.q_network.trainable_variables))


def initialize_environment_and_agent():
    print("Initializing environment and agent...")
    env = BankEnv()
    action_size = env.action_space.n
    state_shape = env.observation_space.shape

    print("Loading the RNN model...")
    load_path ="rnn_model/best_rnn.h5"
    best_rnn_model = load_model(load_path)

    dqn_agent = DQNAgent(action_size, state_shape, best_rnn_model)
    return env, dqn_agent


def train_dqn_agent(env, dqn_agent, num_episodes=1000, gamma=0.99):
    print("Training DQN agent...")
    for episode in range(num_episodes):
        print(f"Episode {episode + 1}/{num_episodes}")
        state = env.reset()
        done = False
        while not done:
            action = env.action_space.sample()
            next_state, reward, done, _ = env.step(action)
            dqn_agent.learn(np.array([state]), np.array([action]), np.array([reward]), np.array([next_state]), done,
                            gamma)
            state = next_state
    print("Training of the DQN agent is complete!")

if __name__ == "__main__":
    env, dqn_agent = initialize_environment_and_agent()
    train_dqn_agent(env, dqn_agent)
    print("The program has finished executing.")