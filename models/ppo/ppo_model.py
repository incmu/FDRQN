import numpy as np
import tensorflow as tf
from keras import layers


class PPOModel:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = 0.99  # Discount factor
        self.alpha = 0.001  # Learning rate for actor
        self.beta = 0.001  # Learning rate for critic
        self.epsilon = 0.1  # Clip factor for PPO

        self.actor = self.build_actor()
        self.critic = self.build_critic()

    def build_actor(self):
        state_input = layers.Input(shape=(self.state_size,))
        advantage = layers.Input(shape=(1,))
        old_prediction = layers.Input(shape=(self.action_size,))

        x = layers.Dense(64, activation='relu')(state_input)
        x = layers.Dense(64, activation='relu')(x)
        out_actions = layers.Dense(self.action_size, activation='softmax', name='output')(x)

        model = tf.keras.models.Model(inputs=[state_input, advantage, old_prediction], outputs=[out_actions])
        model.compile(optimizer=tf.keras.optimizers.Adam(lr=self.alpha),
                      loss=[self.ppo_loss(advantage=advantage, old_prediction=old_prediction)])
        return model

    def build_critic(self):
        state_input = layers.Input(shape=(self.state_size,))
        x = layers.Dense(64, activation='relu')(state_input)
        x = layers.Dense(64, activation='relu')(x)
        value_output = layers.Dense(1, activation=None)(x)

        model = tf.keras.models.Model(inputs=state_input, outputs=value_output)
        model.compile(optimizer=tf.keras.optimizers.Adam(lr=self.beta), loss='mse')
        return model

    def ppo_loss(self, advantage, old_prediction):
        def loss(y_true, y_pred):
            prob = y_true * y_pred
            old_prob = y_true * old_prediction
            r = prob / (old_prob + 1e-10)
            clip_val = tf.clip_by_value(r, 1 - self.epsilon, 1 + self.epsilon)
            return -tf.reduce_mean(tf.minimum(r * advantage, clip_val * advantage))

        return loss

    def act(self, state):
        # Use the actor model to output a probability distribution over actions
        policy = self.actor.predict([state, np.zeros((1, 1)), np.zeros((1, self.action_size))])[0]
        return np.random.choice(self.action_size, p=policy)

    def learn(self, states, actions, rewards, next_states, dones):
        # Update actor and critic models
        advantages, target_values = self.compute_advantages(states, rewards, next_states, dones)
        old_predictions = self.actor.predict(states)

        # Training Actor Model
        self.actor.fit([states, advantages, old_predictions], actions, epochs=10, verbose=0)

        # Training Critic Model
        self.critic.fit(states, target_values, epochs=10, verbose=0)

    def compute_advantages(self, states, rewards, next_states, dones):
        # Compute advantages and target values for actor-critic
        advantages = np.zeros((len(rewards), 1))
        target_values = np.zeros((len(rewards), 1))
        running_add = 0
        for t in reversed(range(len(rewards))):
            if dones[t]:
                running_add = 0
            delta = rewards[t] + self.gamma * running_add * (1 - dones[t]) - self.critic.predict(states[t])[0]
            running_add = delta + self.gamma * running_add * (1 - dones[t])
            advantages[t][0] = delta
            target_values[t][0] = running_add
        return advantages, target_values
