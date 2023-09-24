import gym
import numpy as np
from gym import spaces
from keras import backend as K
from keras.optimizers import Adam
from sklearn.metrics import precision_score

from data_preprocessing import preprocess_data


class BankEnv(gym.Env):
    def __init__(self):
        super(BankEnv, self).__init__()

        X_train, X_test, y_train, y_test, X_val, y_val = preprocess_data()

        self.X = X_train
        self.y = y_train
        self.current_step = 0

        num_actions = len(np.unique(self.y))
        self.action_space = spaces.Discrete(num_actions)

        low_bound = np.min(self.X)
        high_bound = np.max(self.X)
        self.observation_space = spaces.Box(low=low_bound, high=high_bound, shape=(self.X.shape[1],))

        # Initialize the optimizer
        self.optimizer = Adam(learning_rate=0.001)

    def step(self, action):
        observation = self.X[self.current_step]
        reward = self.calculate_reward(action, observation, self.y[self.current_step])

        self.current_step += 1
        done = self.current_step >= len(self.X)

        self.adjust_model_parameters(reward)

        return observation, reward, done, {}

    def reset(self):
        self.current_step = 0
        return self.X[self.current_step]

    def calculate_reward(self, action, observation, label):
        y_pred_label = np.array([action])
        y_val = np.array([label])

        try:
            precision = precision_score(y_val, y_pred_label, average='weighted', zero_division=1)
        except Exception as e:
            print(f"Error in calculating precision: {e}")
            return 0

        return precision

    def adjust_model_parameters(self, reward):
        current_lr = float(K.get_value(self.optimizer.learning_rate))

        if reward > 6000:
            new_lr = current_lr * 1.1
        elif reward < 5150:
            new_lr = current_lr * 0.9
        else:
            new_lr = current_lr

        K.set_value(self.optimizer.learning_rate, new_lr)
