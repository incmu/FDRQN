from keras.callbacks import Callback
import numpy as np
from keras.src.metrics import accuracy
from sklearn.metrics import precision_score, recall_score, accuracy_score
from keras import backend as K

class CustomRewardCallback(Callback):
    def __init__(self, validation_data, metrics_weights={'precision': 0.5, 'recall': 1.0}, accuracy_threshold=0.9,
                 penalty_threshold=720, reward_adjustment=10, learning_rate_adjustment={'increase': 1.1, 'decrease': 0.9},
                 is_rnn=False):
        super(CustomRewardCallback, self).__init__()
        self.metrics_weights = metrics_weights
        self.accuracy_threshold = accuracy_threshold
        self.penalty_threshold = penalty_threshold
        self.reward_adjustment = reward_adjustment
        self.learning_rate_adjustment = learning_rate_adjustment
        self.validation_data = validation_data
        self.is_rnn = is_rnn
        self.memory = []

    def adjust_learning_rate(self, new_lr):
        K.set_value(self.model.optimizer.lr, new_lr)

    def on_epoch_end(self, epoch, logs=None):
        super().on_epoch_end(epoch, logs)
        X_val, y_val = self.validation_data

        if len(y_val.shape) > 1 and y_val.shape[1] > 1:
            y_val = np.argmax(y_val, axis=1)
        else:
            y_val = y_val.flatten()

        if self.is_rnn:
            X_val = self.adjust_input_shape(X_val)

        y_pred_labels = np.argmax(self.model.predict(X_val), axis=1)

        try:
            precision = precision_score(y_val, y_pred_labels, average='weighted', zero_division=1)
            recall = recall_score(y_val, y_pred_labels, average='weighted')
            accuracy = accuracy_score(y_val, y_pred_labels)
        except Exception as e:
            print(f"Error in calculating metrics: {e}")
            return

        R_t = self.calculate_reward(precision, recall, y_pred_labels, y_val)

        self.memory.append({'epoch': epoch, 'total_reward': R_t})
        self.adjust_learning_rate_based_on_reward(R_t)

    def adjust_input_shape(self, X_val):
        num_samples, timesteps, num_features = X_val.shape
        timesteps = 1
        num_features_per_timestep = num_features // timesteps
        return np.reshape(X_val, (num_samples, timesteps, num_features_per_timestep))

    def calculate_reward(self, precision, recall, y_pred_labels, y_val):
        P_t = np.sum(y_pred_labels == y_val)
        N_t = np.sum(y_pred_labels != y_val)
        R_t = self.metrics_weights['precision'] * precision + self.metrics_weights['recall'] * recall

        if accuracy > self.accuracy_threshold:
            R_t += self.reward_adjustment
        if N_t > self.penalty_threshold:
            R_t -= self.penalty_threshold * N_t
        return R_t

    def adjust_learning_rate_based_on_reward(self, R_t):
        current_lr = K.get_value(self.model.optimizer.lr)
        if R_t > self.penalty_threshold:
            new_lr = current_lr * self.learning_rate_adjustment['increase']
        elif R_t < self.penalty_threshold:
            new_lr = current_lr * self.learning_rate_adjustment['decrease']
        else:
            new_lr = current_lr

        self.adjust_learning_rate(new_lr)
        print(f"New learning rate: {new_lr}")
