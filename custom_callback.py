#custom callback
from keras.callbacks import Callback
from sklearn.metrics import precision_score, recall_score, accuracy_score
from keras import backend as K
import numpy as np


class CustomRewardCallback(Callback):
    def __init__(self, validation_data, alpha=1.0, beta=0.5, gamma=1.0, delta=2.0, episode=None, memory=None, is_rnn=False):
        super(CustomRewardCallback, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.episode = episode
        self.memory = memory if memory is not None else []
        self.validation_data = validation_data
        self.is_rnn = is_rnn

    def adjust_learning_rate(self, new_lr):
        K.set_value(self.model.optimizer.lr, new_lr)

    def on_epoch_end(self, epoch, logs=None):
        super().on_epoch_end(epoch, logs)

        X_val, y_val = self.validation_data

        # Check the shape of y_val and flatten if necessary
        if len(y_val.shape) > 1 and y_val.shape[1] > 1:
            y_val = np.argmax(y_val, axis=1)
        else:
            y_val = y_val.flatten()

        # Reshape the input data if the model is an RNN
        if self.is_rnn:
            num_samples = X_val.shape[0]
            num_elements = X_val.size
            timesteps = num_elements // num_samples
            num_features_per_timestep = 1  # assuming 1 feature per timestep

            # Ensure that the reshape is valid
            assert num_samples * timesteps * num_features_per_timestep == num_elements

            X_val = np.reshape(X_val, (num_samples, timesteps, num_features_per_timestep))

        y_pred = self.model.predict(X_val)
        y_pred_labels = np.argmax(y_pred, axis=1)

        try:
            precision = precision_score(y_val, y_pred_labels, average='weighted', zero_division=1)
            recall = recall_score(y_val, y_pred_labels, average='weighted')
            accuracy = accuracy_score(y_val, y_pred_labels)
        except Exception as e:
            print(f"Error in calculating metrics: {e}")
            return

        P_t = np.sum(y_pred_labels == y_val)
        N_t = np.sum(y_pred_labels != y_val)
        R_t = self.alpha * P_t + self.beta * precision + self.gamma * recall

        if accuracy > 0.906202377661045:
            R_t += 10
        if N_t > 720:
            R_t -= self.delta * N_t

        print(f"P_t: {P_t}")
        print(f"N_t: {N_t}")
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        print(f"Accuracy: {accuracy}")
        print(f"R_t: {R_t}")

        self.memory.append({'episode': self.episode, 'total_reward': R_t})

        # Dynamic learning rate adjustment based on R_t
        current_lr = K.get_value(self.model.optimizer.lr)
        if R_t > 4787.319567338135:
            new_lr = current_lr * 1.1  # Increase learning rate
        elif R_t < 4265.283841154123:
            new_lr = current_lr * 0.9  # Decrease learning rate
        else:
            new_lr = current_lr  # Keep learning rate constant

        self.adjust_learning_rate(new_lr)
        print(f"New learning rate: {new_lr}")

