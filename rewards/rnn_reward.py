import numpy as np
import tensorflow as tf
from sklearn.metrics import precision_score, recall_score, accuracy_score
from keras import backend as K
from preprocessing.rnn_processing import train_dataset,test_dataset



class RewardCallback(tf.keras.callbacks.Callback):
    def __init__(self, test_dataset, alpha=1.0, beta=1.0, gamma=1.0, delta=1.0):
        super(RewardCallback, self).__init__()
        self.test_dataset = test_dataset
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.memory = []
        self.episode = 0

    def on_epoch_end(self, epoch, logs=None):
        super().on_epoch_end(epoch, logs)
        self.episode += 1

        # Handle tf.data.Dataset
        y_true = []
        y_pred_list = []
        for X_val, y_val in self.test_dataset:
            y_pred = self.model.predict(X_val)
            y_pred_labels = np.argmax(y_pred, axis=1)
            y_pred_list.extend(y_pred_labels)
            y_true.extend(y_val.numpy().flatten())

        y_true = np.array(y_true)
        y_pred_list = np.array(y_pred_list)

        # Metrics Calculation
        try:
            precision = precision_score(y_true, y_pred_list, average='weighted', zero_division=1)
            recall = recall_score(y_true, y_pred_list, average='weighted')
            accuracy = accuracy_score(y_true, y_pred_list)
        except Exception as e:
            print(f"Error in calculating metrics: {e}")
            return

        P_t = np.sum(y_pred_list == y_true)
        N_t = np.sum(y_pred_list != y_true)
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
        if R_t > 6000:
            new_lr = current_lr * 1.1  # Increase learning rate
        elif R_t < 5150:
            new_lr = current_lr * 0.9  # Decrease learning rate
        else:
            new_lr = current_lr  # Keep learning rate constant

        K.set_value(self.model.optimizer.lr, new_lr)
        print(f"New learning rate: {new_lr}")


