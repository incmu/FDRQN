import keras
import numpy as np
import pandas as pd
import tensorflow as tf
import random
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from keras.callbacks import LearningRateScheduler
from keras.layers import BatchNormalization
from keras.src.layers import Reshape

from custom_callback import CustomRewardCallback
from data_preprocessing import preprocess_data
from dqn import DQNAgent
from enivornment_gym import BankEnv

# Set random seed for reproducibility
np.random.seed(0)
tf.random.set_seed(0)
random.seed(0)

# Preprocess data and label encode target variable
X_train, X_test, y_train, y_test, X_val, y_val = preprocess_data()
# Assuming X_train is a DataFrame
X_train = X_train.astype(np.float32)
y_train = y_train.astype(np.int32)
X_val = X_val.astype(np.float32)
y_val = y_val.astype(np.int32)

# Initialization
ensemble = []
num_feelers = 8
learning_rate = 0.0001
alpha = 1.0
beta = 0.5
memory = []

def load_dqn_agent():
    # Initialize the environment and get the action size and state shape
    env = BankEnv()
    action_size = env.action_space.n
    state_shape = env.observation_space.shape

    # Specify the path to load the DQN model weights
    load_dqn_agent = 'dqn_model/dqn_model_weights_.keras'

    try:
        # Try to instantiate the DQNAgent with the loaded model weights
        dqn_agent = DQNAgent(action_size, state_shape, load_dqn_agent)
        print("DQN model weights loaded successfully!")
        return dqn_agent  # Return the loaded agent
    except (FileNotFoundError, Exception) as e:
        # If loading fails, print an error message and instantiate the DQNAgent without loading the weights
        print(f"Failed to load DQN model weights: {e}")
        print("Initializing DQNAgent without pre-trained weights.")
        return None

# Ensemble Prediction Function
def ensemble_predict(ensemble, X):
    predictions = [model.predict(X) for model in ensemble]
    avg_prediction = np.mean(predictions, axis=0)
    return np.argmax(avg_prediction, axis=1)

# Evaluate Ensemble Function
def evaluate_ensemble(ensemble, X, y):
    predictions = [model.predict(X) for model in ensemble]
    avg_prediction = np.mean(predictions, axis=0)
    y_pred_labels = np.argmax(avg_prediction, axis=1)
    y = np.array(y).ravel()
    reward = np.sum(y_pred_labels == y) / len(y)
    return reward

# Define Learning Rate Scheduler
def scheduler(epoch, lr):
    if epoch < 10:
        return lr
    else:
        return lr * 0.9

def create_feeler_model(input_shape, num_classes, reshaped_output_shape):
    model = Sequential([
        # Input layer
        Dense(512, activation='tanh', kernel_initializer="he_uniform",
              kernel_regularizer='l2', input_shape=input_shape),
        Dropout(0.3),
        BatchNormalization(),

        # Hidden layers
        Dense(128, activation='relu', kernel_initializer="he_uniform", kernel_regularizer='l2'),
        Dropout(0.1),
        BatchNormalization(),

        # Reshape layer
        Reshape(target_shape=reshaped_output_shape),

        # Output layer
        Dense(num_classes, kernel_initializer="he_uniform", activation='softmax')
    ])

    return model

def train_feelers(X_train, y_train, X_val, y_val, num_feelers, learning_rate, scheduler, alpha, beta, memory):
    lr_schedule_callback = LearningRateScheduler(scheduler)
    reward_callback = CustomRewardCallback(validation_data=(X_val, y_val), alpha=alpha, beta=beta,
                                           memory=memory, is_rnn=False)

    input_shape = (X_train.shape[1],)
    num_classes = len(np.unique(y_train))

    # Create a list to store the top 4 FNN models with the highest accuracy
    top_models = []
    top_accuracies = []
    reshaped_output_shape =(128,)
    for i in range(num_feelers):
        print(f"Training model {i + 1}/{num_feelers}")
        model = create_feeler_model(input_shape, num_classes, reshaped_output_shape)
        model.compile(optimizer=Adam(learning_rate=learning_rate),
                      loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        model.fit(X_train, y_train, epochs=1, batch_size=32, validation_data=(X_val, y_val),
                  callbacks=[lr_schedule_callback, reward_callback])

        loss, accuracy = model.evaluate(X_val, y_val)
        print(f"FNN Loss: {loss}")
        print(f"FNN Accuracy: {accuracy}")

        # Keep track of the top 4 models with the highest accuracy
        if len(top_models) < 4:
            top_models.append(model)
            top_accuracies.append(accuracy)
        else:
            min_accuracy_index = np.argmin(top_accuracies)
            if accuracy > top_accuracies[min_accuracy_index]:
                top_models[min_accuracy_index] = model
                top_accuracies[min_accuracy_index] = accuracy

    # Save the top 4 FNN models
    for i, model in enumerate(top_models):
        save_path = f"fnn_model/best_fnn_{i+1}.keras"
        model.save(save_path)

    return top_models

