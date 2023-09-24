# rnn.py
import keras
from keras.models import Sequential
from keras.layers import Dense, LSTM, BatchNormalization
from keras.optimizers import RMSprop
from keras.callbacks import EarlyStopping, LearningRateScheduler
from keras.utils import to_categorical  # Corrected import
from matplotlib import pyplot as plt
import custom_callback
from custom_callback import CustomRewardCallback
import numpy as np


from feelers import get_top_feelers
from data_preprocessing import preprocess_data

# Initialization
alpha = 1.0
beta = 0.5
learning_rate = 0.0001
memory = []

# Preprocess data
X_train, X_test, y_train, y_test, X_val, y_val = preprocess_data()


# Define Learning Rate Scheduler
def scheduler(epoch, lr):
    if epoch < 10:
        return lr
    else:
        return lr * 0.9


def create_rnn_model(input_shape, num_classes):
    model = Sequential()
    model.add(LSTM(128, activation='tanh', input_shape=input_shape))
    model.add(Dense(2, activation='tanh'))
    model.add(BatchNormalization())
    model.add(Dense(num_classes, activation='softmax'))
    return model


def train_rnn(X_train, y_train, X_val, y_val):
    num_classes = len(np.unique(y_train))
    # One-hot encode the target labels
    y_train = to_categorical(y_train, num_classes=num_classes)
    y_val = to_categorical(y_val, num_classes=num_classes)

    top5_feelers = get_top_feelers()

    # Ensure X_train and X_val are 3D arrays
    if len(X_train.shape) == 2:
        X_train = np.expand_dims(X_train, axis=1)
    if len(X_val.shape) == 2:
        X_val = np.expand_dims(X_val, axis=1)
    # Create the RNN model once
    rnn_model = create_rnn_model((1, X_train.shape[2]), num_classes)

    # Average the weights and biases from the final Dense layer of the top 5 feelers
    averaged_weights = np.mean([feeler.layers[-1].get_weights()[0] for feeler in top5_feelers], axis=0)
    averaged_biases = np.mean([feeler.layers[-1].get_weights()[1] for feeler in top5_feelers], axis=0)

    # Ensure the dimensions match before setting weights
    if rnn_model.layers[1].get_weights()[0].shape == averaged_weights.shape:
        rnn_model.layers[1].set_weights([averaged_weights, averaged_biases])
    else:
        print("Dimension mismatch, unable to set weights.")

    # Define Learning Rate Scheduler Callback
    lr_schedule_callback = LearningRateScheduler(scheduler)

    # Reinforcement Learning Loop on the RNN
    stale_count = 0
    last_reward = None

    # Init Val Loss
    training_losses = []
    validation_losses = []

    best_model = None
    best_accuracy = 0
    learning_rate = 0.0001

    for episode in range(10):

        reward_callback = CustomRewardCallback(validation_data=(X_val, y_val), alpha=alpha, beta=beta, episode=episode,
                                               memory=memory, is_rnn=True)
        optimizer = RMSprop(learning_rate=0.0001)
        rnn_model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        history = rnn_model.fit(X_train, y_train, epochs=1, batch_size=32, validation_data=(X_val, y_val),
                                callbacks=[EarlyStopping(patience=3), reward_callback, lr_schedule_callback])

        # Evaluate the RNN
        loss, accuracy = rnn_model.evaluate(X_val, y_val)
        print(f"RNN Loss: {loss}")
        print(f"RNN Accuracy: {accuracy}")

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = keras.models.clone_model(rnn_model)
            best_model.set_weights(rnn_model.get_weights())



        if len(memory) > 0:
            latest_reward = memory[-1]['total_reward']
            if last_reward is not None and abs(last_reward - latest_reward) < 0.001:
                stale_count += 1
                if stale_count >= 3:
                    print("Model is stale. Stopping training.")
                    break
            else:
                stale_count = 0

            training_losses.extend(history.history['loss'])
            validation_losses.extend(history.history['val_loss'])

            last_reward = latest_reward
            if latest_reward >= 6221:  # Threshold for your task
                learning_rate *= 1.2
            else:
                learning_rate *= 0.5

            learning_rate = min(0.01, max(0.0001, learning_rate))
        else:
            print("Memory is empty, unable to check the latest reward.")

        print(f"New learning rate: {learning_rate}")
    save_path = "rnn_model/best_rnn.h5"
    best_model.save(save_path)
    return best_model