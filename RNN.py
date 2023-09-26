# rnn.py
import keras
from keras.models import Sequential
from keras.layers import Dense, LSTM, BatchNormalization
from keras.optimizers import RMSprop
from keras.callbacks import EarlyStopping, LearningRateScheduler
from keras.utils import to_categorical
import numpy as np

from custom_callback import CustomRewardCallback
from data_preprocessing import preprocess_data
from keras.models import load_model  # Import load_model

# Initialization
alpha = 1.0
beta = 0.5
learning_rate = 0.0001
memory = []

# Preprocess data
X_train, X_test, y_train, y_test, X_val, y_val = preprocess_data()
X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
X_val = np.reshape(X_val, (X_val.shape[0], 1, X_val.shape[1]))

# Define Learning Rate Scheduler
def scheduler(epoch, lr):
    if epoch < 10:
        return lr
    else:
        return lr * 0.9


def create_rnn_model(num_classes):
    # Load one of the FNN models to get the input shape for the RNN
    sample_fnn_model = load_model('fnn_model/best_fnn_1.keras')
    # Get the output shape of the last layer (before the output layer)
    input_shape = sample_fnn_model.layers[-2].output_shape[1:]

    # Adjust the input shape for LSTM
    input_shape = (51, 1)

    model = Sequential()

    # Add the initial LSTM layers
    model.add(LSTM(128, activation='tanh', input_shape=input_shape, return_sequences=True))
    model.add(LSTM(128, activation='tanh'))

    # Load the paths of the top FNN models
    top_fnn_models_paths = [
        'fnn_model/best_fnn_1.keras', 'fnn_model/best_fnn_2.keras',
        'fnn_model/best_fnn_3.keras', 'fnn_model/best_fnn_4.keras'
    ]

    # Iterate over each FNN model path and transfer the layers
    for i, model_path in enumerate(top_fnn_models_paths):
        # Load the FNN model
        top_fnn_model = load_model(model_path)

        # Go through the FNN model layers and add them to the RNN model
        for layer in top_fnn_model.layers:
            if isinstance(layer, Dense) and layer != top_fnn_model.layers[-1]:  # Exclude the output layer
                new_dense_layer = Dense(
                    units=layer.units,
                    activation=layer.activation,
                    kernel_initializer=layer.kernel_initializer,
                    bias_initializer=layer.bias_initializer,
                    name=f'dense_transfer_{i}_{layer.name}'  # Adjust the name to include the FNN model index
                )
                # Add the new dense layer to the RNN model
                model.add(new_dense_layer)
            elif isinstance(layer, BatchNormalization):
                # Add the BatchNormalization layer to the RNN model
                model.add(layer)

    # Add the final BatchNormalization and output layers
    model.add(BatchNormalization())
    model.add(Dense(num_classes, activation='softmax'))

    return model


def train_rnn(X_train, y_train, X_val, y_val):
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_val = np.reshape(X_val, (X_val.shape[0], X_val.shape[1], 1))

    # One-hot encode the target labels
    reward_callback = CustomRewardCallback(validation_data=(X_val, y_val), alpha=alpha, beta=beta,
                                           memory=memory, is_rnn=True)

    num_classes = len(np.unique(y_train))
    y_train = to_categorical(y_train, num_classes=num_classes)
    y_val = to_categorical(y_val, num_classes=num_classes)


    # Create an empty list to store the top FNN models
    top_fnn_models = []

    # Create the RNN model once
    rnn_model = create_rnn_model(num_classes)

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
        print("Training RNN...")
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

    # Save the best RNN model
    save_path = "rnn_model/best_rnn.keras"
    best_model.save(save_path)
    return best_model
