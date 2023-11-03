import numpy as np
import pandas as pd
import tensorflow as tf
import random
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from keras.callbacks import LearningRateScheduler
from keras.src.layers import BatchNormalization

from custom_callback import CustomRewardCallback
from data_preprocessing import preprocess_data

# Set random seed for reproducibility
np.random.seed(0)
tf.random.set_seed(0)
random.seed(0)

# Preprocess data and label encode target variable
X_train, X_test, y_train, y_test, X_val, y_val = preprocess_data()

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


def create_feeler_model(input_shape, num_classes):
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

        # Output layer
        Dense(num_classes, kernel_initializer="he_uniform", activation='softmax')
    ])

    return model


def train_feelers(X_train, y_train, X_val, y_val, num_feelers, learning_rate, scheduler, alpha, beta, memory):
    ensemble
    lr_schedule_callback = LearningRateScheduler(scheduler)
    reward_callback = CustomRewardCallback(validation_data=(X_val, y_val), alpha=alpha, beta=beta,
                                           memory=memory, is_rnn=False)

    input_shape = (X_train.shape[1],)
    num_classes = len(np.unique(y_train))

    for i in range(num_feelers):
        print(f"Training model {i + 1}/{num_feelers}")
        model = create_feeler_model(input_shape, num_classes)
        model.compile(optimizer=Adam(learning_rate=learning_rate),
                      loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        model.fit(X_train, y_train, epochs=1, batch_size=32, validation_data=(X_val, y_val),
                  callbacks=[lr_schedule_callback, reward_callback])
        ensemble.append(model)

def get_top_feelers():
    ensemble
    rewards = [evaluate_ensemble([model], X_test, y_test) for model in ensemble]
    top_indices = np.argsort(rewards)[-4:]
    return [ensemble[i] for i in top_indices]


if __name__ == "__main__":
    # ... (your existing code for tuning and printing optimal number of layers)

    # Assuming best_models is a list containing your 16 trained models
    best_models = get_top_feelers()  # replace with actual list of your models

    from sklearn.metrics import classification_report, confusion_matrix
    import seaborn as sns
    import matplotlib.pyplot as plt

    for i, model in enumerate(best_models):
        print(f"Evaluating Model {i + 1}")

        # Predict the classes of the test set
        y_pred = model.predict(X_test)
        y_pred_classes = np.argmax(y_pred, axis=1)

        # 1. Print Classification Report for Precision, Recall, and F1-Score
        print(classification_report(y_test, y_pred_classes))

        # 2. Generate and visualize the Confusion Matrix
        cm = confusion_matrix(y_test, y_pred_classes)

        # Visualize the confusion matrix using seaborn heatmap
        plt.figure(figsize=(10, 7))
        sns.heatmap(cm, annot=True, fmt="d")
        plt.xlabel('Predicted')
        plt.ylabel('Truth')
        plt.title(f'Model {i + 1} - Confusion Matrix')
        plt.show()
