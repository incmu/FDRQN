import tensorflow as tf
import os
from rewards.rnn_reward import RewardCallback
from preprocessing.rnn_processing import vocab_size_input, vocab_size_target, input_padded, target_padded

# Define constants
output_dir = 'processed_rnn_data'
embedding_dim = 256
rnn_units = 1024
batch_size = 64

# Print Shapes
print("Input Shape:", input_padded.shape)
print("Target Shape:", target_padded.shape)

# Load Processed TensorFlow Datasets
train_dataset = tf.data.experimental.load(os.path.join(output_dir, 'train_dataset'))
test_dataset = tf.data.experimental.load(os.path.join(output_dir, 'test_dataset'))

# Define the RNN model
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=vocab_size_input,
                              output_dim=embedding_dim,
                              mask_zero=True),
    tf.keras.layers.SimpleRNN(rnn_units,
                              return_sequences=True),  # Ensure that the RNN layer returns sequences
    tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(30, activation='softmax'))
    # Adjusted the output dimension to match the target vocabulary size
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Initialize RewardCallback
reward_callback = RewardCallback(test_dataset=test_dataset)

# Model Checkpoint to save the best weights
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath='best_model_rnn.h5',
    save_weights_only=True,
    monitor='val_loss',
    mode='min',
    save_best_only=True)

# Train the model with callbacks
history = model.fit(train_dataset, epochs=3, validation_data=test_dataset, callbacks=[reward_callback, model_checkpoint_callback])

# Load best weights
model.load_weights('best_model_rnn.h5')

# Evaluate the model on the test dataset
test_loss, test_accuracy = model.evaluate(test_dataset)
print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)
