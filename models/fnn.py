import tensorflow as tf
import os
from rewards.fnn_reward import RewardCallback

# Define constants
output_dir = 'processed_fnn_data'
embedding_dim = 256
hidden_units = 512
batch_size = 64

# Load Processed TensorFlow Datasets
train_dataset = tf.data.Dataset.load(os.path.join(output_dir, 'train_dataset')).batch(batch_size)
test_dataset = tf.data.Dataset.load(os.path.join(output_dir, 'test_dataset')).batch(batch_size)

def reshape_labels(x, y):
    y = y[:, 1]  # Use the second element of each label array as the label
    return x, y

train_dataset = train_dataset.map(reshape_labels)
test_dataset = test_dataset.map(reshape_labels)


# Define the FNN model
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=10000,  # Set this to the size of your vocabulary
                              output_dim=embedding_dim,
                              mask_zero=True),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(hidden_units, activation='relu'),
    tf.keras.layers.Dense(10000, activation='softmax')  # Set this to the size of your vocabulary
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Initialize RewardCallback
reward_callback = RewardCallback(test_dataset=test_dataset)

# Model Checkpoint to save the best weights
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath='fnn_model.h5',
    save_weights_only=True,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True)

for x, y in train_dataset.take(1):
    print(x.shape)  # Should print (64, 17)
    print(y.shape)  # Should print (64,)

# Train the model with callbacks
history = model.fit(train_dataset, epochs=3, validation_data=test_dataset, callbacks=[reward_callback, model_checkpoint_callback])

# Load best weights
model.load_weights('fnn_model.h5')

# Evaluate the model on the test dataset
test_loss, test_accuracy = model.evaluate(test_dataset)
print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)
