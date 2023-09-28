import tensorflow as tf
from transformers import GPT2Tokenizer, TFGPT2LMHeadModel
import os
from rewards.nlp_reward import RewardCallback

# Define the directory where processed data is saved
output_dir = 'processed_data'

# Load Processed TensorFlow Datasets
train_dataset = tf.data.experimental.load(os.path.join(output_dir, 'train_dataset'))
test_dataset = tf.data.experimental.load(os.path.join(output_dir, 'test_dataset'))

# Load Pre-trained GPT-2 model
model = TFGPT2LMHeadModel.from_pretrained('gpt2')

# Define optimizer, loss, and metric
optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')

# Compile the model
model.compile(optimizer=optimizer, loss=[loss, *[None] * model.config.n_layer], metrics=[metric])

# Initialize RewardCallback
reward_callback = RewardCallback(test_dataset=test_dataset.batch(16))

# Model Checkpoint to save the best weights
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath='best_model_nlp.h5',
    save_weights_only=True,
    monitor='val_loss',
    mode='min',
    save_best_only=True)

# Train the model with callbacks
history = model.fit(train_dataset.shuffle(1000).batch(16), epochs=3, validation_data=test_dataset.batch(16), callbacks=[reward_callback, model_checkpoint_callback])

# Load best weights
model.load_weights('best_model_nlp.h5')

# Evaluate the model on the test dataset
test_loss, test_accuracy = model.evaluate(test_dataset.batch(16))
print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)
