import pandas as pd
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
import os

# Define file paths
input_file_path = 'datasets/js_dataset/javas.csv'
output_dir = 'processed_rnn_data'

# Load Dataset
df = pd.read_csv(input_file_path)

# Initialize Tokenizer for input texts
input_tokenizer = Tokenizer(filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', lower=True, oov_token='<OOV>')
input_tokenizer.fit_on_texts(df['nlp language'].values)

# Tokenize and pad input texts
input_sequences = input_tokenizer.texts_to_sequences(df['nlp language'].values)
input_padded = pad_sequences(input_sequences, padding='post')

# Vocabulary size should include the OOV token
vocab_size_input = len(input_tokenizer.word_index) + 1

# Initialize Tokenizer for target texts
target_tokenizer = Tokenizer(filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', lower=False, oov_token='<OOV>')
target_tokenizer.fit_on_texts(df['target_variable'].values)

# Tokenize and pad target texts
target_sequences = target_tokenizer.texts_to_sequences(df['target_variable'].values)
target_padded = pad_sequences(target_sequences, padding='post')

# Vocabulary size should include the OOV token
vocab_size_target = len(target_tokenizer.word_index) + 1

# Train-Test Split
input_train, input_test, target_train, target_test = train_test_split(input_padded, target_padded, test_size=0.2, random_state=42)

# Convert to TensorFlow Dataset and batch
batch_size = 64
train_dataset = tf.data.Dataset.from_tensor_slices((input_train, target_train)).shuffle(len(input_train)).batch(batch_size)
test_dataset = tf.data.Dataset.from_tensor_slices((input_test, target_test)).batch(batch_size)

# Save processed data
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Save train_dataset and test_dataset for later use
tf.data.Dataset.save(train_dataset, os.path.join(output_dir, 'train_dataset'))
tf.data.Dataset.save(test_dataset, os.path.join(output_dir, 'test_dataset'))

# Optionally, save tokenizers and vocabulary sizes for later use
