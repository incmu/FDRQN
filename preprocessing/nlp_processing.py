import pandas as pd
import re
from sklearn.model_selection import train_test_split
from transformers import GPT2Tokenizer
import tensorflow as tf
import os

# Define file paths
input_file_path = 'datasets/js_dataset/javas.csv'
output_dir = 'processed_nlp_data'

# Load Dataset
df = pd.read_csv(input_file_path)


# Perform basic text cleaning
def clean_text(text):
    # Remove quotes
    text = text.replace('"', '')

    # Remove leading and trailing whitespaces
    text = text.strip()

    # Replace multiple whitespaces with a single whitespace
    text = re.sub(r'\s+', ' ', text)

    return text


# Apply text cleaning function to both input and target columns
df['nlp language'] = df['nlp language'].apply(clean_text)
df['target_variable'] = df['target_variable'].apply(clean_text)

# Initialize the GPT2 tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# Tokenize input texts and targets
input_encodings = tokenizer(df['nlp language'].tolist(), truncation=True, padding=True, max_length=512)
target_encodings = tokenizer(df['target_variable'].tolist(), truncation=True, padding=True, max_length=512)

# Train-Test Split
input_train, input_test, target_train, target_test = train_test_split(input_encodings['input_ids'],
                                                                      target_encodings['input_ids'], test_size=0.2,
                                                                      random_state=42)

# Convert to TensorFlow Dataset
train_dataset = tf.data.Dataset.from_tensor_slices((input_train, target_train))
test_dataset = tf.data.Dataset.from_tensor_slices((input_test, target_test))

# Save processed data (optional)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Save train_dataset and test_dataset for later use
tf.data.experimental.save(train_dataset, os.path.join(output_dir, 'train_dataset'))
tf.data.experimental.save(test_dataset, os.path.join(output_dir, 'test_dataset'))
