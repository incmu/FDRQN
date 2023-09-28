import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical

# Load the dataset
file_path = "datasets/js_dataset/javas.csv"
df = pd.read_csv(file_path)

# Extract the input and target variables
x = df['nlp language']
y = df['target_variable']

# Tokenize the input and target variables
input_tokenizer = Tokenizer(filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n')
input_tokenizer.fit_on_texts(x)
x_seq = input_tokenizer.texts_to_sequences(x)
x_pad = pad_sequences(x_seq, padding='post')

# Tokenize and encode the target variable
target_tokenizer = Tokenizer(filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', char_level=True)
target_tokenizer.fit_on_texts(y)
y_seq = target_tokenizer.texts_to_sequences(y)
y_pad = pad_sequences(y_seq, padding='post')
y_cat = to_categorical(y_pad)

# Split the dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x_pad, y_cat, test_size=0.2, random_state=42)

# Convert test datasets to tf.data.Dataset
batch_size = 32  # You can set your own batch size
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(batch_size)
test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size)
