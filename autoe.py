import pandas as pd
import numpy as np
from keras.models import Model
from keras.layers import Input, Dense
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split


def load_data(file_path):
    # Load data from .csv file
    df = pd.read_csv(file_path)
    input_texts = df['nlp language'].values
    target_texts = df['target_variable'].values
    return input_texts, target_texts


def tokenize_and_pad_sequences(input_texts):
    # Tokenize the input_texts
    tokenizer = Tokenizer(char_level=True)
    tokenizer.fit_on_texts(input_texts)
    input_sequences = tokenizer.texts_to_sequences(input_texts)

    # Pad sequences to the same length
    sequence_length = max(len(seq) for seq in input_sequences)
    padded_sequences = pad_sequences(input_sequences, maxlen=sequence_length, padding='post')
    return padded_sequences, sequence_length, tokenizer


def one_hot_encode_targets(target_texts, num_classes):
    tokenizer = Tokenizer(char_level=True)
    tokenizer.fit_on_texts(target_texts)
    target_sequences = tokenizer.texts_to_sequences(target_texts)
    target_sequence_length = max(len(seq) for seq in target_sequences)
    padded_target_sequences = pad_sequences(target_sequences, maxlen=target_sequence_length, padding='post')
    one_hot_targets = to_categorical(padded_target_sequences, num_classes=num_classes)
    return one_hot_targets, target_sequence_length, tokenizer


def train_autoencoder(padded_sequences, sequence_length):
    input_layer = Input(shape=(sequence_length,))
    encoded = Dense(50, activation='relu')(input_layer)
    decoded = Dense(sequence_length, activation='sigmoid')(encoded)
    autoencoder = Model(input_layer, decoded)
    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
    autoencoder.fit(padded_sequences, padded_sequences, epochs=50, batch_size=256, shuffle=True)
    encoder = Model(input_layer, encoded)
    return encoder


def train_nlp_model():
    file_path = 'datasets/js_dataset/javas.csv'
    input_texts, target_texts = load_data(file_path)
    padded_sequences, sequence_length, input_tokenizer = tokenize_and_pad_sequences(input_texts)
    num_classes = len(set(char for seq in target_texts for char in seq)) + 1  # +1 for zero padding
    one_hot_targets, target_sequence_length, target_tokenizer = one_hot_encode_targets(target_texts, num_classes)

    encoder = train_autoencoder(padded_sequences, sequence_length)

    # Returning the additional objects for further use
    return encoder, padded_sequences, one_hot_targets, input_tokenizer, target_tokenizer, sequence_length, target_sequence_length
