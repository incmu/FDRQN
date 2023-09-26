import numpy as np
from keras.layers import Embedding, LSTM, Dense, Input, Masking
from keras.models import Model
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import autoe  # Ensure that autoe.py is in the same directory or adjust the import accordingly


class NLPModel:
    def __init__(self, input_dim, output_dim, embedding_dim=64, lstm_units=100):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.embedding_dim = embedding_dim
        self.lstm_units = lstm_units
        self.build_model()

    def build_model(self):
        # Encoder
        encoder_input = Input(shape=(None,))
        encoder_embedding = Embedding(input_dim=self.input_dim, output_dim=self.embedding_dim)(encoder_input)
        encoder_masking = Masking(mask_value=0)(encoder_embedding)
        encoder_lstm, state_h, state_c = LSTM(self.lstm_units, return_state=True)(encoder_masking)
        encoder_states = [state_h, state_c]

        # Decoder
        decoder_input = Input(shape=(None,))
        decoder_embedding = Embedding(input_dim=self.output_dim, output_dim=self.embedding_dim)(decoder_input)
        decoder_masking = Masking(mask_value=0)(decoder_embedding)
        decoder_lstm = LSTM(self.lstm_units, return_sequences=True, return_state=True)
        decoder_outputs, _, _ = decoder_lstm(decoder_masking, initial_state=encoder_states)
        decoder_dense = Dense(self.output_dim, activation='softmax')
        decoder_outputs = decoder_dense(decoder_outputs)

        # Model
        self.model = Model([encoder_input, decoder_input], decoder_outputs)
        optimizer = Adam(learning_rate=0.001)
        self.model.summary()
        self.model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    def train_model(self, X_train, y_train, epochs=10, batch_size=64):
        self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)

    def evaluate_model(self, X_test, y_test):
        loss, accuracy = self.model.evaluate(X_test, y_test)
        print(f'Model Loss: {loss}')
        print(f'Model Accuracy: {accuracy}')


def train_nlp():
    # Retrieve encoder, tokenized input, one-hot targets, and tokenizers from autoe
    encoder, padded_sequences, one_hot_targets, input_tokenizer, target_tokenizer, sequence_length, target_sequence_length = autoe.train_nlp_model()

    # Define input_dim and output_dim
    input_dim = len(input_tokenizer.word_index) + 1  # +1 for padding token
    output_dim = len(target_tokenizer.word_index) + 1  # +1 for padding token

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(padded_sequences, one_hot_targets, test_size=0.2, random_state=42)
    print("X_train shape:", X_train.shape)
    print("y_train shape:", y_train.shape)
    # Adjust the output_dim to match the third dimension of y_train
    output_dim = y_train.shape[2]

    # Initialize and build the Seq2Seq model with the adjusted output_dim
    seq2seq_model = NLPModel(input_dim=input_dim, output_dim=output_dim)


    # Train and evaluate the model
    seq2seq_model.train_model([X_train, X_train], y_train)  # Modified as per the model’s expected input
    seq2seq_model.evaluate_model([X_test, X_test], y_test)  # Modified as per the model’s expected input


# Call the function to train the model

