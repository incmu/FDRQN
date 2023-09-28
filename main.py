import numpy as np

from preprocessing.nlp_data import load_and_preprocess_data  # Assuming this is where your preprocessed data and config are stored
from models.nlp.nlp_model import NLPModel  # Import your NLP model class
from models.seq2seq.seq2seq_model import Seq2SeqModel  # Import your Seq2Seq model class
from models.rnn.rnn_model import RNNModel  # Import your RNN model class

# Load and preprocess data
file_path = 'datasets/js_dataset/javas.csv'  # Specify the actual file path for your data
data = load_and_preprocess_data(file_path)
X = data['X']
Y = data['Y']
config = data['config']  # Added this line to get the config

nlp_model = NLPModel(data['config'], input_length=data['X'].shape[1], num_classes=data['Y'].shape[1])
  # Use config to initialize NLPModel

print("Shape of X:", data['X'].shape)
print("Shape of Y:", data['Y'].shape)


nlp_model.train(data['X'], data['Y'], epochs=10, batch_size=32)  # Adjust epochs and batch_size as needed
  # Adjust epochs and batch_size as needed

# Get NLP Model output to feed into Seq2Seq Model
nlp_output = nlp_model.predict(X)
nlp_output_sequence_length = nlp_output.shape[1]
Y_sequence_length = Y.shape[1]
embedding_dim = 256


# Initialize and Train Seq2Seq Model (Assuming you have an appropriate model and data preparation)
seq2seq_model = Seq2SeqModel(input_shape=(nlp_output_sequence_length, embedding_dim), num_classes=config['num_classes'])
seq2seq_model.train(nlp_output, Y, epochs=10, batch_size=32)  # Adjust epochs and batch_size as needed

# Get Seq2Seq Model output to feed into RNN Model
seq2seq_output = seq2seq_model.predict(nlp_output)

# Initialize and Train RNN Model
rnn_model = RNNModel(input_shape=seq2seq_output.shape, num_classes=config['num_classes'])
rnn_model.train(seq2seq_output, Y, epochs=10, batch_size=32)  # Adjust epochs and batch_size as needed

print("Training Complete!")

# You can now use nlp_model, seq2seq_model, and rnn_model for predictions.
