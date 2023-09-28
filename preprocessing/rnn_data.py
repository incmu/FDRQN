import pandas as pd
import numpy as np
from keras.utils import to_categorical

def load_and_preprocess_data(file_path, seq_length):
    # Load CSV file
    data_frame = pd.read_csv(file_path)

    # Combine all code snippets into a large string
    text_data = " ".join(data_frame['target_variable'].values)

    # Create a mapping of unique characters to integers, and a reverse mapping
    chars = sorted(list(set(text_data)))
    char_to_int = dict((c, i) for i, c in enumerate(chars))
    int_to_char = dict((i, c) for i, c in enumerate(chars))

    # Prepare the dataset of input to output pairs encoded as integers
    dataX = []
    dataY = []
    for i in range(0, len(text_data) - seq_length, 1):
        seq_in = text_data[i:i + seq_length]
        seq_out = text_data[i + seq_length]
        dataX.append([char_to_int[char] for char in seq_in])
        dataY.append(char_to_int[seq_out])

    # Reshape X to be [samples, time steps, features] and normalize
    X = np.reshape(dataX, (len(dataX), seq_length, 1)) / float(len(chars))

    # One-hot encode the output variable
    Y = to_categorical(dataY)

    # Create a configuration dictionary
    config = {
        'seq_length': seq_length,
        'num_chars': len(chars),
        'num_classes': len(chars),
        'input_shape': X.shape[1:]
    }

    # Return data and configuration as a dictionary
    return {
        'X': X,
        'Y': Y,
        'config': config
    }
