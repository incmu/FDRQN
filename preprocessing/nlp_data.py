import pandas as pd
import nltk
import string
from nltk.corpus import stopwords
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical

# Downloading nltk datasets
nltk.download('punkt')
nltk.download('stopwords')


def load_and_preprocess_data(file_path):
    # Load CSV file
    data_frame = pd.read_csv(file_path)

    # Extracting 'nlp language' as input and 'target_variable' as output
    X = data_frame['nlp language'].values
    Y = data_frame['target_variable'].values

    # Define Preprocessing function for the input
    def preprocess_text(text):
        # Tokenization
        tokens = nltk.word_tokenize(text)
        # Lowercasing
        tokens = [word.lower() for word in tokens]
        # Removing punctuation and stopwords
        tokens = [word for word in tokens if word not in string.punctuation and word not in stopwords.words('english')]
        return ' '.join(tokens)  # Join tokens back into a string

    # Preprocess Input Text Data
    preprocessed_X = [preprocess_text(text) for text in X]

    # Tokenization and Padding for the input
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(preprocessed_X)
    sequences_X = tokenizer.texts_to_sequences(preprocessed_X)
    padded_sequences_X = pad_sequences(sequences_X)

    # Encoding the output variable (target_variable)
    label_encoder = LabelEncoder()
    integer_encoded_Y = label_encoder.fit_transform(Y)
    one_hot_Y = to_categorical(integer_encoded_Y)

    # Returning preprocessed data and configuration
    config = {
        'vocab_size': len(tokenizer.word_index) + 1,
        'input_length': padded_sequences_X.shape[1],
        'num_classes': one_hot_Y.shape[1]
    }
    return {
        'X': padded_sequences_X,
        'Y': one_hot_Y,
        'config': config
    }