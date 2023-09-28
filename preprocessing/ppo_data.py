import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split


def load_and_preprocess_data(file_path):
    # Load CSV file
    data_frame = pd.read_csv(file_path)

    # Extract the 'nlp lang' and 'code snippet' columns
    nlp_lang_data = data_frame['nlp lang']
    code_snippet_data = data_frame['code snippet']

    # Tokenize the 'nlp lang' and 'code snippet' columns
    tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n')
    tokenizer.fit_on_texts(nlp_lang_data + code_snippet_data)

    # Convert text to sequences
    nlp_lang_seq = tokenizer.texts_to_sequences(nlp_lang_data)
    code_snippet_seq = tokenizer.texts_to_sequences(code_snippet_data)

    # Pad sequences to the same length
    nlp_lang_seq = tf.keras.preprocessing.sequence.pad_sequences(nlp_lang_seq, padding='post')
    code_snippet_seq = tf.keras.preprocessing.sequence.pad_sequences(code_snippet_seq, padding='post')

    # Train-test split
    nlp_lang_train, nlp_lang_val, code_snippet_train, code_snippet_val = train_test_split(
        nlp_lang_seq, code_snippet_seq, test_size=0.2, random_state=42)

    # Create a configuration dictionary
    config = {
        'vocab_size': len(tokenizer.word_index) + 1,  # Adding 1 because of reserved 0 index
        'input_length': nlp_lang_seq.shape[1]
    }

    # Return data and configuration as a dictionary
    return {
        'nlp_lang_train': nlp_lang_train,
        'code_snippet_train': code_snippet_train,
        'nlp_lang_val': nlp_lang_val,
        'code_snippet_val': code_snippet_val,
        'config': config
    }
