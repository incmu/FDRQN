import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


def load_and_preprocess_data(file_path):
    # Load the dataset
    df = pd.read_csv(file_path)

    # Combine nlp language and target_variable columns to create a full code snippet
    full_snippets = df['nlp language'] + ' ' + df['target_variable']

    # Tokenize the code snippets
    tokenizer = Tokenizer(filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', lower=False, split=' ')
    tokenizer.fit_on_texts(full_snippets)

    # Convert text to sequences
    sequences = tokenizer.texts_to_sequences(full_snippets)

    # Find the maximum sequence length
    max_seq_length = max(len(seq) for seq in sequences)

    # Pad sequences to the maximum sequence length
    padded_sequences = pad_sequences(sequences, maxlen=max_seq_length, padding='post')

    # The state for DQN could be represented as the sequence of tokens up to the current token
    # The action would be the next token in the sequence

    states = []
    actions = []
    for seq in padded_sequences:
        for i in range(len(seq) - 1):
            states.append(seq[:i + 1])
            actions.append(seq[i + 1])

    # Pad states sequences to the maximum sequence length
    states = pad_sequences(states, maxlen=max_seq_length, padding='post')

    return states, actions, tokenizer


file_path = "path_to_your_dataset/javas.csv"
states, actions, tokenizer = load_and_preprocess_data(file_path)

print('States shape:', states.shape)
print('Actions length:', len(actions))
