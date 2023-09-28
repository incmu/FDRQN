import os
import subprocess


def train_fnn():
    print("Training FNN model...")
    os.system("python models/fnn.py")


def train_rnn():
    print("Training RNN model...")
    os.system("python models/rnn.py")


def train_nlp():
    print("Training NLP model...")
    os.system("python models/nlp.py")


def main():
    print("Starting the training of models...")

    train_fnn()
    print("FNN model training completed.")

    train_rnn()
    print("RNN model training completed.")

    train_nlp()
    print("NLP model training completed.")

    print("All models have been trained successfully.")


if __name__ == "__main__":
    main()
