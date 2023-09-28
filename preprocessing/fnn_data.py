import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer

def load_and_preprocess_data(file_path):
    # Load dataset
    df = pd.read_csv(file_path)

    # Extract features and labels
    X = df['nlp language']
    y = df['target_variable']

    # Convert the text data to TF-IDF vectors
    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(X).toarray()

    # One-hot encode the labels
    label_binarizer = LabelBinarizer()
    y = label_binarizer.fit_transform(y)

    # Split the data into training and validation sets
    x_train, x_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create a configuration dictionary
    config = {
        'num_features': X.shape[1],  # Number of features after TF-IDF
        'num_classes': y.shape[1],   # Number of classes (one-hot encoded labels)
    }

    # Return preprocessed data and configuration as a dictionary
    return {
        'x_train': x_train,
        'y_train': y_train,
        'x_val': x_val,
        'y_val': y_val,
        'config': config,
    }
