import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline

def preprocessor():
    # Assuming your data file is a CSV
    data_file_path = 'datasets/js_dataset/javas.csv'
    df = pd.read_csv(data_file_path, delimiter=';')

    # Identify categorical columns
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

    # Remove target column from the list of categorical columns
    categorical_cols.remove('y')

    # Define the preprocessing for numeric and categorical features
    numeric_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    numeric_transformer = StandardScaler()

    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # We create the column transformer that will allow us to preprocess the data
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_cols)
        ])

    # We will preprocess the data
    X = df.drop('y', axis=1)
    y = df['y']

    # We will split the data
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    # Applying transformations to X_train, X_val, and X_test
    X_train = preprocessor.fit_transform(X_train)
    X_val = preprocessor.transform(X_val)
    X_test = preprocessor.transform(X_test)

    # Encoding the target variable
    encoder = OneHotEncoder()
    y_train = encoder.fit_transform(y_train.values.reshape(-1, 1)).toarray()
    y_val = encoder.transform(y_val.values.reshape(-1, 1)).toarray()
    y_test = encoder.transform(y_test.values.reshape(-1, 1)).toarray()

    return X_train, X_test, y_train, y_test, X_val, y_val
