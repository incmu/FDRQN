import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from ucimlrepo import fetch_ucirepo


def preprocess_data():
    # Fetch dataset
    dry_bean_dataset = fetch_ucirepo(id=222)

    # Data as pandas dataframes
    X = dry_bean_dataset.data.features
    y = dry_bean_dataset.data.targets

    # Identify categorical columns
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()

    # Initialize ColumnTransformer with OneHotEncoder for categorical columns and StandardScaler for the rest
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(), categorical_cols),
            ('num', StandardScaler(), X.columns.difference(categorical_cols))
        ])

    # Split the data into training, validation, and test sets
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.2, random_state=42)

    # Apply transformations to X_train, X_val, and X_test
    X_train = preprocessor.fit_transform(X_train).astype(np.float32)
    X_val = preprocessor.transform(X_val).astype(np.float32)
    X_test = preprocessor.transform(X_test).astype(np.float32)

    # Label encode y
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(y_train).reshape(-1)
    y_val = label_encoder.transform(y_val).reshape(-1)
    y_test = label_encoder.transform(y_test).reshape(-1)

    return X_train, X_test, y_train, y_test, X_val, y_val


# Example usage
#X_train, X_test, y_train, y_test, X_val, y_val = preprocess_data()

# Print shapes and types for verification
#print("X_train type and shape:", type(X_train), X_train.shape)
#print("y_train type and shape:", type(y_train), y_train.shape)
#print("X_val type and shape:", type(X_val), X_val.shape)
#print("y_val type and shape:", type(y_val), y_val.shape)
#print("X_test type and shape:", type(X_test), X_test.shape)
#print("y_test type and shape:", type(y_test), y_test.shape)
