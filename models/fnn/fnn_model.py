import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout

class FNNModel:
    def __init__(self, input_dim, num_classes):
        # Initialize FNN-specific components and parameters
        self.model = self.build_model(input_dim, num_classes)

    def build_model(self, input_dim, num_classes):
        # Define a simple FNN model
        model = Sequential()
        model.add(Dense(128, input_dim=input_dim, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(num_classes, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    def train(self, x_train, y_train, epochs, batch_size, callbacks=[]):
        # Train the FNN model using the provided data
        self.model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, callbacks=callbacks)

    def predict(self, input_data):
        # Make predictions using the trained FNN model
        return self.model.predict(input_data)
