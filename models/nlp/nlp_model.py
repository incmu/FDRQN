import tensorflow as tf
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense

class NLPModel:
    def __init__(self, model_config):
        self.vocab_size = model_config.get("vocab_size", 10000)
        self.embedding_dim = model_config.get("embedding_dim", 128)
        self.input_length = model_config.get("input_length", 100)
        self.num_classes = model_config.get("num_classes", 2)
        self.model = self.build_model()

    def build_model(self):
        model = Sequential()
        model.add(Embedding(input_dim=self.vocab_size, output_dim=self.embedding_dim, input_length=self.input_length))
        model.add(LSTM(units=512, activation='tanh', kernel_initializer='he_uniform', kernel_regularizer='l2'))
        model.add(Dense(self.num_classes, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    def train(self, x_train, y_train, epochs=10, batch_size=32, callbacks=[]):
        self.model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, callbacks=callbacks)

    def predict(self, input_data):
        return self.model.predict(input_data)

# Example of usage
model_config = {
    "vocab_size": 5000,
    "embedding_dim": 64,
    "input_length": 120,
    "num_classes": 3
}
nlp_model = NLPModel(model_config=model_config)
