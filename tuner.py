import shutil
import uuid
import numpy as np
from keras.src.layers import activation, LeakyReLU, PReLU, ThresholdedReLU, Activation
from keras_tuner import HyperModel, Hyperband
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization, LSTM
from keras.optimizers import Adam
from keras.optimizers import RMSprop


from data_preprocessing import preprocess_data  # Make sure to import your actual data_preprocessing module

# Load and preprocess data
X_train, X_test, y_train, y_test, X_val, y_val = preprocess_data()

shutil.rmtree('output/', ignore_errors=True)


# Feeler HyperModel with Dense layers
class FeelerHyperModel(HyperModel):
    def __init__(self, input_shape, num_classes):
        self.input_shape = input_shape
        self.num_classes = num_classes

    def build(self, hp):
        model = Sequential()

        # Define choices
        activation_choices = ['tanh', 'relu', 'elu', 'exponential', 'sigmoid', 'selu',
                              'softsign', 'softplus', 'gelu', 'leakyrelu', 'prelu']
        kernel_initializer_choices = ['he_uniform', 'orthogonal', 'glorot_uniform']

        # Input layer
        activation_type = hp.Choice('input_activation', values=activation_choices)
        model.add(Dense(units=hp.Int('input_units', min_value=256, max_value=512, step=256),
                        activation='linear',  # Set to linear temporarily
                        kernel_initializer=hp.Choice('input_kernel_initializer', values=kernel_initializer_choices),
                        kernel_regularizer=('l2'),
                        input_shape=self.input_shape))
        model.add(Dropout(hp.Float('input_dropout', min_value=0.1, max_value=0.5, step=0.1)))

        if activation_type == 'leakyrelu':
            model.add(LeakyReLU(alpha=hp.Float('input_leakyrelu_alpha', min_value=0.01, max_value=0.3, step=0.01)))
        elif activation_type == 'prelu':
            model.add(PReLU())
        elif activation_type == 'thresholdedrelu':
            model.add(ThresholdedReLU(theta=hp.Float('input_thresholdedrelu_theta', min_value=1e-3, max_value=1.0, step=1e-3)))
        else:
            model.add(Activation(activation_type))

        model.add(BatchNormalization())

        # Hidden layers
        for i in range(hp.Int('num_layers', 1, 5)):
            activation_type = hp.Choice(f'activation_{i}', values=activation_choices)
            model.add(Dense(units=hp.Int(f'units_{i}', min_value=128, max_value=512, step=128),
                            activation='linear',  # Set to linear temporarily
                            kernel_initializer=hp.Choice(f'kernel_initializer_{i}', values=kernel_initializer_choices),
                            kernel_regularizer=('l2')))
            model.add(Dropout(hp.Float(f'dropout_{i}', min_value=0.1, max_value=0.5, step=0.1)))

            if activation_type == 'leakyrelu':
                model.add(LeakyReLU(alpha=hp.Float(f'leakyrelu_alpha_{i}', min_value=0.01, max_value=0.3, step=0.01)))
            elif activation_type == 'prelu':
                model.add(PReLU())
            elif activation_type == 'thresholdedrelu':
                model.add(ThresholdedReLU(theta=hp.Float(f'thresholdedrelu_theta_{i}', min_value=1e-3, max_value=1.0, step=1e-3)))
            else:
                model.add(Activation(activation_type))

            model.add(BatchNormalization())

        # Output layer
        model.add(Dense(self.num_classes,
                        kernel_initializer=hp.Choice('output_kernel_initializer', values=kernel_initializer_choices),
                        activation='softmax'))

        model.compile(optimizer=Adam(hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])),
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])

        return model
# RNN HyperModel with LSTM layers
class RnnHyperModel(HyperModel):
    def __init__(self, input_shape, num_classes):
        self.input_shape = input_shape
        self.num_classes = num_classes

    def build(self, hp):
        model = Sequential()

        # Define choices
        activation_choices = ['tanh', 'relu', 'elu', 'exponential', 'sigmoid', 'selu',
                              'softsign', 'softplus', 'gelu', 'leakyrelu', 'prelu']
        kernel_initializer_choices = ['he_uniform', 'orthogonal', 'glorot_uniform']

        # Input LSTM layer
        input_activation = hp.Choice('input_activation', values=activation_choices)
        model.add(LSTM(units=hp.Int('input_units', min_value=64, max_value=256, step=64),
                       activation='linear',  # Set to linear temporarily
                       kernel_initializer=hp.Choice('input_kernel_initializer', values=kernel_initializer_choices),
                       input_shape=self.input_shape,
                       return_sequences=True))
        model.add(Dropout(hp.Float('input_dropout', min_value=0.1, max_value=0.5, step=0.1)))

        if input_activation == 'leakyrelu':
            model.add(LeakyReLU(alpha=hp.Float('input_leakyrelu_alpha', min_value=0.01, max_value=0.3, step=0.01)))
        elif input_activation == 'prelu':
            model.add(PReLU())
        else:
            model.add(Activation(input_activation))

        model.add(BatchNormalization())

        # Hidden LSTM layers
        num_layers = hp.Int('num_layers', 1, 5)
        for i in range(num_layers):
            activation = hp.Choice(f'activation_{i}', values=activation_choices)
            model.add(LSTM(units=hp.Int(f'units_{i}', min_value=64, max_value=256, step=64),
                           activation='linear',  # Set to linear temporarily
                           kernel_initializer=hp.Choice(f'kernel_initializer_{i}', values=kernel_initializer_choices),
                           return_sequences=True if i < num_layers - 1 else False,
                           kernel_regularizer=('l2')))
            model.add(Dropout(hp.Float(f'dropout_{i}', min_value=0.1, max_value=0.5, step=0.1)))

            if activation == 'leakyrelu':
                model.add(LeakyReLU(alpha=hp.Float(f'leakyrelu_alpha_{i}', min_value=0.01, max_value=0.3, step=0.01)))
            elif activation == 'prelu':
                model.add(PReLU())
            else:
                model.add(Activation(activation))

            model.add(BatchNormalization())

        # Output layer
        model.add(Dense(self.num_classes,
                        activation='softmax',
                        kernel_initializer=hp.Choice('output_kernel_initializer', values=kernel_initializer_choices)))

        model.compile(optimizer=RMSprop(learning_rate=hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])),
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
        return model


def tune_models(hypermodel, model_type):
    input_shape = (X_train.shape[1],) if model_type == "Feeler" else (X_train.shape[1], 1)
    num_classes = len(np.unique(y_train))

    tuner = Hyperband(
        hypermodel,
        objective='val_accuracy',
        max_epochs=100,
        factor=3,
        directory=f'output/{model_type}_tune_{uuid.uuid4().hex}',
        project_name='kerastuner'
    )

    tuner.search(X_train, y_train, epochs=50, validation_data=(X_val, y_val))

    best_model = tuner.get_best_models(num_models=1)[0]
    best_hyperparameters = tuner.get_best_hyperparameters()[0]

    return best_model, best_hyperparameters


if __name__ == "__main__":
    # Tune Feeler model
    feeler_hypermodel = FeelerHyperModel(input_shape=(X_train.shape[1],), num_classes=len(np.unique(y_train)))
    best_feeler_model, best_feeler_hyperparameters = tune_models(feeler_hypermodel, "Feeler")
    optimal_num_layers_feeler = best_feeler_hyperparameters.get('num_layers')
    print("Optimal number of layers for Feeler Model:", optimal_num_layers_feeler)

    # Tune RNN model
    rnn_hypermodel = RnnHyperModel(input_shape=(X_train.shape[1], 1), num_classes=len(np.unique(y_train)))
    best_rnn_model, best_rnn_hyperparameters = tune_models(rnn_hypermodel, "RnnModel")
    optimal_num_layers_rnn = best_rnn_hyperparameters.get('num_layers')
    print("Optimal number of layers for RNN Model:", optimal_num_layers_rnn)
