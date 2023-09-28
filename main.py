# Import your models, agents, preprocessors, and other necessary components
from models.nlp import nlp_model
from models.nlp.nlp_model import NLPModel
from models.rnn.rnn_model import RNNModel
from models.fnn.fnn_model import FNNModel
from agents.nlp_agent import NLPAgent
from agents.rnn_agent import RNNModelAgent
from agents.fnn_agent import FNNAgent
from communication import communications
from preprocessing.nlp_data import load_and_preprocess_data as load_and_preprocess_nlp_data
from preprocessing.rnn_data import load_and_preprocess_data as load_and_preprocess_rnn_data
from preprocessing.fnn_data import load_and_preprocess_data as load_and_preprocess_fnn_data
from environment import reward

# File Path
file_path = "datasets/js_dataset/javas.csv"

# Load and preprocess data
nlp_data = load_and_preprocess_nlp_data(file_path)
rnn_data = load_and_preprocess_rnn_data(file_path,100)
fnn_data = load_and_preprocess_fnn_data(file_path)

# Initialize Models
# Define the configuration dictionary
nlp_model_config = {
    "vocab_size": 10000,
    "embedding_dim": 64,
    "input_length": 120,
    "num_classes": 3
}

# Instantiate the NLPModel with the configuration dictionary
nlp_model_instance = NLPModel(model_config=nlp_model_config)
rnn_model_config = {
    "input_shape": rnn_data['config']['input_shape'],
    "num_classes": rnn_data['config']['num_classes']
}
fnn_model = {
    "input_dim" : fnn_data['config']['num_features'],
    "num_classes" : fnn_data['config']['num_classes']
}


# Add CustomRewardCallback to your models (assuming validation_data is available)
#reward_callback = reward(validation_data=fnn_data['validation_data'])
#nlp_model.model.callbacks.append(reward_callback)
#rnn_model.model.callbacks.append(reward_callback)
#fnn_model.model.callbacks.append(reward_callback)

# Initialize Agents
nlp_agent = NLPAgent(nlp_model_config, environment=None, data_file_path=file_path)
rnn_agent = RNNModelAgent(rnn_model_config, environment=None, data_file_path=file_path)
fnn_agent = FNNAgent(fnn_model, environment=None, data_file_path=file_path)

