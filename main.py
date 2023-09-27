# Import your models, agents, preprocessors, and other necessary components
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
rnn_data = load_and_preprocess_rnn_data(file_path)
fnn_data = load_and_preprocess_fnn_data(file_path)

# Initialize Models
nlp_model = NLPModel(nlp_data['config'])
rnn_model = RNNModel(rnn_data['config'])
fnn_model = FNNModel(fnn_data['config'])

# Add CustomRewardCallback to your models (assuming validation_data is available)
reward_callback = reward(validation_data=fnn_data['validation_data'])
nlp_model.model.callbacks.append(reward_callback)
rnn_model.model.callbacks.append(reward_callback)
fnn_model.model.callbacks.append(reward_callback)

# Initialize Agents
nlp_agent = NLPAgent(nlp_model, environment=None, data_file_path=file_path)
rnn_agent = RNNModelAgent(rnn_model, environment=None, data_file_path=file_path)
fnn_agent = FNNAgent(fnn_model, environment=None, data_file_path=file_path)

# Initialize CommunicationManager
communication_manager = communications([nlp_agent, rnn_agent, fnn_agent])

# Start communication and learning process
communication_manager.start_communication_and_learning()
