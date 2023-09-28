# Import your models, agents, preprocessors, environments, and other necessary components
from environments.db_environment import DatabaseEnvironment
from agents.miniAgents.nlp_agent import NLPAgent
from agents.miniAgents.rnn_agent import RNNModelAgent
from agents.miniAgents.fnn_agent import FNNAgent
from agents.combinedAgents.nlnn_agent import NLNNAgent
from agents.superAgent import SuperAgent
from preprocessing.nlp_data import load_and_preprocess_data as load_and_preprocess_nlp_data
from preprocessing.rnn_data import load_and_preprocess_data as load_and_preprocess_rnn_data
from preprocessing.fnn_data import load_and_preprocess_data as load_and_preprocess_fnn_data

# File Path
file_path = "datasets/js_dataset/javas.csv"

# Load and preprocess data
nlp_data = load_and_preprocess_nlp_data(file_path)
rnn_data = load_and_preprocess_rnn_data(file_path, 100)
fnn_data = load_and_preprocess_fnn_data(file_path)

# Initialize Models Configurations
nlp_model_config = {
    "vocab_size": 10000,
    "embedding_dim": 64,
    "input_length": 120,
    "num_classes": 3
}

rnn_model_config = {
    "input_shape": rnn_data['config']['input_shape'],
    "num_classes": rnn_data['config']['num_classes']
}

fnn_model_config = {
    "input_dim": fnn_data['config']['num_features'],
    "num_classes": fnn_data['config']['num_classes']
}

# Initialize Database Environment
db_environment = DatabaseEnvironment()

# Initialize Mini Agents
nlp_agent = NLPAgent(nlp_model_config, environment=db_environment, data_file_path=file_path)
rnn_agent = RNNModelAgent(rnn_model_config, environment=db_environment, data_file_path=file_path)
fnn_agent = FNNAgent(fnn_model_config, environment=db_environment, data_file_path=file_path)

# Initialize Combined Agents
nlnn_agent = NLNNAgent(nlp_model_config=nlp_model_config, rnn_model_config=rnn_model_config,
                       environment=db_environment, data_file_path=file_path)

big_agent_config = {
    "nlnn_agent": nlnn_agent,
    "fnn_agent": fnn_agent
}

state_size = 10  # Define appropriately
action_size = 5  # Define appropriately

ppo_agent_config = {
    "state_size": state_size,
    "action_size": action_size,
    "environment": db_environment,
    "data_file_path": file_path
}

# Initialize Super Agent
super_agent = SuperAgent(big_agent_config, ppo_agent_config, db_environment)

# Run the Super Agent
total_reward = super_agent.run_episode()

print("Total Reward for Episode:", total_reward)
