import matplotlib.pyplot as plt
import gym
#import feelers
#from RNN import train_rnn, preprocessor
#import dqn  # Importing the dqn module
from nlp import train_nlp

#NLP
# Call the function to train the NLP model
nlp_model = train_nlp()

















# Preprocess data and get training, validation, and test sets
X_train, X_test, y_train, y_test, X_val, y_val = preprocessor()
print("|*********************************|\n"
      "|*********************************|\n"
      "|*********************************|\n"
      "|**********Beginning FNN**********|\n"
      "|*********************************|\n"
      "|*********************************|\n"
      "|*********************************|\n"
      )
# Train feelers and get top 5 feelers
feelers.train_feelers(X_train, y_train, X_val, y_val, feelers.num_feelers, feelers.learning_rate, feelers.scheduler,
                      feelers.alpha, feelers.beta, feelers.memory)

print("|*********************************|\n"
      "|*********************************|\n"
      "|*********************************|\n"
      "|**********Beginning RNN**********|\n"
      "|*********************************|\n"
      "|*********************************|\n"
      "|*********************************|\n"
      )

rnn_model = train_rnn(X_train, y_train, X_val, y_val)

print("|*********************************|\n"
      "|*********************************|\n"
      "|*********************************|\n"
      "|**********Beginning DQN**********|\n"
      "|*********************************|\n"
      "|*********************************|\n"
      "|*********************************|\n"
      )

# Initialize the environment and the DQN agent using the functions from dqn module
env, dqn_agent = dqn.initialize_environment_and_agent()

# Train the DQN agent using the function from dqn module
dqn.train_dqn_agent(env, dqn_agent)

print("DQN Training complete.")
