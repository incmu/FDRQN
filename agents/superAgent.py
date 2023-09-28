from agents.bigAgents.bigAgent import BigAgent
from agents.miniAgents.ppo_agent import PPOAgent
from environments.db_environment import DatabaseEnvironment

class SuperAgent:
    def __init__(self, big_agent_config, ppo_agent_config, environment):
        self.big_agent = BigAgent(**big_agent_config)
        self.ppo_agent = PPOAgent(**ppo_agent_config)
        self.environment = environment

    def act(self, state):
        # BigAgent decides actions based on the state
        big_agent_action, big_agent_prob = self.big_agent.act_with_prob(state)

        # PPOAgent decides actions based on the state
        ppo_agent_action, ppo_agent_prob = self.ppo_agent.act_with_prob(state)

        # Combine, compare or evaluate both actions and decide the final action
        # Let the PPO agent decide based on the probability comparison
        final_action = self.evaluate_actions(big_agent_action, big_agent_prob, ppo_agent_action, ppo_agent_prob)
        return final_action

    def evaluate_actions(self, big_agent_action, big_agent_prob, ppo_agent_action, ppo_agent_prob):
        # Compare the probabilities and let PPOAgent decide the final action
        if big_agent_prob > ppo_agent_prob:
            return big_agent_action
        else:
            return ppo_agent_action

    def learn(self, state, action, reward, next_state, done):
        # Let both BigAgent and PPOAgent learn from the experience
        self.big_agent.learn(state, action, reward, next_state)
        self.ppo_agent.learn([state], [action], [reward], [next_state], [done])

    def run_episode(self):
        state = self.environment.reset()
        done = False
        total_reward = 0
        while not done:
            action = self.act(state)
            next_state, reward, done = self.environment.step(action)
            self.learn(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
        return total_reward

