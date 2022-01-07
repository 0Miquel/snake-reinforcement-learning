import gym
import gym_snake
from AgentNN import AgentNN
import matplotlib.pyplot as plt
import numpy as np

total_games = 300

results = []

#Experiment 1
"""agents = [
AgentNN(batch_size = 1000, hidden_layers=[256], dropout=0, epsilon=1, reward_type=1),
AgentNN(batch_size = 1000, hidden_layers=[64,64], dropout=0, epsilon=1, reward_type=1),
AgentNN(batch_size=1000, hidden_layers=[100, 100, 100], dropout=0, epsilon=1, reward_type=1)
]"""
#Experiment 2
"""agents = [
AgentNN(batch_size = 1000, hidden_layers=[256], dropout=0, epsilon=1, reward_type=1),
AgentNN(batch_size = 500, hidden_layers=[256], dropout=0, epsilon=1, reward_type=1),
AgentNN(batch_size = 100, hidden_layers=[256], dropout=0, epsilon=1, reward_type=1),
AgentNN(batch_size = 0, hidden_layers=[256], dropout=0, epsilon=1, reward_type=1)
]"""
#Experiment 3
"""agents = [
AgentNN(batch_size = 1000, hidden_layers=[256], dropout=0, epsilon=1, reward_type=1),
AgentNN(batch_size = 1000, hidden_layers=[256], dropout=0.3, epsilon=1, reward_type=1),
AgentNN(batch_size = 1000, hidden_layers=[256], dropout=0.5, epsilon=1, reward_type=1)
]"""
#Experiment 4
agents = [
AgentNN(batch_size = 1000, hidden_layers=[256], dropout=0, epsilon=1, reward_type=1),
AgentNN(batch_size = 1000, hidden_layers=[256], dropout=0, epsilon=0.66, reward_type=1),
AgentNN(batch_size = 1000, hidden_layers=[256], dropout=0, epsilon=0.33, reward_type=1),
AgentNN(batch_size = 1000, hidden_layers=[256], dropout=0, epsilon=1, decrease_rate = 0.02 ,reward_type=1)
]
#Experiment 5
"""agents = [
AgentNN(batch_size = 1000, hidden_layers=[256], dropout=0, epsilon=1, reward_type=1),
AgentNN(batch_size = 1000, hidden_layers=[256], dropout=0, epsilon=1, reward_type=0),
AgentNN(batch_size = 1000, hidden_layers=[256], dropout=0, epsilon=1, reward_type=2)
]"""

for i in range(len(agents)):
    agent = agents[i]
    env = gym.make('Snake-16x16-v0')

    observation = env.reset()

    score = 0
    n_games = 1
    mean_scores = []
    total_score = 0
    while(n_games <= total_games):
        state = agent.get_state(env)
        action = agent.get_action(state, env)

        previous_head = env.env.grid.snakes[0]._deque[0]

        observation, reward, done, info = env.step(action)

        if reward == 1:
            score = score + 1
            agent.update_memory(reward)
        elif reward != -2:  # doesnt eat an apple
            reward = agent.get_reward(env.env.grid.snakes[0]._deque[0], previous_head, list(env.env.grid.apples._set)[0], done, reward)

        next_state = agent.get_state(env)
        agent.store_experience(state, action, reward, next_state, done)
        agent.short_train(state, action, reward, next_state, done)

        if done:
            env.reset()
            agent.decrease_epsilon()
            agent.update_memory(reward)
            agent.long_train()

            total_score += score
            mean_scores.append(total_score / n_games)
            n_games = n_games + 1
            score = 0

    env.close()
    results.append(mean_scores)


plt.plot(np.arange(total_games), results[0], label="Epsilon 1")
plt.plot(np.arange(total_games), results[1], label="Epsilon 0.66")
plt.plot(np.arange(total_games), results[2], label="Epsilon 0.33")
plt.plot(np.arange(total_games), results[3], label="Epsilon 1-0.02")
plt.title("Epsilon comparison")
plt.xlabel('Number of Games')
plt.ylabel('Mean Score')
plt.legend()
plt.show()
#plt.savefig()