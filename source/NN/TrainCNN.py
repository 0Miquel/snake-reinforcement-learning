import gym
import gym_snake
from AgentCNN import AgentNN
from plot import plot

path = "../models/model.pth"

agent = AgentNN(batch_size = 1000)
env = gym.make('Snake-16x16-v0')



score = 0
n_games = 1
plot_scores = []
plot_mean_scores = []
total_score = 0

state = env.reset()
env.render()
while(True):
    action = agent.get_action(state, env)

    previous_head = env.env.grid.snakes[0]._deque[0]

    next_state, reward, done, info = env.step(action)

    if reward == 1:
        score = score + 1
    #elif reward != -2:  # doesnt eat an apple
    #    reward = agent.get_reward(env.env.grid.snakes[0]._deque[0], previous_head, list(env.env.grid.apples._set)[0], done)

    #print(reward)
    env.render()
    agent.store_experience(state, action, reward, next_state, done)
    agent.short_train(state, action, reward, next_state, done)

    state = next_state
    if done:
        env.reset()
        agent.decrease_epsilon()
        agent.long_train(reward)
        agent.save_model(path, score)

        plot_scores.append(score)
        total_score += score
        mean_score = total_score / n_games
        plot_mean_scores.append(mean_score)
        plot(plot_scores, plot_mean_scores)

        n_games = n_games + 1
        score = 0

env.close()