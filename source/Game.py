import time
import gym
import gym_snake
from Agent import Agent
from copy import copy

agent = Agent('lnn', batch_size=1000)

env = gym.make('Snake-16x16-v0')
#Snake-16x16-v0
#Breakout-v0
#MsPacman-v0
observation = env.reset()
n_games = 0
while(True):
    env.render()

    state = agent.get_state(env)
    action = agent.get_action(state, env, n_games)
    # 0 = straight
    # 1 = right
    # 2 = left


    previous_head = env.env.grid.snakes[0]._deque[0]

    observation, reward, done, info = env.step(action) # take a random action
    print(action)
    #if reward != 10: #doesnt eat an apple
    #   reward = agent.get_reward(env.env.grid.snakes[0]._deque[0], previous_head, list(env.env.grid.apples._set)[0], done)

    next_state = agent.get_state(env)
    agent.store_experience(state, action, reward, next_state, done)
    agent.short_train(state, action, reward, next_state, done)
    if done:
        env.reset()
        agent.decrease_epsilon()
        agent.long_train()
        n_games = n_games + 1

    time.sleep(0.05)

env.close()