import gym
import gym_snake
from AgentNN import AgentNN

path = "../models/model.pth"

agent = AgentNN(batch_size = 1000)
env = gym.make('Snake-16x16-v0')

observation = env.reset()
env.render()
score = 0
while(True):
    state = agent.get_state(env)
    action = agent.get_action(state, env)

    previous_head = env.env.grid.snakes[0]._deque[0]

    observation, reward, done, info = env.step(action)

    if reward != 10:  # doesnt eat an apple
        reward = agent.get_reward(env.env.grid.snakes[0]._deque[0], previous_head, list(env.env.grid.apples._set)[0], done)
    else:
        score = score + 1
    #print(reward)
    env.render()
    next_state = agent.get_state(env)
    agent.store_experience(state, action, reward, next_state, done)
    agent.short_train(state, action, reward, next_state, done)

    if done:
        env.reset()
        agent.decrease_epsilon()
        agent.long_train()
        agent.save_model(path, score)
        score = 0

env.close()