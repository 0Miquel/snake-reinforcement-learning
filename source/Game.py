import time
import gym
import gym_snake
from Agent import Agent

agent = Agent('lnn', batch_size=100)

env = gym.make('Snake-16x16-v0')
#Snake-16x16-v0
#Breakout-v0
#MsPacman-v0
observation = env.reset()

while(True):
    env.render()

    state = agent.get_state(env)
    action = agent.get_action(state, env)
    observation, reward, done, info = env.step(action) # take a random action
    next_state = agent.get_state(env)
    agent.store_experience(state, action, reward, next_state, done)



    if done:
        env.reset()
        agent.decrease_epsilon()
        agent.train()

    time.sleep(0.05)

env.close()