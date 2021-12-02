import time
import gym
import gym_snake
from Agent import Agent

agent = Agent(0,0)

env = gym.make('Snake-16x16-v0')
#Snake-16x16-v0
#Breakout-v0
#MsPacman-v0
observation = env.reset()

for i in range(1000):
    env.render()
    right = True
    observation, reward, done, info = env.step(env.action_space.sample()) # take a random action
    if done:
        env.reset()
    state = agent.get_state(env)
    print(state)
    time.sleep(0.05)
    #print(i)
env.close()
