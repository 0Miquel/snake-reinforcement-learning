import gym
import gym_snake
import time
from AgentNN import AgentNN

path = "../models/LNN-46-0_9-1000-0_4-0_005-r1-s1.pth"
agent = AgentNN(path = path)

env = gym.make('Snake-16x16-v0')
observation = env.reset()
env.render()

while(True):
    state = agent.get_state(env)
    action = agent.get_action(state, env)

    observation, reward, done, info = env.step(action)
    env.render()

    if done:
        env.reset()
    time.sleep(0.05)
env.close()
