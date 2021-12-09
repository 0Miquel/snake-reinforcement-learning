import time
import gym
import QTable
import gym_snake
from Agent import Agent

agent = Agent('tabular', 0)
env = gym.make('Snake-16x16-v0')
#Snake-16x16-v0
#Breakout-v0
#MsPacman-v0
observation = env.reset()

mode = 'tabular'
# mode = 'QNN'
lastState = None
action = None
qTable = {}
reward = 0
for i in range(10000):
    env.render()
    right = True
    if mode == 'tabular':
        action, lastState = QTable.qLearning(agent, env, qTable, lastState, action, reward)
    else:
        action = env.action_space.sample()
    observation, reward, done, info = env.step(action) # take a random action
    if done:
        agent.decrease_epsilon()
        env.reset()
    #state = agent.get_state(env)
    #print(state)
    time.sleep(0.05)
    #print(i)
env.close()
