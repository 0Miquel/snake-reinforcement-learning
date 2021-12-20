import time
import gym
import QTable
from source.TabularMethod.Agent import Agent
import pickle
import numpy as np

import matplotlib.pyplot as plt
#from helper import plot

env = gym.make('Snake-16x16-v0')
#Snake-16x16-v0
#Breakout-v0
#MsPacman-v0
observation = env.reset()
#usePickle = True
usePickle = False
#savePickle = True
savePickle = False
mode = 'tabular'
# mode = 'QNN'
lastState = None
action = None
qTable = {}
reward = 0
plt.ion()
# Lists for plotting results
scoreList = []
meanScoreList = []
score = 0
meanScore = 0
numberOfGames = 1
totalScore = 0


if usePickle is False:
    agent = Agent('tabular', 0)
else: # if we pick the qTable, we don't want randomness
    agent = Agent('tabular', 0, epsilon=0.0)
if usePickle is True:
    file = open("qTable.txt", "rb")
    qTable = pickle.load(file)
    file.close()

env.render()
#for i in range(10000):
while True:
    right = True
    if mode == 'tabular':
        snakeHead = env.env.grid.snakes[0]._deque[-1]
        done, score = QTable.qLearning(agent, env, qTable, snakeHead, score)
    else:
        action = env.action_space.sample()

    if done:
        numberOfGames = numberOfGames + 1
        if usePickle is False:
            agent.decrease_epsilon()
        env.reset()

        #Plot results
        scoreList.append(score)
        totalScore = totalScore + score
        meanScore = totalScore / numberOfGames
        meanScoreList.append(meanScore)
        plt.plot(np.array(scoreList))
        plt.plot(meanScoreList)
        plt.title('Results of training')
        plt.xlabel('Number of Games')
        plt.ylabel('Score')
        plt.show(block=False)
        score = 0

    time.sleep(0.02)

plt.savefig('ResultsQlearningTabular2.png')
env.close()

if savePickle is True:
    file = open("qTable.txt", "wb")
    pickle.dump(qTable, file)
    file.close()