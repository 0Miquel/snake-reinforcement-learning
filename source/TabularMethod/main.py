import time
import gym
import QTable
from source.TabularMethod.Agent import Agent
import pickle
import numpy as np
import gym_snake

import matplotlib.pyplot as plt

def loadQTable(qTable, mode):
    agent = Agent('tabular', 0, epsilon=0.0)
    if mode == 'qlearning':
        file = open("qTableReward1-0_9-1-0_1-0_9.txt", "rb")
    else:
        file = open("qTableSarsa.txt", "rb")
    qTable = pickle.load(file)
    file.close()
    return agent, qTable


def storeQTable(qTable, mode):
    if mode == 'qlearning':
        file = open("qTableReward1-0_9-1-0_1-0_9.txt", "wb")
    else:
        file = open("qTableSarsa.txt", "wb")
    pickle.dump(qTable, file)
    file.close()


def plot(score, totalScore, scoreList, meanScoreList, numberOfGames):
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


if __name__ == "__main__":
    qTable = {}
    agent = Agent('tabular', gamma = 0.9, epsilon = 1, rewardType = 2)

    mode = 'sarsa'
    action = None
    actualState = None

    # Plot
    plt.ion()
    scoreList = []
    meanScoreList = []
    score = 0
    numberOfGames = 1
    totalScore = 0

    load = False
    store = True

    env = gym.make('Snake-16x16-v0')

    observation = env.reset()

    env.render()

    if load is True:
        agent, qTable = loadQTable(qTable, mode)

    #for i in range(10000):
    while numberOfGames < 400:
    #while True:
        snakeHead = env.env.grid.snakes[0]._deque[-1]
        if mode == 'qlearning':
            done, score = QTable.qLearning(agent, env, qTable, snakeHead, score)
        else:
            done, score, action, actualState = QTable.sarsa(action, agent, env, qTable, snakeHead, score, actualState)

        if done:
            numberOfGames = numberOfGames + 1
            if load is False:
                agent.decrease_epsilon()
                print("Epsilon = ", agent.epsilon)
            env.reset()
            totalScore = totalScore + score
            plot(score, totalScore, scoreList, meanScoreList, numberOfGames)

            score = 0

        #time.sleep(0.02)

    if mode == 'qlearning':
        plt.savefig('ResultsQlearningTabularReward2-0_9-1-0_1-0_9-400Games_3.png')
    else:
        plt.savefig('ResultsSarsaTabularReward1-0_9-1-0_1-0_9-400Games.png')
    env.close()

    if store is True:
        storeQTable(qTable, mode)
