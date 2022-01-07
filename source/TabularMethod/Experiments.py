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


def plot(score, totalScore, scoreList, meanScoreList, numberOfGames, value):
    '''scoreList.append(score)
    totalScore = totalScore + score
    meanScore = totalScore / numberOfGames
    meanScoreList.append(meanScore)'''
    #plt.plot(np.array(scoreList))
    plt.plot(meanScoreList, label = "Epsilon " + str(value))
    plt.title('Results different epsilon values Sarsa')
    plt.legend()
    plt.xlabel('Number of Games')
    plt.ylabel('Mean score')
    plt.show(block=False)


if __name__ == "__main__":
    qTable = {}

    #agent = Agent('tabular', gamma = 0.9, epsilon = 1, rewardType = 2)

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
    store = False

    env = gym.make('Snake-16x16-v0')

    observation = env.reset()

    env.render()

    '''if load is True:
        agent, qTable = loadQTable(qTable, mode)'''
    rewardType = [0,1,2]
    epsilonValues = [1, 0.66, 0.33]

    for i in range(len(epsilonValues)):
        qTable = {}
        action = None
        actualState = None
        agent = Agent('tabular', gamma=0.9, epsilon=epsilonValues[i], rewardType=0, stateType=0)
        scoreList = []
        meanScoreList = []
        score = 0
        numberOfGames = 1
        totalScore = 0
        maxScore = 0
        while numberOfGames < 400:
        #while True:
            snakeHead = env.env.grid.snakes[0]._deque[-1]
            if mode == 'qlearning':
                done, score = QTable.qLearning(agent, env, qTable, snakeHead, score)
            else:
                done, score, action, actualState = QTable.sarsa(action, agent, env, qTable, snakeHead, score, actualState)

            if score > maxScore:
                maxScore = score

            if done:
                numberOfGames = numberOfGames + 1
                if load is False:
                    agent.decrease_epsilon()
                    print("Epsilon = ", agent.epsilon)
                env.reset()
                totalScore = totalScore + score
                meanScore = totalScore / numberOfGames
                print("Max_score =", maxScore, "score =", score, ", MeanScore=", meanScore, ", TotalScore =", totalScore, ", NumberGames =", numberOfGames)
                meanScoreList.append(meanScore)
                score = 0

            #time.sleep(0.02)
        plot(score, totalScore, scoreList, meanScoreList, numberOfGames, epsilonValues[i])

    #plt.legend()
    plt.savefig('ResultsSarsaEpsilon-x-1-0_1-0_9_400Games-Reward0.png') #primer valor és reward o epsilon
    #plt.savefig('ResultsQlearningStates-0.66-1-0_1-0_9_400Games-Reward1.png')  # primer valor és reward o epsilon

    '''
    if mode == 'qlearning':
        plt.savefig('ResultsQlearningTabularReward2-0_9-1-0_1-0_9-400Games_3.png')
    else:
        plt.savefig('ResultsSarsaTabularReward1-0_9-1-0_1-0_9-400Games.png')
    env.close()'''
    '''
    if store is True:
        storeQTable(qTable, mode)'''
