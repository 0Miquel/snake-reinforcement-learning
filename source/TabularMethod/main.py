import time
import gym
import QTable
from source.TabularMethod.Agent import Agent
import pickle
import numpy as np
import gym_snake


def loadQTable(qTable, mode, gamma, epsilon, reward, state):
    agent = Agent(gamma=gamma, epsilon=epsilon, rewardType=reward, stateType=state)
    if mode == 'qlearning':
        file = open("qTableQlearning.txt", "rb")
    else:
        file = open("qTableSarsa.txt", "rb")
    qTable = pickle.load(file)
    file.close()
    return agent, qTable


def storeQTable(qTable, mode):
    if mode == 'qlearning':
        file = open("qTableReward.txt", "wb")
    else:
        file = open("qTableSarsa.txt", "wb")
    pickle.dump(qTable, file)
    file.close()

if __name__ == "__main__":
    env = gym.make('Snake-16x16-v0')
    observation = env.reset()
    env.render()

    qTable = {}
    action = None
    actualState = None

    score = 0
    numberOfGames = 1
    totalScore = 0
    maxScore = 0

    #Choose algorithm and if you want to load a dictionary or train a new one, you can also store it
    mode = 'qlearning' #qlearning or sarsa
    load = True
    store = False

    if load is True:
        #you pass qTable, mode, gamma, espilon, rewardType, stateType
        agent, qTable = loadQTable(qTable, mode, 0.5, 0, 0, 0)
    else:
        agent = Agent(gamma=0.9, epsilon=0.66, rewardType=0, stateType=0, learningRate=0.1)  # rewardType can be value from 0 to 3

    while numberOfGames < 1000:
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
            score = 0

        time.sleep(0.01)

    if store is True:
        storeQTable(qTable, mode)
