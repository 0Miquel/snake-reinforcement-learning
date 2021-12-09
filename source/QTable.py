import numpy as np
from Agent import Agent
import time
import gym

def qLearning(agent, env, qTable, lastState, lastAction, reward):
    learning_rate = 0.1  # usually used 0.1, it states the importance of the new information gained
    discount_factor = 0.9 # if close to 1 more importance to long term rewards, else only considers low term reward

    actualState = agent.get_state(env) # getting the actual state from the environment
    actualState = actualState.astype(str)
    actualState = ''.join(actualState) # convert actualState to string

    '''if actualState in qTable:
        actualRewards = qTable[actualState] #array of three qvalues, each for a diferent actin
    else:
        qTable[actualState] = [0,0,0]'''
    if actualState not in qTable: #add new state to the dictionary if it wasn't there
        qTable[actualState] = [0, 0, 0]

    lastState = qTable[lastState] #array of three qvalues, each for a different action

    # we calculate the new value of the qTable, using the reward from the previous action and updating it on the table
    qTable[lastState][lastAction] = qTable[lastState][lastAction] + \
            learning_rate * (reward + discount_factor*max(qTable[actualState]) - qTable[lastState])

    lastState = actualState # update new state

    newAction = agent.get_action(actualState, env) # get action for the new state

    return newAction, lastState
