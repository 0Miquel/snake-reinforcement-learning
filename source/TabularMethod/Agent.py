import numpy as np
import random
import math
from collections import deque

class Agent:
    def __init__(self, gamma = 0.9, epsilon = 0.4, rewardType = 0, stateType=0, learningRate=0.1):
        self.epsilon = epsilon #1 = random move 100%, 0 = no random moves
        self.gamma = gamma
        self.rewardType = rewardType
        self.stateType = stateType
        self.learningRate = learningRate

    def get_state(self, env):
        if self.stateType == 0:
            apple = list(env.env.grid.apples._set)[0]  # columna, fila
            head = env.env.grid.snakes[0]._deque[-1]
            queue = list(env.env.grid.snakes[0]._deque)[:-1]

            # apple's direction from snake's head
            apple_left = apple[0] < head[0]  # left
            apple_right = apple[0] > head[0]  # right
            apple_up = apple[1] < head[1]  # up
            apple_down = apple[1] > head[1]  # down

            # snake's direction
            dir_up = env.env.grid.snakes[0]._direction == 0  # up
            dir_right = env.env.grid.snakes[0]._direction == 1  # right
            dir_down = env.env.grid.snakes[0]._direction == 2  # down
            dir_left = env.env.grid.snakes[0]._direction == 3  # left

            # danger's direction
            def colission(position, queue):
                return position in queue or (position[0] > 15 or position[0] < 0 or position[1] > 15 or position[1] < 0)

            danger_left = (dir_up and colission((head[0] - 1, head[1]), queue) or
                           dir_down and colission((head[0] + 1, head[1]), queue) or
                           dir_left and colission((head[0], head[1] + 1), queue) or
                           dir_right and colission((head[0], head[1] - 1), queue))
            danger_right = (dir_up and colission((head[0] + 1, head[1]), queue) or
                            dir_down and colission((head[0] - 1, head[1]), queue) or
                            dir_left and colission((head[0], head[1] - 1), queue) or
                            dir_right and colission((head[0], head[1] + 1), queue))
            danger_straight = (dir_up and colission((head[0], head[1] - 1), queue) or
                               dir_down and colission((head[0], head[1] + 1), queue) or
                               dir_left and colission((head[0] - 1, head[1]), queue) or
                               dir_right and colission((head[0] + 1, head[1]), queue))

            return np.array([apple_left, apple_right, apple_up, apple_down, dir_left, dir_right,
                    dir_up, dir_down, danger_left, danger_right, danger_straight], dtype=int)
        else:
            apple = list(env.env.grid.apples._set)[0]  # columna, fila
            apple_x = apple[0]
            apple_y = apple[1]
            head = env.env.grid.snakes[0]._deque[-1]
            head_x = head[0]
            head_y = head[1]
            queue = list(env.env.grid.snakes[0]._deque)[:-1]

            # snake's direction
            dir_up = env.env.grid.snakes[0]._direction == 0  # up
            dir_right = env.env.grid.snakes[0]._direction == 1  # right
            dir_down = env.env.grid.snakes[0]._direction == 2  # down
            dir_left = env.env.grid.snakes[0]._direction == 3  # left

            # danger's direction
            def colission(position, queue):
                return position in queue or (position[0] > 15 or position[0] < 0 or position[1] > 15 or position[1] < 0)

            danger_left = (dir_up and colission((head[0] - 1, head[1]), queue) or
                           dir_down and colission((head[0] + 1, head[1]), queue) or
                           dir_left and colission((head[0], head[1] + 1), queue) or
                           dir_right and colission((head[0], head[1] - 1), queue))
            danger_right = (dir_up and colission((head[0] + 1, head[1]), queue) or
                            dir_down and colission((head[0] - 1, head[1]), queue) or
                            dir_left and colission((head[0], head[1] - 1), queue) or
                            dir_right and colission((head[0], head[1] + 1), queue))
            danger_straight = (dir_up and colission((head[0], head[1] - 1), queue) or
                               dir_down and colission((head[0], head[1] + 1), queue) or
                               dir_left and colission((head[0] - 1, head[1]), queue) or
                               dir_right and colission((head[0] + 1, head[1]), queue))
            return np.array([apple_x, apple_y, head_x, head_y, dir_left, dir_right,
                    dir_up, dir_down, danger_left, danger_right, danger_straight], dtype=int)


    def get_action(self, state, env, qTable=None):
        random_move = np.random.choice([False, True], p=[1 - self.epsilon, self.epsilon])
        if random_move: #random move
            move = env.action_space.sample()
        else: #model move
            pred = qTable[state]
            move = pred.index(max(pred))
        return move

    def decrease_epsilon(self):
        if self.epsilon > 0:
            self.epsilon = self.epsilon - 0.01
            if self.epsilon < 0:
                self.epsilon = 0

    def getReward(self, env, action, done, snakeHeadBefore, appleEaten):
        reward = 0
        if self.rewardType == 0: # if dies -100 if eats apple + 100, if gets closer to apple + 10, if gets further - 10
            if done is True:
                reward = reward - 100
            else:
                if appleEaten is True:
                    reward = reward + 100
                else:
                    if env.env.grid.done_apple is True:
                        reward = reward + 100
                    else:
                        snakeHeadActual = env.env.grid.snakes[0]._deque[-1]
                        applePosition = list(env.env.grid.apples._set)[0]
                        '''# Euclidean disance
                        distBefore = math.sqrt(pow((snakeHeadBefore[0] - applePosition[0]), 2) + pow(
                            (snakeHeadBefore[1] - applePosition[1]), 2))
                        distActual = math.sqrt(pow((snakeHeadActual[0] - applePosition[0]), 2) + pow(
                            (snakeHeadActual[1] - applePosition[1]), 2))'''
                        # Manhattan distance
                        distActual = sum(abs(val1 - val2) for val1, val2 in zip(applePosition, snakeHeadActual))
                        distBefore = sum(abs(val1 - val2) for val1, val2 in zip(applePosition, snakeHeadBefore))

                        if distActual < distBefore:
                            reward = reward + 10
                        else:
                            reward = reward - 10

        elif self.rewardType == 1: # if dies -100 if eats apple + 100
            if done is True:
                reward = reward - 100
            else:
                if appleEaten is True:
                    reward = reward + 100

        elif self.rewardType == 2: # if dies -100 if eats apple + 100, if survives + 5
            if done is True:
                reward = reward - 100
            else:
                if appleEaten is True:
                    reward = reward + 100
                else:
                    reward = 5
        elif self.rewardType == 2: # if dies -100 if eats apple + 100, if survives but doesn't eat - 5
            if done is True:
                reward = reward - 1000
            else:
                if appleEaten is True:
                    reward = reward + 1000
                else:
                    reward = -1
        return reward