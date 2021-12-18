import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import math

from QNN import Q_LNN, Q_CNN
from collections import deque

class Agent:
    def __init__(self, model_type, gamma = 0.9, batch_size = 1, epsilon = 0.4):
        self.epsilon = epsilon #1 = random move 100%, 0 = no random moves
        self.gamma = gamma

        self.batch_size = batch_size
        self.model_type = model_type
        if self.model_type == 'lnn':
            self.model = Q_LNN()
            self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
            self.loss_fn = nn.MSELoss()
        elif self.model_type == 'cnn':
            self.model = Q_CNN()

        self.memory = deque(maxlen=self.batch_size)

    def get_state(self, env):
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


    def get_action(self, state, env, n_games, qTable=None):
        """random_move = np.random.choice([False, True], p=[1 - self.epsilon, self.epsilon])
        if random_move: #random move
            move = env.action_space.sample()
        else: #model move
            if self.model_type == 'lnn' or self.model_type == 'cnn':
                pred = self.model(torch.tensor(state, dtype=torch.float))
                move = torch.argmax(pred).item()
            elif self.model_type == 'tabular':
                pred = qTable[state]
                move = pred.index(max(pred))"""

        self.epsilon = 80 - n_games
        if random.randint(0, 200) < self.epsilon:
            move = env.action_space.sample()
        else:
            if self.model_type == 'lnn' or self.model_type == 'cnn':
                pred = self.model(torch.tensor(state, dtype=torch.float))
                move = torch.argmax(pred).item()

        return move


    def get_reward(self, head, previous_head, apple, gameOver):
        if gameOver:
            reward = -100
        else:
            #euclidian distance
            #previous_distance = math.sqrt(((previous_head[0] - apple[0]) ** 2) + ((previous_head[1] - apple[1]) ** 2))
            #distance = math.sqrt(((head[0] - apple[0]) ** 2) + ((head[1] - apple[1]) ** 2))
            #manhattan distance
            distance = sum(abs(val1 - val2) for val1, val2 in zip(apple, head))
            previous_distance = sum(abs(val1 - val2) for val1, val2 in zip(apple, previous_head))
            if distance > previous_distance:
                reward = -1
            else:
                reward = 1

        return reward

    def decrease_epsilon(self):
        if self.epsilon > 0.01:
            self.epsilon = self.epsilon - 0.001

    def store_experience(self, state, action, reward, next_state, gameOver):
        self.memory.append((state, action, reward, next_state, gameOver))

    #replay experience
    def long_train(self):
        sample = random.sample(self.memory, len(self.memory))
        state, action, reward, next_state, gameOver = zip(*sample)
        self.train_nn(state, action, reward, next_state, gameOver)

    def short_train(self, state, action, reward, next_state, gameOver):
        self.train_nn(state, action, reward, next_state, gameOver)

    def train_nn(self, state, action, reward, next_state, gameOver):
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)
        if len(state.shape) == 1:
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            gameOver = (gameOver,)
        ###########################################################################
        #model.train()
        pred = self.model(state)
        target = pred.clone()
        for idx in range(len(gameOver)):
            Q_new = reward[idx]
            if not gameOver[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))

            target[idx][action[idx].item()] = Q_new

        ##########################################
        self.optimizer.zero_grad()
        loss = self.loss_fn(target, pred)
        loss.backward()
        self.optimizer.step()