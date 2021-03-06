import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import math

from QNN import Q_LNN, Q_CNN
from collections import deque

MAX_MEMORY = 100000

class AgentNN:
    def __init__(self, path = None, gamma = 0.9, batch_size = 1000, epsilon = 1, decrease_rate = 0.01, hidden_layers=[256], dropout=0, reward_type = 0, state_type = 0):
        self.epsilon = epsilon #1 = random move 100%, 0 = no random moves
        self.decrease_rate = decrease_rate
        self.gamma = gamma

        self.reward_type = reward_type
        self.state_type = state_type

        self.max_score = -1

        self.batch_size = batch_size
        self.short_memory = []
        self.memory = deque(maxlen=MAX_MEMORY)

        self.model = Q_LNN(hidden_size=hidden_layers, dropout=dropout)
        if path is not None:
            self.epsilon = 0
            self.model.load_state_dict(torch.load(path))
            self.model.eval()

        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.loss_fn = nn.MSELoss()

    def get_state(self, env):
        def colission(position, queue):
            return position in queue or (position[0] > 15 or position[0] < 0 or position[1] > 15 or position[1] < 0)

        apple = list(env.env.grid.apples._set)[0]  # columna, fila
        head = env.env.grid.snakes[0]._deque[-1]
        queue = list(env.env.grid.snakes[0]._deque)[:-1]

        if self.state_type == 0:
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

            state = np.array([apple_left, apple_right, apple_up, apple_down, dir_left, dir_right,
                      dir_up, dir_down, danger_left, danger_right, danger_straight], dtype=int)

        elif self.state_type == 1:
            apple_x = apple[0]/15
            apple_y = apple[1]/15
            head_x = head[0]/15
            head_y = head[1]/15

            # snake's direction
            dir_up = env.env.grid.snakes[0]._direction == 0  # up
            dir_right = env.env.grid.snakes[0]._direction == 1  # right
            dir_down = env.env.grid.snakes[0]._direction == 2  # down
            dir_left = env.env.grid.snakes[0]._direction == 3  # left

            # danger's direction
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

            state = np.array([apple_x, apple_y, head_x, head_y, dir_left, dir_right,
                              dir_up, dir_down, danger_left, danger_right, danger_straight], dtype=int)

        return state


    def get_action(self, state, env, qTable=None):
        random_move = np.random.choice([False, True], p=[1 - self.epsilon, self.epsilon])
        if random_move: #random move
            move = env.action_space.sample()
        else: #model move
            pred = self.model(torch.tensor(state, dtype=torch.float))
            move = torch.argmax(pred).item()

        return move

    def get_reward(self, head, previous_head, apple, gameOver, reward):
        if gameOver:
            new_reward = -1
        elif reward == 1:
            new_reward = 1
        else:
            if self.reward_type == 1:
                new_reward = 0
            elif self.reward_type == 0:
                # euclidian distance
                # previous_distance = math.sqrt(((previous_head[0] - apple[0]) ** 2) + ((previous_head[1] - apple[1]) ** 2))
                # distance = math.sqrt(((head[0] - apple[0]) ** 2) + ((head[1] - apple[1]) ** 2))
                # manhattan distance
                distance = sum(abs(val1 - val2) for val1, val2 in zip(apple, head))
                previous_distance = sum(abs(val1 - val2) for val1, val2 in zip(apple, previous_head))
                if distance > previous_distance:
                    new_reward = -0.1
                else:
                    new_reward = 0.1
                    # reward = 0.6 / distance
            elif self.reward_type == 2:
                new_reward = 0.1
            elif self.reward_type == 3:
                new_reward = -0.1

        return new_reward

    def decrease_epsilon(self):
        if self.epsilon > 0:
            self.epsilon = self.epsilon - self.decrease_rate
            self.epsilon = round(self.epsilon, 3)
            #print(self.epsilon)

    def save_model(self, path, score):
        if score > self.max_score:
            self.max_score = score
            torch.save(self.model.state_dict(), path)
            #print(f'New model saved with score: {score}')

    def store_experience(self, state, action, reward, next_state, gameOver):
        self.short_memory.append((state, action, reward, next_state, gameOver))

    def update_memory(self, reward):
        if reward != -2:
            self.memory.extendleft(self.short_memory)
        else:
            print("Timeout")

        self.short_memory = []

    #replay experiences
    def replay_experiences(self):
        if len(self.memory) > self.batch_size:
            sample = random.sample(self.memory, self.batch_size)
        else:
            sample = self.memory

        if list(sample) != []:
            state, action, reward, next_state, gameOver = zip(*sample)
            self.train(state, action, reward, next_state, gameOver)

    def train(self, state, action, reward, next_state, gameOver):
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)
        if len(state.shape) == 1:
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            gameOver = (gameOver, )

        pred = self.model(state)
        target = pred.clone()
        for i in range(len(gameOver)):
            Q_new = reward[i] + self.gamma * torch.max(self.model(next_state[i])) * (1 - gameOver[i])
            target[i][action[i].item()] = Q_new

        self.optimizer.zero_grad()
        loss = self.loss_fn(target, pred)
        loss.backward()
        self.optimizer.step()