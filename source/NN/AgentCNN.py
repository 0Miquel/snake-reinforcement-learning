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
    def __init__(self, path = None, gamma = 0.9, batch_size = 1000, epsilon = 1, decrease_rate = 0.01, reward_type = 1):
        self.epsilon = epsilon #1 = random move 100%, 0 = no random moves
        self.decrease_rate = decrease_rate
        self.gamma = gamma

        self.max_score = -1
        self.reward_type = reward_type

        self.batch_size = batch_size

        self.short_memory = []
        self.memory = deque(maxlen=MAX_MEMORY)
        self.good_memory = deque(maxlen=MAX_MEMORY)
        self.bad_memory = deque(maxlen=MAX_MEMORY)

        self.model = Q_CNN()
        if path is not None:
            self.epsilon = 0
            self.model.load_state_dict(torch.load(path))
            self.model.eval()

        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0001)
        self.loss_fn = nn.MSELoss()


    def get_action(self, state, env, qTable=None):
        random_move = np.random.choice([False, True], p=[1 - self.epsilon, self.epsilon])
        if random_move: #random move
            move = env.action_space.sample()
        else: #model move
            state = torch.tensor(state, dtype=torch.float)#.permute(2, 0, 1)
            pred = self.model(state)
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

    #replay experience
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
        if len(state.shape) == 2:
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            gameOver = (gameOver, )
        #state = state.permute(0, 3, 1, 2)
        #next_state = next_state.permute(0, 3, 1, 2)

        pred = self.model(state)
        target = pred.clone()
        for i in range(len(gameOver)):
            Q_new = reward[i] + self.gamma * torch.max(self.model(next_state[i])) * (1 - gameOver[i])
            target[i][action[i].item()] = Q_new

        self.optimizer.zero_grad()
        loss = self.loss_fn(target, pred)
        loss.backward()
        self.optimizer.step()

