import numpy as np
import random
import torch
from QNN import Q_LNN, Q_CNN

class Agent:
    def __init__(self, model_type, gamma, batch_size = 1, epsilon = 1):
        self.epsilon = epsilon #1 = random move 100%, 0 = no random moves
        self.gamma = gamma

        self.batch_size = batch_size
        self.model_type = model_type
        if self.model_type == 'lnn':
            self.model = Q_LNN()
        elif self.mode_type == 'cnn':
            self.model = Q_CNN()


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


    def get_action(self, state, env):
        random_move = np.random.choice([False, True], p=[1 - self.epsilon, self.epsilon])
        if random_move: #random move
            move = env.action_space.sample()
        else: #model move
            if self.model_type == 'lnn' or self.model_type == 'cnn':
                pred = self.model(state)
                move = torch.argmax(pred).item()
            elif self.model_type == 'tabular':
                pred = self.model[state]
                move = pred.index(max(pred))
        return move


    def decrease_epsilon(self):
        self.epsilon = self.epsilon - 0.01


    def train(self):
        pass