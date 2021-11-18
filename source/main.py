import time
import gym
import numpy as np
import gym_snake

def colission(position, queue):
    return position in queue or (position[0] > 15 or position[0] < 0 or position[1] > 15 or position[1] < 0)


def get_state(env):
    apple = list(env.env.grid.apples._set)[0] # columna, fila
    head = env.env.grid.snakes[0]._deque[-1]
    queue = list(env.env.grid.snakes[0]._deque)[:-1]

    #apple's direction from snake's head
    apple_left = apple[0] < head[0] #left
    apple_right = apple[0] > head[0] #right
    apple_up = apple[1] < head[1] #up
    apple_down = apple[1] > head[1] #down

    #snake's direction
    dir_up = env.env.grid.snakes[0]._direction == 0 #up
    dir_right = env.env.grid.snakes[0]._direction == 1 #right
    dir_down = env.env.grid.snakes[0]._direction == 2 #down
    dir_left = env.env.grid.snakes[0]._direction == 3 #left

    # danger's direction
    danger_left = (dir_up and colission((head[0] - 1,head[1]), queue) or
                       dir_down and colission((head[0] + 1,head[1]), queue) or
                       dir_left and colission((head[0],head[1] + 1), queue) or
                       dir_right and colission((head[0],head[1] - 1), queue))
    danger_right = (dir_up and colission((head[0] + 1,head[1]), queue) or
                       dir_down and colission((head[0] - 1,head[1]), queue) or
                       dir_left and colission((head[0],head[1] - 1), queue) or
                       dir_right and colission((head[0],head[1] + 1), queue))
    danger_straight = (dir_up and colission((head[0],head[1] - 1), queue) or
                           dir_down and colission((head[0],head[1] + 1), queue) or
                           dir_left and colission((head[0] - 1,head[1]), queue) or
                           dir_right and colission((head[0] + 1,head[1]), queue))

    return [apple_left, apple_right, apple_up, apple_down, dir_left, dir_right,
            dir_up, dir_down, danger_left, danger_right, danger_straight]



env = gym.make('Snake-16x16-v0')
#Snake-16x16-v0
#Breakout-v0
#MsPacman-v0
observation = env.reset()

for i in range(1000):
    env.render()
    right = True
    observation, reward, done, info = env.step(env.action_space.sample()) # take a random action
    if done:
        env.reset()
    state = get_state(env)
    print(state)
    time.sleep(0.05)
    #print(i)
env.close()
