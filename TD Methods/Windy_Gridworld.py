import numpy as np
import numpy.random as rand


WORLD_HEIGHT = 7

WORLD_WIDTH = 10


WIND = [0, 0, 0, 1, 1, 1, 2, 2, 1, 0]


ACTION_UP = 0
ACTION_DOWN = 1
ACTION_LEFT = 2
ACTION_RIGHT = 3
ACTION_UP_RIGHT = 4
ACTION_UP_LEFT = 5
ACTION_DOWN_RIGHT = 6
ACTION_DOWN_LEFT = 7


EPSILON = 0.1

ALPHA = 0.5


REWARD = -1.0

START = [3, 0]
GOAL = [3, 7]
ACTIONS = [ACTION_UP, ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT, ACTION_UP_RIGHT, ACTION_UP_LEFT, ACTION_DOWN_RIGHT, ACTION_DOWN_LEFT]


def step(state, action):
    
    i, j = state
    if action == ACTION_UP:
        return [min(i + 1 + WIND[j], WORLD_HEIGHT-1), j]
    elif action == ACTION_DOWN:
        return [max(min(i - 1 + WIND[j], WORLD_HEIGHT-1), 0), j]
    elif action == ACTION_LEFT:
        return [min(i + WIND[max(j - 1,0)], WORLD_HEIGHT-1), max(j - 1, 0)]
    elif action == ACTION_RIGHT:
        return [min(i + WIND[j], WORLD_HEIGHT-1), min(j + 1, WORLD_WIDTH - 1)]
    elif action == ACTION_UP_RIGHT:
        return[min(i + 1 + WIND[min(j+1, WORLD_WIDTH-1)], WORLD_HEIGHT-1), min(j + 1, WORLD_WIDTH - 1)]
    elif action == ACTION_UP_LEFT:
        return[min(i + 1 + WIND[max(j-1,0)], WORLD_HEIGHT-1), max(j - 1, 0)]
    elif action == ACTION_DOWN_RIGHT:
        return[max(min(i - 1 + WIND[min(j + 1, WORLD_WIDTH - 1)], WORLD_HEIGHT-1), 0), min(j + 1, WORLD_WIDTH-1)]
    elif action == ACTION_DOWN_LEFT:
        return[max(min(i - 1 + WIND[max(j - 1, 0)], WORLD_HEIGHT-1), 0), max(j - 1, 0)]
    else:
        assert False

def episode(q_value):
    time = 0
    state = START

    if rand.binomial(1,EPSILON) == 1:
        action = rand.choice(ACTIONS)
    else:
        values = q_value[state[0], state[1], :]
        action = rand.choice([action for action, value in enumerate(values) if value == np.max(values)])
        
    while state != GOAL:
        
        next_state = step(state, action)
        
        if rand.binomial(1, EPSILON) == 1:
            next_action = rand.choice(ACTIONS)
        else:
            values = q_value[next_state[0], next_state[1], :]
            next_action = rand.choice([action for action, value in enumerate(values) if value == np.max(values)])

            q_value[state[0], state[1], action] += \
            ALPHA * (REWARD + q_value[next_state[0], next_state[1], next_action] -
                     q_value[state[0], state[1], action])
        state = next_state
        action = next_action
        time += 1
    return time

def run():
    q_value = np.zeros((WORLD_HEIGHT, WORLD_WIDTH, len(ACTIONS)))
    episodes = 500

    ep = 0

    while ep < episodes:
        episode(q_value)
        ep += 1
        
    print(q_value[3, 7])

run()