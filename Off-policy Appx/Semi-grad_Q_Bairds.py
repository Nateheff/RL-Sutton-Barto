import numpy as np
import numpy.random as rand
import torch.nn as nn
import torch

features = torch.tensor(
    [
        [2, 0, 0, 0, 0, 0, 0, 1],  # Upper state 1
        [0, 2, 0, 0, 0, 0, 0, 1],  # Upper state 2
        [0, 0, 2, 0, 0, 0, 0, 1],  # Upper state 3
        [0, 0, 0, 2, 0, 0, 0, 1],  # Upper state 4
        [0, 0, 0, 0, 2, 0, 0, 1],  # Upper state 5
        [0, 0, 0, 0, 0, 2, 0, 1],  # Upper state 6
        [0, 0, 0, 0, 0, 0, 1, 2],  # Lower state
    ],
    dtype=torch.float,
    requires_grad=False
)
weights = torch.tensor(
    [
        [1, 1],
        [1, 1],
        [1, 1],
        [1, 1],
        [1, 1],
        [1, 1],
        [10, 10],
        [1, 1],
    ],
    dtype=torch.float,
    requires_grad=True
)
actions = [0,1]
EPSILON = 1/7
GAMMA = 0.99
ALPHA = 0.05

reward = 0

def behavior():
    if rand.binomial(1,EPSILON):
        return 0,rand.choice(6)
    else:
        return 1,6
    
def q_value(state, action):
    return features[state] @ weights[:,action]

def run(n_steps):
    global weights, features
    state = 0
    action,next_state = behavior()
    
    value = q_value(state, action)

    #reward from behavior action + discount * target policy estimate of next state
    q_target = reward + GAMMA * torch.max(q_value(next_state, 0), q_value(next_state,1))
    #target - behavior estimate
    delta = q_target - value
    value.backward()
    with torch.no_grad():
        weights += ALPHA * delta * weights.grad
        weights.grad.zero_()

    step = 1
    state = next_state
    while step < n_steps:
        action, next_state = behavior()

        value = q_value(state, action)
        q_target = reward + ALPHA * torch.max(q_value(next_state, 0), q_value(next_state, 1))
        delta = q_target - value

        value.backward()
        with torch.no_grad():
            weights += ALPHA * delta * weights.grad
            weights.grad.zero_()

        step += 1
        state = next_state



run(1000)
print(weights)

values = []
for i, feature in enumerate(features):
    for i in range(2):
        value = feature @ weights[:, i]
        values.append(value.item())
print(values)