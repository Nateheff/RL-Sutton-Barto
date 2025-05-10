import numpy as np
import math
import matplotlib.pyplot as plt

EPSILON = 0.1
ALPHA = 0.1
GAMMA = 0.9

start_policy = []
start_uniform = []

values = np.zeros((10000,2))
values_uniform = np.zeros((10000,2))
b = 3
actions = [0,1]
prob_term = 0.1

prob_b_chosen = 3/len(values)
prob_b_taken = 1/3

start = 4999
reward = np.random.normal(0,1)

def policy():
    pass

def uniform():
    pass

def on_policy(state):

    if np.random.binomial(1, EPSILON) == 1:
        action = np.random.choice(actions)
    else:
        action_values = values[state]
        

        action = np.random.choice([action for action, value in enumerate(action_values) if value == np.max(action_values)])
    return action


def episode():

    #initialze S
    state = start
    #for on-policy:
    for i in range(10000):
        #Choose action
        action = on_policy(state)
            #Get next b random states
        next_states = np.random.randint(0,len(values),size=(3))
            #Go to some state in b
        next_state = np.random.choice(next_states)
            #Update value func
        values[state, action] += ALPHA * (reward + GAMMA * max(values[next_state]) - values[state, action])
        for state in range(len(values)):
            
            values_uniform[state,action] = prob_b_taken * (reward + GAMMA * max(values_uniform[next_state]) - values_uniform[state,action])
        if np.random.binomial(1, prob_term) == 1:
            state = start
        else:
            state = next_state
        #for uniform:

            #
    
        start_uniform.append(sum(values_uniform[4999]))
        start_policy.append(sum(values[start]))
            


episode()

plt.plot(start_policy)
plt.plot(start_uniform)

plt.show()
