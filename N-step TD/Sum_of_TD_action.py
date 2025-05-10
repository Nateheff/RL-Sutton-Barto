import numpy as np
import queue

q_values = np.zeros((5,5,4))

rewards = np.random.randint(0,5, (5,5))

ALPHA = 0.1
WIDTH = 5
HEIGHT = 5

start = [0,0]

episodes = 100
EPSILON = 0.1
GAMMA = 0.9

UP = 0
DOWN = 1
RIGHT = 2
LEFT = 3

actions = [UP, DOWN, LEFT, RIGHT]

def policy(state):
    action = 0
    if np.random.choice([0,1], p=[1-EPSILON, EPSILON]) == 1:
        action = np.random.choice(actions)
    else:
        values = q_values[state[0], state[1],:]
        if values.any():
            action = np.argmax(values)
        else:
            action = np.random.choice(actions)

    return action
    

def n_step(start, episodes, steps):
    ep = 0

    while ep < episodes:
        ep +=1
        reward_store = []
        state_store = []
        t = 0
        T = 50
        state = [0,0]

        next_state = state
        reward = 0
        
        while t < T:
            
            action = policy(state)
           
            
            i, j = state
            
            if action == UP:
                next_state =  [min(i + 1, HEIGHT-1), j]
            elif action == DOWN:
                next_state = [max(min(i - 1, HEIGHT-1), 0), j]
            elif action == LEFT:
                next_state = [min(i , HEIGHT-1), max(j - 1, 0)]
            elif action == RIGHT:
                next_state = [min(i, HEIGHT-1), min(j + 1, WIDTH - 1)]
            if next_state == state:
                reward = 0
            else:
                reward = rewards[next_state[0], next_state[1]]
            
            if t < steps:
                
                state.append(action)
                
                state_store.append(state)
                reward = reward 
                reward_store.append(reward)
                state = next_state
            else:
                
                # print(q_values[state[0], state[1], action])
                reward_total = GAMMA * sum(reward_store) + GAMMA**steps * q_values[state[0], state[1], action]
                
                update_state = state_store.pop(0)
                # print(q_values[update_state[0], update_state[1], update_state[2]])
                q_values[update_state[0], update_state[1], update_state[2]] += ALPHA * (reward_total - q_values[update_state[0], update_state[1], update_state[2]])
                # print(q_values[update_state[0], update_state[1], update_state[2]])
                state.append(action)
                state_store.append(state)
                reward_store.append(reward)
                
                reward_store.pop(0)
                state = next_state
            t+=1
                 
                
                
    #   First three steps: catch up
    
    #   Go next state after catch up

    # Accumulate reward

    # Update V

    # 

n_step([0,0,0], 10000, 3)

print(rewards)
print(np.sum(q_values,axis=-1))

n_step([0,0,0], 10, 3)