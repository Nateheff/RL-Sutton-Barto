import numpy as np
import math
HEIGHT = 6
WIDTH = 9
EPSILON = 0.1
ALPHA = 0.1
GAMMA = 0.9
K = 0.1

grid = np.zeros((HEIGHT,WIDTH))
t = np.zeros((HEIGHT,WIDTH))

UP = 0
DOWN = 1
RIGHT = 2
LEFT = 3

actions = [UP, DOWN, RIGHT, LEFT]
visited_taken = []
start = [0,8]
terminal = [5,3]

valid_row3 = [0]

q_values = np.zeros((HEIGHT, WIDTH, len(actions)))


model = {}
state = start
reward = 0
episode = 0


def policy(state):
    if np.random.binomial(1, EPSILON) == 1:
        action = np.random.choice(actions)
    else:
        values = q_values[state[0], state[1], :]
        print(values)
        values = [value + K*math.sqrt(t[state[0], state[1]]) for value in values]

        action = np.random.choice([action for action, value in enumerate(values) if value == np.max(values)])
    return action

def step(state, action):
    
    i,j = state

    if action == UP:
        next_state =  [min(i + 1, HEIGHT-1), j]
    elif action == DOWN:
        next_state = [max(min(i - 1, HEIGHT-1), 0), j]
    elif action == LEFT:
        next_state = [i, max(j - 1, 0)]
    elif action == RIGHT:
        next_state = [i, min(j + 1, WIDTH - 1)]
    
    if next_state[0] == 3 and next_state[1] not in valid_row3:
        next_state = state

    return next_state


def dyna_p(n):
    global t, state, episode, reward
    #{[state[0], state[1], action]: [reward, next_state[0], next_state[1]}
    
    while True:
        
        action = policy(state)
        visit_take = [state, action]
        visited_taken.append(visit_take)
        next_state = step(state, action)
        if next_state == terminal:
            reward = 1
        update = ALPHA * (reward + GAMMA * max(q_values[next_state[0], next_state[1]]) - q_values[state[0], state[1], action])
        q_values[state[0], state[1], action] += update
        model[state[0], state[1], action] = [reward, next_state]
        for i in range(n):
            plan_state_ind = np.random.randint(0,len(visited_taken))
            #[state_row, state_column, action]
            plan_taken = visited_taken[plan_state_ind]
            plan_state = plan_taken[0]
            plan_action = plan_taken[1]
            plan_reward, plan_next_state = model[plan_state[0], plan_state[1], plan_action]
            q_values[plan_state[0], plan_state[1], plan_action] += ALPHA * (plan_reward + GAMMA * max(q_values[plan_next_state[0], plan_next_state[1]]) - q_values[plan_state[0], plan_state[1], action])
        t += 1
        t[state[0], state[1]] -= 1
        state = next_state
        if state == terminal:
            reward = 0
            state = start
            episode += 1
            t = np.zeros((HEIGHT,WIDTH))
        # episode += 1
        if episode == 5:
            break
        # elif episode == 20:
        #     valid_row3.append(WIDTH-1)


dyna_p(5)
print('done')
dyna_p(5)
print(np.max(q_values), np.min(q_values))
print(q_values)