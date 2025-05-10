import numpy
import numpy.random as random

def walk():
    values = [0.5, 0.5, 0.5, 0.5, 0.5]
    alpha = 0.025
    for i in range(100):
        state = 2
        next_state = state
        
        while state >=0 and state <= 4:
            
            rand = random.random()
            if rand > 0.5:
                next_state += 1
            else:
                next_state -= 1

            value_next = 0
            reward = 0
            if state == 0:
                next_state = -1
            elif state == 4:
                reward = 1
                next_state = 5
            else:
                value_next = values[next_state]
            
            values[state] += alpha * (reward + value_next - values[state])
            state = next_state
        print(values)
    

walk()