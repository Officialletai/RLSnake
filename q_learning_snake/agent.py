import random
import numpy as np
import pickle

class agent():
    def __init__(self, window_x, window_y, block_size):

        self.window_x = window_x
        self.window_y = window_y
        self.block_size = block_size


        self.discount_rate = 0.90
        self.learning_rate = 0.02

        self.epsilon = 1.0
        self.epsilon_discount_rate = 0.999
        self.min_epsilon = 0.002

        self.action_map = {
            0 : "UP",
            1 : "DOWN",
            2 : "LEFT",
            3 : "RIGHT"
        }

    def pick_direction(self):
        
        if random.random() < self.eps:
            action = random.choice['UP', 'DOWN', 'LEFT', 'RIGHT']
            
        # otherwise exploit by using q value table
    


        

