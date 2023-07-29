import random
import numpy as np
import pickle
from environment import environment

class agent():
    def __init__(self):

        # defining the constants used for bellman equation 
        # q learning
        # these parameters will need to be optimised, typically best by machine,
        # but for now, we'll crudely fine tune by hand to get an acceptable learning rate
        self.learning_rate = 0.01
        self.discount_rate = 0.995
        
        # this is the exploitation vs exploration part of the reinforcement learning
        # start with random (exploratory) moves and eventually almost only explot 
        self.epsilon = 1.0
        self.epsilon_discount_rate = 0.9992
        # this is minimum chance to explore
        self.min_epsilon = 0.0001
        
        # total number of games we want to train over - pick one that lets the algorithm converge
        self.max_epochs = 10000

        # we can define the entire state of the game using 12 observations 
        # is the snake moving:
        # up
        # down
        # left
        # right 
        # in what direction is the apple from the snake:
        # up
        # down
        # left
        # right 
        # is there immediate danger in the following direction:
        # up
        # down
        # left
        # right 

        # we use the q-value table to map out every possible state and record the best action to take in that state:
        # however, there are 2^12 states with 4 different directions, so we need to make the table following table:
        self.table = np.zeros((2,2,2,2, 2,2,2,2, 2,2,2,2 ,4))

        # this is the environment that we can simulate for the agent
        self.environment = environment()
        
        # scores to display later
        self.scores = []



    def pick_direction(self, state):
        # off path - sometimes we want to explore, so we pick a random choice
        if random.random() < self.epsilon:
            return random.choice([0, 1, 2, 3])

        else:
            # otherwise, we want to pick the best possible move
            return np.argmax(self.table[state])

    def q_learn(self):

        # setting gamespeed
        self.environment.fps.tick(self.environment.game_speed)

        # going through each play through
        for i in range(0, self.max_epochs):
            # initialising environment
            self.environment = environment()
            # another break method
            self.environment.uneventful_move = 0
            # setting epoch value for display purposes
            self.environment.epoch = i

            # if we displayed every iteration it would be too much, so we only do it every 25 iterations
            if i % 25 == 0 and i != 0:
                print(f"Epochs: {i}, average_score: {np.mean(self.scores)}, median_score: {np.median(self.scores)}, highest_score: {np.amax(self.scores)}, epsilon_value: {self.epsilon}")
                self.scores = []
            
            # save learning stuff every so often
            if (i < 500 and i % 10 == 0) or (i >= 500 and i < 1000 and i % 200 == 0) or (i >= 1000 and i % 500 == 0):
                # wb means write and binary
                with open(f'C:/Users/offic/Desktop/A lifetime of work/RLSnake/q_learning_snake_dqn/training_data/{i}.pickle', 'wb') as file:
                    '''C:/Users/rhinz/OneDrive - Imperial College London/Desktop/Snake/RLSnake/q_learning_snake/training_data/'''
                    pickle.dump(self.table, file)
                # dump all the values into a pickle
                # alternative to json
     
            # set the state as a variable and save to current_state            
            current_state = self.environment.get_state()
            # make sure that the epsilon never goes below the min when we include discount
            self.epsilon = max(self.epsilon * self.epsilon_discount_rate, self.min_epsilon)
            # is it finished
            finished = False
            # while not finished
            while not finished:
                # pick an action based off the q value table
                action = self.pick_direction(current_state)
                # save the values from the step function so that we can use them in the q-learning formula
                new_state, reward, finished = self.environment.step(action)


                # if we are on the 1000th uneventful move, we are stuck in a loop, so we punish the snake
                # and then later we will break the loop
                if self.environment.uneventful_move == 1000:
                    reward = -10

                # Q-learning formula / Bellman equation
                self.table[current_state][action] = (1 - self.learning_rate)\
                    * self.table[current_state][action] + self.learning_rate\
                    * (reward + self.discount_rate * max(self.table[new_state])) 
                
                # update the state so that we can use the new q tablue value
                current_state = new_state

                # break the loop
                if self.environment.uneventful_move == 1000:
                    break

            # storing the important values to look for history usage
            self.scores.append(self.environment.score)

            


    


        

