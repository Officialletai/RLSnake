import random
import numpy as np
import pickle
from environment import environment

class agent():
    def __init__(self):

        self.discount_rate = 0.95
        self.learning_rate = 0.01

        self.epsilon = 1.0
        self.epsilon_discount_rate = 0.992
        self.min_epsilon = 0.001
        
        self.max_epochs = 3000
        self.table = np.zeros((2,2,2,2, 2,2,2,2, 2,2,2,2 ,4))
        self.environment = environment()
        self.scores = []
        self.survived = []



    def pick_direction(self, state):
        # off path - sometimes we want to explore, so we pick a random choice
        if random.random() < self.epsilon:
            return random.choice([0, 1, 2, 3])

        else:
            return np.argmax(self.table[state])

    def q_learn(self):

        self.environment.fps.tick(self.environment.game_speed)
    
        for i in range(0, self.max_epochs):
            # initialising environment
            self.environment = environment()
            # another break method
            self.environment.uneventful_move = 0
            # setting epoch value for display purposes
            self.environment.epoch = i

            # if we displayed every iteration it would be too much, so we only do it every 25 iterations
            if i % 25 == 0:
                print(f"Epochs: {i}, score: {np.mean(self.scores)}, survived: {np.mean(self.survived)}, epsilon_value: {self.epsilon}, learning_rate: {self.learning_rate}")
                self.scores = []
                self.survived = []
            
            # save learning stuff every so often
            if (i < 500 and i % 10 == 0) or (i >= 500 and i < 1000 and i % 200 == 0) or (i >= 1000 and i % 500 == 0):
                # wb means write and binary
                with open(f'/home/tai/Documents/Snake/q_learning_snake/pickle/{i}.pickle', 'wb') as file:
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


                    
                if self.environment.uneventful_move == 200:
                    reward = -10

                # Q-learning formula or Bellman equation
                self.table[current_state][action] = (1 - self.learning_rate)\
                    * self.table[current_state][action] + self.learning_rate\
                    * (reward + self.discount_rate * max(self.table[new_state])) 
                
                # update the state so that we can use the new q tablue value
                current_state = new_state

                # break the loop again
                if self.environment.uneventful_move == 1000:
                    break

            # storing the important values to look for history usage
            self.scores.append(self.environment.score)
            self.survived.append(self.environment.number_of_rounds)
            


    


        

