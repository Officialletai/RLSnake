import random
import numpy as np
from dqn_model import DQN
from environment import environment
import torch
import torch.optim as optim
import torch.nn.functional as F


class DQNAgent:
    def __init__(self, input_size=12, output_size=4):
        
        self.environment = environment(display_game=False)
        self.model = DQN(input_size,output_size)

        #hyperparameters
        self.epsilon = 1.0
        self.epsilon_min = 0.1
        self.epsilon_decay_rate = 0.999
        self.max_epochs = 1000
        self.learning_rate = 0.05
        self.batch_size = 64


    def pick_action(self):
        if random.random() < self.epsilon:
            #random action
            pass
        else:
            #pick action based on dqn output
            pass

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        #memory
        pass

    def replay(self):
        minibatch = random.sample(self.memory, self.batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = self.model.forward(state)
            if done == True:
                target[0][action] = reward
            else:
                Q_future = max(self.model.forward(next_state)[0])
                target[0][action] = reward + self.learning_rate * Q_future
                
            #function will update the model using randomly sampled experiences from the memory
        pass

    def dqn_learn(self):

        for i in range(self.max_epochs):
            self.environment = environment(display_game = False)
            self.environment.epoch = i
            self.environment.uneventful_move = 0

            current_state = self.environment.get_state()
            self.espilon = max(self.epsilon*self.epsilon_decay_rate,self.epsilon_min)
            finished = False

            while not finished:
                action = self.pick_action(current_state)
                next_state, reward, finished= self.environment.step(action)

                if self.environment.uneventful_move == 600:
                    reward = -10

                self.remember(current_state, action, reward, next_state, finished)
                current_state = next_state

                if self.environment.uneventful_move == 600:
                    break

            self.score.append(self.environment.score)

            self.replay

        pass

    def load(self):
        #load the weights
        pass

    def save(self):
        #save the weights
        pass