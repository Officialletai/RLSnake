import random
import numpy as np
from model import DQN
from environment import environment
import torch
import torch.optim as optim
import torch.nn.functional as F


class DQNAgent:
    def __init__(self, input_size=12, output_size=4):

        #hyperparameters
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay_rate = 0.95
        self.max_epochs = 1000
        self.discount_factor = 0.995
        self.batch_size = 64
        self.learning_rate = 0.01

        self.environment = environment(display_game=False)
        self.model = DQN(input_size,output_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr = self.learning_rate)
        self.memory = []
        self.scores = []
        self.q_values = []



    def pick_action(self, state):
        if random.random() < self.epsilon:
            return random.choice([0,1,2,3])
        else:
            
            move = torch.argmax(self.model.forward(state)).item()
            return move

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        #memory

    def replay(self):
        minibatch = random.sample(self.memory, self.batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = self.model.forward(state)
            if done == True:
                target[0][action] = reward
            else:
                Q_future = max(self.model.forward(next_state)[0])
                target[0][action] = reward + self.discount_factor * Q_future
                
            loss = torch.sum(target - self.model.forward(state))^(2)
            loss.backward()
            self.optimizer.step()
            #function will update the model using randomly sampled experiences from the memory
        pass

    def dqn_learn(self):

        for i in range(self.max_epochs):
            self.environment = environment(display_game=False)
            self.environment.epoch = i
            self.environment.uneventful_move = 0

            if i % 25 == 0 and i != 0:
                print(
                    f"Epochs: {i}, average_score: {np.mean(self.scores)}, median_score: {np.median(self.scores)}, highest_score: {np.amax(self.scores)}, epsilon_value: {self.epsilon}, avg_q_value: {np.mean(self.q_values)}"
                )
                self.scores = []

            raw_state = self.environment.get_state()
            current_state = torch.tensor(raw_state, dtype=torch.float32)
            print(len(current_state.shape) == 1)
            
            self.epsilon = max(self.epsilon * self.epsilon_decay_rate ,self.epsilon_min)
            finished = False

            while not finished:
                action = self.pick_action(current_state)
                next_state, reward, finished= self.environment.step(action)

                if self.environment.uneventful_move == 600:
                    reward = -10

                self.remember(current_state, action, reward, next_state, finished)
                current_state = torch.tensor(next_state, dtype=torch.float32)

                if self.environment.uneventful_move == 600:
                    break

            self.scores.append(self.environment.score)
            self.replay


    def load(self):
        #load the weights
        pass

    def save(self):
        #save the weights
        pass