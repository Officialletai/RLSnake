from collections import deque
import math
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
        self.epsilon_min = 0.001
        self.epsilon_decay_rate = 0.9992
        self.max_epochs = 10000
        self.discount_factor = 0.995
        self.batch_size = 64
        self.learning_rate = 0.0025
        
        self.display_game = True
        self.environment = environment(display_game=self.display_game)
        self.model = DQN(input_size,output_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr = self.learning_rate)
        self.memory = deque(maxlen=10000)  # Store 10000 most recent experiences
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

        minibatch_size = max(64, len(self.memory) // 10)  # Batch size is 10% of current memory size
        minibatch = random.sample(self.memory, minibatch_size)

        # if (len(self.memory) > self.batch_size):
        #     minibatch = random.sample(self.memory, self.batch_size)
        # else:
        #     minibatch = self.memory

        
        states, actions, rewards, next_states, dones = zip(*minibatch)

        state = torch.tensor(states, dtype=torch.float)
        next_state = torch.tensor(next_states, dtype=torch.float)
        action = torch.tensor(actions, dtype=torch.long)
        reward = torch.tensor(rewards, dtype=torch.float)
        done = dones

        if(len(state.shape) == 1):
            state = torch.unsqueeze(state,0)
            next_state = torch.unsqueeze(next_state,0)
            action = torch.unsqueeze(action,0)
            reward = torch.unsqueeze(reward,0)
            done = (done, )
        
        prediction = self.model.forward(state)
        target = prediction.clone()

        for i in range(len(done)):
            if done == True:
                Q_new = reward[i]
            else:
                Q_new = reward[i] + self.discount_factor * torch.max(self.model.forward(next_state[i]))
            
            
            
            target[i][actions[i]] = Q_new
            self.q_values.append(float(Q_new))

        self.optimizer.zero_grad()
        loss = F.mse_loss(target, prediction)
        loss.backward()
        self.optimizer.step()


    def dqn_learn(self):

        for i in range(self.max_epochs):
            self.environment = environment(display_game=self.display_game)
            self.environment.epoch = i
            self.environment.uneventful_move = 0

            if i % 25 == 0 and i != 0:
                print(
                    f"Epochs: {i}, average_score: {np.mean(self.scores)}, median_score: {np.median(self.scores)}, highest_score: {np.amax(self.scores)}, epsilon_value: {self.epsilon}, avg_q_value: {np.mean(self.q_values)}"
                )
                self.scores = []

            raw_state = self.environment.get_state()
            current_state = torch.tensor(raw_state, dtype=torch.float32)
            #print(len(current_state.shape) == 1)
            
            self.epsilon = max(self.epsilon * self.epsilon_decay_rate ,self.epsilon_min)
            #self.epsilon = 0.1 + (0.95 - 0.05) * math.exp(-1. * i / 150)  # Exponential decay
            finished = False

            while not finished:
                action = self.pick_action(current_state)
                next_state, reward, finished= self.environment.step(action)



                # if self.environment.uneventful_move == 600:
                #     reward = -10 * (self.environment.uneventful_move // 100)  # Increase penalty for successive uneventful moves
                

                self.remember(raw_state, action, reward, next_state, finished)
                raw_state = next_state
                current_state = torch.tensor(raw_state, dtype=torch.float32)

                # if self.environment.uneventful_move == 600:
                #     break

            self.scores.append(self.environment.score)
            self.replay()


    def load(self):
        #load the weights
        pass

    def save(self):
        #save the weights
        pass