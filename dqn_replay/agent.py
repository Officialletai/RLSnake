import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque

from environment import environment

class DeepQNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(DeepQNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, output_size)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        return self.fc2(x)

class DQNAgent:
    def __init__(self, input_size=12, output_size=4, lr=0.000025, gamma=0.995, max_epochs=10000, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995, memory_size=10000):

        self.max_epochs = max_epochs
        self.discount_factor = gamma
        self.learning_rate = lr
        self.memory = deque(maxlen=memory_size)
        self.model = DeepQNetwork(input_size, output_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.loss_fn = nn.MSELoss()
        self.scores = []

        self.display_game = True
        self.batch_size = 256

        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

    def pick_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, 3)
        else:
            with torch.no_grad():
                return torch.argmax(self.model(state)).item()

    def train(self, batch):
        states, actions, rewards, next_states, dones = zip(*batch)
        states = torch.stack(states)
        actions = torch.tensor(actions)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        next_states = torch.stack(next_states)
        dones = torch.tensor(dones).float()

        current_Q_values = self.model(states).gather(1, actions.unsqueeze(1)).squeeze()
        next_Q_values = self.model(next_states).max(1)[0]
        target_Q_values = rewards + self.discount_factor * next_Q_values * (1 - dones)
        
        loss = self.loss_fn(current_Q_values, target_Q_values.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


    def learn(self):
        for i in range(self.max_epochs):
            self.environment = environment(display_game=self.display_game)
            self.environment.epoch = i
            self.environment.uneventful_move = 0

            state = self.environment.get_state()
            state = torch.tensor(state, dtype=torch.float32)

            done = False

            while not done:
                action = self.pick_action(torch.tensor(state, dtype=torch.float32))
                next_state, reward, done = self.environment.step(action)

                next_state = torch.tensor(next_state, dtype=torch.float32)

                self.memory.append((state, action, reward, next_state, done))
                state = next_state

                if len(self.memory) >= self.batch_size:
                    self.train(random.sample(self.memory, self.batch_size))
            
            self.epsilon = max(self.epsilon_end, self.epsilon_decay * self.epsilon)

            self.scores.append(self.environment.score)

            if i % 25 == 0:
                print(
                    f"Epochs: {i}, average_score: {np.mean(self.scores)}, median_score: {np.median(self.scores)}, highest_score: {np.amax(self.scores)}, epsilon_value: {self.epsilon}"
                )                
                self.scores = []

if __name__ == "__main__":
    DQNAgent().learn()
