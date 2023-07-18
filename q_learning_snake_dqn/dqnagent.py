import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque

from environment import environment

class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, output_size)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        return self.fc2(x)

class DQNAgent:
    def __init__(self, input_size=12, output_size=4, lr=0.0025, gamma=0.995, max_epochs=10000, epsilon_start=1.0, epsilon_min=0.001, epsilon_decay_rate=0.999992, display_game=True):

        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay_rate = epsilon_decay_rate
        self.max_epochs = max_epochs
        self.discount_factor = gamma
        self.batch_size = 64
        self.learning_rate = lr
        self.display_game = display_game
        self.environment = environment(display_game=self.display_game)
        self.model = DQN(input_size, output_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr = self.learning_rate)
        self.scores = []
        self.q_values = []

    def pick_action(self, state):
        if random.random() < self.epsilon:
            return random.choice([0,1,2,3])
        else:
            with torch.no_grad():
                move = torch.argmax(self.model(state)).item()
            return move

    def train(self, state, action, reward, next_state, done):
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        next_state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)
        reward = torch.tensor(reward, dtype=torch.float32).unsqueeze(0)
        action = torch.tensor(action, dtype=torch.long).unsqueeze(0)
        done = torch.tensor(int(done), dtype=torch.float32).unsqueeze(0)

        # Compute Q(s, a)
        q_values = self.model(state)
        q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1)

        # Compute Q(s', a')
        with torch.no_grad():
            next_q_values = self.model(next_state)
            next_q_value = next_q_values.max(1)[0]

        # Compute the target of the Q values
        target = reward + (1 - done) * self.discount_factor * next_q_value

        # Compute loss
        loss = (q_value - target.detach()).pow(2).mean()

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update epsilon
        self.epsilon = max(self.epsilon * self.epsilon_decay_rate, self.epsilon_min)

        # Save q values for monitoring
        self.q_values.append(float(target))


    def dqn_learn(self):
        for i in range(self.max_epochs):
            self.environment = environment(display_game=self.display_game)
            self.environment.epoch = i
            self.environment.uneventful_move = 0

            state = self.environment.get_state()
            done = False

            while not done:
                action = self.pick_action(torch.tensor(state, dtype=torch.float32))
                next_state, reward, done = self.environment.step(action)

                self.train(state, action, reward, next_state, done)
                state = next_state

            self.scores.append(self.environment.score)

            if i % 25 == 0 and i != 0:
                print(
                    f"Epochs: {i}, average_score: {np.mean(self.scores)}, median_score: {np.median(self.scores)}, highest_score: {np.amax(self.scores)}, epsilon_value: {self.epsilon}, avg_q_value: {np.mean(self.q_values)}"
                )
                self.scores = []

if __name__ == "__main__":
    DQNAgent().dqn_learn()
