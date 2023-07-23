import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque

from environment import environment

class Policy(nn.Module):
    def __init__(self, input_size, output_size):
        super(Policy, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, output_size)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        # difference is that we need output to get normalised in range 0-1 as opposed to any q-value
        if len(x.shape) == 1:  # If the input is a 1D tensor
            return torch.softmax(self.fc2(x), dim=0)
        else:  # If the input is a 2D tensor
            return torch.softmax(self.fc2(x), dim=1)

class PolicyAgent:
    def __init__(self, input_size=12, output_size=4, lr=0.0025, gamma=0.995, max_epochs=10000):


        self.max_epochs = max_epochs
        self.discount_factor = gamma
        self.learning_rate = lr
        self.display_game = True
        self.environment = environment(display_game=self.display_game)
        self.model = Policy(input_size, output_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr = self.learning_rate)
        self.scores = []

    def pick_action(self, state):
        action_probs = self.model(state)
        action_distribution = torch.distributions.Categorical(action_probs)
        action = action_distribution.sample()
        return action.item(), action_distribution.log_prob(action)

    def train(self, episode):
        G = 0
        policy_loss = []

        for state, action, reward, log_prob in reversed(episode):
            G = reward + self.discount_factor * G
            policy_loss.append(-log_prob * G)

        loss = torch.stack(policy_loss).sum()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


    def reinforce_learn(self):
        for i in range(self.max_epochs):
            self.environment = environment(display_game=self.display_game)
            self.environment.epoch = i
            self.environment.uneventful_move = 0

            state = self.environment.get_state()
            done = False

            episode = []  # to store states, actions, and rewards for this episode

            while not done:
                action, log_prob = self.pick_action(torch.tensor(state, dtype=torch.float32))
                next_state, reward, done = self.environment.step(action)

                episode.append((state, action, reward, log_prob))
                state = next_state

            self.train(episode)

            self.scores.append(self.environment.score)

            if i % 25 == 0 and i != 0:
                print(
                    f"Epochs: {i}, average_score: {np.mean(self.scores)}, median_score: {np.median(self.scores)}, highest_score: {np.amax(self.scores)}"
                )
                self.scores = []

if __name__ == "__main__":
    PolicyAgent().reinforce_learn()