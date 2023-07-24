import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque

from environment import environment

class Actor(nn.Module):
    def __init__(self, input_size, output_size):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, output_size)

    def forward(self, state):
        x = torch.relu(self.fc1(state))

        if len(x.shape) == 1: 
            return torch.softmax(self.fc2(x), dim=0)
        else:
            return torch.softmax(self.fc2(x), dim=1)

class Critic(nn.Module):
    def __init__(self, input_size):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        return self.fc2(x)


class ActorCriticAgent:
    def __init__(self, input_size=12, output_size=4, lr=0.000025, gamma=0.995, max_epochs=10000):

        self.max_epochs = max_epochs
        self.discount_factor = gamma
        self.learning_rate = lr
        self.display_game = True
        self.environment = environment(display_game=self.display_game)
        self.actor = Actor(input_size, output_size)
        self.critic = Critic(input_size)
        self.optimizer_actor = optim.Adam(self.actor.parameters(), lr=self.learning_rate)
        self.optimizer_critic = optim.Adam(self.critic.parameters(), lr=self.learning_rate)
        self.scores = []

    def pick_action(self, state):
        action_probs = self.actor(state)
        action_distribution = torch.distributions.Categorical(action_probs)
        action = action_distribution.sample()
        return action.item(), action_distribution.log_prob(action)


    def train(self, state, action, reward, next_state, done):
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        next_state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)
        reward = torch.tensor(reward, dtype=torch.float32).unsqueeze(0)
        done = torch.tensor(int(done), dtype=torch.float32).unsqueeze(0)
        action = torch.tensor(action, dtype=torch.long).unsqueeze(0)

        # Compute advantage
        target_value = reward + (1 - done) * self.discount_factor * self.critic(next_state)
        current_value = self.critic(state)
        advantage = target_value - current_value

        # Compute actor loss
        log_prob = torch.log(self.actor(state))[0, action]
        actor_loss = -log_prob * advantage.detach()

        # Compute critic loss
        critic_loss = advantage.pow(2).mean()

        # Optimize the actor
        self.optimizer_actor.zero_grad()
        actor_loss.backward()
        self.optimizer_actor.step()

        # Optimize the critic
        self.optimizer_critic.zero_grad()
        critic_loss.backward()
        self.optimizer_critic.step()



    def actor_critic_learn(self):
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

                self.train(state, action, reward, next_state, done)

                episode.append((state, action, reward, log_prob))
                state = next_state

            # switch from training every episode to training every move
            # this iss due to unbiased estimate returnafter every step, but you can do either
            #self.train(state, action, reward, next_state, done)

            self.scores.append(self.environment.score)

            if i % 25 == 0 and i != 0:
                print(
                    f"Epochs: {i}, average_score: {np.mean(self.scores)}, median_score: {np.median(self.scores)}, highest_score: {np.amax(self.scores)}"
                )
                self.scores = []

if __name__ == "__main__":
    ActorCriticAgent().actor_critic_learn()