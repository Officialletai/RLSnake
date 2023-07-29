import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import time

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


class PPOAgent:
    def __init__(self, input_size=12, output_size=4, lr=0.000025, gamma=0.995, max_epochs=10000, clip_epsilon=0.4, update_epochs=5):

        self.max_epochs = max_epochs
        self.discount_factor = gamma
        self.learning_rate = lr
        self.display_game = True
        self.environment = environment(display_game=self.display_game)
        self.actor = Actor(input_size, output_size)
        self.actor_old = Actor(input_size, output_size)
        self.actor_old.load_state_dict(self.actor.state_dict())

        self.critic = Critic(input_size)
        self.critic_old = Critic(input_size)
        self.critic_old.load_state_dict(self.critic.state_dict())


        self.optimizer_actor = optim.Adam(self.actor.parameters(), lr=self.learning_rate)
        self.optimizer_critic = optim.Adam(self.critic.parameters(), lr=self.learning_rate)
        self.scores = []

        self.clip_epsilon = clip_epsilon
        self.update_epochs = update_epochs


    def pick_action(self, state, policy_old=True):
        if policy_old:
            action_probs = self.actor_old(state)
        else:
            action_probs = self.actor(state)
        action_distribution = torch.distributions.Categorical(action_probs)
        action = action_distribution.sample()
        log_prob = action_distribution.log_prob(action)
        return action.item(), log_prob
    

    def update_old_policy(self):
        self.actor_old.load_state_dict(self.actor.state_dict())
        self.critic_old.load_state_dict(self.critic.state_dict())


    def train(self, trajectories):
        #start_time = time.time()
        batch_size = len(trajectories) // 10

        for _ in range(self.update_epochs):
            minibatch = random.sample(trajectories, batch_size)

            for state, action, reward, next_state, done, old_log_prob in minibatch:
                state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                next_state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)
                reward = torch.tensor(reward, dtype=torch.float32).unsqueeze(0)
                done = torch.tensor(int(done), dtype=torch.float32).unsqueeze(0)
                action = torch.tensor(action, dtype=torch.long).unsqueeze(0)
                old_log_prob = torch.tensor(old_log_prob, dtype=torch.float32).unsqueeze(0)

                # Compute advantage
                target_value = reward + (1 - done) * self.discount_factor * self.critic_old(next_state)
                current_value = self.critic(state)
                advantage = target_value - current_value

                # Compute actor loss
                log_prob = torch.log(self.actor(state))[0, action]
                ratio = torch.exp(log_prob - old_log_prob)
                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantage
                actor_loss = -torch.min(surr1, surr2)

                # Compute critic loss
                critic_loss = advantage.pow(2).mean()

                # Combine the actor and critic losses
                total_loss = actor_loss + critic_loss

                # Optimize the actor and critic
                self.optimizer_actor.zero_grad()
                self.optimizer_critic.zero_grad()
                total_loss.backward()
                self.optimizer_actor.step()
                self.optimizer_critic.step()

        #print(f"Time taken for training: {time.time() - start_time} seconds")
                

        self.update_old_policy()


    def ppo_learn(self):
        for i in range(self.max_epochs):
            self.environment = environment(display_game=self.display_game)
            self.environment.epoch = i
            self.environment.uneventful_move = 0

            state = self.environment.get_state()
            done = False

            trajectories = []  # to store states, actions, rewards, and old log probs for this episode

            while not done:
                action, log_prob = self.pick_action(torch.tensor(state, dtype=torch.float32))
                next_state, reward, done = self.environment.step(action)

                trajectories.append((state, action, reward, next_state, done, log_prob))
                state = next_state

            self.train(trajectories)

            self.scores.append(self.environment.score)

            if i % 25 == 0 and i != 0:
                print(
                    f"Epochs: {i}, average_score: {np.mean(self.scores)}, median_score: {np.median(self.scores)}, highest_score: {np.amax(self.scores)}"
                )
                self.scores = []


if __name__ == "__main__":
    PPOAgent().ppo_learn()