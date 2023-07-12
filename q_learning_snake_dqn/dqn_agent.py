import random
import numpy as np
from dqn_model import DQN
from environment import environment
import torch
import torch.optim as optim
import torch.nn.functional as F


class DQNAgent:
    def __init__(self, input_size=12, output_size=4):
        self.input_size = input_size
        self.output_size = output_size
        self.learning_rate = 0.001
        self.discount_rate = 0.99
        self.epsilon = 1.0
        self.epsilon_decay_rate = 0.9995
        self.min_epsilon = 0.03
        self.batch_size = 32
        self.max_epochs = 1000
        self.memory = []
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DQN(input_size, output_size).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.environment = environment(display_game=False)
        self.scores = []
        self.q_values = []

    def pick_direction(self, state):
        if random.random() < self.epsilon:
            return random.choice([0, 1, 2, 3])
        else:
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.model(state_tensor)
            return torch.argmax(q_values).item()

    def remember(self, state, action, reward, new_state, done):
        self.memory.append((state, action, reward, new_state, done))

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        batch = random.sample(self.memory, self.batch_size)
        state_batch, action_batch, reward_batch, new_state_batch, done_batch = zip(*batch)
        state_batch = torch.tensor(state_batch, dtype=torch.float32).to(self.device)
        action_batch = torch.tensor(action_batch, dtype=torch.long).unsqueeze(1).to(self.device)
        reward_batch = torch.tensor(reward_batch, dtype=torch.float32).unsqueeze(1).to(self.device)
        new_state_batch = torch.tensor(new_state_batch, dtype=torch.float32).to(self.device)
        done_batch = torch.tensor(done_batch, dtype=torch.float32).unsqueeze(1).to(self.device)

        q_values = self.model(state_batch)
        self.q_values.append(q_values.detach().cpu().numpy())
        next_q_values = self.model(new_state_batch)
        max_next_q_values = torch.max(next_q_values, dim=1, keepdim=True)[0]
        target_q_values = reward_batch + self.discount_rate * max_next_q_values * (1 - done_batch)

        predicted_q_values = torch.gather(q_values, 1, action_batch)

        loss = F.mse_loss(predicted_q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.min_epsilon:
            self.epsilon *= self.epsilon_decay_rate

    def dqn_learn(self):
        for i in range(0, self.max_epochs):
            self.environment = environment(display_game=False)
            self.environment.uneventful_move = 0
            self.environment.epoch = i

            if i % 25 == 0 and i != 0:
                print(
                    f"Epochs: {i}, average_score: {np.mean(self.scores)}, median_score: {np.median(self.scores)}, highest_score: {np.amax(self.scores)}, epsilon_value: {self.epsilon}, avg_q_value: {np.mean(self.q_values)}"
                )
                self.scores = []

            current_state = self.environment.get_state()
            self.epsilon = max(self.epsilon * self.epsilon_decay_rate, self.min_epsilon)
            finished = False

            while not finished:
                action = self.pick_direction(current_state)
                new_state, reward, finished = self.environment.step(action)

                if self.environment.uneventful_move == 600:
                    reward = -10

                self.remember(current_state, action, reward, new_state, finished)
                current_state = new_state

                if self.environment.uneventful_move == 600:
                    break

            self.scores.append(self.environment.score)
            self.replay()
