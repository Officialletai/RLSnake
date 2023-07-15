#from dqn_agent import DQNAgent

# run to train the model
#DQNAgent().dqn_learn()

import numpy as np
from agent import DQNAgent
import itertools

# Define the hyperparameter combinations to search
learning_rates = [0.005, 0.0025]
discount_rates = [0.9, 0.95]
epsilon_decay_rates = [0.9975, 0.99975]
min_epsilons = [0.05]
batch_sizes = [1024, 256, 64]

# Perform grid search
results = []
best_score = float("-inf")
best_hyperparameters = {}

for learning_rate, discount_rate, epsilon_decay_rate, min_epsilon, batch_size in itertools.product(
    learning_rates, discount_rates, epsilon_decay_rates, min_epsilons, batch_sizes
):
    agent = DQNAgent(input_size=12, output_size=4)
    agent.learning_rate = learning_rate
    agent.discount_rate = discount_rate
    agent.epsilon_decay_rate = epsilon_decay_rate
    agent.min_epsilon = min_epsilon
    agent.batch_size = batch_size

    agent.dqn_learn()

    average_score = np.mean(agent.scores)
    results.append((learning_rate, discount_rate, epsilon_decay_rate, min_epsilon, batch_size, average_score))

    if average_score > best_score:
        best_score = average_score
        best_hyperparameters = {
            "learning_rate": learning_rate,
            "discount_rate": discount_rate,
            "epsilon_decay_rate": epsilon_decay_rate,
            "min_epsilon": min_epsilon,
            "batch_size": batch_size,
        }

# Print the results and best hyperparameters
for result in results:
    learning_rate, discount_rate, epsilon_decay_rate, min_epsilon, batch_size, average_score = result
    print(
        f"Hyperparameters: learning_rate={learning_rate}, discount_rate={discount_rate}, epsilon_decay_rate={epsilon_decay_rate}, min_epsilon={min_epsilon}, batch_size={batch_size}, average_score={average_score}"
    )

print("\nBest Hyperparameters:")
print(best_hyperparameters)
