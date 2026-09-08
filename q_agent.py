"""
q_agent.py
Tabular Q-learning agent. Each drone runs its OWN independent Q-learner
(fully decentralized / distributed multi-agent RL), coordinating only
through the environment's shared coverage map -- no central controller,
no shared Q-table. This is what makes the framework scalable: adding
more drones adds more independent learners, not more central state.
"""

import numpy as np
from collections import defaultdict


class QLearningAgent:
    def __init__(self, n_actions=5, alpha=0.15, gamma=0.95,
                 epsilon=1.0, epsilon_min=0.05, epsilon_decay=0.995,
                 seed=0):
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.rng = np.random.default_rng(seed)
        self.q_table = defaultdict(lambda: np.zeros(n_actions))

    def select_action(self, state):
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(0, self.n_actions))
        return int(np.argmax(self.q_table[state]))

    def update(self, state, action, reward, next_state, done):
        best_next = 0.0 if done else np.max(self.q_table[next_state])
        td_target = reward + self.gamma * best_next
        td_error = td_target - self.q_table[state][action]
        self.q_table[state][action] += self.alpha * td_error

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def policy_size(self):
        return len(self.q_table)
