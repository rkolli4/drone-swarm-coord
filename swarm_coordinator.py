"""
swarm_coordinator.py
Orchestrates N independent Q-learning agents operating in the shared
SwarmGridEnv. Coordination is implicit/distributed: agents never see
each other's Q-tables, only the shared coverage map exposed through
the environment's state encoding (nearest-unmapped-target direction).
This mirrors real distributed swarm systems where each vehicle plans
locally but reacts to a shared situational picture.
"""

from q_agent import QLearningAgent


class SwarmCoordinator:
    def __init__(self, env):
        self.env = env
        self.agents = [
            QLearningAgent(n_actions=5, seed=i) for i in range(env.n_drones)
        ]

    def run_episode(self, train=True):
        states = self.env.reset()
        done = False
        episode_reward = [0.0] * self.env.n_drones
        trajectory = {i: [self.env.positions[i]] for i in range(self.env.n_drones)}

        while not done:
            actions = [agent.select_action(s) for agent, s in zip(self.agents, states)]
            next_states, rewards, done, info = self.env.step(actions)

            if train:
                for i, agent in enumerate(self.agents):
                    agent.update(states[i], actions[i], rewards[i], next_states[i], done)

            for i in range(self.env.n_drones):
                episode_reward[i] += rewards[i]
                trajectory[i].append(self.env.positions[i])

            states = next_states

        if train:
            for agent in self.agents:
                agent.decay_epsilon()

        return {
            "episode_reward": episode_reward,
            "coverage": info["coverage"],
            "steps": self.env.steps_taken,
            "trajectory": trajectory,
        }
