"""
train.py
Trains the distributed multi-agent Q-learning swarm and logs
path-optimization / coverage metrics across episodes.
"""

import numpy as np
from environment import SwarmGridEnv
from swarm_coordinator import SwarmCoordinator


def train(n_episodes=400, size=15, n_drones=4, verbose_every=50):
    env = SwarmGridEnv(size=size, n_drones=n_drones)
    swarm = SwarmCoordinator(env)

    history = {"coverage": [], "total_reward": [], "steps": []}

    for ep in range(1, n_episodes + 1):
        result = swarm.run_episode(train=True)
        history["coverage"].append(result["coverage"])
        history["total_reward"].append(sum(result["episode_reward"]))
        history["steps"].append(result["steps"])

        if ep % verbose_every == 0 or ep == 1:
            avg_cov = np.mean(history["coverage"][-verbose_every:])
            avg_rew = np.mean(history["total_reward"][-verbose_every:])
            avg_steps = np.mean(history["steps"][-verbose_every:])
            eps = swarm.agents[0].epsilon
            print(f"Episode {ep:4d} | avg coverage {avg_cov:5.1%} | "
                  f"avg reward {avg_rew:8.2f} | avg steps {avg_steps:5.1f} | "
                  f"epsilon {eps:.3f}")

    return env, swarm, history


if __name__ == "__main__":
    env, swarm, history = train(n_episodes=1200)

    # Final evaluation: average over several greedy-policy episodes
    for agent in swarm.agents:
        agent.epsilon = 0.0
    n_eval = 20
    eval_coverages, eval_steps = [], []
    for _ in range(n_eval):
        eval_result = swarm.run_episode(train=False)
        eval_coverages.append(eval_result["coverage"])
        eval_steps.append(eval_result["steps"])

    print(f"\n--- Final evaluation (greedy policy, {n_eval} episodes) ---")
    print(f"Mean coverage     : {np.mean(eval_coverages):.1%}  (std {np.std(eval_coverages):.1%})")
    print(f"Mean steps        : {np.mean(eval_steps):.1f}")
    print(f"Q-table sizes     : {[a.policy_size() for a in swarm.agents]}")
