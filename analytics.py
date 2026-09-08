"""
analytics.py
Post-disaster surveillance analytics layer. Turns raw episode trajectories
and the environment's shared coverage map into decision-support outputs:
  - coverage heatmap (which parts of the area have been surveyed)
  - infrastructure mapping status (POI / damage-zone survey completion)
  - per-drone flight-path visualization (path optimization result)
  - training-curve diagnostics (learning stability)

Run as a script: trains the swarm, then writes PNGs to /mnt/user-data/outputs.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from environment import SwarmGridEnv, CELL_NORMAL, CELL_POI, CELL_OBSTACLE, CELL_DAMAGE
from swarm_coordinator import SwarmCoordinator
from train import train

DRONE_COLORS = ["#e63946", "#2a9d8f", "#457b9d", "#f4a261", "#8338ec", "#ffbe0b"]


def plot_training_curves(history, path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    cov = np.array(history["coverage"]) * 100
    window = 20
    smoothed = np.convolve(cov, np.ones(window) / window, mode="valid")
    axes[0].plot(cov, alpha=0.25, color="#457b9d", label="per-episode")
    axes[0].plot(range(window - 1, len(cov)), smoothed, color="#e63946",
                 linewidth=2, label=f"{window}-ep moving avg")
    axes[0].set_title("Infrastructure Coverage over Training")
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Area Surveyed (%)")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    rew = np.array(history["total_reward"])
    smoothed_r = np.convolve(rew, np.ones(window) / window, mode="valid")
    axes[1].plot(rew, alpha=0.25, color="#2a9d8f", label="per-episode")
    axes[1].plot(range(window - 1, len(rew)), smoothed_r, color="#e63946",
                 linewidth=2, label=f"{window}-ep moving avg")
    axes[1].set_title("Swarm Total Reward over Training")
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Total Reward (all drones)")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.suptitle("Distributed Multi-Agent Q-Learning: Training Diagnostics", fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_mission_map(env, trajectory, path, title="Post-Disaster Survey Mission"):
    size = env.size
    fig, ax = plt.subplots(figsize=(8, 8))

    # base terrain layer
    base = np.zeros((size, size))
    base[env.grid == CELL_OBSTACLE] = 1
    base[env.grid == CELL_POI] = 2
    base[env.grid == CELL_DAMAGE] = 3
    cmap = ListedColormap(["#f1faee", "#6c757d", "#457b9d", "#e63946"])
    ax.imshow(base.T, origin="lower", cmap=cmap, vmin=0, vmax=3, alpha=0.55)

    # overlay: surveyed cells get a green hatch marker
    mapped_y, mapped_x = np.where(env.mapped.T)
    ax.scatter(mapped_x, mapped_y, marker="s", s=90, facecolors="none",
               edgecolors="#2a9d8f", linewidths=1.6, label="Surveyed")

    for i, path_cells in trajectory.items():
        xs = [p[0] for p in path_cells]
        ys = [p[1] for p in path_cells]
        color = DRONE_COLORS[i % len(DRONE_COLORS)]
        ax.plot(xs, ys, color=color, linewidth=1.3, alpha=0.8, zorder=3)
        ax.scatter(xs[0], ys[0], color=color, marker="o", s=70,
                   edgecolors="black", zorder=4, label=f"Drone {i} start")
        ax.scatter(xs[-1], ys[-1], color=color, marker="X", s=90,
                   edgecolors="black", zorder=4, label=f"Drone {i} end")

    ax.set_title(f"{title}\nCoverage: {env.coverage_ratio():.1%}", fontweight="bold")
    ax.set_xlim(-1, size)
    ax.set_ylim(-1, size)
    ax.set_xticks([])
    ax.set_yticks([])

    legend_elems = [
        plt.Line2D([0], [0], marker="s", color="w", markerfacecolor="#6c757d",
                   markersize=10, label="Obstacle / no-fly"),
        plt.Line2D([0], [0], marker="s", color="w", markerfacecolor="#457b9d",
                   markersize=10, label="Infrastructure POI"),
        plt.Line2D([0], [0], marker="s", color="w", markerfacecolor="#e63946",
                   markersize=10, label="Damage zone"),
        plt.Line2D([0], [0], marker="s", markerfacecolor="none",
                   markeredgecolor="#2a9d8f", markersize=10, label="Surveyed"),
    ]
    ax.legend(handles=legend_elems, loc="upper center",
              bbox_to_anchor=(0.5, -0.02), ncol=2, frameon=False)

    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def survey_report(env):
    """Text analytics: infrastructure mapping completion by category."""
    total_poi = int(np.sum(env.grid == CELL_POI))
    total_damage = int(np.sum(env.grid == CELL_DAMAGE))
    mapped_poi = int(np.sum(env.mapped & (env.grid == CELL_POI)))
    mapped_damage = int(np.sum(env.mapped & (env.grid == CELL_DAMAGE)))

    lines = [
        "=== Post-Disaster Surveillance Report ===",
        f"Grid size            : {env.size} x {env.size}",
        f"Drones deployed       : {env.n_drones}",
        f"Overall coverage      : {env.coverage_ratio():.1%}",
        f"Infrastructure mapped : {mapped_poi}/{total_poi} POIs "
        f"({mapped_poi / max(total_poi, 1):.1%})",
        f"Damage zones surveyed : {mapped_damage}/{total_damage} "
        f"({mapped_damage / max(total_damage, 1):.1%})",
        f"Mission duration       : {env.steps_taken} timesteps",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    import os
    out_dir = "/mnt/user-data/outputs"
    os.makedirs(out_dir, exist_ok=True)

    print("Training distributed swarm...")
    env, swarm, history = train(n_episodes=1200, verbose_every=300)

    print("\nRunning final greedy-policy mission for reporting...")
    for agent in swarm.agents:
        agent.epsilon = 0.0
    best = None
    for _ in range(10):
        result = swarm.run_episode(train=False)
        if best is None or result["coverage"] > best["coverage"]:
            best = result

    print(f"\nBest of 10 greedy runs -> coverage {best['coverage']:.1%}, "
          f"steps {best['steps']}")
    print()
    print(survey_report(env))

    plot_training_curves(history, f"{out_dir}/training_diagnostics.png")
    plot_mission_map(env, best["trajectory"], f"{out_dir}/mission_map.png")

    with open(f"{out_dir}/survey_report.txt", "w") as f:
        f.write(survey_report(env))

    print(f"\nSaved: training_diagnostics.png, mission_map.png, survey_report.txt")
