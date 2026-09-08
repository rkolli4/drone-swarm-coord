"""
environment.py
Geospatial grid-world environment for multi-drone swarm coordination.

The grid represents a post-disaster area. Each cell can be:
  - normal terrain
  - infrastructure POI (road junction, building, utility node) that needs mapping
  - obstacle (debris, no-fly zone)
  - damage zone (higher survey priority -- simulates disaster surveillance)

Drones move through the grid, and the environment tracks which cells have
been surveyed by ANY drone (shared "world map"), enabling emergent
division of labor between distributed agents.
"""

import numpy as np

ACTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1), (0, 0)]  # N, S, W, E, hover
ACTION_NAMES = ["N", "S", "W", "E", "HOVER"]

CELL_NORMAL = 0
CELL_POI = 1
CELL_OBSTACLE = 2
CELL_DAMAGE = 3


class SwarmGridEnv:
    def __init__(self, size=15, n_drones=4, n_poi=12, n_obstacles=10,
                 n_damage=8, seed=42):
        self.size = size
        self.n_drones = n_drones
        self.rng = np.random.default_rng(seed)

        self.grid = np.zeros((size, size), dtype=np.int8)
        self._scatter(CELL_OBSTACLE, n_obstacles)
        self._scatter(CELL_POI, n_poi)
        self._scatter(CELL_DAMAGE, n_damage)

        # Shared coverage map: has ANY drone surveyed this cell?
        self.mapped = np.zeros((size, size), dtype=bool)
        self.total_targets = int(np.sum(self.grid != CELL_NORMAL))

        self.positions = self._init_positions()
        self.last_actions = [4] * n_drones  # start hovering
        self.steps_taken = 0

    def _scatter(self, cell_type, count):
        placed = 0
        while placed < count:
            x, y = self.rng.integers(0, self.size, size=2)
            if self.grid[x, y] == CELL_NORMAL:
                self.grid[x, y] = cell_type
                placed += 1

    def _init_positions(self):
        pos = []
        for _ in range(self.n_drones):
            while True:
                x, y = self.rng.integers(0, self.size, size=2)
                if self.grid[x, y] != CELL_OBSTACLE and (x, y) not in pos:
                    pos.append((int(x), int(y)))
                    break
        return pos

    def reset(self):
        self.mapped[:] = False
        self.positions = self._init_positions()
        self.last_actions = [4] * self.n_drones
        self.steps_taken = 0
        return self.get_states()

    def get_state(self, agent_id):
        """Compact state: compass direction to nearest unmapped target +
        immediate obstacle/edge occupancy in each of the 4 cardinal cells +
        the agent's previous action. The previous-action term is what makes
        this Markovian enough for tabular Q-learning to avoid N<->S / E<->W
        oscillation cycles that pure position-based states are prone to."""
        x, y = self.positions[agent_id]
        target_dir = self._nearest_unmapped_direction(x, y)
        blocked = tuple(self._is_blocked(x + dx, y + dy) for dx, dy in ACTIONS[:4])
        return (target_dir,) + blocked + (self.last_actions[agent_id],)

    def get_states(self):
        return [self.get_state(i) for i in range(self.n_drones)]

    def _nearest_unmapped_direction(self, x, y):
        """Coarse compass direction (0-8) toward the nearest un-surveyed target cell."""
        targets = np.argwhere((self.grid != CELL_NORMAL) & (~self.mapped))
        if len(targets) == 0:
            return 8  # "done" direction
        dists = np.abs(targets[:, 0] - x) + np.abs(targets[:, 1] - y)
        tx, ty = targets[np.argmin(dists)]
        dx, dy = np.sign(tx - x), np.sign(ty - y)
        mapping = {(-1, -1): 0, (-1, 0): 1, (-1, 1): 2,
                   (0, -1): 3, (0, 0): 4, (0, 1): 5,
                   (1, -1): 6, (1, 0): 7, (1, 1): 8}
        return mapping.get((int(dx), int(dy)), 4)

    def _is_blocked(self, x, y):
        if not (0 <= x < self.size and 0 <= y < self.size):
            return True
        return self.grid[x, y] == CELL_OBSTACLE

    def _nearest_unmapped_dist(self, x, y):
        targets = np.argwhere((self.grid != CELL_NORMAL) & (~self.mapped))
        if len(targets) == 0:
            return 0
        dists = np.abs(targets[:, 0] - x) + np.abs(targets[:, 1] - y)
        return int(np.min(dists))

    def step(self, actions):
        """actions: list of action indices, one per drone."""
        rewards = [0.0] * self.n_drones
        new_positions = list(self.positions)
        old_dists = [self._nearest_unmapped_dist(x, y) for x, y in self.positions]

        for i, a in enumerate(actions):
            dx, dy = ACTIONS[a]
            x, y = self.positions[i]
            nx, ny = x + dx, y + dy
            if not (0 <= nx < self.size and 0 <= ny < self.size):
                rewards[i] -= 2.0  # boundary penalty
                continue
            if self.grid[nx, ny] == CELL_OBSTACLE:
                rewards[i] -= 3.0  # obstacle / no-fly penalty
                continue
            new_positions[i] = (nx, ny)

        # collision penalty (two drones landing on same cell)
        occ = {}
        for i, p in enumerate(new_positions):
            occ.setdefault(p, []).append(i)
        for p, idxs in occ.items():
            if len(idxs) > 1:
                for i in idxs:
                    rewards[i] -= 5.0

        self.positions = new_positions
        self.last_actions = list(actions)

        for i, (x, y) in enumerate(self.positions):
            rewards[i] -= 0.1  # step cost -> encourages efficient paths

            # potential-based shaping: reward getting closer to the nearest
            # un-surveyed target, penalize moving away from it
            new_dist = self._nearest_unmapped_dist(x, y)
            rewards[i] += 0.5 * (old_dists[i] - new_dist)

            cell = self.grid[x, y]
            if cell != CELL_NORMAL and not self.mapped[x, y]:
                self.mapped[x, y] = True
                bonus = 15.0 if cell == CELL_DAMAGE else 10.0  # prioritize disaster zones
                rewards[i] += bonus

        self.steps_taken += 1
        coverage = np.sum(self.mapped) / max(self.total_targets, 1)
        done = coverage >= 1.0 or self.steps_taken >= 300

        return self.get_states(), rewards, done, {"coverage": coverage}

    def coverage_ratio(self):
        return float(np.sum(self.mapped) / max(self.total_targets, 1))
