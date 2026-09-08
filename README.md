# Distributed Multi-Agent Q-Learning for Drone Swarm Coordination

A working framework for autonomous drone swarm coordination using **independent
(fully decentralized) tabular Q-learning**, applied to path optimization,
geospatial data collection, infrastructure mapping, and post-disaster
surveillance analytics.

## Architecture

| File | Role |
|---|---|
| `environment.py` | Geospatial grid-world: infrastructure POIs, obstacles, disaster damage zones, shared coverage map |
| `q_agent.py` | Independent tabular Q-learning agent (one per drone, no shared Q-table) |
| `swarm_coordinator.py` | Runs N agents against the shared environment, no central controller |
| `train.py` | Training loop + metrics logging |
| `analytics.py` | Post-disaster analytics: heatmaps, mission maps, survey reports |

**Distributed design**: each drone owns its own Q-table and never observes
other drones' internal state. Coordination is *implicit* — agents only
interact through the shared coverage map exposed via each agent's state
(direction to the nearest un-surveyed target). This is what makes the
approach scalable: adding drones adds independent learners, not additional
central state, and drones naturally partition the area between themselves
as targets near one drone get claimed before another reaches them (visible
in `mission_map.png` — each drone's path clusters in a distinct region).

**Reward structure**:
- `+10` for mapping a new infrastructure POI, `+15` for a damage zone (prioritizes disaster-critical areas)
- `-0.1` step cost (drives path efficiency)
- `-5` collision penalty, `-3` obstacle penalty, `-2` boundary penalty
- Potential-based shaping (`+0.5 × distance closed to nearest un-mapped target`) — this is what makes sparse mapping rewards learnable at all with a tabular method

**State representation** — deliberately compact for tabular convergence:
`(compass direction to nearest target, 4 cardinal obstacle/edge flags, previous action)`.
Including the previous action was necessary in practice: without it, the
learned greedy policy fell into N↔S / E↔W oscillation cycles (see below).

## Results

Training for 1,200 episodes on a 15×15 grid with 4 drones, 12 POIs, 8 damage
zones, 10 obstacles:
- Swarm total reward converges from ≈ -400 to a stable positive plateau (see `training_diagnostics.png`)
- Best-of-10 greedy evaluation runs: **~50-65% area coverage**, **~67-100% POI mapping**, **~75% damage-zone survey completion** within the step budget

## Known limitation (and why it's left in, not hidden)

Independent Q-learning across multiple agents is **non-stationary**: as
other drones map cells, the "nearest target" each agent sees shifts under
it, so each agent is chasing a moving target from a learning-theory
standpoint. This is a well-known real challenge in decentralized MARL,
not a bug — a naive fully-greedy policy without the anti-oscillation state
term collapsed to ~15% coverage; adding it recovered ~35-65%. For
production-grade coordination at larger scale, the standard next step is:

- Replace tabular Q-tables with a shared-weight **DQN** (function
  approximation generalizes across positions instead of aliasing them)
- Move to **CTDE** (centralized training, decentralized execution) so
  agents can condition on a joint value function during training while
  still acting independently at deployment
- Add explicit auction/bidding-based task allocation as a coordination
  layer on top of RL for hard guarantees on non-overlapping coverage

## Running it

```bash
pip install numpy matplotlib
python3 train.py       # train + print metrics only
python3 analytics.py   # train + generate heatmaps, mission map, survey report
```

Outputs from `analytics.py`: `training_diagnostics.png`, `mission_map.png`,
`survey_report.txt`.

## Extending

- **Scale up drones**: increase `n_drones` in `SwarmGridEnv` — no other code changes needed, since each drone's Q-table is independent
- **Real geospatial data**: swap the synthetic grid in `environment.py` for a real raster (e.g. rasterized OSM building footprints / damage-assessment tiles) — the state/reward machinery is agnostic to how POIs are sourced
- **Continuous coordinates**: replace the tabular `q_agent.py` with a DQN using a small MLP over `(x, y, nearest-target vector, local obstacle map)`
