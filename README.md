# Cold-Chain MARL

Reproduction of the multi-agent RL core from Khanna et al., *Generative AI and
Blockchain-Integrated Multi-Agent Framework for Resilient and Sustainable Fruit
Cold-Chain Logistics* (Foods 2025, 14(17):3004). Five heterogeneous agents —
routing, temperature, spoilage, inventory, delivery — trained with **centralized
training, decentralized execution (CTDE)** on a synthetic farm-to-retail
cold-chain environment, fidelity-first against the paper's Algorithm boxes 1-5.

## Status: MARL core complete

All five agents live (verified trained-vs-random on held-out seeds, full joint run):

| Agent | Algorithm (paper box) | Action | Metric | Trained vs random |
|-------|----------------------|--------|--------|-------------------|
| temperature | DDPG (Alg 2) | continuous setpoint | temp_deviation | +98% |
| routing | DQN (Alg 1, see deviations) | discrete next-hop | route_cost | +60% |
| spoilage | frozen GraphSAGE encoder + DDPG head (Alg 3) | continuous risk threshold | fn_rate | +81% |
| inventory | DDPG (Alg 4) | continuous order qty | inventory_cost | +76% |
| delivery | MADDPG, shared centralized critic (Alg 5) | discrete schedule slot × 3 vehicles | delivery_cost | +43% |

Paper-wide mechanisms, implemented across all agents:

- **Dynamic context-aware Pareto weights** `ω_j = c_j / Σ c_k` in every reward
  (each Alg box's "Context-Aware Weights"); static priority coefficients folded
  into the weighted cost terms.
- **Shared intention buffer + coordination penalty ρ** (`core/intention.py`):
  declare → detect conflicts → ρ → clear, each step. Delivery vehicles are the
  live conflict source (slot collisions); trained vehicles learn distinct slots
  (conflict rate 0.00).
- **CTDE**: every agent acts on local observations; only delivery uses a shared
  critic `Q(joint_obs, joint_act)`, per Alg 5.

## Run

```
uv run python -m training.train                                   # full joint run (all 5)
uv run python -m training.train --agents temperature routing      # subset
uv run python -m training.train --agents delivery_0 delivery_1 delivery_2   # delivery group
uv run python -m training.pretrain_spoilage                       # regen GNN encoder artifact
uv run ruff check core env training data
```

Training writes learning curves to `artifacts/reward_curve.csv` and agent modules
to `artifacts/modules/`, then prints trained-vs-random margins per learner
(delivery evaluated as one block — the three vehicles share one MADDPG group).

## Notebooks

- `notebooks/training_report.ipynb` — per-learner curves, trained-vs-random bars,
  delivery coordination panel (slot histograms, conflict/SLA rates).
- `notebooks/agent_behavior.ipynb` — one greedy episode, **trained vs random on
  the same seed** side by side: temperature-vs-band, route on the supply graph
  (with transit-shortest and Alg1-cost-shortest baselines + path-cost analysis),
  spoilage prediction tracking, inventory control, delivery slot coordination,
  cumulative returns.
- `notebooks/dataset_report.ipynb` — offline synthetic dataset summary.

## Architecture

- `core/` — framework-agnostic domain logic: config, state (shipment + per-vehicle
  `VehicleState`), dynamics, Arrhenius spoilage, supply graph, observations,
  spaces, disruption noise, `IntentionBuffer`.
- `env/` — `ColdChainParallelEnv` (PettingZoo) and `ColdChainTrainingEnv`
  (per-agent reward methods live on the env class, including the dynamic-Pareto
  weighting).
- `training/` — `agents.py` (Agent protocol, `FrozenAgent`, `RandomAgent`,
  `DDPGAgent`, `DQNAgent`, `SpoilageAgent`), `maddpg.py` (`MADDPGDelivery`
  shared-critic group + per-vehicle `DeliveryHandle`), `gnn.py` +
  `pretrain_spoilage.py` (GraphSAGE encoder), `loop.py` (CTDE loop), `config.py`
  (per-agent algorithm registry), `evaluate.py`, `train.py`.
- `data/` — offline synthetic dataset (generation, schema, loader).

Stack: PyTorch + **TorchRL** loss modules / replay buffers (`DDPGLoss`, `DQNLoss`,
`TensorDictReplayBuffer`, `SoftUpdate`) plus a hand-built MADDPG (TorchRL has no
turnkey MADDPG loss), driven by a hand-written CTDE loop. torch-geometric for the
spoilage GNN. No Ray/RLlib.

Determinism: same seed → bit-identical runs. Inventory demand and delivery use
isolated RNG streams (dedicated generator / torch RNG only), so adding an agent
never perturbs the others' worlds.

## Deviations from the paper (documented)

- **routing**: paper uses *tabular* Q-learning (Alg 1); impl uses **DQN** —
  tabular discretization of the routing observation is impractical.
- **Non-paper reward shaping** (load-bearing, kept outside the Pareto term):
  routing delivery bonus (Alg 1 reward is pure penalty; the bonus is the only
  positive signal preventing wait-forever), temperature deviation term + step
  penalty (Alg 2 lists only energy + spoilage).
- **Dynamic-weight form**: literal `−Σ ω_j·raw_j` degenerates in sim units
  (emissions dominate); implemented as `−Σ ω_j·c_j` with the static priority
  coefficients folded into `c_j = α_j·raw_j`.
- **Delivery scope**: n vehicles are a scoped scheduling group over the shared
  single-shipment world (retailer tasks + scenario-derived routes), not a
  multi-shipment fleet — matches Alg 5's "agents = vehicle controllers" without a
  core sim rewrite. Gumbel-softmax discrete actors (paper does not specify the
  discrete-actor mechanism).
- **Intention buffer scope**: the paper's buffer coordinates instances of the
  same agent type; in this scoped world only delivery is multi-instance, so it is
  the sole live conflict source — the other four declare into the buffer but
  cannot conflict by definition.
- Known deferred items: temperature observation lacks `T_ambient` + `fruit_type`
  (paper state), inventory observation carries a redundant static
  `predicted_demand`, routing action aliasing (`idx % out_edges`; masked DQN is
  the fix).

## Not in scope yet (next phases)

GenAI layer (transformer demand forecast + LLM disruption/negotiation, Alg 6-7),
blockchain layer (Solidity contracts, Alg 8-18), model serving + containerization,
and final integration against the paper's claimed numbers (−50% spoilage, −35%
energy, −25% emissions, −30% travel).
