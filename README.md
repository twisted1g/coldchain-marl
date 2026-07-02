# Cold-Chain MARL

Reproduction of the multi-agent RL core from Khanna et al., *Generative AI and
Blockchain-Integrated Multi-Agent Framework for Resilient and Sustainable Fruit
Cold-Chain Logistics* (Foods 2025). Five modular agents — routing, temperature,
spoilage, inventory, delivery — trained with **centralized training, decentralized
execution (CTDE)** on a synthetic farm-to-retail cold-chain environment.

## Status

Trainable agents (verified, trained-vs-random on held-out seeds):

| Agent | Algorithm | Action | Metric | Trained vs random |
|-------|-----------|--------|--------|-------------------|
| temperature | DDPG | continuous setpoint | temp_deviation | ~+70% lower deviation |
| routing | DQN | discrete next-hop | route_cost | ~+20% lower cost |

Remaining agents (spoilage, inventory, delivery) are frozen (fixed-action) until
unfrozen one at a time.

## Run

```
uv run python -m training.train --agents temperature            # single agent
uv run python -m training.train --agents temperature routing    # joint CTDE run
uv run ruff check core env training data
```
Then open `notebooks/training_report.ipynb` for per-learner curves and
trained-vs-random checks (the notebook reads which agents were trained from the
saved curve, so it works for single or joint runs).

## Architecture

- `core/` — framework-agnostic domain logic (config, state, dynamics, spoilage,
  graph, observations, spaces, noise).
- `env/` — `ColdChainParallelEnv` (PettingZoo) and `ColdChainTrainingEnv`
  (per-agent reward shaping; reward methods live on the env class).
- `training/` — training only: `agents.py` (Agent protocol, `FrozenAgent`,
  `RandomAgent`, `DDPGAgent`, `DQNAgent`), `loop.py` (CTDE loop), `config.py`
  (per-agent algorithm registry), `evaluate.py`, `train.py`.

Stack: PyTorch + **TorchRL** loss modules / replay buffers (`DDPGLoss`,
`DQNLoss`, `TensorDictReplayBuffer`, `SoftUpdate`) driven by a hand-written CTDE
loop. No Ray/RLlib.

## Deviations from the paper

Intentional, for runnability on a single shared training stack:

- **Stack**: paper describes five separate per-agent algorithms coordinated under
  CTDE. Implemented as TorchRL loss modules inside one custom CTDE loop (not Ray
  RLlib — RLlib's single-Learner-per-Algorithm model cannot host heterogeneous
  algorithms cleanly).
- **routing**: paper uses *tabular* Q-learning (Algorithm 1); impl uses **DQN**
  (deep) for compatibility with the shared TorchRL stack.
- **temperature**: paper uses DDPG (Algorithm 2) — matched.
- **Not yet implemented** (paper features, left as seams): shared intention buffer
  + conflict resolution, shared critic for delivery, GNN encoder for spoilage,
  dynamic context-aware Pareto reward weights (`# PHASE 4`), blockchain, LLM
  negotiation, generative-AI scenario simulation.
