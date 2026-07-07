from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from core.config import OBS_FIELDS_BY_AGENT
from env.training_env import DEFAULT_MAX_STEPS, ColdChainTrainingEnv
from training.agents import Agent, DDPGAgent, DQNAgent, FrozenAgent, SpoilageAgent

SEED = 0
NUM_ITERATIONS = 150
EPISODES_PER_ITERATION = 40
EVAL_EPISODES = 30

AGENTS = list(OBS_FIELDS_BY_AGENT)
ALGO = {
    "temperature": "DDPG",
    "routing": "DQN",
    "spoilage": "SPOILAGE_GNN",
    "inventory": "DDPG",
}
METRIC = {
    "temperature": ("temp_deviation", "min"),
    "routing": ("route_cost", "min"),
    "spoilage": ("fn_rate", "min"),
    "inventory": ("inventory_cost", "min"),
}

LEARNERS = ["temperature", "routing", "spoilage", "inventory"]

FRUIT = "banana"
TRAIN_SEED = 1000
EVAL_SEED = 90_000
COMPARE_SEED = 500_000

DDPG_CFG: dict[str, Any] = {
    "hidden": [64, 64],
    "lr": 3e-4,
    "gamma": 0.99,
    "tau": 0.005,
    "batch_size": 256,
    "buffer_capacity": 100_000,
    "warmup": 256,
    "noise_sigma": 0.1,
}

DQN_CFG: dict[str, Any] = {
    "hidden": [64, 64],
    "lr": 1e-3,
    "gamma": 0.99,
    "tau": 0.005,
    "batch_size": 256,
    "buffer_capacity": 100_000,
    "warmup": 256,
    "eps_start": 1.0,
    "eps_end": 0.05,
    "eps_decay_steps": 2000,
}

SPOILAGE_CFG: dict[str, Any] = dict(DDPG_CFG)

ALGO_CFG = {"DDPG": DDPG_CFG, "DQN": DQN_CFG, "SPOILAGE_GNN": SPOILAGE_CFG}

ARTIFACTS = Path(__file__).resolve().parent.parent / "artifacts"
MODULES_DIR = ARTIFACTS / "modules"
SPOILAGE_ENCODER_PATH = MODULES_DIR / "spoilage_gnn" / "encoder.pt"
CURVE_CSV = ARTIFACTS / "reward_curve.csv"


def module_dir(agent: str) -> Path:
    return MODULES_DIR / agent


def env_config(base_seed: int, learners: list[str]) -> dict[str, Any]:
    return {"fruit": FRUIT, "max_steps": DEFAULT_MAX_STEPS, "base_seed": base_seed, "learners": list(learners)}


def _build_learner(agent: str, env: ColdChainTrainingEnv) -> Agent:
    obs_dim = int(np.prod(env.observation_space(agent).shape))
    algo = ALGO[agent]
    if algo == "DDPG":
        return DDPGAgent(obs_dim, env.action_space(agent), ALGO_CFG["DDPG"])
    if algo == "DQN":
        return DQNAgent(obs_dim, env.action_space(agent), ALGO_CFG["DQN"])
    if algo == "SPOILAGE_GNN":
        return SpoilageAgent(
            obs_dim, env.action_space(agent), ALGO_CFG["SPOILAGE_GNN"], SPOILAGE_ENCODER_PATH
        )
    raise NotImplementedError(f"Algorithm {algo!r} for agent {agent!r} not implemented yet")


def build_agents(env: ColdChainTrainingEnv, learners: list[str]) -> dict[str, Agent]:
    """All agents for the CTDE loop: learners get their algorithm, rest frozen."""
    agents: dict[str, Agent] = {}
    for agent in AGENTS:
        if agent in learners:
            agents[agent] = _build_learner(agent, env)
        else:
            agents[agent] = FrozenAgent(env.action_space(agent))
    return agents


def load_agents(env: ColdChainTrainingEnv, learners: list[str]) -> dict[str, Agent]:
    """Frozen backdrop with trained learner modules loaded from artifacts."""
    agents = build_agents(env, learners)
    for agent in learners:
        agents[agent].load(module_dir(agent))
    return agents
