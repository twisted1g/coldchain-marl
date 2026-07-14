from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from core import config as core_config
from core.config import DELIVERY_AGENTS, OBS_FIELDS_BY_AGENT
from env.training_env import DEFAULT_MAX_STEPS, ColdChainTrainingEnv
from training.marl.agents import Agent, DDPGAgent, DQNAgent, FrozenAgent, SpoilageAgent
from training.marl.maddpg import DeliveryHandle, MADDPGDelivery

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
    **dict.fromkeys(DELIVERY_AGENTS, "MADDPG"),
}
METRIC = {
    "temperature": ("temp_deviation", "min"),
    "routing": ("route_cost", "min"),
    "spoilage": ("fn_rate", "min"),
    "inventory": ("inventory_cost", "min"),
    **dict.fromkeys(DELIVERY_AGENTS, ("delivery_cost", "min")),
}
# Random routing never reaches the target, so its per-step route_cost is not
# comparable; the trained-vs-random check uses return (delivery bonus + cost).
COMPARE_METRIC = {**METRIC, "routing": ("return", "max")}

LEARNERS = ["temperature", "routing", "spoilage", "inventory", *DELIVERY_AGENTS]

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

MADDPG_CFG: dict[str, Any] = {
    k: v for k, v in DDPG_CFG.items() if k != "noise_sigma"
} | {
    "gumbel_tau_start": 1.0,
    "gumbel_tau_end": 0.3,
    "gumbel_tau_decay_steps": 30_000,
}

ALGO_CFG = {
    "DDPG": DDPG_CFG,
    "DQN": DQN_CFG,
    "SPOILAGE_GNN": SPOILAGE_CFG,
    "MADDPG": MADDPG_CFG,
}

ARTIFACTS = Path(__file__).resolve().parent.parent / "artifacts"
MODULES_DIR = ARTIFACTS / "modules"
SPOILAGE_ENCODER_PATH = MODULES_DIR / "spoilage_gnn" / "encoder.pt"
FORECASTER_PATH = MODULES_DIR / "forecaster" / "forecaster.pt"
CURVE_CSV = ARTIFACTS / "reward_curve.csv"


def module_dir(agent: str) -> Path:
    return MODULES_DIR / agent


def env_config(
    base_seed: int, learners: list[str], forecaster: Path | None = None
) -> dict[str, Any]:
    cfg: dict[str, Any] = {
        "fruit": FRUIT,
        "max_steps": DEFAULT_MAX_STEPS,
        "base_seed": base_seed,
        "learners": list(learners),
    }
    if forecaster is not None:
        cfg["forecaster"] = forecaster
    return cfg


def _build_learner(agent: str, env: ColdChainTrainingEnv) -> Agent:
    obs_dim = int(np.prod(env.observation_space(agent).shape))
    algo = ALGO[agent]
    if algo == "DDPG":
        return DDPGAgent(obs_dim, env.action_space(agent), ALGO_CFG["DDPG"])
    if algo == "DQN":
        return DQNAgent(obs_dim, env.action_space(agent), ALGO_CFG["DQN"])
    if algo == "SPOILAGE_GNN":
        return SpoilageAgent(
            obs_dim,
            env.action_space(agent),
            ALGO_CFG["SPOILAGE_GNN"],
            SPOILAGE_ENCODER_PATH,
        )
    raise NotImplementedError(
        f"Algorithm {algo!r} for agent {agent!r} not implemented yet"
    )


def _build_delivery_group(
    env: ColdChainTrainingEnv, delivery_learners: list[str]
) -> dict[str, Agent]:
    obs_dim = int(np.prod(env.observation_space(delivery_learners[0]).shape))
    group = MADDPGDelivery(
        n=len(delivery_learners),
        obs_dim=obs_dim,
        n_slots=core_config.N_DELIVERY_WINDOWS,
        cfg=ALGO_CFG["MADDPG"],
    )
    return {name: DeliveryHandle(group, i) for i, name in enumerate(delivery_learners)}


def build_agents(env: ColdChainTrainingEnv, learners: list[str]) -> dict[str, Agent]:
    """All agents for the CTDE loop: learners get their algorithm, rest frozen.

    Delivery learners share one MADDPGDelivery group (paper Alg 5 shared critic).
    """
    delivery_learners = [a for a in AGENTS if a in learners and a in DELIVERY_AGENTS]
    delivery_agents = (
        _build_delivery_group(env, delivery_learners) if delivery_learners else {}
    )

    agents: dict[str, Agent] = {}
    for agent in AGENTS:
        if agent in delivery_agents:
            agents[agent] = delivery_agents[agent]
        elif agent in learners:
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
