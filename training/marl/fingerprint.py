"""Behavior fingerprint: trained-block metrics + exact obs-stream hashes.

Run before and after a refactor; diff the output. Any numeric drift means
the refactor changed behavior.
"""

import hashlib

import numpy as np
import torch

from core.config import (
    DELIVERY_AGENTS,
    INVENTORY_AGENTS,
    ROUTING_AGENTS,
    SPOILAGE_AGENTS,
    TEMPERATURE_AGENTS,
)
from env.training_env import ColdChainTrainingEnv
from llm.scenarios import load_bank
from training.config import (
    COMPARE_METRIC,
    COMPARE_SEED,
    build_agents,
    env_config,
    load_agents,
)
from training.marl.evaluate import rollout

np.random.seed(0)
torch.manual_seed(0)

BLOCKS = {
    "temperature": list(TEMPERATURE_AGENTS),
    "routing": list(ROUTING_AGENTS),
    "spoilage": list(SPOILAGE_AGENTS),
    "inventory": list(INVENTORY_AGENTS),
    "delivery": list(DELIVERY_AGENTS),
}

print("== trained rollout metrics (COMPARE_SEED, 2 episodes) ==")
for name, block in BLOCKS.items():
    metric_key, _ = COMPARE_METRIC[block[0]]
    env = ColdChainTrainingEnv(env_config(COMPARE_SEED, block))
    try:
        agents = load_agents(env, block)
    except FileNotFoundError:
        print(f"{name:<12} no checkpoint, skipped")
        continue
    vals = [rollout(env, agents, a, 2, metric_key) for a in block]
    ret = float(np.mean([v[0] for v in vals]))
    met = float(np.mean([v[1] for v in vals]))
    print(f"{name:<12} return={ret:.6f} {metric_key}={met:.6f}")

print("== frozen obs-stream hash, clean episode ==")
env = ColdChainTrainingEnv(env_config(4242, list(TEMPERATURE_AGENTS)))
agents = build_agents(env, [])
obs, _ = env.reset()
h = hashlib.sha256()
done = False
while not done:
    for a in sorted(obs):
        h.update(obs[a].tobytes())
    actions = {a: agents[a].act(obs[a], explore=False) for a in agents}
    obs, rewards, term, trunc, _ = env.step(actions)
    h.update(np.float64(sum(rewards.values())).tobytes())
    done = term["__all__"] or trunc["__all__"]
print("clean:", h.hexdigest())

print("== frozen obs-stream hash, every 7th scenario ==")
bank = load_bank("data/scenarios/bank.json")
cfg = env_config(4242, [*TEMPERATURE_AGENTS, "inventory_0"])
cfg["scenario_bank"] = "data/scenarios/bank.json"
env = ColdChainTrainingEnv(cfg)
agents = build_agents(env, [])
for scenario in bank[::7]:
    obs, _ = env.reset(options={"scenario_id": scenario.id})
    h = hashlib.sha256()
    done = False
    while not done:
        for a in sorted(obs):
            h.update(obs[a].tobytes())
        actions = {a: agents[a].act(obs[a], explore=False) for a in agents}
        obs, rewards, term, trunc, _ = env.step(actions)
        h.update(np.float64(sum(rewards.values())).tobytes())
        done = term["__all__"] or trunc["__all__"]
    print(f"{scenario.id}: {h.hexdigest()[:16]}")
