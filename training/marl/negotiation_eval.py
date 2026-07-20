"""Evaluate paper Alg 6 (LLM-mediated negotiation) on delivery slot conflicts.

Rolls out the trained delivery block on held-out seeds twice: as-is
(conflicting slot claims pay the env penalty) and with a mediator that
intercepts conflicting claims before env.step and replaces them with the
negotiated assignment (failed negotiations leave the conflict standing).

Usage:
    uv run python -m training.marl.negotiation_eval --greedy
    uv run python -m training.marl.negotiation_eval --model <lm-studio-model>
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from env.training_env import ColdChainTrainingEnv
from llm.client import LLMConfig, OpenAICompatClient
from llm.mediation import SlotMediator
from training.config import FORECASTER_PATH, SEED, env_config, load_agents

NEGO_SEED = 800_000
DEFAULT_EPISODES = 30
DEFAULT_ROUNDS = 3

ARM_METRICS = ("conflict", "slot_cost", "delay", "sla_violated")


def _run_arm(
    mediator: SlotMediator | None,
    episodes: int,
    forecaster: Path | None,
    tag: str | None,
) -> dict[str, float]:
    block = list(DELIVERY_AGENTS)
    env = ColdChainTrainingEnv(env_config(NEGO_SEED, block, forecaster))
    agents = load_agents(env, block, tag)
    values: dict[str, list[float]] = defaultdict(list)
    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        while not done:
            actions = {a: agents[a].act(obs[a], explore=False) for a in agents}
            if mediator is not None:
                actions = mediator.resolve(actions, env.world_state)
            obs, _, terminated, truncated, infos = env.step(actions)
            for a in block:
                for key in ARM_METRICS:
                    values[key].append(float(infos[a][key]))
            done = terminated["__all__"] or truncated["__all__"]
    return {key: float(np.mean(values[key])) for key in ARM_METRICS}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=DEFAULT_EPISODES)
    parser.add_argument(
        "--rounds", type=int, default=DEFAULT_ROUNDS, help="negotiation limit T"
    )
    parser.add_argument(
        "--greedy",
        action="store_true",
        help="rule-based mediator instead of the LLM (no server needed)",
    )
    parser.add_argument("--model", default=None)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--forecaster", action="store_true")
    parser.add_argument(
        "--tag", default=None, help="load suffixed delivery module variant"
    )
    args = parser.parse_args()

    np.random.seed(SEED)
    torch.manual_seed(SEED)
    forecaster = FORECASTER_PATH if args.forecaster else None

    client = None
    if not args.greedy:
        overrides: dict[str, Any] = {"temperature": 0.2}
        if args.model:
            overrides["model"] = args.model
        if args.base_url:
            overrides["base_url"] = args.base_url
        client = OpenAICompatClient(LLMConfig.from_env(**overrides))

    mediator = SlotMediator(client, args.rounds)
    baseline = _run_arm(None, args.episodes, forecaster, args.tag)
    mediated = _run_arm(mediator, args.episodes, forecaster, args.tag)
    if client is not None:
        client.close()

    label = "greedy" if args.greedy else "llm"
    print(
        f"\ndelivery slot negotiation ({label} mediator, T={args.rounds}, "
        f"{args.episodes} episodes, seed {NEGO_SEED})"
    )
    print(f"{'metric':<14}{'baseline':>10}{'mediated':>10}{'delta':>9}")
    for key in ARM_METRICS:
        base, med = baseline[key], mediated[key]
        delta = f"{(med - base) / abs(base):+.0%}" if abs(base) >= 1e-9 else "   --"
        print(f"{key:<14}{base:>10.3f}{med:>10.3f}{delta:>9}")

    mean_rounds = float(np.mean(mediator.rounds)) if mediator.rounds else 0.0
    print(
        f"\nnegotiation: {mediator.events} conflicts, "
        f"{mediator.agreements} agreements (mean {mean_rounds:.1f} rounds), "
        f"{mediator.failures} failures ({mediator.errors} mediator errors), "
        f"{mediator.negotiations} negotiations run, "
        f"{mediator.cache_hits} cache hits"
    )


if __name__ == "__main__":
    main()
