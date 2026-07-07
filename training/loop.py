from __future__ import annotations

from env.training_env import ColdChainTrainingEnv
from training.agents import Agent


def collect_and_learn(
    env: ColdChainTrainingEnv,
    agents: dict[str, Agent],
    n_episodes: int,
    *,
    explore: bool = True,
    learn: bool = True,
) -> None:
    """CTDE loop: all agents act each step, store transitions, then update."""
    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False
        while not done:
            actions = {a: agents[a].act(obs[a], explore=explore) for a in agents}
            next_obs, rewards, terminated, truncated, _ = env.step(actions)
            for a in agents:
                agents[a].observe(
                    obs[a], actions[a], rewards[a], next_obs[a], terminated[a], truncated[a]
                )
                if learn:
                    agents[a].update()
            obs = next_obs
            done = terminated["__all__"] or truncated["__all__"]
