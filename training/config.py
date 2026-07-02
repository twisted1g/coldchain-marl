from __future__ import annotations

from pathlib import Path

from ray.rllib.algorithms.sac import SACConfig
from ray.rllib.core.rl_module.multi_rl_module import MultiRLModuleSpec
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.tune.registry import register_env

from core.config import OBS_FIELDS_BY_AGENT
from env.training_env import ColdChainTrainingEnv
from training.module import FixedActionRLModule

SEED = 0
NUM_ITERATIONS = 25
EVAL_EPISODES = 10

AGENTS = list(OBS_FIELDS_BY_AGENT)
# PHASE 3b: add continuous-action agents here (e.g. "inventory") to train them.
LEARNERS = ["temperature"]
FROZEN = [a for a in AGENTS if a not in LEARNERS]
ENV_NAME = "coldchain_training"

ENV_CONFIG = {"fruit": "strawberry", "max_steps": 20, "base_seed": 1000, "learners": LEARNERS}
EVAL_ENV_CONFIG = {**ENV_CONFIG, "base_seed": 90_000}
COMPARE_ENV_CONFIG = {**ENV_CONFIG, "base_seed": 500_000}

ARTIFACTS = Path(__file__).resolve().parent.parent / "artifacts"
MODULES_DIR = ARTIFACTS / "modules"
CURVE_CSV = ARTIFACTS / "reward_curve.csv"


def module_dir(agent: str) -> Path:
    return MODULES_DIR / agent


def build_config() -> SACConfig:
    register_env(ENV_NAME, lambda cfg: ParallelPettingZooEnv(ColdChainTrainingEnv(cfg)))

    module_specs = {}
    for agent in AGENTS:
        if agent in LEARNERS:
            module_specs[agent] = RLModuleSpec(model_config={"fcnet_hiddens": [64, 64]})
        else:
            module_specs[agent] = RLModuleSpec(module_class=FixedActionRLModule)

    return (
        SACConfig()
        .environment(ENV_NAME, env_config=ENV_CONFIG)
        .framework("torch")
        .env_runners(num_env_runners=0, rollout_fragment_length=1)
        .multi_agent(
            policies=set(AGENTS),
            policy_mapping_fn=lambda agent_id, *a, **k: agent_id,
            policies_to_train=LEARNERS,
        )
        .training(
            num_steps_sampled_before_learning_starts=256,
            target_network_update_freq=1,
            replay_buffer_config={
                "type": "MultiAgentPrioritizedEpisodeReplayBuffer",
                "capacity": 100_000,
                "alpha": 0.6,
                "beta": 0.4,
            },
        )
        .rl_module(rl_module_spec=MultiRLModuleSpec(rl_module_specs=module_specs))
        .debugging(seed=SEED)
    )
