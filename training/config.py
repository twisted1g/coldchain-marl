from __future__ import annotations

from pathlib import Path

from ray.rllib.algorithms.sac import SACConfig
from ray.rllib.core.rl_module.multi_rl_module import MultiRLModuleSpec
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.tune.registry import register_env

from training.env import TemperatureTrainingEnv
from training.module import FixedActionRLModule

SEED = 0
NUM_ITERATIONS = 25
EVAL_EPISODES = 10

LEARNER = "temperature"
FROZEN = ["routing", "spoilage", "inventory", "delivery"]
ENV_NAME = "coldchain_temperature"

ENV_CONFIG = {"fruit": "strawberry", "max_steps": 20, "base_seed": 1000}
EVAL_ENV_CONFIG = {**ENV_CONFIG, "base_seed": 90_000}
COMPARE_ENV_CONFIG = {**ENV_CONFIG, "base_seed": 500_000}

ARTIFACTS = Path(__file__).resolve().parent.parent / "artifacts"
MODULE_DIR = ARTIFACTS / "temp_skeleton_module"
CURVE_CSV = ARTIFACTS / "temp_skeleton_reward_curve.csv"
CURVE_PNG = ARTIFACTS / "temp_skeleton_reward_curve.png"


def build_config() -> SACConfig:
    register_env(ENV_NAME, lambda cfg: ParallelPettingZooEnv(TemperatureTrainingEnv(cfg)))

    module_specs = {LEARNER: RLModuleSpec(model_config={"fcnet_hiddens": [64, 64]})}
    for agent in FROZEN:
        module_specs[agent] = RLModuleSpec(module_class=FixedActionRLModule)

    return (
        SACConfig()
        .environment(ENV_NAME, env_config=ENV_CONFIG)
        .framework("torch")
        .env_runners(num_env_runners=0, rollout_fragment_length=1)
        .multi_agent(
            policies={LEARNER, *FROZEN},
            policy_mapping_fn=lambda agent_id, *a, **k: agent_id,
            # PHASE 3b: add the frozen agents here and swap their FixedActionRLModule
            # specs for trainable RLModuleSpec()s to unfreeze them.
            policies_to_train=[LEARNER],
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
