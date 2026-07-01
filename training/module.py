from __future__ import annotations

import numpy as np
import tree
from gymnasium.spaces import Box, Discrete

from ray.rllib.core.columns import Columns
from ray.rllib.core.rl_module import RLModule
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.annotations import override
from ray.rllib.utils.spaces.space_utils import batch as batch_func


class FixedActionRLModule(RLModule):
    """Non-trainable module emitting a constant no-op action.

    Freezes the non-temperature agents: Discrete spaces map to index 0, Box
    spaces to their midpoint.
    """

    def _fixed_action(self) -> np.ndarray:
        space = self.action_space
        if isinstance(space, Discrete):
            return np.int64(0)
        if isinstance(space, Box):
            return ((space.low + space.high) / 2.0).astype(space.dtype)
        raise TypeError(f"Unsupported action space: {space}")

    @override(RLModule)
    def _forward(self, batch, **kwargs):
        batch_size = len(tree.flatten(batch[SampleBatch.OBS])[0])
        actions = batch_func([self._fixed_action() for _ in range(batch_size)])
        return {Columns.ACTIONS: actions}

    @override(RLModule)
    def _forward_train(self, *args, **kwargs):
        raise NotImplementedError("FixedActionRLModule is frozen.")

    def compile(self, *args, **kwargs):
        pass
