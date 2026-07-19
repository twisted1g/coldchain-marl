from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np
import torch
from gymnasium.spaces import Box, Discrete
from tensordict import TensorDict
from torch import nn
from torchrl.data import Categorical, LazyTensorStorage, TensorDictReplayBuffer
from torchrl.modules import Actor, QValueActor, ValueOperator
from torchrl.objectives import DDPGLoss, DQNLoss
from torchrl.objectives.utils import SoftUpdate

from core.world.graph import build_supply_chain
from core.world.graph_features import SPOILAGE_NODE_FEATURES, static_edge_index
from training.marl.gnn import GNN_EMBED_DIM, SpoilageGNN

Action = np.ndarray | np.integer | int


def mlp(dims: list[int], *, final_activation: nn.Module | None = None) -> nn.Sequential:
    layers: list[nn.Module] = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers.append(nn.ReLU())
    if final_activation is not None:
        layers.append(final_activation)
    return nn.Sequential(*layers)


def linear_decay(start: float, end: float, step: int, decay_steps: int) -> float:
    frac = min(1.0, step / decay_steps)
    return start + frac * (end - start)


def _transition_td(
    obs: np.ndarray,
    action: torch.Tensor,
    reward: float,
    next_obs: np.ndarray,
    terminated: bool,
    truncated: bool,
) -> TensorDict:
    return TensorDict(
        {
            "observation": torch.as_tensor(obs, dtype=torch.float32),
            "action": action,
            "next": TensorDict(
                {
                    "observation": torch.as_tensor(next_obs, dtype=torch.float32),
                    "reward": torch.tensor([reward], dtype=torch.float32),
                    "done": torch.tensor([terminated or truncated], dtype=torch.bool),
                    "terminated": torch.tensor([terminated], dtype=torch.bool),
                },
                batch_size=[],
            ),
        },
        batch_size=[],
    )


@runtime_checkable
class Agent(Protocol):
    """One decision-maker in the CTDE loop."""

    def act(self, obs: np.ndarray, *, explore: bool) -> Action: ...

    def observe(
        self,
        obs: np.ndarray,
        action: Action,
        reward: float,
        next_obs: np.ndarray,
        terminated: bool,
        truncated: bool,
    ) -> None: ...

    def update(self) -> dict[str, float]: ...

    def save(self, path: Path) -> None: ...

    def load(self, path: Path) -> None: ...


class _StatelessAgent:
    """Base for policies that neither learn nor persist anything."""

    def observe(self, *args: Any, **kwargs: Any) -> None:
        return None

    def update(self) -> dict[str, float]:
        return {}

    def save(self, path: Path) -> None:
        return None

    def load(self, path: Path) -> None:
        return None


class FrozenAgent(_StatelessAgent):
    """Non-trainable policy emitting a constant action: Discrete->0, Box->midpoint."""

    def __init__(self, action_space: Box | Discrete) -> None:
        if isinstance(action_space, Discrete):
            self._action: Action = np.int64(0)
        elif isinstance(action_space, Box):
            mid = (action_space.low + action_space.high) / 2.0
            self._action = mid.astype(action_space.dtype)
        else:
            raise TypeError(f"Unsupported action space: {action_space}")

    def act(self, obs: np.ndarray, *, explore: bool) -> Action:
        return self._action


class RandomAgent(_StatelessAgent):
    """Uniform-random policy over the action space, for trained-vs-random checks."""

    def __init__(self, action_space: Box | Discrete) -> None:
        self._action_space = action_space

    def act(self, obs: np.ndarray, *, explore: bool) -> Action:
        return self._action_space.sample()


class _ScaledActor(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden: list[int],
        low: np.ndarray,
        high: np.ndarray,
    ) -> None:
        super().__init__()
        self.net = mlp([obs_dim, *hidden, act_dim], final_activation=nn.Tanh())
        self.register_buffer("_low", torch.as_tensor(low, dtype=torch.float32))
        self.register_buffer("_high", torch.as_tensor(high, dtype=torch.float32))

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        unit = self.net(obs)
        return self._low + (unit + 1.0) / 2.0 * (self._high - self._low)


class QNet(nn.Module):
    """State-action value net over concatenated inputs; also the MADDPG critic
    with joint obs/action dims."""

    def __init__(self, obs_dim: int, act_dim: int, hidden: list[int]) -> None:
        super().__init__()
        self.net = mlp([obs_dim + act_dim, *hidden, 1])

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([obs, action], dim=-1))


class DDPGAgent:
    """Continuous-action DDPG agent (paper Algorithm 2) on TorchRL's DDPGLoss."""

    def __init__(self, obs_dim: int, action_space: Box, cfg: dict[str, Any]) -> None:
        self._act_space = action_space
        low, high = action_space.low, action_space.high
        act_dim = int(np.prod(action_space.shape))
        hidden = list(cfg["hidden"])

        actor_net = _ScaledActor(obs_dim, act_dim, hidden, low, high)
        self._actor = Actor(
            module=actor_net, in_keys=["observation"], out_keys=["action"]
        )
        qval = ValueOperator(
            module=QNet(obs_dim, act_dim, hidden), in_keys=["observation", "action"]
        )

        self._loss = DDPGLoss(actor_network=self._actor, value_network=qval)
        self._loss.make_value_estimator(gamma=cfg["gamma"])
        self._updater = SoftUpdate(self._loss, tau=cfg["tau"])
        self._opt = torch.optim.Adam(self._loss.parameters(), lr=cfg["lr"])

        self._rb = TensorDictReplayBuffer(
            storage=LazyTensorStorage(cfg["buffer_capacity"])
        )
        self._batch_size = cfg["batch_size"]
        self._warmup = cfg["warmup"]
        self._sigma = cfg["noise_sigma"] * (high - low)
        self._low = low
        self._high = high

    def act(self, obs: np.ndarray, *, explore: bool) -> np.ndarray:
        td = TensorDict(
            {"observation": torch.as_tensor(obs, dtype=torch.float32)}, batch_size=[]
        )
        with torch.no_grad():
            action = self._actor(td)["action"].numpy()
        if explore:
            action = action + np.random.normal(0.0, self._sigma).astype(np.float32)
        return np.clip(action, self._low, self._high).astype(np.float32)

    def observe(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_obs: np.ndarray,
        terminated: bool,
        truncated: bool,
    ) -> None:
        self._rb.add(
            _transition_td(
                obs,
                torch.as_tensor(action, dtype=torch.float32),
                reward,
                next_obs,
                terminated,
                truncated,
            )
        )

    def update(self) -> dict[str, float]:
        if len(self._rb) < self._warmup:
            return {}
        batch = self._rb.sample(self._batch_size)
        out = self._loss(batch)
        loss = out["loss_actor"] + out["loss_value"]
        self._opt.zero_grad()
        loss.backward()
        self._opt.step()
        self._updater.step()
        return {
            "loss_actor": out["loss_actor"].detach().item(),
            "loss_value": out["loss_value"].detach().item(),
        }

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self._actor.state_dict(), path / "actor.pt")

    def load(self, path: Path) -> None:
        self._actor.load_state_dict(torch.load(path / "actor.pt", weights_only=True))


class DQNAgent:
    """Discrete-action DQN agent (paper Algorithm 1) with epsilon-greedy exploration."""

    def __init__(
        self, obs_dim: int, action_space: Discrete, cfg: dict[str, Any]
    ) -> None:
        self._n = int(action_space.n)
        net = mlp([obs_dim, *list(cfg["hidden"]), self._n])
        self._qnet = QValueActor(
            module=net,
            in_keys=["observation"],
            spec=Categorical(self._n),
            action_space="categorical",
        )
        self._loss = DQNLoss(self._qnet, action_space="categorical", double_dqn=True)
        self._loss.make_value_estimator(gamma=cfg["gamma"])
        self._updater = SoftUpdate(self._loss, tau=cfg["tau"])
        self._opt = torch.optim.Adam(self._loss.parameters(), lr=cfg["lr"])

        self._rb = TensorDictReplayBuffer(
            storage=LazyTensorStorage(cfg["buffer_capacity"])
        )
        self._batch_size = cfg["batch_size"]
        self._warmup = cfg["warmup"]
        self._eps_start = cfg["eps_start"]
        self._eps_end = cfg["eps_end"]
        self._eps_decay = cfg["eps_decay_steps"]
        self._steps = 0

    def _epsilon(self) -> float:
        return linear_decay(
            self._eps_start, self._eps_end, self._steps, self._eps_decay
        )

    def act(self, obs: np.ndarray, *, explore: bool) -> np.integer:
        if explore:
            self._steps += 1
            if np.random.random() < self._epsilon():
                return np.int64(np.random.randint(self._n))
        td = TensorDict(
            {"observation": torch.as_tensor(obs, dtype=torch.float32)}, batch_size=[]
        )
        with torch.no_grad():
            action = self._qnet(td)["action"]
        return np.int64(int(action.argmax()) if action.ndim else int(action))

    def observe(
        self,
        obs: np.ndarray,
        action: np.integer,
        reward: float,
        next_obs: np.ndarray,
        terminated: bool,
        truncated: bool,
    ) -> None:
        self._rb.add(
            _transition_td(
                obs,
                torch.tensor(int(action), dtype=torch.long),
                reward,
                next_obs,
                terminated,
                truncated,
            )
        )

    def update(self) -> dict[str, float]:
        if len(self._rb) < self._warmup:
            return {}
        batch = self._rb.sample(self._batch_size)
        out = self._loss(batch)
        loss = out["loss"]
        self._opt.zero_grad()
        loss.backward()
        self._opt.step()
        self._updater.step()
        return {"loss": loss.detach().item(), "epsilon": self._epsilon()}

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self._qnet.state_dict(), path / "qnet.pt")

    def load(self, path: Path) -> None:
        self._qnet.load_state_dict(torch.load(path / "qnet.pt", weights_only=True))


class SharedHandle:
    """Per-instance adapter over one shared agent (paper Alg 4: symmetric
    instances, one policy). All handles feed the shared replay buffer; index 0
    updates and persists so one env step triggers one gradient step."""

    def __init__(self, agent: Agent, index: int) -> None:
        self._agent = agent
        self._index = index

    def act(self, obs: np.ndarray, *, explore: bool) -> Action:
        return self._agent.act(obs, explore=explore)

    def observe(
        self,
        obs: np.ndarray,
        action: Action,
        reward: float,
        next_obs: np.ndarray,
        terminated: bool,
        truncated: bool,
    ) -> None:
        self._agent.observe(obs, action, reward, next_obs, terminated, truncated)

    def update(self) -> dict[str, float]:
        return self._agent.update() if self._index == 0 else {}

    def save(self, path: Path) -> None:
        if self._index == 0:
            self._agent.save(path)

    def load(self, path: Path) -> None:
        if self._index == 0:
            self._agent.load(path)


class SpoilageAgent:
    """Spoilage agent (paper Alg 3): frozen GraphSAGE encoder feeding a DDPG head."""

    def __init__(
        self, obs_dim: int, action_space: Box, cfg: dict[str, Any], encoder_path: Path
    ) -> None:
        self._n_nodes = obs_dim // SPOILAGE_NODE_FEATURES
        self._encoder = SpoilageGNN()
        self._encoder.load_state_dict(torch.load(encoder_path, weights_only=True))
        self._encoder.eval()
        for p in self._encoder.parameters():
            p.requires_grad_(False)
        graph = build_supply_chain(np.random.default_rng(0))
        self._edge_index = torch.as_tensor(static_edge_index(graph), dtype=torch.long)
        self._ddpg = DDPGAgent(GNN_EMBED_DIM, action_space, cfg)

    def _encode(self, obs: np.ndarray) -> np.ndarray:
        x = torch.as_tensor(obs, dtype=torch.float32).reshape(
            self._n_nodes, SPOILAGE_NODE_FEATURES
        )
        with torch.no_grad():
            z = self._encoder(x, self._edge_index)
        return z.squeeze(0).numpy()

    def act(self, obs: np.ndarray, *, explore: bool) -> np.ndarray:
        return self._ddpg.act(self._encode(obs), explore=explore)

    def observe(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_obs: np.ndarray,
        terminated: bool,
        truncated: bool,
    ) -> None:
        self._ddpg.observe(
            self._encode(obs),
            action,
            reward,
            self._encode(next_obs),
            terminated,
            truncated,
        )

    def update(self) -> dict[str, float]:
        return self._ddpg.update()

    def save(self, path: Path) -> None:
        self._ddpg.save(path)

    def load(self, path: Path) -> None:
        self._ddpg.load(path)
