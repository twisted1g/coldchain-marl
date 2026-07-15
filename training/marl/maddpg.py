from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from tensordict import TensorDict
from torch import nn
from torchrl.data import LazyTensorStorage, TensorDictReplayBuffer

from training.marl.agents import _linear_decay, _mlp


class _CentralQNet(nn.Module):
    def __init__(
        self, joint_obs_dim: int, joint_act_dim: int, hidden: list[int]
    ) -> None:
        super().__init__()
        self.net = _mlp([joint_obs_dim + joint_act_dim, *hidden, 1])

    def forward(self, joint_obs: torch.Tensor, joint_act: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([joint_obs, joint_act], dim=-1))


class MADDPGDelivery:
    """Cooperative MADDPG group (paper Alg 5): N Gumbel actors, one shared critic."""

    def __init__(self, n: int, obs_dim: int, n_slots: int, cfg: dict[str, Any]) -> None:
        self._n = n
        self._obs_dim = obs_dim
        self._n_slots = n_slots
        hidden = list(cfg["hidden"])

        self._actors = nn.ModuleList(
            _mlp([obs_dim, *hidden, n_slots]) for _ in range(n)
        )
        self._critic = _CentralQNet(n * obs_dim, n * n_slots, hidden)
        self._target_actors = copy.deepcopy(self._actors)
        self._target_critic = copy.deepcopy(self._critic)
        for p in self._target_actors.parameters():
            p.requires_grad_(False)
        for p in self._target_critic.parameters():
            p.requires_grad_(False)

        self._actor_opt = torch.optim.Adam(self._actors.parameters(), lr=cfg["lr"])
        self._critic_opt = torch.optim.Adam(self._critic.parameters(), lr=cfg["lr"])

        self._rb = TensorDictReplayBuffer(
            storage=LazyTensorStorage(cfg["buffer_capacity"])
        )
        self._batch_size = cfg["batch_size"]
        self._warmup = cfg["warmup"]
        self._gamma = cfg["gamma"]
        self._tau = cfg["tau"]
        self._gumbel_start = cfg["gumbel_tau_start"]
        self._gumbel_end = cfg["gumbel_tau_end"]
        self._gumbel_decay = cfg["gumbel_tau_decay_steps"]
        self._steps = 0

        self._pending: dict[int, tuple[np.ndarray, int, float, np.ndarray, bool]] = {}
        self._need_update = False

    def _gumbel_tau(self) -> float:
        return _linear_decay(
            self._gumbel_start, self._gumbel_end, self._steps, self._gumbel_decay
        )

    def act(self, i: int, obs: np.ndarray, *, explore: bool) -> np.integer:
        td_obs = torch.as_tensor(obs, dtype=torch.float32)
        with torch.no_grad():
            logits = self._actors[i](td_obs)
            if explore:
                sample = F.gumbel_softmax(logits, tau=self._gumbel_tau(), hard=True)
                slot = int(sample.argmax())
            else:
                slot = int(logits.argmax())
        return np.int64(slot)

    def observe(
        self,
        i: int,
        obs: np.ndarray,
        action: np.integer,
        reward: float,
        next_obs: np.ndarray,
        terminated: bool,
        truncated: bool,
    ) -> None:
        self._pending[i] = (obs, int(action), reward, next_obs, terminated or truncated)
        if len(self._pending) == self._n:
            self._store_joint()
            self._pending = {}
            self._need_update = True

    def _store_joint(self) -> None:
        obs_l, act_l, reward_l, next_l, done_l = zip(
            *(self._pending[i] for i in range(self._n)), strict=True
        )
        obs = np.stack(obs_l)
        act = np.array(act_l, dtype=np.int64)
        reward = float(sum(reward_l))
        next_obs = np.stack(next_l)
        done = any(done_l)
        self._rb.add(
            TensorDict(
                {
                    "obs": torch.as_tensor(obs, dtype=torch.float32),
                    "act": torch.as_tensor(act, dtype=torch.long),
                    "reward": torch.tensor([reward], dtype=torch.float32),
                    "next_obs": torch.as_tensor(next_obs, dtype=torch.float32),
                    "done": torch.tensor([done], dtype=torch.bool),
                },
                batch_size=[],
            )
        )

    def update(self) -> dict[str, float]:
        if not self._need_update:
            return {}
        self._need_update = False
        if len(self._rb) < self._warmup:
            return {}

        batch = self._rb.sample(self._batch_size)
        obs = batch["obs"]
        act = batch["act"]
        reward = batch["reward"]
        next_obs = batch["next_obs"]
        done = batch["done"].float()
        b = obs.shape[0]

        joint_obs = obs.reshape(b, self._n * self._obs_dim)
        joint_next_obs = next_obs.reshape(b, self._n * self._obs_dim)
        act_oh = F.one_hot(act, self._n_slots).float()
        joint_act = act_oh.reshape(b, self._n * self._n_slots)

        with torch.no_grad():
            next_oh = [
                F.one_hot(
                    self._target_actors[i](next_obs[:, i, :]).argmax(-1), self._n_slots
                ).float()
                for i in range(self._n)
            ]
            next_joint_act = torch.stack(next_oh, dim=1).reshape(
                b, self._n * self._n_slots
            )
            target_q = self._target_critic(joint_next_obs, next_joint_act)
            y = reward + self._gamma * (1.0 - done) * target_q

        q = self._critic(joint_obs, joint_act)
        critic_loss = F.mse_loss(q, y)
        self._critic_opt.zero_grad()
        critic_loss.backward()
        self._critic_opt.step()

        self._steps += 1
        gumbel_tau = self._gumbel_tau()
        actor_loss = torch.zeros(())
        for i in range(self._n):
            logits_i = self._actors[i](obs[:, i, :])
            gs_i = F.gumbel_softmax(logits_i, tau=gumbel_tau, hard=True)
            parts = [gs_i if j == i else act_oh[:, j, :] for j in range(self._n)]
            joint_act_i = torch.cat(parts, dim=-1)
            actor_loss = actor_loss - self._critic(joint_obs, joint_act_i).mean()
        self._actor_opt.zero_grad()
        actor_loss.backward()
        self._actor_opt.step()

        self._soft_update()
        return {
            "loss_critic": critic_loss.detach().item(),
            "loss_actor": actor_loss.detach().item() / self._n,
        }

    def _soft_update(self) -> None:
        targets = [*self._target_actors.parameters(), *self._target_critic.parameters()]
        sources = [*self._actors.parameters(), *self._critic.parameters()]
        with torch.no_grad():
            for tp, p in zip(targets, sources, strict=True):
                tp.mul_(1.0 - self._tau).add_(self._tau * p)

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"actors": self._actors.state_dict(), "critic": self._critic.state_dict()},
            path / "maddpg.pt",
        )

    def load(self, path: Path) -> None:
        ckpt = torch.load(path / "maddpg.pt", weights_only=True)
        self._actors.load_state_dict(ckpt["actors"])
        self._critic.load_state_dict(ckpt["critic"])
        self._target_actors.load_state_dict(self._actors.state_dict())
        self._target_critic.load_state_dict(self._critic.state_dict())


class DeliveryHandle:
    """Per-vehicle adapter over a shared MADDPG group (vehicle 0 persists it)."""

    def __init__(self, group: MADDPGDelivery, index: int) -> None:
        self._group = group
        self._index = index

    def act(self, obs: np.ndarray, *, explore: bool) -> np.integer:
        return self._group.act(self._index, obs, explore=explore)

    def observe(
        self,
        obs: np.ndarray,
        action: np.integer,
        reward: float,
        next_obs: np.ndarray,
        terminated: bool,
        truncated: bool,
    ) -> None:
        self._group.observe(
            self._index, obs, action, reward, next_obs, terminated, truncated
        )

    def update(self) -> dict[str, float]:
        return self._group.update()

    def save(self, path: Path) -> None:
        if self._index == 0:
            self._group.save(path)

    def load(self, path: Path) -> None:
        if self._index == 0:
            self._group.load(path)
