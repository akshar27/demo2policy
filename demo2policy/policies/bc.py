"""Behavior cloning: an MLP regressing the expert's next action from the observation.

Action target is *relative* (ee_target - ee_pos), so the network never has to memorise absolute workspace coordinates
and its mistakes do not accumulate in an absolute frame; the policy re-reads the observation every step (closed loop).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from ..data.demo import Demo
from .base import Action

MAX_T = 120.0  # normalisation for the optional time feature


@dataclass
class BCConfig:
    hidden: int = 256
    layers: int = 3
    epochs: int = 300
    lr: float = 1e-3
    batch_size: int = 256
    weight_decay: float = 1e-5
    use_time: bool = False       # add t/MAX_T to the observation (ablation: breaks phase ambiguity, hurts closed-loop reactivity)
    obs_noise: float = 0.0       # gaussian noise (in normalised obs units) added at train time; a cheap robustness trick
    seed: int = 0


def _features(obs: np.ndarray, t: np.ndarray | None, use_time: bool) -> np.ndarray:
    x = np.asarray(obs, dtype=np.float32)
    if use_time:
        x = np.concatenate([x, (np.asarray(t, dtype=np.float32) / MAX_T)[..., None]], axis=-1)
    return x


def _targets(demos: list[Demo]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    X, T, Y = [], [], []
    for d in demos:
        X.append(d.obs)
        T.append(np.arange(len(d)))
        rel = d.ee_target - d.obs[:, 0:3]
        Y.append(np.concatenate([rel, d.gripper[:, None], d.yaw[:, None]], axis=1))
    return np.concatenate(X), np.concatenate(T), np.concatenate(Y).astype(np.float32)


def _mlp(inp: int, out: int, hidden: int, layers: int) -> nn.Sequential:
    mods, d = [], inp
    for _ in range(layers):
        mods += [nn.Linear(d, hidden), nn.ReLU()]
        d = hidden
    mods.append(nn.Linear(d, out))
    return nn.Sequential(*mods)


class BCPolicy:
    def __init__(self, net: nn.Module, x_mu, x_sd, y_mu, y_sd, cfg: BCConfig, name: str = "bc"):
        self.net, self.cfg, self.name = net.eval(), cfg, name
        self.x_mu, self.x_sd, self.y_mu, self.y_sd = (np.asarray(a, dtype=np.float32) for a in (x_mu, x_sd, y_mu, y_sd))

    def reset(self) -> None:
        pass

    @torch.no_grad()
    def act(self, obs: np.ndarray, t: int) -> Action:
        x = _features(obs, np.array(t), self.cfg.use_time)
        x = torch.from_numpy((x - self.x_mu) / self.x_sd)[None]
        y = self.net(x)[0].numpy() * self.y_sd + self.y_mu
        return Action(np.asarray(obs[0:3], dtype=float) + y[0:3], float(np.clip(y[3], 0.0, 1.0)), float(y[4]))

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"cfg": asdict(self.cfg), "state": self.net.state_dict(), "x_mu": self.x_mu, "x_sd": self.x_sd,
                    "y_mu": self.y_mu, "y_sd": self.y_sd, "name": self.name}, path)

    @classmethod
    def load(cls, path: str | Path) -> "BCPolicy":
        ck = torch.load(path, weights_only=False)
        cfg = BCConfig(**ck["cfg"])
        n_in = len(ck["x_mu"])
        net = _mlp(n_in, 5, cfg.hidden, cfg.layers)
        net.load_state_dict(ck["state"])
        return cls(net, ck["x_mu"], ck["x_sd"], ck["y_mu"], ck["y_sd"], cfg, ck.get("name", "bc"))


def train_bc(demos: list[Demo], cfg: BCConfig | None = None, log: list | None = None) -> BCPolicy:
    cfg = cfg or BCConfig()
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    obs, t, y = _targets(demos)
    x = _features(obs, t, cfg.use_time)
    x_mu, x_sd = x.mean(0), x.std(0) + 1e-6
    y_mu, y_sd = y.mean(0), y.std(0) + 1e-6
    X = torch.from_numpy((x - x_mu) / x_sd)
    Y = torch.from_numpy((y - y_mu) / y_sd)
    net = _mlp(x.shape[1], 5, cfg.hidden, cfg.layers)
    opt = torch.optim.AdamW(net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, cfg.epochs)
    n = len(X)
    for ep in range(cfg.epochs):
        perm = torch.from_numpy(rng.permutation(n))
        total = 0.0
        for i in range(0, n, cfg.batch_size):
            idx = perm[i:i + cfg.batch_size]
            xb = X[idx]
            if cfg.obs_noise > 0:
                xb = xb + cfg.obs_noise * torch.randn_like(xb)
            loss = nn.functional.mse_loss(net(xb), Y[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        sched.step()
        if log is not None:
            log.append(total / n)
    return BCPolicy(net, x_mu, x_sd, y_mu, y_sd, cfg, name=f"bc[n={len(demos)}{',t' if cfg.use_time else ''}]")
