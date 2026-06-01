"""From-scratch optimizer implementations used in the blog figures."""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover - torch is expected for full experiments
    torch = None


def _is_torch_tensor(value: object) -> bool:
    return torch is not None and isinstance(value, torch.Tensor)


def _zeros_like(value):
    if _is_torch_tensor(value):
        return torch.zeros_like(value)
    return np.zeros_like(value)


def _sqrt(value):
    if _is_torch_tensor(value):
        return torch.sqrt(value)
    return np.sqrt(value)


def _numel(value) -> int:
    if _is_torch_tensor(value):
        return int(value.numel())
    return int(np.size(value))


def count_param_values(params: Sequence) -> int:
    """Return the number of scalar parameter values."""
    return sum(_numel(p) for p in params)


class OptimizerBase:
    """Small readable optimizer base class with explicit memory accounting."""

    state_multiplier = 0

    def step(self, params: Sequence, grads: Sequence) -> None:
        raise NotImplementedError

    def state_values(self, params: Sequence) -> int:
        return self.state_multiplier * count_param_values(params)


class SGD(OptimizerBase):
    """Vanilla SGD: no memory."""

    state_multiplier = 0

    def __init__(self, lr: float):
        self.lr = lr

    def step(self, params: Sequence, grads: Sequence) -> None:
        for param, grad in zip(params, grads):
            param -= self.lr * grad


class Momentum(OptimizerBase):
    """SGD with direction memory via velocity."""

    state_multiplier = 1

    def __init__(self, lr: float, beta: float = 0.9):
        self.lr = lr
        self.beta = beta
        self.velocity = None

    def step(self, params: Sequence, grads: Sequence) -> None:
        if self.velocity is None:
            self.velocity = [_zeros_like(p) for p in params]

        for i, (param, grad) in enumerate(zip(params, grads)):
            self.velocity[i] = self.beta * self.velocity[i] + grad
            param -= self.lr * self.velocity[i]


class RMSProp(OptimizerBase):
    """RMSProp: remembers gradient scale through squared-gradient averages."""

    state_multiplier = 1

    def __init__(self, lr: float, beta: float = 0.9, eps: float = 1e-8):
        self.lr = lr
        self.beta = beta
        self.eps = eps
        self.sq_avg = None

    def step(self, params: Sequence, grads: Sequence) -> None:
        if self.sq_avg is None:
            self.sq_avg = [_zeros_like(p) for p in params]

        for i, (param, grad) in enumerate(zip(params, grads)):
            self.sq_avg[i] = self.beta * self.sq_avg[i] + (1.0 - self.beta) * (grad**2)
            param -= self.lr * grad / (_sqrt(self.sq_avg[i]) + self.eps)


class Adam(OptimizerBase):
    """Adam: remembers both direction and scale.

    Optional `weight_decay` behaves like L2-style regularization by adding
    weight decay into gradients before Adam's adaptive update.
    """

    state_multiplier = 2

    def __init__(
        self,
        lr: float,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0
        self.m = None
        self.v = None

    def step(self, params: Sequence, grads: Sequence) -> None:
        if self.m is None:
            self.m = [_zeros_like(p) for p in params]
            self.v = [_zeros_like(p) for p in params]

        self.t += 1
        for i, (param, grad) in enumerate(zip(params, grads)):
            adjusted_grad = grad
            if self.weight_decay:
                adjusted_grad = adjusted_grad + self.weight_decay * param

            self.m[i] = self.beta1 * self.m[i] + (1.0 - self.beta1) * adjusted_grad
            self.v[i] = self.beta2 * self.v[i] + (1.0 - self.beta2) * (adjusted_grad**2)

            m_hat = self.m[i] / (1.0 - self.beta1**self.t)
            v_hat = self.v[i] / (1.0 - self.beta2**self.t)
            param -= self.lr * m_hat / (_sqrt(v_hat) + self.eps)


class AdamW(OptimizerBase):
    """AdamW: Adam memory with decoupled weight decay."""

    state_multiplier = 2

    def __init__(
        self,
        lr: float,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0
        self.m = None
        self.v = None

    def step(self, params: Sequence, grads: Sequence) -> None:
        if self.m is None:
            self.m = [_zeros_like(p) for p in params]
            self.v = [_zeros_like(p) for p in params]

        self.t += 1
        for i, (param, grad) in enumerate(zip(params, grads)):
            if self.weight_decay:
                param -= self.lr * self.weight_decay * param

            self.m[i] = self.beta1 * self.m[i] + (1.0 - self.beta1) * grad
            self.v[i] = self.beta2 * self.v[i] + (1.0 - self.beta2) * (grad**2)

            m_hat = self.m[i] / (1.0 - self.beta1**self.t)
            v_hat = self.v[i] / (1.0 - self.beta2**self.t)
            param -= self.lr * m_hat / (_sqrt(v_hat) + self.eps)


def optimizer_state_report(params: Sequence) -> dict[str, int]:
    """Memory-overhead report in units of scalar values."""
    param_count = count_param_values(params)
    return {
        "Model parameters": param_count,
        "SGD state values": 0,
        "Momentum state values": param_count,
        "RMSProp state values": param_count,
        "Adam state values": 2 * param_count,
        "AdamW state values": 2 * param_count,
    }

