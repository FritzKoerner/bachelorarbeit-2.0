"""MultiCategoricalDistribution: rsl-rl v5.x Distribution for multi-discrete action spaces.

Each action dimension is an independent Categorical distribution with `num_choices`
categories.  The MLP outputs `num_dims * num_choices` logits, which are split into
per-dimension logit vectors.

Example (4 binary action dims):
    output_dim = 4, num_choices = 2
    MLP outputs 8 logits → 4 independent Categorical(logits_i) distributions
    sample() returns (batch, 4) integer tensor in {0, ..., num_choices-1}
"""

import torch
import torch.nn as nn
from torch.distributions import Categorical

from rsl_rl.modules.distribution import Distribution


class MultiCategoricalDistribution(Distribution):
    """Multi-discrete distribution: N independent categorical variables."""

    def __init__(self, output_dim: int, num_choices: int = 2) -> None:
        super().__init__(output_dim)
        self.num_dims = output_dim
        self.num_choices = num_choices
        self._dists: list[Categorical] | None = None

    def update(self, mlp_output: torch.Tensor) -> None:
        # mlp_output: (batch, num_dims * num_choices)
        logits = mlp_output.reshape(-1, self.num_dims, self.num_choices)
        self._dists = [Categorical(logits=logits[:, i]) for i in range(self.num_dims)]

    def sample(self) -> torch.Tensor:
        # Returns (batch, num_dims) integer tensor
        return torch.stack([d.sample() for d in self._dists], dim=-1).float()

    def deterministic_output(self, mlp_output: torch.Tensor) -> torch.Tensor:
        logits = mlp_output.reshape(-1, self.num_dims, self.num_choices)
        return logits.argmax(dim=-1).float()

    def as_deterministic_output_module(self) -> nn.Module:
        return _MultiCategoricalDeterministicOutput(self.num_dims, self.num_choices)

    @property
    def input_dim(self) -> int:
        return self.num_dims * self.num_choices

    @property
    def mean(self) -> torch.Tensor:
        # For binary {0,1}: mean = P(action=1).  Generalizes to expected value.
        probs = torch.stack([d.probs for d in self._dists], dim=1)  # (batch, num_dims, num_choices)
        values = torch.arange(self.num_choices, device=probs.device, dtype=probs.dtype)
        return (probs * values).sum(dim=-1)  # (batch, num_dims)

    @property
    def std(self) -> torch.Tensor:
        # Variance of categorical = E[X^2] - E[X]^2
        probs = torch.stack([d.probs for d in self._dists], dim=1)
        values = torch.arange(self.num_choices, device=probs.device, dtype=probs.dtype)
        mean = (probs * values).sum(dim=-1)
        mean_sq = (probs * values ** 2).sum(dim=-1)
        return torch.sqrt(mean_sq - mean ** 2 + 1e-8)

    @property
    def entropy(self) -> torch.Tensor:
        return sum(d.entropy() for d in self._dists)

    @property
    def params(self) -> tuple[torch.Tensor, ...]:
        # Store all logits as a single tensor for rollout storage
        logits = torch.stack([d.logits for d in self._dists], dim=1)  # (batch, num_dims, num_choices)
        return (logits.reshape(-1, self.num_dims * self.num_choices),)

    def log_prob(self, outputs: torch.Tensor) -> torch.Tensor:
        # outputs: (batch, num_dims) integer actions stored as float
        actions = outputs.long()
        return sum(
            self._dists[i].log_prob(actions[:, i]) for i in range(self.num_dims)
        )

    def kl_divergence(
        self, old_params: tuple[torch.Tensor, ...], new_params: tuple[torch.Tensor, ...]
    ) -> torch.Tensor:
        old_logits = old_params[0].reshape(-1, self.num_dims, self.num_choices)
        new_logits = new_params[0].reshape(-1, self.num_dims, self.num_choices)
        kl = torch.zeros(old_logits.shape[0], device=old_logits.device)
        for i in range(self.num_dims):
            old_d = Categorical(logits=old_logits[:, i])
            new_d = Categorical(logits=new_logits[:, i])
            kl += torch.distributions.kl_divergence(old_d, new_d)
        return kl


class _MultiCategoricalDeterministicOutput(nn.Module):
    """Export-friendly module: argmax over logit pairs."""

    def __init__(self, num_dims: int, num_choices: int):
        super().__init__()
        self.num_dims = num_dims
        self.num_choices = num_choices

    def forward(self, mlp_output: torch.Tensor) -> torch.Tensor:
        logits = mlp_output.reshape(-1, self.num_dims, self.num_choices)
        return logits.argmax(dim=-1).float()
