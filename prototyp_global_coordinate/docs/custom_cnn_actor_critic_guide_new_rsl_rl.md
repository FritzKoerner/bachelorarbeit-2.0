# CNN ActorCritic for Depth + State Observations (New rsl-rl)

Guide for implementing a CNN-based actor-critic that processes **128×128 depth maps**
through 3 shared CNN layers alongside the existing **17-dimensional state vector**, with
2 separate actor MLP layers and 2 separate critic MLP layers, trained with PPO.

**Target version:** rsl-rl development version (`~/Repos/rsl_rl`) — uses `TensorDict`
observations, `PPO.construct_algorithm()` factory, and built-in `CNNModel`.

---

## Architecture overview

```
                       ┌────────────────────────────┐
  depth (1, 128, 128) ─┤  3 × Conv2d  (shared CNN)  ├─── cnn_latent (128)
                       └────────────────────────────┘
                                                        ┌──────────────────┐
                               ┌─── concat ────────────►│  2 × Linear      │──► actions (4)
  state (17) ──────────────────┤    (128 + 17 = 145)    │  (actor MLP)     │
                               │                        └──────────────────┘
                               │                        ┌──────────────────┐
                               └─── concat ────────────►│  2 × Linear      │──► value (1)
                                    (128 + 17 = 145)    │  (critic MLP)    │
                                                        └──────────────────┘
```

The new rsl-rl already ships `CNNModel` and a `share_cnn_encoders` option. This means
the architecture above can be achieved **purely through configuration** — no custom model
class required. The guide covers both approaches:

- [**Approach A: Config-only**](#approach-a-config-only-recommended) — use built-in `CNNModel`
- [**Approach B: Custom model**](#approach-b-custom-model-class) — for full control

---

## Key differences from v2.2.4

| Aspect | v2.2.4 (installed) | New version (`~/Repos/rsl_rl`) |
|--------|-------------------|-------------------------------|
| Observations | Flat `(B, num_obs)` tensor | `TensorDict` with named groups |
| Model creation | `eval("ActorCritic")(num_obs, ...)` | `PPO.construct_algorithm(obs, env, cfg)` |
| Actor/Critic | Single `ActorCritic` class | Separate `actor: MLPModel` + `critic: MLPModel` |
| CNN support | None (must build custom) | Built-in `CNNModel` class |
| CNN sharing | Manual | `share_cnn_encoders: True` in config |
| Class resolution | `eval()` — needs builtins hack | `resolve_callable()` — finds classes by name |
| Config format | `"policy": {...}` | Separate `"actor": {...}`, `"critic": {...}` |

---

## Step 1: Update the environment to return TensorDict

The new rsl-rl expects `get_observations()` and `step()` to return a `TensorDict`
with named observation groups instead of a flat tensor.

### 1a. Add imports

```python
from tensordict import TensorDict
```

### 1b. Update `__init__`

```python
def __init__(self, num_envs, env_cfg, obs_cfg, reward_cfg, show_viewer=False):
    # ... existing code ...

    # Image observation dimensions
    self.depth_res = obs_cfg.get("depth_res", 128)
    self.num_state_obs = obs_cfg["num_state_obs"]  # 17

    # num_obs is no longer meaningful as a single int since observations
    # are multi-modal. Keep it for max_episode_length etc. but the runner
    # reads shapes from the TensorDict directly.
    self.num_obs = self.num_state_obs  # runner doesn't use this for the new version
    self.num_actions = env_cfg["num_actions"]
```

### 1c. Add camera in `build()`

```python
def build(self):
    # ... existing scene creation ...

    # Downward-facing depth camera (before scene.build)
    self.camera = self.scene.add_camera(
        res=(self.depth_res, self.depth_res),
        pos=(0, 0, 3),
        lookat=(0, 0, 0),
        fov=70,
        GUI=False,
    )

    # ... scene.build() and rest of build ...

    # Pre-allocate observation buffers
    self.state_buf = torch.zeros(
        (self.num_envs, self.num_state_obs), device=gs.device, dtype=gs.tc_float
    )
    self.depth_buf = torch.zeros(
        (self.num_envs, 1, self.depth_res, self.depth_res), device=gs.device, dtype=gs.tc_float
    )
```

### 1d. Update `_compute_obs()` to return TensorDict

```python
def _compute_obs(self) -> TensorDict:
    s = self.obs_scales

    # State vector (unchanged from current implementation)
    self.state_buf[:] = torch.cat([
        torch.clip(self.rel_pos * s["rel_pos"], -1, 1),     # 3
        self.base_quat,                                       # 4
        torch.clip(self.base_lin_vel * s["lin_vel"], -1, 1), # 3
        torch.clip(self.base_ang_vel * s["ang_vel"], -1, 1), # 3
        self.last_actions,                                     # 4
    ], dim=-1)  # (num_envs, 17)

    # Depth image
    _, depth, _, _ = self.camera.render(depth=True)
    # depth shape: (H, W) for single env, or (num_envs, H, W) for batched
    self.depth_buf[:, 0] = torch.clamp(depth / 20.0, 0.0, 1.0)  # normalise to [0, 1]

    return TensorDict({
        "state": self.state_buf,   # (num_envs, 17)    — 1D, fed to MLP
        "depth": self.depth_buf,   # (num_envs, 1, 128, 128) — 2D, fed to CNN
    }, batch_size=[self.num_envs])
```

### 1e. Update `get_observations()`

```python
def get_observations(self) -> TensorDict:
    return self._compute_obs()
```

### 1f. Update `step()` return

```python
def step(self, actions):
    # ... existing logic (PID, physics, rewards, resets) ...

    obs = self._compute_obs()
    self.last_actions[:] = self.actions[:]
    return obs, self.rew_buf, self.reset_buf, self.extras
```

### 1g. Update `reset()`

```python
def reset(self):
    self.reset_buf[:] = True
    self.reset_idx(torch.arange(self.num_envs, device=gs.device))
    return self._compute_obs()
```

### 1h. Drop the old extras["observations"]["critic"] pattern

In v2.2.4, the critic observations were passed through `extras["observations"]["critic"]`.
The new version reads critic observations directly from the `TensorDict` via `obs_groups`.
Remove lines like:

```python
# DELETE these from step() and get_observations():
self.extras["observations"]["critic"] = self.obs_buf
```

The `extras` dict now only needs `"time_outs"` and optionally `"log"` / `"episode"`.

### 1i. Config changes

```python
obs_cfg = {
    "num_state_obs": 17,
    "depth_res": 128,
    "obs_scales": {
        "rel_pos":  1 / 15.0,
        "lin_vel":  1 / 5.0,
        "ang_vel":  1 / 3.14159,
    },
}
```

---

## Step 2: Update the training config

The new rsl-rl uses separate `"actor"` and `"critic"` sections instead of a single
`"policy"` block.

```python
def get_train_cfg(exp_name, max_iterations):
    return {
        "algorithm": {
            "class_name": "PPO",
            "clip_param": 0.2,
            "desired_kl": 0.01,
            "entropy_coef": 0.005,
            "gamma": 0.99,
            "lam": 0.95,
            "learning_rate": 3e-4,
            "max_grad_norm": 1.0,
            "num_learning_epochs": 5,
            "num_mini_batches": 4,
            "schedule": "adaptive",
            "use_clipped_value_loss": True,
            "value_loss_coef": 1.0,
            "share_cnn_encoders": True,   # ← actor CNN weights shared with critic
            "rnd_cfg": None,
            "symmetry_cfg": None,
        },

        "actor": {
            "class_name": "CNNModel",     # ← built-in CNN + MLP model
            "hidden_dims": [256, 256],    # 2 actor MLP layers
            "activation": "elu",
            "obs_normalization": True,    # normalise 1D state obs
            "distribution_cfg": {
                "class_name": "GaussianDistribution",
                "init_std": 1.0,
            },
            "cnn_cfg": {                  # 3 shared CNN layers
                "depth": {                # key must match TensorDict obs group name
                    "output_channels": [32, 64, 128],
                    "kernel_size": [8, 4, 3],
                    "stride": [4, 2, 1],
                    "padding": "zeros",
                    "norm": "batch",
                    "activation": "elu",
                    "global_pool": "avg",
                    "flatten": True,
                },
            },
        },

        "critic": {
            "class_name": "CNNModel",     # same class — CNN shared via share_cnn_encoders
            "hidden_dims": [256, 256],    # 2 critic MLP layers
            "activation": "elu",
            "obs_normalization": True,
            # NO cnn_cfg needed — PPO.construct_algorithm injects actor.cnns
            # NO distribution_cfg — critic outputs scalar deterministically
        },

        "obs_groups": {
            "actor":  ["state", "depth"],  # actor sees state + depth
            "critic": ["state", "depth"],  # critic sees same (can differ for asymmetric)
        },

        "num_steps_per_env": 24,    # reduced from 100 for memory (see Section 5)
        "save_interval": 100,
        "check_for_nan": True,
        "multi_gpu": None,

        "runner": {
            "experiment_name": exp_name,
            "max_iterations": max_iterations,
        },
    }
```

### What happens under the hood

`PPO.construct_algorithm()` (line 472–506 of `ppo.py`) does this:

```python
# 1. Resolve classes
actor_class = resolve_callable("CNNModel")   # → rsl_rl.models.CNNModel
critic_class = resolve_callable("CNNModel")  # → rsl_rl.models.CNNModel

# 2. Validate observation groups
obs_groups = resolve_obs_groups(obs, {"actor": ["state", "depth"], "critic": ["state", "depth"]})

# 3. Create actor (builds 3 CNN layers + 2 MLP layers)
actor = CNNModel(obs, obs_groups, "actor", num_actions=4, **cfg["actor"])

# 4. Share CNN encoders
cfg["critic"]["cnns"] = actor.cnns  # critic reuses actor's CNN nn.ModuleDict

# 5. Create critic (reuses shared CNN + builds 2 separate MLP layers)
critic = CNNModel(obs, obs_groups, "critic", output_dim=1, **cfg["critic"])
```

The CNN `nn.ModuleDict` is literally the same Python object in both models. When PPO
computes loss and calls `loss.backward()`, gradients flow through the shared CNN from
both the surrogate (actor) loss and the value (critic) loss.

---

## Step 3: Update the training script

```python
# train_rl.py
import copy
from tensordict import TensorDict
from rsl_rl.runners import OnPolicyRunner
import genesis as gs
from envs.coordinate_landing_env import CoordinateLandingEnv


def main():
    # ... argparse unchanged ...

    gs.init(backend=gs.gpu, precision="32", logging_level="warning")

    env_cfg, obs_cfg, reward_cfg = get_cfgs()
    train_cfg = get_train_cfg(args.exp_name, args.max_iterations)

    env = CoordinateLandingEnv(
        num_envs=args.num_envs,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        show_viewer=args.vis,
    )
    env.build()

    # New version: pass deep copy (OnPolicyRunner still mutates via .pop())
    runner = OnPolicyRunner(env, copy.deepcopy(train_cfg), log_dir, device=gs.device)
    runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=True)
```

No class registration hacks needed. `resolve_callable("CNNModel")` automatically searches
the `rsl_rl` package and finds `rsl_rl.models.CNNModel`.

---

## Approach A: Config-only (recommended)

Everything above **is** the config-only approach. The built-in `CNNModel` + `share_cnn_encoders`
gives exactly the requested architecture:

- **3 shared CNN layers**: defined in `actor.cnn_cfg.depth`
- **2 actor MLP layers**: `actor.hidden_dims = [256, 256]`
- **2 critic MLP layers**: `critic.hidden_dims = [256, 256]`
- **CNN sharing**: `algorithm.share_cnn_encoders = True`

### CNN layer math (128×128 input, no padding)

```
Conv2d(1→32,  k=8, s=4) + BN + ELU  →  (B, 32,  31, 31)
Conv2d(32→64, k=4, s=2) + BN + ELU  →  (B, 64,  14, 14)
Conv2d(64→128, k=3, s=1) + BN + ELU →  (B, 128, 12, 12)
AdaptiveAvgPool2d(1)                  →  (B, 128,  1,  1)
Flatten                               →  (B, 128)
```

### With padding="zeros" (same-ish padding)

```
Conv2d(1→32,  k=8, s=4, p=2) + BN + ELU  →  (B, 32, 32, 32)
Conv2d(32→64, k=4, s=2, p=1) + BN + ELU  →  (B, 64, 16, 16)
Conv2d(64→128, k=3, s=1, p=1) + BN + ELU →  (B, 128, 16, 16)
AdaptiveAvgPool2d(1)                       →  (B, 128, 1, 1)
Flatten                                    →  (B, 128)
```

Both produce a **128-dim** CNN latent after global average pooling.

### Full model structure

```
Actor (CNNModel):
  cnns.depth:  Conv2d(1→32) → BN → ELU → Conv2d(32→64) → BN → ELU → Conv2d(64→128) → BN → ELU → AvgPool → Flatten
  normalizer:  EmpiricalNormalization(17)     ← state normalisation
  mlp:         Linear(145→256) → ELU → Linear(256→256) → ELU → Linear(256→4)
  distribution: GaussianDistribution(std=1.0)

Critic (CNNModel):
  cnns.depth:  ← SHARED with actor (same nn.Module)
  normalizer:  EmpiricalNormalization(17)     ← separate normaliser instance
  mlp:         Linear(145→256) → ELU → Linear(256→256) → ELU → Linear(256→1)
```

**Total parameters:** ~345K (CNN: ~120K shared, actor MLP: ~103K, critic MLP: ~103K)

### CNN configuration options reference

The `cnn_cfg` dict for each 2D observation group accepts:

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `output_channels` | `list[int]` | **required** | Channels per conv layer |
| `kernel_size` | `int \| list[int]` | **required** | Kernel size(s) |
| `stride` | `int \| list[int]` | `1` | Stride(s) |
| `dilation` | `int \| list[int]` | `1` | Dilation(s) |
| `padding` | `str` | `"none"` | `"none"`, `"zeros"`, `"reflect"`, `"replicate"`, `"circular"` |
| `norm` | `str \| list[str]` | `"none"` | `"none"`, `"batch"`, `"layer"` |
| `activation` | `str` | `"elu"` | Any supported activation |
| `max_pool` | `bool \| list[bool]` | `False` | MaxPool2d(k=3, s=2) after layer |
| `global_pool` | `str` | `"none"` | `"none"`, `"max"`, `"avg"` |
| `flatten` | `bool` | `True` | Flatten to 1D (required for MLP head) |

---

## Approach B: Custom model class

If you need architecture changes that `CNNModel` doesn't support (e.g., skip connections,
attention, a projection layer between CNN and MLP, or a fundamentally different fusion
strategy), you can write a custom model.

### The model interface

The new rsl-rl creates models via `PPO.construct_algorithm()`:

```python
actor = actor_class(obs, obs_groups, "actor", num_actions, **cfg["actor"])
critic = critic_class(obs, obs_groups, "critic", 1, **cfg["critic"])
```

Your class must accept this constructor signature and implement the `MLPModel` interface.
The simplest approach is to **subclass `MLPModel`** (like `CNNModel` does).

### Required interface

| Method / Property | Signature | Purpose |
|-------------------|-----------|---------|
| `forward(obs, masks, hidden_state, stochastic_output)` | `TensorDict → Tensor` | Full forward pass |
| `get_latent(obs, masks, hidden_state)` | `TensorDict → Tensor` | Pre-MLP feature vector |
| `reset(dones)` | `Tensor → None` | Reset recurrent states |
| `get_hidden_state()` | `→ HiddenState` | Current RNN state (None for FF) |
| `update_normalization(obs)` | `TensorDict → None` | Update running stats |
| `get_output_log_prob(outputs)` | `Tensor → Tensor` | Log-prob under distribution |
| `get_kl_divergence(old, new)` | `Tensor, Tensor → Tensor` | KL between distributions |
| `output_mean` | `property → Tensor` | Distribution mean |
| `output_std` | `property → Tensor` | Distribution std |
| `output_entropy` | `property → Tensor` | Distribution entropy |
| `output_distribution_params` | `property → Tensor` | Concatenated dist params |

### Example: custom model with projection layer

Create `prototyp_global_coordinate/networks/actor_critic_cnn.py`:

```python
"""
Custom CNN model that adds a projection layer between the CNN encoder
and the MLP head, with optional dropout on the CNN latent.

Extends MLPModel to inherit distribution handling, normalisation, and
all PPO-required interface methods. Only overrides get_latent() and
_get_latent_dim() to inject the CNN path.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from tensordict import TensorDict
from rsl_rl.models.mlp_model import MLPModel
from rsl_rl.modules import CNN, HiddenState


class CustomCNNActorCritic(MLPModel):
    """CNN + MLP model with configurable projection and dropout."""

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int,
        hidden_dims: tuple[int, ...] | list[int] = (256, 256),
        activation: str = "elu",
        obs_normalization: bool = False,
        distribution_cfg: dict | None = None,
        # Custom CNN parameters
        cnn_cfg: dict | None = None,
        cnns: nn.ModuleDict | None = None,     # for sharing
        cnn_latent_dim: int = 128,
        cnn_dropout: float = 0.0,
    ) -> None:
        # ── Separate 1D and 2D observation groups ───────────────
        active_groups = obs_groups[obs_set]
        self.obs_groups_2d = []
        obs_groups_1d = []
        obs_dim_1d = 0

        for group in active_groups:
            shape = obs[group].shape
            if len(shape) == 4:  # (B, C, H, W)
                self.obs_groups_2d.append(group)
            elif len(shape) == 2:  # (B, D)
                obs_groups_1d.append(group)
                obs_dim_1d += shape[-1]

        # ── Create or reuse CNN encoders ────────────────────────
        if cnns is not None:
            self.cnns = cnns if isinstance(cnns, nn.ModuleDict) else nn.ModuleDict(cnns)
        else:
            if cnn_cfg is None:
                raise ValueError("Must provide cnn_cfg or cnns")
            cnn_modules = {}
            for group in self.obs_groups_2d:
                cfg = cnn_cfg.get(group, cnn_cfg)
                h, w = obs[group].shape[2], obs[group].shape[3]
                c = obs[group].shape[1]
                cnn_modules[group] = CNN(input_dim=(h, w), input_channels=c, **cfg)
            self.cnns = nn.ModuleDict(cnn_modules)

        # Compute raw CNN output dim (after flatten)
        raw_cnn_dim = sum(int(cnn.output_dim) for cnn in self.cnns.values())

        # ── Projection layer (custom addition) ──────────────────
        from rsl_rl.utils import resolve_nn_activation
        act = resolve_nn_activation(activation)
        self.cnn_proj = nn.Sequential(
            nn.Linear(raw_cnn_dim, cnn_latent_dim),
            act,
            nn.Dropout(cnn_dropout) if cnn_dropout > 0 else nn.Identity(),
        )
        self.cnn_latent_dim = cnn_latent_dim

        # ── Initialise parent MLPModel ──────────────────────────
        # Parent reads self.obs_dim from _get_obs_dim and self._get_latent_dim()
        # to size the MLP correctly.
        super().__init__(
            obs, obs_groups, obs_set, output_dim,
            hidden_dims, activation, obs_normalization, distribution_cfg,
        )

    def _get_obs_dim(self, obs, obs_groups, obs_set):
        """Override: return only 1D obs groups so parent MLP sizes correctly."""
        active = obs_groups[obs_set]
        groups_1d = []
        dim = 0
        for g in active:
            if len(obs[g].shape) == 2:
                groups_1d.append(g)
                dim += obs[g].shape[-1]
        return groups_1d, dim

    def _get_latent_dim(self) -> int:
        """Override: MLP input = 1D obs dim + CNN projected latent dim."""
        return self.obs_dim + self.cnn_latent_dim

    def get_latent(
        self, obs: TensorDict, masks: torch.Tensor | None = None, hidden_state: HiddenState = None
    ) -> torch.Tensor:
        """Build combined feature vector from 1D state and 2D depth."""
        # 1D path: concatenate + normalise (parent handles this)
        latent_1d = super().get_latent(obs)
        # 2D path: CNN encode + project
        cnn_outs = [self.cnns[g](obs[g]) for g in self.obs_groups_2d]
        latent_cnn = self.cnn_proj(torch.cat(cnn_outs, dim=-1))
        # Fuse
        return torch.cat([latent_1d, latent_cnn], dim=-1)
```

### Registering the custom class

`resolve_callable()` supports several formats:

```python
# Option 1: colon-separated module:class (recommended)
"actor": {
    "class_name": "networks.actor_critic_cnn:CustomCNNActorCritic",
    ...
}

# Option 2: fully qualified dotted path
"actor": {
    "class_name": "networks.actor_critic_cnn.CustomCNNActorCritic",
    ...
}
```

Both work as long as `networks/` is importable from the working directory
(add `networks/__init__.py`).

### Sharing CNN in the custom model

`PPO.construct_algorithm()` already handles this (line 495–496):

```python
if cfg["algorithm"].pop("share_cnn_encoders", None):
    cfg["critic"]["cnns"] = actor.cnns
```

Your custom model receives the shared `cnns` parameter via `**cfg["critic"]` and
reuses it. No extra code needed — just ensure your `__init__` accepts a `cnns` kwarg.

---

## Step 4: Evaluation script changes

The evaluation script needs the same environment changes (TensorDict output). The
policy loading works the same:

```python
runner = OnPolicyRunner(env, copy.deepcopy(train_cfg), log_dir=None, device=gs.device)
runner.load(checkpoint_path)
policy = runner.alg.get_policy()  # returns the actor MLPModel/CNNModel
policy.eval()

obs = env.get_observations().to(device)
with torch.inference_mode():
    actions = policy(obs, stochastic_output=False)  # deterministic
```

---

## Step 5: Performance considerations

### Memory budget

| Component | 256 envs × 24 steps | 4096 envs × 24 steps |
|-----------|--------------------|-----------------------|
| State buffer | 0.4 MB | 6.7 MB |
| Depth buffer (128²) | 192 MB | 3,072 MB |
| Rollout storage (depth) | 4.6 GB | **73.7 GB** ✗ |
| Rollout storage (state) | 9.4 MB | 150 MB |

**4096 envs with 128×128 depth is infeasible.** The rollout storage alone needs 74 GB.

### Recommended configurations

| Scenario | `num_envs` | `depth_res` | `num_steps_per_env` | Rollout depth storage |
|----------|-----------|-------------|--------------------|-----------------------|
| Dev/debug | 4 | 128 | 24 | 72 MB |
| Small run | 64 | 128 | 24 | 1.2 GB |
| Medium run | 256 | 64 | 24 | 288 MB |
| Large run | 512 | 64 | 24 | 576 MB |

**Recommendation for first experiments:** `num_envs=64, depth_res=64, num_steps_per_env=24`.

### Reducing resolution

Change `depth_res` to 64 in `obs_cfg`. The CNN automatically adapts:

```
Conv2d(1→32,  k=8, s=4) + BN + ELU  →  (B, 32, 15, 15)
Conv2d(32→64, k=4, s=2) + BN + ELU  →  (B, 64,  6,  6)
Conv2d(64→128, k=3, s=1) + BN + ELU →  (B, 128, 4,  4)
AdaptiveAvgPool2d(1)                  →  (B, 128, 1,  1)
Flatten                               →  (B, 128)
```

Same 128-dim latent, 4× less rendering and storage cost.

### Frame skipping

Render depth every N physics steps and reuse the cached image:

```python
def _compute_obs(self):
    # ... state computation unchanged ...

    if self.episode_length_buf[0] % self.render_interval == 0:
        _, depth, _, _ = self.camera.render(depth=True)
        self.depth_buf[:, 0] = torch.clamp(depth / 20.0, 0.0, 1.0)
    # else: reuse self.depth_buf from last render

    return TensorDict({"state": self.state_buf, "depth": self.depth_buf}, ...)
```

---

## Step 6: Training tips

### 1. Start with state-only, then add depth

Train the current 17-dim state policy to convergence first. Then add the CNN path
and continue training. This gives the MLP heads a good baseline before the CNN features
stabilise.

To do this with the new config, switch between `MLPModel` and `CNNModel`:

```python
# Phase 1: state-only
"actor": {"class_name": "MLPModel", "hidden_dims": [256, 256], ...}
"obs_groups": {"actor": ["state"], "critic": ["state"]}

# Phase 2: add depth
"actor": {"class_name": "CNNModel", "hidden_dims": [256, 256], "cnn_cfg": {...}, ...}
"obs_groups": {"actor": ["state", "depth"], "critic": ["state", "depth"]}
```

### 2. Separate learning rates for CNN and MLP

After the runner creates the algorithm, override the optimizer:

```python
runner = OnPolicyRunner(env, copy.deepcopy(train_cfg), log_dir, device=gs.device)

# Replace the default single-LR optimizer with parameter groups
actor = runner.alg.actor
critic = runner.alg.critic
runner.alg.optimizer = torch.optim.Adam([
    {"params": actor.cnns.parameters(), "lr": 1e-4},      # shared CNN (slow)
    {"params": actor.mlp.parameters(), "lr": 3e-4},        # actor MLP
    {"params": list(actor.distribution.parameters()), "lr": 3e-4},
    {"params": critic.mlp.parameters(), "lr": 3e-4},       # critic MLP
    # Note: critic.cnns is the same object as actor.cnns — don't add twice
])
```

### 3. Depth normalisation

Raw depth values range from 0 to ∞. Two good options:

```python
# Linear clamp (simple)
depth_norm = torch.clamp(depth / max_depth, 0.0, 1.0)

# Log-depth (preserves close-range detail)
depth_norm = torch.log1p(depth) / torch.log1p(torch.tensor(max_depth))
```

### 4. Asymmetric actor-critic

Give the critic extra information the actor can't see (e.g., ground-truth distance):

```python
"obs_groups": {
    "actor":  ["state", "depth"],                # only visual + proprioception
    "critic": ["state", "depth", "privileged"],  # + ground truth target info
}
```

Add a `"privileged"` key to the TensorDict with extra state info.

---

## Step 7: File structure

```
prototyp_global_coordinate/
  train_rl.py                     ← updated config (Step 2-3)
  eval_rl.py                      ← updated for TensorDict (Step 4)
  envs/
    coordinate_landing_env.py     ← TensorDict observations + camera (Step 1)
  networks/                       ← only needed for Approach B
    __init__.py
    actor_critic_cnn.py
  docs/
    custom_cnn_actor_critic_guide_new_rsl_rl.md  ← this guide
```

---

## Step 8: Checklist

- [ ] Install new rsl-rl from `~/Repos/rsl_rl` (`pip install -e ~/Repos/rsl_rl`)
- [ ] Add `tensordict` dependency (`pip install tensordict`)
- [ ] Update `coordinate_landing_env.py`:
  - [ ] Return `TensorDict` from `_compute_obs()`, `get_observations()`, `step()`, `reset()`
  - [ ] Add Genesis camera in `build()`
  - [ ] Remove `extras["observations"]["critic"]` pattern
- [ ] Update `train_rl.py`:
  - [ ] New config format with `"actor"` / `"critic"` sections
  - [ ] `copy.deepcopy(train_cfg)` when creating runner
- [ ] Update `eval_rl.py` for new runner API
- [ ] Reduce `num_envs` to 64–256 and `num_steps_per_env` to 24
- [ ] Smoke test: `python train_rl.py -B 4 -v --max_iterations 2`
- [ ] Verify TensorBoard logs show value/surrogate loss converging

---

## Appendix: Migrating an existing v2.2.4 checkpoint

The checkpoint format changed (v2.2.4 stores one `model_state_dict`, new version stores
separate `model_state_dict` for actor and `critic_state_dict` for critic). To load old
weights into the new model, manually map the state dict:

```python
old_ckpt = torch.load("logs/old/model_300.pt", weights_only=False)
old_sd = old_ckpt["model_state_dict"]

# Map old ActorCritic keys → new separate actor/critic keys
actor_sd = {k.replace("actor.", ""): v for k, v in old_sd.items() if k.startswith("actor.")}
critic_sd = {k.replace("critic.", ""): v for k, v in old_sd.items() if k.startswith("critic.")}

# Load into new models (strict=False to skip CNN keys that don't exist in old model)
runner.alg.actor.mlp.load_state_dict(actor_sd, strict=False)
runner.alg.critic.mlp.load_state_dict(critic_sd, strict=False)
```

This only transfers the MLP weights. CNN weights start fresh, which is fine since the old
model had no CNN.
