# Custom CNN ActorCritic for Depth + State Observations

Guide for implementing a custom `ActorCritic` model that processes **128×128 depth maps**
through a shared CNN encoder alongside the existing **17-dimensional state vector**, trained
with PPO via `rsl-rl v2.2.4`.

---

## Why a custom model?

The stock `ActorCritic` (from `rsl_rl.modules`) only accepts flat 1D observation vectors.
Our drone landing task will add a forward-facing depth camera, producing a `(1, 128, 128)`
depth image per environment on every step. A CNN encoder must compress this image into a
compact latent vector before it can be combined with the existing state observations and
fed into the actor and critic heads.

### Architecture at a glance

```
                     ┌─────────────────────────┐
  depth (1,128,128) ─┤  3 × Conv2d  (shared)   ├─── cnn_latent (128)
                     └─────────────────────────┘
                                                   ┌──────────────────┐
                              ┌─── concat ────────►│  2 × Linear      │──► actions (4)
  state (17)  ────────────────┤    (128+17=145)    │  (actor head)    │
                              │                    └──────────────────┘
                              │                    ┌──────────────────┐
                              └─── concat ────────►│  2 × Linear      │──► value (1)
                                   (128+17=145)    │  (critic head)   │
                                                   └──────────────────┘
```

**Key design decisions:**

| Decision | Rationale |
|----------|-----------|
| **Shared** CNN encoder | Depth features useful for both policy and value estimation. Halves CNN memory and compute. Gradients from both heads improve representations. |
| **Separate** MLP heads | Actor and critic have different objectives (actions vs. value). Separate heads prevent interference in the final layers. |
| CNN outputs a **128-dim** latent | Small enough to not dominate the state signal; large enough to capture spatial structure. Tunable. |

---

## 1. The rsl-rl v2.2.4 interface contract

`OnPolicyRunner` creates the model via (line 38–41 of `on_policy_runner.py`):

```python
actor_critic_class = eval(self.policy_cfg.pop("class_name"))
actor_critic = actor_critic_class(
    num_actor_obs, num_critic_obs, self.env.num_actions, **self.policy_cfg
)
```

The `PPO` algorithm then calls these methods on the model during training:

| Method | Called during | Must do |
|--------|-------------|---------|
| `act(obs, **kwargs)` | Rollout collection | Update distribution, return sampled actions |
| `evaluate(critic_obs, **kwargs)` | Rollout + update | Return value estimate `(B, 1)` |
| `get_actions_log_prob(actions)` | Rollout + update | Return log-prob of given actions `(B,)` |
| `act_inference(obs)` | Evaluation | Return deterministic actions (distribution mean) |
| `update_distribution(obs)` | Called by `act()` | Build the action distribution from observations |
| `reset(dones)` | After each step | Reset recurrent state (no-op for feedforward) |

**Properties** read by PPO/Runner:

| Property | Type | Used for |
|----------|------|----------|
| `action_mean` | `Tensor` | KL divergence computation, symmetry loss |
| `action_std` | `Tensor` | KL divergence computation, logging |
| `entropy` | `Tensor` | Entropy bonus in PPO loss |
| `is_recurrent` | `bool` | Selects mini-batch generator (flat vs. trajectory) |

### Observation flow

```
OnPolicyRunner.learn():
    obs, extras = env.get_observations()        # obs shape: what env returns
    critic_obs = extras["observations"]["critic"] # can differ from obs

    actions = alg.act(obs, critic_obs)           # PPO.act calls actor_critic.act(obs)
                                                 # and actor_critic.evaluate(critic_obs)
```

**Important:** in v2.2.4, `obs` and `critic_obs` are raw tensors (not TensorDicts). The
runner passes them as flat `(num_envs, num_obs)` tensors. For image observations, the
environment must pack state and image data into a single flat tensor — the model unpacks it.

---

## 2. Packing observations: flat tensor layout

Since v2.2.4's runner only handles flat tensors, the environment must flatten the depth
image and concatenate it with the state vector. The model reconstructs the image internally.

### Tensor layout convention

```
obs = [state_vector(17) | depth_flat(1×128×128=16384)]
       ◄─── 17 ────────►  ◄──────── 16384 ────────────►

total num_obs = 17 + 16384 = 16401
```

### Environment changes (`coordinate_landing_env.py`)

The following snippets show the **additions** needed. Existing code stays unchanged.

#### In `__init__`: update observation count

```python
# Before (state-only):
# self.num_obs = obs_cfg["num_obs"]  # 17

# After (state + depth):
self.depth_res = obs_cfg.get("depth_res", 128)        # image resolution
self.num_state_obs = obs_cfg["num_state_obs"]          # 17
self.num_obs = self.num_state_obs + self.depth_res ** 2  # 17 + 16384 = 16401
```

#### In `build()`: add a downward-facing camera

```python
# After scene.build():
self.camera = self.scene.add_camera(
    res=(self.depth_res, self.depth_res),
    pos=(0, 0, 3),
    lookat=(0, 0, 0),
    fov=70,
    GUI=False,
)
# NOTE: if adding camera post-build, use the bypass:
# self.camera = self.scene._visualizer.add_camera(
#     res=(self.depth_res, self.depth_res), ..., env_idx=0, debug=False
# )
# self.camera.build()
```

#### In `_compute_obs()`: pack state + depth

```python
def _compute_obs(self):
    s = self.obs_scales
    state = torch.cat([
        torch.clip(self.rel_pos * s["rel_pos"], -1, 1),
        self.base_quat,
        torch.clip(self.base_lin_vel * s["lin_vel"], -1, 1),
        torch.clip(self.base_ang_vel * s["ang_vel"], -1, 1),
        self.last_actions,
    ], dim=-1)  # (num_envs, 17)

    # Render depth — returns (rgb, depth, seg, normal)
    _, depth, _, _ = self.camera.render(depth=True)  # depth: (H, W) or (num_envs, H, W)
    depth_flat = depth.reshape(self.num_envs, -1)    # (num_envs, 16384)

    # Normalise depth to [0, 1] range (clip far values)
    max_depth = 20.0  # metres — tune to your scene
    depth_flat = torch.clamp(depth_flat / max_depth, 0.0, 1.0)

    return torch.cat([state, depth_flat], dim=-1)  # (num_envs, 16401)
```

#### Config changes (`obs_cfg`)

```python
obs_cfg = {
    "num_state_obs": 17,
    "num_obs": 16401,          # 17 + 128*128
    "depth_res": 128,
    "obs_scales": { ... },     # unchanged — only applies to state portion
}
```

---

## 3. The custom model

Create `prototyp_global_coordinate/networks/actor_critic_cnn.py`:

```python
"""
ActorCriticCNN: shared CNN encoder + separate actor/critic MLP heads.

Processes packed observations: [state(17) | depth_flat(1×H×W)].
The CNN encodes the depth image; the result is concatenated with the
state vector and fed into per-head MLPs.

Implements the rsl-rl v2.2.4 ActorCritic interface so it works as a
drop-in replacement with OnPolicyRunner and PPO.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal


class ActorCriticCNN(nn.Module):
    is_recurrent = False

    def __init__(
        self,
        num_actor_obs: int,
        num_critic_obs: int,
        num_actions: int,
        # --- custom params (passed via **policy_cfg) ---
        num_state_obs: int = 17,
        depth_res: int = 128,
        cnn_channels: list[int] = [32, 64, 128],
        cnn_kernel_sizes: list[int] = [8, 4, 3],
        cnn_strides: list[int] = [4, 2, 1],
        cnn_latent_dim: int = 128,
        actor_hidden_dims: list[int] = [256, 256],
        critic_hidden_dims: list[int] = [256, 256],
        activation: str = "elu",
        init_noise_std: float = 1.0,
        **kwargs,
    ):
        if kwargs:
            print(
                "ActorCriticCNN.__init__ got unexpected arguments, "
                "which will be ignored: " + str(list(kwargs.keys()))
            )
        super().__init__()

        self.num_state_obs = num_state_obs
        self.depth_res = depth_res
        self.num_actions = num_actions

        # ── Activation ──────────────────────────────────────────────
        activations = {
            "elu": nn.ELU, "relu": nn.ReLU, "tanh": nn.Tanh,
            "leaky_relu": nn.LeakyReLU, "selu": nn.SELU,
        }
        act_cls = activations.get(activation, nn.ELU)

        # ── Shared CNN encoder (3 Conv layers) ──────────────────────
        #
        # Input: (B, 1, depth_res, depth_res)  — single-channel depth
        # Output: (B, cnn_latent_dim)
        #
        cnn_layers: list[nn.Module] = []
        in_channels = 1  # depth map is single-channel
        for out_ch, ks, stride in zip(cnn_channels, cnn_kernel_sizes, cnn_strides):
            cnn_layers.append(nn.Conv2d(in_channels, out_ch, ks, stride))
            cnn_layers.append(nn.BatchNorm2d(out_ch))
            cnn_layers.append(act_cls())
            in_channels = out_ch

        cnn_layers.append(nn.AdaptiveAvgPool2d(1))   # (B, C, 1, 1)
        cnn_layers.append(nn.Flatten())               # (B, C)
        self.cnn = nn.Sequential(*cnn_layers)

        # Compute CNN output dim (= last channel count after global pool)
        cnn_out_dim = cnn_channels[-1]

        # Linear projection to fixed latent size
        self.cnn_proj = nn.Sequential(
            nn.Linear(cnn_out_dim, cnn_latent_dim),
            act_cls(),
        )

        # ── MLP input dimension ─────────────────────────────────────
        mlp_input_dim = num_state_obs + cnn_latent_dim

        # ── Actor head (2 hidden layers → actions) ──────────────────
        actor_layers: list[nn.Module] = []
        prev_dim = mlp_input_dim
        for hdim in actor_hidden_dims:
            actor_layers.append(nn.Linear(prev_dim, hdim))
            actor_layers.append(act_cls())
            prev_dim = hdim
        actor_layers.append(nn.Linear(prev_dim, num_actions))
        self.actor = nn.Sequential(*actor_layers)

        # ── Critic head (2 hidden layers → scalar value) ────────────
        critic_layers: list[nn.Module] = []
        prev_dim = mlp_input_dim
        for hdim in critic_hidden_dims:
            critic_layers.append(nn.Linear(prev_dim, hdim))
            critic_layers.append(act_cls())
            prev_dim = hdim
        critic_layers.append(nn.Linear(prev_dim, 1))
        self.critic = nn.Sequential(*critic_layers)

        # ── Action noise (learnable, state-independent) ─────────────
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution: Normal | None = None
        Normal.set_default_validate_args(False)

        print(f"CNN encoder: {self.cnn}")
        print(f"CNN projection: {self.cnn_proj}")
        print(f"Actor MLP: {self.actor}")
        print(f"Critic MLP: {self.critic}")

    # ── Observation unpacking ───────────────────────────────────────

    def _unpack_obs(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Split packed flat obs into state vector and depth image.

        Args:
            obs: (B, num_state_obs + depth_res²) flat tensor

        Returns:
            state: (B, num_state_obs)
            depth: (B, 1, depth_res, depth_res)
        """
        state = obs[:, :self.num_state_obs]
        depth_flat = obs[:, self.num_state_obs:]
        depth = depth_flat.reshape(-1, 1, self.depth_res, self.depth_res)
        return state, depth

    def _encode(self, obs: torch.Tensor) -> torch.Tensor:
        """Encode packed observation into combined feature vector.

        Args:
            obs: (B, num_state_obs + depth_res²)

        Returns:
            features: (B, num_state_obs + cnn_latent_dim)
        """
        state, depth = self._unpack_obs(obs)
        cnn_features = self.cnn_proj(self.cnn(depth))  # (B, cnn_latent_dim)
        return torch.cat([state, cnn_features], dim=-1)

    # ── rsl-rl v2.2.4 interface ─────────────────────────────────────

    def reset(self, dones=None):
        """No-op for feedforward models."""
        pass

    def forward(self):
        raise NotImplementedError

    @property
    def action_mean(self) -> torch.Tensor:
        return self.distribution.mean

    @property
    def action_std(self) -> torch.Tensor:
        return self.distribution.stddev

    @property
    def entropy(self) -> torch.Tensor:
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, observations: torch.Tensor):
        """Build Gaussian action distribution from observations."""
        features = self._encode(observations)
        mean = self.actor(features)
        std = self.std.expand_as(mean)
        self.distribution = Normal(mean, std)

    def act(self, observations: torch.Tensor, **kwargs) -> torch.Tensor:
        """Sample actions from the policy (used during rollout collection)."""
        self.update_distribution(observations)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        """Log probability of given actions under current distribution."""
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations: torch.Tensor) -> torch.Tensor:
        """Deterministic action (distribution mean), used for evaluation."""
        features = self._encode(observations)
        return self.actor(features)

    def evaluate(self, critic_observations: torch.Tensor, **kwargs) -> torch.Tensor:
        """Compute value estimate (used by critic during training)."""
        features = self._encode(critic_observations)
        return self.critic(features)
```

### What each section does

**`_unpack_obs`** — Splits the flat packed tensor back into a 17-dim state vector and a
`(1, 128, 128)` depth image. This is the inverse of the environment's packing in
`_compute_obs()`.

**`_encode`** — Runs the shared CNN on the depth image, projects to a fixed-size latent,
then concatenates with the state vector. Both actor and critic call this, so the CNN
weights receive gradients from both heads.

**Shared CNN (3 layers):**
```
Conv2d(1→32, k=8, s=4)  +  BN  +  ELU    →  (B, 32, 31, 31)
Conv2d(32→64, k=4, s=2)  +  BN  +  ELU    →  (B, 64, 14, 14)
Conv2d(64→128, k=3, s=1)  +  BN  +  ELU    →  (B, 128, 12, 12)
AdaptiveAvgPool2d(1)                        →  (B, 128, 1, 1)
Flatten                                     →  (B, 128)
Linear(128→128) + ELU                       →  (B, 128)  ← cnn_latent
```

**Actor head (2 layers):** `Linear(145→256) + ELU → Linear(256→256) + ELU → Linear(256→4)`

**Critic head (2 layers):** `Linear(145→256) + ELU → Linear(256→256) + ELU → Linear(256→1)`

---

## 4. Training config changes

```python
def get_train_cfg(exp_name, max_iterations):
    return {
        "algorithm": {
            "class_name": "PPO",
            "clip_param": 0.2,
            "desired_kl": 0.01,
            "entropy_coef": 0.005,       # slightly > 0 encourages exploration
            "gamma": 0.99,
            "lam": 0.95,
            "learning_rate": 3e-4,
            "max_grad_norm": 1.0,
            "num_learning_epochs": 5,
            "num_mini_batches": 4,
            "schedule": "adaptive",
            "use_clipped_value_loss": True,
            "value_loss_coef": 1.0,
        },
        "policy": {
            "class_name": "ActorCriticCNN",  # ← our custom class
            "num_state_obs": 17,
            "depth_res": 128,
            "cnn_channels": [32, 64, 128],
            "cnn_kernel_sizes": [8, 4, 3],
            "cnn_strides": [4, 2, 1],
            "cnn_latent_dim": 128,
            "actor_hidden_dims": [256, 256],
            "critic_hidden_dims": [256, 256],
            "activation": "elu",
            "init_noise_std": 1.0,
        },
        # ... runner, num_steps_per_env, etc. unchanged
    }
```

---

## 5. Registering the class with OnPolicyRunner

`OnPolicyRunner` resolves `class_name` via `eval()` (line 38). For `eval("ActorCriticCNN")`
to work, the class must be in scope at the call site — which is inside `on_policy_runner.py`
in the `rsl_rl` package.

### Option A: Inject into builtins (simplest, recommended)

In `train_rl.py`, before creating the runner:

```python
import builtins
from networks.actor_critic_cnn import ActorCriticCNN
builtins.ActorCriticCNN = ActorCriticCNN

runner = OnPolicyRunner(env, train_cfg, log_dir, device=gs.device)
```

This makes `ActorCriticCNN` resolvable by `eval()` from any module.

### Option B: Monkey-patch the rsl-rl module namespace

```python
import rsl_rl.modules
from networks.actor_critic_cnn import ActorCriticCNN
rsl_rl.modules.ActorCriticCNN = ActorCriticCNN

# Also patch the runner's import namespace so eval() finds it:
import rsl_rl.runners.on_policy_runner as runner_module
runner_module.ActorCriticCNN = ActorCriticCNN
```

### Option C: Use a fully qualified class name

If the module is importable:

```python
"policy": {
    "class_name": "networks.actor_critic_cnn.ActorCriticCNN",
    ...
}
```

This only works if `eval()` can resolve it — which requires the module path to be
importable. Since `train_rl.py` runs from `prototyp_global_coordinate/`, the `networks`
package must be importable from there (add `__init__.py` if needed).

**Note:** `eval()` with a dotted path won't work directly. You'd need to modify the
runner or use Option A/B. The simplest approach is **Option A**.

---

## 6. Evaluation compatibility

The existing `eval_rl.py` uses `get_inference_policy()` which calls
`actor_critic.act_inference(obs)`. This already works with our model since we implemented
`act_inference`. However, `eval_rl.py` also loads checkpoints via `runner.load()` which
calls `actor_critic.load_state_dict()`. As long as the model class is registered (same
as during training), loading works transparently.

Make sure `eval_rl.py` also registers the class:
```python
import builtins
from networks.actor_critic_cnn import ActorCriticCNN
builtins.ActorCriticCNN = ActorCriticCNN
```

---

## 7. Performance considerations

### GPU memory

| Component | Parameters (approx.) |
|-----------|---------------------|
| CNN (3 conv + BN) | ~120K |
| CNN projection | ~16.5K |
| Actor MLP (2 layers) | ~103K |
| Critic MLP (2 layers) | ~103K |
| Action std | 4 |
| **Total** | **~343K** |

Compare to the current state-only model: ~35K parameters. The CNN adds ~10× more
parameters, but this is still small by deep learning standards.

### Rendering bottleneck

Camera rendering is the likely bottleneck — rendering 4096 depth images at 128×128 every
step is expensive. Strategies to mitigate:

- **Reduce resolution**: 64×64 depth maps may suffice (change `depth_res`). This 4×
  reduces rendering cost and observation size.
- **Skip frames**: Render depth every N steps, reuse the last image in between.
  Update the environment's step function:
  ```python
  if self.episode_length_buf[0] % render_interval == 0:
      self.last_depth = self._render_depth()
  # Always use self.last_depth in _compute_obs
  ```
- **Reduce num_envs**: Use fewer parallel environments (e.g., 512 instead of 4096)
  to fit rendering in GPU memory.
- **Async rendering**: Not supported in Genesis v0.3.13, but worth checking in
  future versions.

### Observation buffer size

With 16401 floats per env × 4096 envs × 4 bytes = ~256 MB for the observation buffer
alone. The rollout storage holds `num_steps_per_env` timesteps, so:

```
256 MB × 100 steps = ~25 GB for rollout storage
```

This will likely **exceed GPU memory**. Solutions:

1. **Reduce `num_steps_per_env`** from 100 to 24 (standard for image-based RL)
2. **Reduce `num_envs`** from 4096 to 256–512
3. **Reduce `depth_res`** from 128 to 64 (cuts storage by 4×)
4. **Reduce `num_mini_batches`** or `num_learning_epochs` (less impact)

A reasonable starting configuration:
```python
num_envs = 256
num_steps_per_env = 24
depth_res = 64
```

This gives: `(17 + 64²) × 256 × 24 × 4 bytes ≈ 100 MB` — manageable.

---

## 8. Training tips for vision-based RL

1. **Pre-train state-only first.** Train the current 17-dim state policy until it
   reliably lands, then add the CNN and fine-tune. This gives the MLP heads a good
   starting point.

2. **Freeze CNN initially.** For the first ~100 iterations, freeze CNN weights
   (`requires_grad=False`) and only train the MLP heads. This prevents early random
   CNN features from destabilising training.
   ```python
   # After creating runner, before learn():
   for p in runner.alg.actor_critic.cnn.parameters():
       p.requires_grad = False
   for p in runner.alg.actor_critic.cnn_proj.parameters():
       p.requires_grad = False
   ```

3. **Lower learning rate for CNN.** Use separate parameter groups:
   ```python
   # In train_rl.py after creating runner:
   runner.alg.optimizer = torch.optim.Adam([
       {"params": runner.alg.actor_critic.cnn.parameters(), "lr": 1e-4},
       {"params": runner.alg.actor_critic.cnn_proj.parameters(), "lr": 1e-4},
       {"params": runner.alg.actor_critic.actor.parameters(), "lr": 3e-4},
       {"params": runner.alg.actor_critic.critic.parameters(), "lr": 3e-4},
       {"params": [runner.alg.actor_critic.std], "lr": 3e-4},
   ])
   ```

4. **Depth normalisation matters.** Raw depth values can range from 0 to infinity.
   Clamp to a max distance (e.g., 20m) and scale to [0, 1]. Consider log-depth:
   `depth_norm = torch.log1p(depth) / torch.log1p(max_depth)`.

5. **Data augmentation.** Random crops, small translations, or noise on depth images
   can improve generalisation. Apply during `_compute_obs()`.

---

## 9. File structure after implementation

```
prototyp_global_coordinate/
  networks/
    __init__.py
    actor_critic_cnn.py     ← the custom model (Section 3)
  envs/
    coordinate_landing_env.py  ← modified to pack depth (Section 2)
  train_rl.py                  ← updated config + class registration (Sections 4–5)
  eval_rl.py                   ← add class registration (Section 6)
  ...
```

---

## 10. Checklist

- [ ] Create `networks/__init__.py` and `networks/actor_critic_cnn.py`
- [ ] Update `obs_cfg` to include `num_state_obs`, `depth_res`, and new `num_obs`
- [ ] Add Genesis camera in `build()`
- [ ] Modify `_compute_obs()` to pack depth into flat tensor
- [ ] Update `train_rl.py` policy config to use `ActorCriticCNN`
- [ ] Register class via `builtins` in `train_rl.py` and `eval_rl.py`
- [ ] Reduce `num_envs` and `num_steps_per_env` for memory
- [ ] Smoke test: `python train_rl.py -B 4 -v --max_iterations 2`
- [ ] Verify checkpoint save/load round-trip
- [ ] Monitor TensorBoard for training stability
