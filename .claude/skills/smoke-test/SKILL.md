---
name: smoke-test
description: Run the 4-env viewer smoke test for the drone-landing prototype in the user's current working directory (or a named one). Exercises env reset, PPO rollout, model update, and W&B log in ~30s — the fastest way to validate an env or reward change before an HPC submission. Accepts optional --env-v2 for obstacle_avoidance.
---

# smoke-test

Runs the canonical pre-HPC validation command from CLAUDE.md:

```bash
python train_rl_wb.py -B 4 -v --max_iterations 5 [--env-v2]
```

## Procedure

1. **Detect prototype** from the user's cwd:
   - Under `prototyp_global_coordinate/` → run there.
   - Under `prototyp_obstacle_avoidance/` → run there; append `--env-v2` if requested or if the last run used v2.
   - Repo root or elsewhere → ask which prototype.

2. **Run** (foreground, let stdout stream so the user sees it):

   ```bash
   cd <prototype_dir> && python train_rl_wb.py -B 4 -v --max_iterations 5 [--env-v2]
   ```

3. **Classify the outcome** — match against the same failure modes the `diagnose-slurm` skill catalogues:
   - Viewer-close `GenesisException` → remind the user not to close the window.
   - `NaN` / unstable-drone messages → the env change destabilized physics; suggest reducing decimation or tightening the instability threshold.
   - Import error → deps missing (`ba_v04` conda env not activated, or a new dep not installed).
   - 5 iterations complete + W&B URL printed → **smoke test OK**; report the URL.

## Notes

- Uses `-v` (viewer), so this only works on a local graphical session. Never run this on HPC — the submission scripts use headless mode.
- Small `-B 4` keeps build time ~30s including CoACD convex decomposition cache warmup.
- `--max_iterations 5` is enough to hit: env reset → rollout → `PPO.update()` → W&B flush. Most regressions manifest in iter 1–2.
- This skill just runs the command and interprets output — it does not modify code or rewrite configs.
