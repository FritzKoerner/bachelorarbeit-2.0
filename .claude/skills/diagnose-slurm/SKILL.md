---
name: diagnose-slurm
description: Classify a SLURM job log from hpc_results/slurm/ against this project's known failure modes (OOM, time limit, NaN physics crash, viewer-close GenesisException, CoACD stall, BatchRenderer-on-Volta abort, CUDA init error). Use when a returned HPC job was killed or produced unexpected output. Input arg can be a path to slurm-*.out/.err or a bare job ID.
---

# diagnose-slurm

Read-only classification of SLURM job logs against the failure modes catalogued in this project's `CLAUDE.md`. Intended to collapse "open log, skim, guess, grep, cross-reference CLAUDE.md" into one command.

## Inputs

- **Path form**: `hpc_results/slurm/slurm-<jobid>.out` or `.err`.
- **Bare ID form**: just `<jobid>` — resolve with `ls hpc_results/slurm/slurm-<jobid>.*`.

If both `.out` and `.err` exist, read both. `.err` holds OOM notices and Python exceptions; `.out` has training stdout.

## Classification table

Grep in order — stop at the first match.

| Signature (regex-ish) | Classification | Documented fix |
|---|---|---|
| `oom_kill event` / final `Killed` after `python` line | **OOM-kill (host RAM)** | If killed during `record_landing.py` → already fixed by streaming writer; if during training → raise SLURM `--mem` or drop `-B` batch size. Reference: CLAUDE.md "Video recording" note. |
| `DUE TO TIME LIMIT` / `CANCELLED AT` | **Walltime exceeded** | Resubmit to `clara-long` (10-day) or lower `--max_iterations`. |
| `NaN` / `inf` in solver / `nan acceleration` | **Physics instability** | Env already zeros unstable drones (tilt > 60°, alt < 0.2m) — but protection is reactive. Suggest reducing decimation or tightening the instability threshold. Reference: past session investigation. |
| `GenesisException` mentioning `viewer.update` / `viewer closed` | **Viewer closed mid-run** | `-v` is dev-only. Resubmit headless. Reference: CLAUDE.md "Viewer mode (-v) crashes on window close". |
| Long hang (>30s) near `CoACD` on first build, no subsequent error | **First-run mesh decomposition** — NOT a failure | Expected ~60s on first run per CLAUDE.md "CoACD convex decomposition"; cached afterward. Do not flag as a bug. |
| `abort()` shortly after startup, no Python traceback | **BatchRenderer on Volta** | `clara`'s V100 lacks compute capability 7.5. Resubmit to `paula` (A30) or set `use_batch_renderer: False`. Reference: CLAUDE.md "BatchRenderer requires Turing+ GPU". |
| `CUDA_VISIBLE_DEVICES` mismatch / `cuda:0` vs `cuda:<n>` device assert | **Multi-GPU device allocation** | Genesis Quadrants always allocates on `cuda:0`. If using `train_rl_multigpu.py`, verify the `CUDA_VISIBLE_DEVICES` remap at the top of the script ran before any CUDA import. |
| `AttributeError: ... set_propellels_rpm` / `set_quat` relative-arg mismatch | **Genesis version bump broke env code** | Repo is pinned to v0.4.3; check if environment somehow got v0.4.4+. Reference: CLAUDE.md "v0.4.4+ latent breakage". |
| None of the above, but file ends cleanly with `Training complete` or similar | **Success** | Report and stop — don't invent failure. |
| None of the above, non-clean end | **Unknown** | Tail last 30 lines of `.err` and top 30 of `.out`; propose a specific next grep. |

## Output format

Short, no framing. One run of the skill produces:

```
Job <id>: <classification>
Cause: <one sentence>
Fix: <one sentence, ideally citing CLAUDE.md section>
[If unknown: tail excerpt for the user to inspect]
```

## Notes

- This is a **read-only** skill. Do not Edit / Write / run shell commands that modify state.
- Do not open `.pt`, `.mp4`, or `.gif` files — binary artifacts may co-exist in the job directory.
- Never resubmit a job or touch HPC scripts (they are hook-blocked anyway per `.claude/settings.json`).
