#!/usr/bin/env python3
"""Extract ALL training/eval data from W&B and TensorBoard into JSON files.

Outputs (in eval_data/):
  wandb_training_{run_name}.json   - Full scalar history for each training run
  wandb_eval_{run_name}.json       - Full scalar history for each eval run
  wandb_eval_table_{run_name}.json - Per-episode eval table for each eval run
  wandb_config_{run_name}.json     - Run config for each run
  tb_training_{run_name}.json      - TensorBoard scalars for each training run
"""

import json
import os
import sys
from pathlib import Path

import wandb
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

OUT_DIR = Path(__file__).parent / "eval_data"
LOGS_DIR = Path(__file__).parent / "logs"
WANDB_PROJECT = "drone-landing"

# Run name -> W&B run ID mapping
TRAINING_RUNS = {
    "drone-landing": "8iahsb8m",
    "increased-rel_pos-scale": "sz6pja68",
    "no-curriculum": "fuw7uhmc",
}

EVAL_RUNS = {
    "drone-landing-eval-iter400": "fxjmkc3e",
    "increased-rel_pos-scale-eval-iter400": "634zhfye",
    "no-curriculum-eval-iter400": "om50timv",
}

# TB log dirs (same names as training runs)
TB_DIRS = {
    "drone-landing": LOGS_DIR / "drone-landing",
    "increased-rel_pos-scale": LOGS_DIR / "increased-rel_pos-scale",
    "no-curriculum": LOGS_DIR / "no-curriculum",
}


def save_json(data, filename):
    path = OUT_DIR / filename
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    print(f"  Saved {path} ({os.path.getsize(path)} bytes)")


def extract_wandb_history(api, run_id, run_name, prefix):
    """Pull full scalar history from a W&B run."""
    print(f"Extracting W&B history: {run_name} ({run_id})")
    run = api.run(f"{WANDB_PROJECT}/{run_id}")
    # samples=10000 to get all rows (max 401 for training, 2 for eval)
    history = run.history(samples=10000, pandas=False)
    save_json(history, f"wandb_{prefix}_{run_name}.json")
    return len(history)


def extract_wandb_config(api, run_id, run_name):
    """Pull run config from W&B."""
    print(f"Extracting W&B config: {run_name}")
    run = api.run(f"{WANDB_PROJECT}/{run_id}")
    config = dict(run.config)
    # Also grab summary metrics
    summary = {}
    for k, v in run.summary.items():
        if not k.startswith("_"):
            try:
                json.dumps(v)
                summary[k] = v
            except (TypeError, ValueError):
                summary[k] = str(v)
    data = {
        "name": run.name,
        "id": run.id,
        "state": run.state,
        "created_at": str(run.created_at),
        "config": config,
        "summary": summary,
    }
    save_json(data, f"wandb_config_{run_name}.json")


def extract_eval_tables(api, run_id, run_name):
    """Download per-episode eval tables from W&B artifacts."""
    print(f"Extracting eval table: {run_name} ({run_id})")
    run = api.run(f"{WANDB_PROJECT}/{run_id}")
    table_found = False
    for art in run.logged_artifacts():
        if art.type == "run_table":
            art_dir = art.download()
            # Find the .table.json file
            for root, dirs, files in os.walk(art_dir):
                for f in files:
                    if f.endswith(".table.json"):
                        fp = os.path.join(root, f)
                        with open(fp) as fh:
                            table_data = json.load(fh)
                        # Extract into a cleaner format
                        columns = table_data["columns"]
                        rows = table_data["data"]
                        episodes = [dict(zip(columns, row)) for row in rows]
                        result = {
                            "run_name": run_name,
                            "run_id": run_id,
                            "columns": columns,
                            "num_episodes": len(episodes),
                            "episodes": episodes,
                        }
                        save_json(result, f"wandb_eval_table_{run_name}.json")
                        table_found = True
    if not table_found:
        print(f"  WARNING: No table artifact found for {run_name}")


def extract_tensorboard(tb_dir, run_name):
    """Read all scalar data from TensorBoard event files."""
    print(f"Extracting TensorBoard: {run_name} ({tb_dir})")
    ea = EventAccumulator(str(tb_dir))
    ea.Reload()

    tags = ea.Tags().get("scalars", [])
    print(f"  Tags: {tags}")

    tb_data = {"run_name": run_name, "tags": tags, "scalars": {}}
    for tag in tags:
        scalars = ea.Scalars(tag)
        tb_data["scalars"][tag] = [
            {"wall_time": s.wall_time, "step": s.step, "value": s.value}
            for s in scalars
        ]
        print(f"    {tag}: {len(scalars)} entries")

    save_json(tb_data, f"tb_training_{run_name}.json")


def main():
    OUT_DIR.mkdir(exist_ok=True)
    api = wandb.Api()

    print("=" * 60)
    print("1. W&B Training Run Histories")
    print("=" * 60)
    for name, rid in TRAINING_RUNS.items():
        n = extract_wandb_history(api, rid, name, "training")
        print(f"    -> {n} rows")

    print()
    print("=" * 60)
    print("2. W&B Eval Run Histories")
    print("=" * 60)
    for name, rid in EVAL_RUNS.items():
        n = extract_wandb_history(api, rid, name, "eval")
        print(f"    -> {n} rows")

    print()
    print("=" * 60)
    print("3. W&B Eval Tables (per-episode)")
    print("=" * 60)
    for name, rid in EVAL_RUNS.items():
        extract_eval_tables(api, rid, name)

    print()
    print("=" * 60)
    print("4. W&B Run Configs")
    print("=" * 60)
    for name, rid in {**TRAINING_RUNS, **EVAL_RUNS}.items():
        extract_wandb_config(api, rid, name)

    print()
    print("=" * 60)
    print("5. TensorBoard Scalars")
    print("=" * 60)
    for name, tb_dir in TB_DIRS.items():
        if tb_dir.exists():
            extract_tensorboard(tb_dir, name)
        else:
            print(f"  SKIP: {tb_dir} does not exist")

    print()
    print("=" * 60)
    print("DONE. All files in:", OUT_DIR)
    print("=" * 60)
    for f in sorted(OUT_DIR.iterdir()):
        print(f"  {f.name:55s} {os.path.getsize(f):>10,} bytes")


if __name__ == "__main__":
    main()
