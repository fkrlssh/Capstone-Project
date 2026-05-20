"""
Append-only training log helpers.

- Global episode index continues across main.py restarts.
- Plot script tracks how many CSV rows were already exported to TensorBoard.
"""
from __future__ import annotations

import csv
import json
import os
from typing import Any, Dict, List, Optional

from config import CFG

EPISODE_LOG_FIELDS = [
    "episode",
    "session_episode",
    "obstacle_density",
    "stop_reason",
    "global_observed",
    "global_visited",
    "global_obstacle",
    "global_unknown",
    "local_window",
    "local_observed",
    "local_visited",
    "local_obstacle",
    "local_unknown",
    "collisions",
    "global_visited_delta",
    "min_target_dist_m",
    "target_progress",
    "coverage_shaping_reward",
    "target_shaping_reward",
    "total_reward",
    "agent_rewards",
]


def episode_log_path() -> str:
    return getattr(CFG, "EPISODE_CSV", "outputs/episode_log_dynamic_v7_clock1_fast_visit.csv")


def plot_state_path() -> str:
    return os.path.join(
        getattr(CFG, "OUTPUT_DIR", "outputs"),
        getattr(CFG, "RL_PLOT_STATE_JSON", "rl_plot_state.json"),
    )


def read_max_global_episode(csv_path: Optional[str] = None) -> int:
    path = csv_path or episode_log_path()
    if not os.path.exists(path):
        return 0
    max_ep = 0
    try:
        with open(path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames or "episode" not in reader.fieldnames:
                return 0
            for row in reader:
                try:
                    max_ep = max(max_ep, int(float(row.get("episode", 0) or 0)))
                except (TypeError, ValueError):
                    continue
    except OSError:
        return 0
    return max_ep


def count_episode_rows(csv_path: Optional[str] = None) -> int:
    path = csv_path or episode_log_path()
    if not os.path.exists(path):
        return 0
    try:
        with open(path, "r", newline="", encoding="utf-8") as f:
            return max(0, sum(1 for _ in csv.DictReader(f)))
    except OSError:
        return 0


def load_plot_state(path: Optional[str] = None) -> Dict[str, Any]:
    path = path or plot_state_path()
    if not os.path.exists(path):
        return {"last_tb_global_step": 0, "last_csv_rows": 0}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return {
            "last_tb_global_step": int(data.get("last_tb_global_step", 0) or 0),
            "last_csv_rows": int(data.get("last_csv_rows", 0) or 0),
        }
    except (OSError, json.JSONDecodeError):
        return {"last_tb_global_step": 0, "last_csv_rows": 0}


def save_plot_state(
    last_tb_global_step: int,
    last_csv_rows: int,
    path: Optional[str] = None,
) -> None:
    path = path or plot_state_path()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "last_tb_global_step": int(last_tb_global_step),
                "last_csv_rows": int(last_csv_rows),
            },
            f,
            indent=2,
            ensure_ascii=False,
        )


def load_episode_rows_raw(csv_path: Optional[str] = None) -> List[Dict[str, str]]:
    path = csv_path or episode_log_path()
    if not os.path.exists(path):
        return []
    with open(path, "r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))
