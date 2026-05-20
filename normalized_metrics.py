"""
Episode / Q-table metrics normalized to [0, 1] for comparable training dashboards.

Used by export_normalized_indicators.py and train_background.py (periodic refresh).
"""
from __future__ import annotations

import csv
import json
import os
from typing import Any, Dict, List, Optional

from config import CFG


def _clip01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def _sym_reward01(reward: float, ceiling: float) -> float:
    """Map reward in [-ceiling, +ceiling] to [0, 1]."""
    c = max(1.0, float(ceiling))
    return _clip01(0.5 + 0.5 * float(reward) / c)


def parse_agent_rewards(raw: str) -> Dict[str, float]:
    if not raw or not str(raw).strip():
        return {}
    try:
        import ast

        data = ast.literal_eval(raw)
        if isinstance(data, dict):
            return {str(k): float(v) for k, v in data.items()}
    except (SyntaxError, ValueError):
        pass
    return {}


def load_episode_rows(csv_path: str, last_n: Optional[int] = None) -> List[Dict[str, Any]]:
    if not os.path.exists(csv_path):
        return []
    rows: List[Dict[str, Any]] = []
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader, start=1):
            stop = str(row.get("stop_reason", ""))
            try:
                episode = int(float(row.get("episode", i) or i))
            except (TypeError, ValueError):
                episode = i
            gv = float(row.get("global_visited", 0.0) or 0.0)
            gvd = float(row.get("global_visited_delta", 0.0) or 0.0)
            tp = float(row.get("target_progress", 0.0) or 0.0)
            mtd = float(row.get("min_target_dist_m", 0.0) or 0.0)
            total_reward = float(row.get("total_reward", 0.0) or 0.0)
            csr = float(row.get("coverage_shaping_reward", 0.0) or 0.0)
            tsr = float(row.get("target_shaping_reward", 0.0) or 0.0)
            collisions = int(float(row.get("collisions", 0) or 0))
            agent_rewards = parse_agent_rewards(row.get("agent_rewards", ""))
            rows.append(
                {
                    "global_step": i,
                    "episode": episode,
                    "stop_reason": stop,
                    "collision_episode": "EPISODE_COLLISION" in stop,
                    "target_reached": stop.startswith("TARGET_REACHED"),
                    "global_visited": gv,
                    "global_visited_delta": gvd,
                    "min_target_dist_m": mtd,
                    "target_progress": tp,
                    "coverage_shaping_reward": csr,
                    "target_shaping_reward": tsr,
                    "total_reward": total_reward,
                    "collisions": collisions,
                    "agent_rewards": agent_rewards,
                }
            )
    if last_n is not None and last_n > 0 and len(rows) > last_n:
        rows = rows[-last_n:]
    return rows


def norm_target_dist(min_dist_m: float) -> float:
    """1 = at target, 0 = at or beyond EPISODE_TARGET_NORM_DIST_M."""
    d = max(0.0, float(min_dist_m))
    ref = max(1.0, float(getattr(CFG, "EPISODE_TARGET_NORM_DIST_M", 200.0)))
    return _clip01(1.0 - d / ref)


def normalize_episode_row(row: Dict[str, Any]) -> Dict[str, float]:
    reward_ceiling = float(getattr(CFG, "METRIC_REWARD_NORM_CEILING", 120.0))
    cov_scale = max(1.0, float(getattr(CFG, "EPISODE_COVERAGE_REWARD_SCALE", 80.0)))
    tgt_scale = max(1.0, float(getattr(CFG, "TARGET_PROXIMITY_REWARD_SCALE", 45.0)))
    delta_cap = float(getattr(CFG, "METRIC_VISITED_DELTA_NORM", 0.15))

    agent_vals = list(row.get("agent_rewards", {}).values())
    min_agent = min(agent_vals) if agent_vals else 0.0

    norm = {
        "norm_global_visited": _clip01(row.get("global_visited", 0.0)),
        "norm_visited_delta": _clip01(float(row.get("global_visited_delta", 0.0)) / delta_cap),
        "norm_target_progress": _clip01(row.get("target_progress", 0.0)),
        "norm_target_dist": norm_target_dist(row.get("min_target_dist_m", 0.0)),
        "norm_collision_free": 0.0 if row.get("collision_episode") else 1.0,
        "norm_success": 1.0 if row.get("target_reached") else 0.0,
        "norm_total_reward": _sym_reward01(row.get("total_reward", 0.0), reward_ceiling),
        "norm_min_agent_reward": _sym_reward01(min_agent, reward_ceiling),
        "norm_coverage_shaping": _clip01(
            float(row.get("coverage_shaping_reward", 0.0)) / cov_scale
        ),
        "norm_target_shaping": _clip01(
            float(row.get("target_shaping_reward", 0.0)) / tgt_scale
        ),
        "norm_low_collisions": _clip01(1.0 - float(row.get("collisions", 0)) / 3.0),
    }

    w = getattr(CFG, "METRIC_COMPOSITE_WEIGHTS", None) or {
        "norm_collision_free": 0.20,
        "norm_success": 0.20,
        "norm_global_visited": 0.15,
        "norm_visited_delta": 0.15,
        "norm_target_progress": 0.15,
        "norm_total_reward": 0.15,
    }
    score = 0.0
    weight_sum = 0.0
    for key, wgt in w.items():
        if key in norm:
            score += float(wgt) * norm[key]
            weight_sum += float(wgt)
    norm["norm_training_score"] = score / weight_sum if weight_sum > 0 else 0.0
    return norm


def load_qtable_norm_stats(qtable_path: str) -> Dict[str, float]:
    if not os.path.exists(qtable_path):
        return {}
    with open(qtable_path, "r", encoding="utf-8") as f:
        q = json.load(f)
    if not q:
        return {}
    vals: List[float] = []
    danger_reverse: List[float] = []
    for state_key, actions in q.items():
        for action, value in actions.items():
            v = float(value)
            vals.append(v)
            if action == "reverse" and "|danger|" in state_key:
                danger_reverse.append(v)
    if not vals:
        return {}

    q_min = min(vals)
    q_max = max(vals)
    span = max(1e-6, q_max - q_min)
    q_mean = sum(vals) / len(vals)
    danger_min = min(danger_reverse) if danger_reverse else q_min

    coll_penalty = abs(float(getattr(CFG, "RL_COLLISION_REWARD", -80.0)))
    return {
        "norm_q_span": _clip01((q_max - q_min) / coll_penalty),
        "norm_q_mean": _sym_reward01(q_mean, coll_penalty),
        "norm_q_danger_escape": _sym_reward01(danger_min, coll_penalty),
        "norm_q_states": _clip01(len(q) / float(getattr(CFG, "METRIC_Q_STATES_NORM", 64.0))),
    }


def build_indicator_records(
    csv_path: Optional[str] = None,
    qtable_path: Optional[str] = None,
    last_n: Optional[int] = None,
) -> List[Dict[str, Any]]:
    csv_path = csv_path or getattr(CFG, "EPISODE_CSV", "outputs/episode_log.csv")
    qtable_path = qtable_path or getattr(CFG, "RL_SAVE_PATH", "outputs/rl_collision_qtable.json")
    rows = load_episode_rows(csv_path, last_n=last_n)
    q_norm = load_qtable_norm_stats(qtable_path)
    out: List[Dict[str, Any]] = []
    for row in rows:
        rec = {
            "episode": int(row["episode"]),
            "global_step": int(row["global_step"]),
            "stop_reason": row["stop_reason"],
        }
        rec.update(normalize_episode_row(row))
        out.append(rec)
    if out and q_norm:
        out[-1].update(q_norm)
    return out


def rolling_mean_dict(
    records: List[Dict[str, Any]], key: str, window: int
) -> List[float]:
    vals = [float(r.get(key, 0.0)) for r in records]
    if window < 1:
        window = 1
    out: List[float] = []
    for i in range(len(vals)):
        start = max(0, i - window + 1)
        chunk = vals[start : i + 1]
        out.append(sum(chunk) / len(chunk))
    return out
