"""
Collision / tabular-RL training metrics -> TensorBoard (and optional PNG).

Data sources:
  - outputs/episode_log_*.csv  (per-episode stop_reason, collisions, rewards, coverage)
  - outputs/rl_collision_qtable.json  (current Q-table snapshot)
  - outputs/rl_training_history.jsonl  (appended each run for Q-table trends over time)

Usage (from 11_test):
  pip install tensorflow
  python plot_collision_learning_tf.py
  python plot_collision_learning_tf.py --window 25
  python plot_collision_learning_tf.py --last-n 100
  tensorboard --logdir outputs/tensorboard/rl_collision

While main.py is running, re-run this script periodically to refresh curves.
"""
from __future__ import annotations

import argparse
import ast
import csv
import json
import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple

from config import CFG
from training_log_utils import load_plot_state, save_plot_state

try:
    import tensorflow as tf
except ImportError as exc:
    raise SystemExit(
        "TensorFlow is required: pip install tensorflow\n"
        f"Original error: {exc}"
    ) from exc


def parse_agent_rewards(raw: str) -> Dict[str, float]:
    if not raw or not str(raw).strip():
        return {}
    try:
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
        for row in reader:
            stop = str(row.get("stop_reason", ""))
            global_step = len(rows) + 1
            try:
                global_episode = int(float(row.get("episode", global_step) or global_step))
            except (TypeError, ValueError):
                global_episode = global_step
            session_episode = row.get("session_episode", "")
            try:
                session_episode = int(float(session_episode)) if str(session_episode).strip() else 0
            except (TypeError, ValueError):
                session_episode = 0
            gv = float(row.get("global_visited", 0.0) or 0.0)
            gvd = row.get("global_visited_delta", "")
            if str(gvd).strip() == "":
                gvd_f = 0.0
            else:
                gvd_f = float(gvd)
            tp = float(row.get("target_progress", 0.0) or 0.0)
            csr = float(row.get("coverage_shaping_reward", 0.0) or 0.0)
            tsr = float(row.get("target_shaping_reward", 0.0) or 0.0)
            rows.append(
                {
                    "global_step": global_step,
                    "episode": global_episode,
                    "session_episode": session_episode,
                    "stop_reason": stop,
                    "collision_episode": "EPISODE_COLLISION" in stop,
                    "target_reached": stop.startswith("TARGET_REACHED"),
                    "timeout": "EPISODE_TIMEOUT" in stop,
                    "global_map_complete": stop.startswith("GLOBAL_MAP_COMPLETE"),
                    "collisions": int(float(row.get("collisions", 0) or 0)),
                    "total_reward": float(row.get("total_reward", 0.0) or 0.0),
                    "global_visited": gv,
                    "global_visited_delta": gvd_f,
                    "min_target_dist_m": float(row.get("min_target_dist_m", 0.0) or 0.0),
                    "target_progress": tp,
                    "coverage_shaping_reward": csr,
                    "target_shaping_reward": tsr,
                    "shaping_reward": csr + tsr,
                    "global_observed": float(row.get("global_observed", 0.0) or 0.0),
                    "local_visited": float(row.get("local_visited", 0.0) or 0.0),
                    "agent_rewards": parse_agent_rewards(row.get("agent_rewards", "")),
                }
            )
    if last_n is not None and last_n > 0 and len(rows) > last_n:
        rows = rows[-last_n:]
    return rows


def load_qtable_stats(qtable_path: str) -> Dict[str, float]:
    if not os.path.exists(qtable_path):
        return {}
    with open(qtable_path, "r", encoding="utf-8") as f:
        q = json.load(f)
    if not q:
        return {}

    all_vals: List[float] = []
    reverse_vals: List[float] = []
    danger_reverse: List[float] = []
    for state_key, actions in q.items():
        for action, value in actions.items():
            v = float(value)
            all_vals.append(v)
            if action == "reverse":
                reverse_vals.append(v)
                if "|danger|" in state_key:
                    danger_reverse.append(v)

    def _mean(xs: List[float]) -> float:
        return float(sum(xs) / len(xs)) if xs else 0.0

    return {
        "q_states": float(len(q)),
        "q_entries": float(sum(len(a) for a in q.values())),
        "q_mean": _mean(all_vals),
        "q_std": float(tf.math.reduce_std(tf.constant(all_vals)).numpy()) if all_vals else 0.0,
        "q_min": float(min(all_vals)) if all_vals else 0.0,
        "q_max": float(max(all_vals)) if all_vals else 0.0,
        "q_reverse_mean": _mean(reverse_vals),
        "q_danger_reverse_mean": _mean(danger_reverse),
        "q_danger_reverse_min": float(min(danger_reverse)) if danger_reverse else 0.0,
    }


def append_qtable_history(history_path: str, stats: Dict[str, float]) -> None:
    if not stats:
        return
    os.makedirs(os.path.dirname(history_path) or ".", exist_ok=True)
    record = {"time": time.time(), **stats}
    with open(history_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_qtable_history(history_path: str) -> List[Dict[str, float]]:
    if not os.path.exists(history_path):
        return []
    out: List[Dict[str, float]] = []
    with open(history_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def rolling_mean(values: List[float], window: int) -> List[float]:
    if window < 1:
        window = 1
    out: List[float] = []
    for i in range(len(values)):
        start = max(0, i - window + 1)
        chunk = values[start : i + 1]
        out.append(sum(chunk) / len(chunk))
    return out


def extract_collision_agent(stop_reason: str) -> str:
    m = re.search(r"agent=(\w+)", stop_reason)
    return m.group(1) if m else ""


def build_episode_tensors(rows: List[Dict[str, Any]], window: int) -> Dict[str, tf.Tensor]:
    n = len(rows)
    if n == 0:
        return {}

    collision_flag = [1.0 if r["collision_episode"] else 0.0 for r in rows]
    success_flag = [1.0 if r["target_reached"] else 0.0 for r in rows]
    total_reward = [r["total_reward"] for r in rows]
    collisions = [float(r["collisions"]) for r in rows]
    global_visited = [r["global_visited"] for r in rows]
    visited_delta = [r["global_visited_delta"] for r in rows]
    target_progress = [r["target_progress"] for r in rows]
    shaping_reward = [r["shaping_reward"] for r in rows]

    roll_collision = rolling_mean(collision_flag, window)
    roll_success = rolling_mean(success_flag, window)
    roll_reward = rolling_mean(total_reward, window)
    roll_visited = rolling_mean(global_visited, window)
    roll_visited_delta = rolling_mean(visited_delta, window)
    roll_target_progress = rolling_mean(target_progress, window)
    roll_shaping = rolling_mean(shaping_reward, window)

    min_agent_reward = []
    for r in rows:
        vals = list(r["agent_rewards"].values())
        min_agent_reward.append(min(vals) if vals else 0.0)

    return {
        "collision_rate_roll": tf.constant(roll_collision, dtype=tf.float32),
        "success_rate_roll": tf.constant(roll_success, dtype=tf.float32),
        "total_reward_roll": tf.constant(roll_reward, dtype=tf.float32),
        "global_visited_roll": tf.constant(roll_visited, dtype=tf.float32),
        "visited_delta_roll": tf.constant(roll_visited_delta, dtype=tf.float32),
        "target_progress_roll": tf.constant(roll_target_progress, dtype=tf.float32),
        "shaping_reward_roll": tf.constant(roll_shaping, dtype=tf.float32),
        "visited_delta": tf.constant(visited_delta, dtype=tf.float32),
        "target_progress": tf.constant(target_progress, dtype=tf.float32),
        "shaping_reward": tf.constant(shaping_reward, dtype=tf.float32),
        "collision_flag": tf.constant(collision_flag, dtype=tf.float32),
        "success_flag": tf.constant(success_flag, dtype=tf.float32),
        "total_reward": tf.constant(total_reward, dtype=tf.float32),
        "collisions": tf.constant(collisions, dtype=tf.float32),
        "min_agent_reward": tf.constant(min_agent_reward, dtype=tf.float32),
        "steps": tf.constant([int(r["global_step"]) for r in rows], dtype=tf.int64),
    }


def write_tensorboard(
    log_dir: str,
    episode_tensors: Dict[str, tf.Tensor],
    q_stats: Dict[str, float],
    q_history: List[Dict[str, float]],
    min_global_step: int = 0,
) -> int:
    """Write episode scalars with global_step; only rows with step > min_global_step."""
    os.makedirs(log_dir, exist_ok=True)
    try:
        writer = tf.summary.create_file_writer(log_dir)
    except Exception as exc:
        print(f"[RL TF] TensorBoard writer failed: {exc}")
        print("  pip install tensorboard")
        return min_global_step

    max_written = min_global_step
    if episode_tensors:
        steps = episode_tensors["steps"]
        n = int(steps.shape[0])
        with writer.as_default():
            for i in range(n):
                step = int(steps[i].numpy())
                if step <= min_global_step:
                    continue
                tf.summary.scalar("episode/collision_flag", episode_tensors["collision_flag"][i], step=step)
                tf.summary.scalar("episode/success_flag", episode_tensors["success_flag"][i], step=step)
                tf.summary.scalar("episode/total_reward", episode_tensors["total_reward"][i], step=step)
                tf.summary.scalar("episode/collisions", episode_tensors["collisions"][i], step=step)
                tf.summary.scalar("episode/min_agent_reward", episode_tensors["min_agent_reward"][i], step=step)
                tf.summary.scalar(
                    "episode/collision_rate_rolling",
                    episode_tensors["collision_rate_roll"][i],
                    step=step,
                )
                tf.summary.scalar(
                    "episode/success_rate_rolling",
                    episode_tensors["success_rate_roll"][i],
                    step=step,
                )
                tf.summary.scalar(
                    "episode/total_reward_rolling",
                    episode_tensors["total_reward_roll"][i],
                    step=step,
                )
                tf.summary.scalar(
                    "episode/global_visited_rolling",
                    episode_tensors["global_visited_roll"][i],
                    step=step,
                )
                if "visited_delta_roll" in episode_tensors:
                    tf.summary.scalar(
                        "episode/visited_delta_rolling",
                        episode_tensors["visited_delta_roll"][i],
                        step=step,
                    )
                    tf.summary.scalar(
                        "episode/target_progress_rolling",
                        episode_tensors["target_progress_roll"][i],
                        step=step,
                    )
                    tf.summary.scalar(
                        "episode/shaping_reward_rolling",
                        episode_tensors["shaping_reward_roll"][i],
                        step=step,
                    )
                max_written = max(max_written, step)

    if q_stats:
        with writer.as_default():
            step = int(time.time())
            for key, value in q_stats.items():
                tf.summary.scalar(f"qtable/{key}", value, step=step)

    if q_history:
        with writer.as_default():
            for i, rec in enumerate(q_history):
                for key, value in rec.items():
                    if key == "time":
                        continue
                    tf.summary.scalar(f"qtable_history/{key}", float(value), step=i)

    writer.flush()
    return max_written


def try_export_png(
    png_path: str,
    rows: List[Dict[str, Any]],
    window: int,
    q_history: List[Dict[str, float]],
) -> bool:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    if not rows and not q_history:
        return False

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    fig.suptitle("Collision RL training (from episode log + Q-table)", fontsize=12)

    if rows:
        x = [int(r["global_step"]) for r in rows]
        coll = [1 if r["collision_episode"] else 0 for r in rows]
        succ = [1 if r["target_reached"] else 0 for r in rows]
        rew = [r["total_reward"] for r in rows]
        vis = [r["global_visited"] * 100.0 for r in rows]
        vd = [r["global_visited_delta"] * 100.0 for r in rows]
        tp = [r["target_progress"] * 100.0 for r in rows]
        shp = [r["shaping_reward"] for r in rows]

        axes[0, 0].plot(x, rolling_mean([float(c) for c in coll], window), label=f"collision (w={window})")
        axes[0, 0].plot(x, rolling_mean([float(s) for s in succ], window), label="success")
        axes[0, 0].set_ylim(-0.05, 1.05)
        axes[0, 0].set_xlabel("global step")
        axes[0, 0].set_ylabel("rate")
        axes[0, 0].legend(loc="best", fontsize=8)
        axes[0, 0].set_title("Rolling collision / success")

        axes[0, 1].plot(x, rew, alpha=0.35, label="total_reward")
        axes[0, 1].plot(x, rolling_mean(rew, window), label="rolling mean")
        axes[0, 1].plot(x, rolling_mean(shp, window), label="shaping roll")
        axes[0, 1].set_xlabel("global step")
        axes[0, 1].set_ylabel("reward")
        axes[0, 1].legend(loc="best", fontsize=8)
        axes[0, 1].set_title("Episode reward (+ shaping)")

        axes[0, 2].plot(x, rolling_mean(vd, window), label="visited_delta % roll")
        axes[0, 2].plot(x, rolling_mean(tp, window), label="target progress % roll")
        axes[0, 2].set_xlabel("global step")
        axes[0, 2].set_ylabel("%")
        axes[0, 2].legend(loc="best", fontsize=8)
        axes[0, 2].set_title("Map expansion & target proximity")

        axes[1, 0].plot(x, vis, alpha=0.4, label="global visited %")
        axes[1, 0].plot(x, rolling_mean(vis, window), label="visited roll")
        axes[1, 0].set_xlabel("global step")
        axes[1, 0].set_ylabel("%")
        axes[1, 0].legend(loc="best", fontsize=8)
        axes[1, 0].set_title("Map visited (global)")

        axes[1, 1].plot(x, [v * 100 for v in vd], alpha=0.35, label="visited_delta %")
        axes[1, 1].plot(x, rolling_mean(vd, window), label="delta roll")
        axes[1, 1].set_xlabel("global step")
        axes[1, 1].set_ylabel("% gained / episode")
        axes[1, 1].legend(loc="best", fontsize=8)
        axes[1, 1].set_title("Visited area gained (episode)")

        if q_history:
            qh_x = list(range(len(q_history)))
            qh_states = [rec.get("q_states", 0) for rec in q_history]
            axes[1, 2].plot(qh_x, qh_states, marker="o")
            axes[1, 2].set_title("Q-table states")
            axes[1, 2].set_xlabel("snapshot #")
            axes[1, 2].set_ylabel("states")
        else:
            axes[1, 2].plot(x, tp, alpha=0.35, label="target progress %")
            axes[1, 2].plot(x, rolling_mean(tp, window), label="progress roll")
            axes[1, 2].set_ylim(-2, 105)
            axes[1, 2].set_xlabel("global step")
            axes[1, 2].set_ylabel("%")
            axes[1, 2].legend(loc="best", fontsize=8)
            axes[1, 2].set_title("Target proximity (episode)")
    else:
        for ax in axes.flat:
            ax.text(0.5, 0.5, "No episode CSV", ha="center")

    plt.tight_layout()
    os.makedirs(os.path.dirname(png_path) or ".", exist_ok=True)
    plt.savefig(png_path, dpi=140)
    plt.close(fig)
    return True


def print_summary(rows: List[Dict[str, Any]], q_stats: Dict[str, float], window: int) -> None:
    n = len(rows)
    print("=" * 60)
    print("[RL TF] Training metrics summary")
    print("=" * 60)
    if n == 0:
        print("No episode rows found.")
    else:
        coll_n = sum(1 for r in rows if r["collision_episode"])
        succ_n = sum(1 for r in rows if r["target_reached"])
        print(f"Episodes in CSV     : {n}")
        print(f"Collision episodes  : {coll_n} ({100.0 * coll_n / n:.1f}%)")
        print(f"TARGET_REACHED      : {succ_n} ({100.0 * succ_n / n:.1f}%)")
        if n >= window:
            tail = rows[-window:]
            tail_coll = sum(1 for r in tail if r["collision_episode"]) / window
            tail_succ = sum(1 for r in tail if r["target_reached"]) / window
            tail_vd = sum(r["global_visited_delta"] for r in tail) / window
            tail_tp = sum(r["target_progress"] for r in tail) / window
            print(f"Last {window} collision rate : {100.0 * tail_coll:.1f}%")
            print(f"Last {window} success rate   : {100.0 * tail_succ:.1f}%")
            print(f"Last {window} visited_delta  : {100.0 * tail_vd:.2f}%")
            print(f"Last {window} target progress: {100.0 * tail_tp:.1f}%")
    if q_stats:
        print(f"Q-table states      : {int(q_stats.get('q_states', 0))}")
        print(f"Q mean / min        : {q_stats.get('q_mean', 0):.3f} / {q_stats.get('q_min', 0):.3f}")
        print(f"Danger reverse min  : {q_stats.get('q_danger_reverse_min', 0):.3f}")
    print("=" * 60)


def main() -> int:
    parser = argparse.ArgumentParser(description="Log collision RL metrics to TensorBoard.")
    parser.add_argument("--csv", default=CFG.EPISODE_CSV, help="Episode log CSV path")
    parser.add_argument("--qtable", default=CFG.RL_SAVE_PATH, help="Q-table JSON path")
    parser.add_argument(
        "--logdir",
        default=os.path.join(CFG.OUTPUT_DIR, "tensorboard", "rl_collision"),
        help="TensorBoard log directory",
    )
    parser.add_argument(
        "--history",
        default=os.path.join(CFG.OUTPUT_DIR, "rl_training_history.jsonl"),
        help="Append-only Q-table snapshot history",
    )
    parser.add_argument("--window", type=int, default=20, help="Rolling mean window (episodes)")
    parser.add_argument(
        "--png",
        default=os.path.join(CFG.OUTPUT_DIR, "rl_training_curves.png"),
        help="Optional matplotlib PNG export",
    )
    parser.add_argument("--no-history", action="store_true", help="Do not append Q-table snapshot")
    parser.add_argument(
        "--last-n",
        type=int,
        default=0,
        help="Plot only the last N CSV rows (0 = all rows, for tail cherry-pick view)",
    )
    parser.add_argument(
        "--full-tb-rewrite",
        action="store_true",
        help="Rewrite all TensorBoard episode steps from CSV (ignore rl_plot_state.json)",
    )
    args = parser.parse_args()

    all_rows = load_episode_rows(args.csv, last_n=0)
    last_n = int(args.last_n) if args.last_n and args.last_n > 0 else 0
    rows = all_rows[-last_n:] if last_n > 0 else all_rows

    q_stats = load_qtable_stats(args.qtable)
    if q_stats and not args.no_history:
        append_qtable_history(args.history, q_stats)
    q_history = load_qtable_history(args.history)

    plot_state = load_plot_state()
    min_tb_step = 0 if args.full_tb_rewrite else int(plot_state.get("last_tb_global_step", 0))
    if args.full_tb_rewrite:
        tb_rows = all_rows
    else:
        tb_rows = [r for r in all_rows if int(r["global_step"]) > min_tb_step]

    episode_tensors = build_episode_tensors(tb_rows, args.window) if tb_rows else {}
    max_tb_step = min_tb_step
    try:
        max_tb_step = write_tensorboard(
            args.logdir,
            episode_tensors,
            q_stats,
            q_history,
            min_global_step=min_tb_step,
        )
    except Exception as exc:
        err = str(exc)
        if "tensorboard" in err.lower() or "TBNotInstalled" in type(exc).__name__:
            print("[RL TF] TensorBoard not installed. Run: pip install tensorboard")
        else:
            raise

    save_plot_state(
        last_tb_global_step=max(max_tb_step, len(all_rows)),
        last_csv_rows=len(all_rows),
    )

    if args.png:
        try_export_png(args.png, rows, args.window, q_history)

    print_summary(all_rows, q_stats, args.window)
    if max_tb_step > min_tb_step or episode_tensors:
        print(
            f"[RL TF] TensorBoard appended steps {min_tb_step + 1}..{max_tb_step} "
            f"(total CSV rows={len(all_rows)})"
        )
        print(f"[RL TF] TensorBoard logdir: {os.path.abspath(args.logdir)}")
        print("  tensorboard --logdir", os.path.abspath(os.path.join(CFG.OUTPUT_DIR, "tensorboard")))
    if last_n > 0:
        print(f"[RL TF] PNG uses last {last_n} rows only (tail view)")
    if args.png and os.path.exists(args.png):
        print(f"[RL TF] PNG saved: {os.path.abspath(args.png)}")

    try:
        from export_normalized_indicators import write_csv, write_jsonl, print_summary
        from normalized_metrics import build_indicator_records
        from config import CFG as _CFG
        from safe_io import atomic_write_json
        import os as _os

        norm_records = build_indicator_records(args.csv, args.qtable)
        norm_csv = getattr(_CFG, "NORMALIZED_INDICATORS_CSV", "outputs/normalized_indicators.csv")
        norm_jsonl = getattr(_CFG, "NORMALIZED_INDICATORS_JSONL", "outputs/normalized_indicators.jsonl")
        write_csv(norm_csv, norm_records, args.window)
        write_jsonl(norm_jsonl, norm_records, args.window)
        print_summary(norm_records, args.window)
        if norm_records:
            tail = norm_records[-min(args.window, len(norm_records)) :]
            summary_path = _os.path.join(
                _CFG.OUTPUT_DIR,
                getattr(_CFG, "NORMALIZED_SUMMARY_JSON", "normalized_summary.json"),
            )
            atomic_write_json(
                summary_path,
                {
                    "episodes": len(norm_records),
                    "window": args.window,
                    "last_episode": norm_records[-1].get("episode"),
                    "averages": {
                        k: sum(float(r.get(k, 0.0)) for r in tail) / len(tail)
                        for k in (
                            "norm_training_score",
                            "norm_collision_free",
                            "norm_success",
                            "norm_global_visited",
                            "norm_target_progress",
                        )
                    },
                },
            )
            print(f"[NORM] Summary: {_os.path.abspath(summary_path)}")
    except Exception as exc:
        print(f"[RL TF] Normalized indicators export skipped: {exc}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
