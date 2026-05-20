"""
Export per-episode normalized training indicators (0–1) from episode CSV + Q-table.

Usage (from 11_test):
  python export_normalized_indicators.py
  python export_normalized_indicators.py --window 20
"""
from __future__ import annotations

import argparse
import csv
import json
import os

from config import CFG
from normalized_metrics import (
    build_indicator_records,
    rolling_mean_dict,
)
from safe_io import atomic_write_json, atomic_write_text, file_lock


def write_csv(path: str, records: list, window: int) -> None:
    if not records:
        return
    import io

    from safe_io import atomic_write_text

    score_roll = rolling_mean_dict(records, "norm_training_score", window)
    fieldnames = sorted(records[0].keys())
    if "norm_training_score_roll" not in fieldnames:
        fieldnames.append("norm_training_score_roll")

    sio = io.StringIO()
    w = csv.DictWriter(sio, fieldnames=fieldnames, extrasaction="ignore")
    w.writeheader()
    for i, rec in enumerate(records):
        row = dict(rec)
        row["norm_training_score_roll"] = f"{score_roll[i]:.6f}"
        w.writerow(row)
    with file_lock(path):
        atomic_write_text(path, sio.getvalue())


def write_jsonl(path: str, records: list, window: int) -> None:
    if not records:
        return
    score_roll = rolling_mean_dict(records, "norm_training_score", window)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    lines = []
    for i, rec in enumerate(records):
        row = dict(rec)
        row["norm_training_score_roll"] = score_roll[i]
        lines.append(json.dumps(row, ensure_ascii=False))
    with file_lock(path):
        atomic_write_text(path, "\n".join(lines) + "\n")


def print_summary(records: list, window: int) -> None:
    n = len(records)
    print("=" * 60)
    print("[NORM] Normalized training indicators")
    print("=" * 60)
    if n == 0:
        print("No episode rows.")
        return
    tail = records[-min(window, n) :]
    keys = [
        "norm_training_score",
        "norm_collision_free",
        "norm_success",
        "norm_global_visited",
        "norm_target_progress",
    ]
    print(f"Episodes            : {n}")
    for key in keys:
        avg = sum(float(r.get(key, 0.0)) for r in tail) / len(tail)
        print(f"Last {len(tail)} avg {key:24s}: {100.0 * avg:.1f}%")
    print("=" * 60)


def main() -> int:
    parser = argparse.ArgumentParser(description="Export normalized RL training indicators.")
    parser.add_argument("--csv", default=CFG.EPISODE_CSV)
    parser.add_argument("--qtable", default=CFG.RL_SAVE_PATH)
    parser.add_argument("--window", type=int, default=20)
    parser.add_argument(
        "--csv-out",
        default=getattr(CFG, "NORMALIZED_INDICATORS_CSV", "outputs/normalized_indicators.csv"),
    )
    parser.add_argument(
        "--jsonl-out",
        default=getattr(CFG, "NORMALIZED_INDICATORS_JSONL", "outputs/normalized_indicators.jsonl"),
    )
    parser.add_argument("--last-n", type=int, default=0)
    args = parser.parse_args()

    last_n = int(args.last_n) if args.last_n > 0 else None
    records = build_indicator_records(args.csv, args.qtable, last_n=last_n)

    write_csv(args.csv_out, records, args.window)
    write_jsonl(args.jsonl_out, records, args.window)

    summary_path = os.path.join(
        getattr(CFG, "OUTPUT_DIR", "outputs"),
        getattr(CFG, "NORMALIZED_SUMMARY_JSON", "normalized_summary.json"),
    )
    if records:
        tail = records[-min(args.window, len(records)) :]
        summary = {
            "episodes": len(records),
            "window": args.window,
            "last_episode": records[-1].get("episode"),
            "averages": {
                k: sum(float(r.get(k, 0.0)) for r in tail) / len(tail)
                for k in (
                    "norm_training_score",
                    "norm_collision_free",
                    "norm_success",
                    "norm_global_visited",
                    "norm_target_progress",
                    "norm_visited_delta",
                )
            },
        }
        atomic_write_json(summary_path, summary)

    print_summary(records, args.window)
    print(f"[NORM] CSV  : {os.path.abspath(args.csv_out)}")
    print(f"[NORM] JSONL: {os.path.abspath(args.jsonl_out)}")
    if records:
        print(f"[NORM] Summary: {os.path.abspath(summary_path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
