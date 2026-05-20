"""
Safe background training supervisor (11_test).

Runs main.py in a child process and periodically exports TensorBoard curves +
normalized indicators without corrupting Q-table / episode CSV (atomic I/O + locks).

Usage (from 11_test, AirSim/UE already running):
  python train_background.py
  python train_background.py --metrics-interval 120 --no-tensorboard
  python train_background.py --stop

Stops gracefully on Ctrl+C; writes outputs/train_background.pid while running.
"""
from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from typing import Optional

from config import CFG

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PID_FILE = os.path.join(getattr(CFG, "OUTPUT_DIR", "outputs"), "train_background.pid")
STATUS_FILE = os.path.join(getattr(CFG, "OUTPUT_DIR", "outputs"), "train_background_status.json")


def _status_write(payload: dict) -> None:
    try:
        from safe_io import atomic_write_json

        os.makedirs(os.path.dirname(STATUS_FILE) or ".", exist_ok=True)
        payload["updated_at"] = time.time()
        atomic_write_json(STATUS_FILE, payload)
    except Exception as exc:
        print(f"[BG] status write failed: {exc}")


def _read_pid() -> Optional[int]:
    if not os.path.exists(PID_FILE):
        return None
    try:
        with open(PID_FILE, "r", encoding="utf-8") as f:
            return int(f.read().strip())
    except (OSError, ValueError):
        return None


def _write_pid(pid: int) -> None:
    os.makedirs(os.path.dirname(PID_FILE) or ".", exist_ok=True)
    with open(PID_FILE, "w", encoding="utf-8") as f:
        f.write(str(pid))


def _clear_pid() -> None:
    try:
        os.remove(PID_FILE)
    except OSError:
        pass


def request_stop() -> bool:
    pid = _read_pid()
    if pid is None:
        print("[BG] No background trainer PID file.")
        return False
    try:
        os.kill(pid, signal.SIGTERM)
        print(f"[BG] Sent SIGTERM to supervisor pid={pid}")
        return True
    except OSError as exc:
        print(f"[BG] Could not stop pid={pid}: {exc}")
        _clear_pid()
        return False


def _run_metrics(export_tb: bool) -> None:
    if export_tb:
        plot_script = os.path.join(SCRIPT_DIR, "plot_collision_learning_tf.py")
        subprocess.run(
            [sys.executable, plot_script],
            cwd=SCRIPT_DIR,
            check=False,
        )
    export_script = os.path.join(SCRIPT_DIR, "export_normalized_indicators.py")
    subprocess.run(
        [sys.executable, export_script],
        cwd=SCRIPT_DIR,
        check=False,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Safe background RL training supervisor.")
    parser.add_argument(
        "--metrics-interval",
        type=float,
        default=float(getattr(CFG, "BG_METRICS_INTERVAL_SEC", 180.0)),
        help="Seconds between metric export runs",
    )
    parser.add_argument("--no-tensorboard", action="store_true", help="Skip plot_collision_learning_tf.py")
    parser.add_argument("--stop", action="store_true", help="Stop running supervisor")
    parser.add_argument(
        "--main-args",
        nargs=argparse.REMAINDER,
        help="Extra args forwarded to main.py (prefix with --)",
    )
    args = parser.parse_args()

    if args.stop:
        return 0 if request_stop() else 1

    existing = _read_pid()
    if existing is not None:
        try:
            os.kill(existing, 0)
            print(f"[BG] Already running (pid={existing}). Use --stop first.")
            return 1
        except OSError:
            _clear_pid()

    main_py = os.path.join(SCRIPT_DIR, "main.py")
    cmd = [sys.executable, main_py]
    if args.main_args:
        cmd.extend(args.main_args)

    child: Optional[subprocess.Popen] = None
    stop_requested = False

    def _handle_sig(signum, frame):
        nonlocal stop_requested
        stop_requested = True
        print(f"\n[BG] Signal {signum} — stopping child main.py …")

    signal.signal(signal.SIGINT, _handle_sig)
    signal.signal(signal.SIGTERM, _handle_sig)

    _write_pid(os.getpid())
    _status_write({"phase": "starting", "main_cmd": cmd})

    try:
        child = subprocess.Popen(cmd, cwd=SCRIPT_DIR)
        print(f"[BG] Started main.py pid={child.pid}")
        _status_write({"phase": "training", "main_pid": child.pid})

        last_metrics = 0.0
        export_tb = not args.no_tensorboard

        while True:
            if stop_requested:
                break
            if child.poll() is not None:
                print(f"[BG] main.py exited code={child.returncode}")
                break
            now = time.time()
            if now - last_metrics >= max(30.0, args.metrics_interval):
                print("[BG] Refreshing metrics (TB + normalized indicators)…")
                _run_metrics(export_tb)
                last_metrics = now
                _status_write(
                    {
                        "phase": "training",
                        "main_pid": child.pid,
                        "last_metrics_at": now,
                    }
                )
            time.sleep(2.0)

        if child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=25)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait(timeout=10)

        print("[BG] Final metrics export…")
        _run_metrics(export_tb)
        code = child.returncode if child.returncode is not None else 0
        _status_write({"phase": "stopped", "main_exit_code": code})
        return int(code) if code is not None else 0
    finally:
        _clear_pid()


if __name__ == "__main__":
    raise SystemExit(main())
