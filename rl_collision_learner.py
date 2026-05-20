import os
import random
from typing import Dict, Optional, Tuple

from config import CFG
from safe_io import atomic_write_json, file_lock, read_json_locked

_PKG_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_PKG_DIR)


# Shield / hard-escape action names map to tabular actions for credit assignment.
ACTION_NAMES = frozenset(
    {
        "safe_side_soft",
        "safe_side_hard",
        "force_left",
        "force_right",
        "reverse",
        "slow_hold",
    }
)

SHIELD_ACTION_ALIASES = {
    "hard_corridor_escape": None,
    "track_hard_corridor_escape": None,
    "shield_side_scrape_escape": "safe_side_hard",
    "safety_reverse_both_blocked": "reverse",
    "safety_force_right": "force_right",
    "safety_force_left": "force_left",
    "no_obstacle_keep_moving": "safe_side_soft",
}


class RLCollisionLearner:
    """
    Online tabular Q-learning for obstacle-avoidance action choice.
    """

    def __init__(self):
        self.q: Dict[str, Dict[str, float]] = {}
        self.last_state: Dict[str, Tuple] = {}
        self.last_action: Dict[str, str] = {}
        self.update_count = 0
        self._updates_since_save = 0

        self.actions = {
            "safe_side_soft": {
                "forward": -0.15,
                "side": 0.85,
                "speed_scale": 0.75,
                "force_side": "safe",
            },
            "safe_side_hard": {
                "forward": -0.40,
                "side": 1.20,
                "speed_scale": 1.00,
                "force_side": "safe",
            },
            "force_left": {
                "forward": -0.30,
                "side": 1.10,
                "speed_scale": 0.95,
                "force_side": "left",
            },
            "force_right": {
                "forward": -0.30,
                "side": 1.10,
                "speed_scale": 0.95,
                "force_side": "right",
            },
            "reverse": {
                "forward": -0.90,
                "side": 0.20,
                "speed_scale": 0.70,
                "force_side": "safe",
            },
            "slow_hold": {
                "forward": -0.65,
                "side": 0.25,
                "speed_scale": 0.65,
                "force_side": "safe",
            },
        }

        self.load()

    def _state_key(self, state: Tuple) -> str:
        return "|".join(str(x) for x in state)

    def _ensure_row(self, key: str) -> Dict[str, float]:
        if key not in self.q:
            self.q[key] = {name: 0.0 for name in self.actions}
        else:
            for name in self.actions:
                self.q[key].setdefault(name, 0.0)
        return self.q[key]

    @staticmethod
    def resolve_learnable_action(executed_name: str, snapshot: Optional[dict] = None) -> str:
        """Map executed (possibly shield) action name to a Q-table action key."""
        name = str(executed_name or "safe_side_hard")
        if name.startswith("safety_balanced_"):
            suffix = name[len("safety_balanced_") :]
            if suffix in ACTION_NAMES:
                return suffix
            return "safe_side_hard"

        alias = SHIELD_ACTION_ALIASES.get(name)
        if alias is not None:
            return alias
        if name in ACTION_NAMES:
            return name

        if name in ("hard_corridor_escape", "track_hard_corridor_escape") and snapshot:
            left = float(snapshot.get("left_clear", 100.0))
            right = float(snapshot.get("right_clear", 100.0))
            return "force_right" if right >= left else "force_left"

        return "safe_side_hard"

    def make_state(self, snapshot, obstacle_dist: float) -> Tuple:
        obstacle = 1 if snapshot.get("obstacle", False) else 0

        left = float(snapshot.get("left_clear", 100.0))
        right = float(snapshot.get("right_clear", 100.0))
        min_clear = min(left, right)

        if min_clear < obstacle_dist * 0.4:
            risk_bin = "danger"
        elif min_clear < obstacle_dist * 0.8:
            risk_bin = "warn"
        else:
            risk_bin = "clear"

        if right > left + 1.0:
            side_bin = "right_clear"
        elif left > right + 1.0:
            side_bin = "left_clear"
        else:
            side_bin = "balanced"

        if obstacle_dist < 9.0:
            depth_bin = "short"
        elif obstacle_dist < 12.0:
            depth_bin = "mid"
        else:
            depth_bin = "long"

        return obstacle, risk_bin, side_bin, depth_bin

    def commit_action(self, agent: str, state: Tuple, executed_action_name: str, snapshot: Optional[dict] = None):
        """Bind (state, action) for the next reward_step / update."""
        learn_name = self.resolve_learnable_action(executed_action_name, snapshot)
        self.last_state[agent] = state
        self.last_action[agent] = learn_name

    def clear_pending(self, agent: str):
        self.last_state.pop(agent, None)
        self.last_action.pop(agent, None)

    def has_pending(self, agent: str) -> bool:
        return agent in self.last_state and agent in self.last_action

    def select_action(self, agent: str, state: Tuple) -> Dict:
        key = self._state_key(state)
        row = self._ensure_row(key)

        if random.random() < CFG.RL_EPSILON:
            action_name = random.choice(list(self.actions.keys()))
        else:
            action_name = max(row, key=row.get)

        action = dict(self.actions[action_name])
        action["name"] = action_name
        return action

    def update(self, agent: str, reward: float, next_state: Tuple) -> bool:
        if agent not in self.last_state or agent not in self.last_action:
            return False

        prev_state = self.last_state[agent]
        action_name = self.last_action[agent]
        prev_key = self._state_key(prev_state)
        next_key = self._state_key(next_state)

        row_prev = self._ensure_row(prev_key)
        row_next = self._ensure_row(next_key)

        old_q = row_prev[action_name]
        next_best = max(row_next.values())
        new_q = old_q + CFG.RL_ALPHA * (reward + CFG.RL_GAMMA * next_best - old_q)
        row_prev[action_name] = new_q

        self.update_count += 1
        self._updates_since_save += 1
        self.clear_pending(agent)
        self.maybe_autosave()
        return True

    def maybe_autosave(self):
        every = int(getattr(CFG, "RL_SAVE_EVERY_UPDATES", 25))
        if every > 0 and self._updates_since_save >= every:
            self.save()

    def save(self):
        try:
            path = CFG.RL_SAVE_PATH
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            with file_lock(path):
                atomic_write_json(path, self.q)
            self._updates_since_save = 0
        except Exception as e:
            print(f"[RL] failed to save q table: {e}")

    @staticmethod
    def resolve_pretrain_policy_path() -> str:
        custom = str(getattr(CFG, "RL_PRETRAIN_POLICY_PATH", "") or "").strip()
        if custom:
            if os.path.isabs(custom):
                return custom
            return os.path.normpath(os.path.join(_PKG_DIR, custom))
        return os.path.normpath(
            os.path.join(_REPO_ROOT, "12_test", "outputs", "rl_collision_qtable.json")
        )

    def _ingest_qtable(self, loaded: dict, label: str, overlay: bool) -> int:
        if not isinstance(loaded, dict):
            return 0
        n = 0
        for key, row in loaded.items():
            if not isinstance(row, dict):
                continue
            if key not in self.q or not overlay:
                self.q[key] = dict(row)
                self._ensure_row(key)
            else:
                dst = self._ensure_row(key)
                for action, value in row.items():
                    dst[action] = float(value)
            n += 1
        if n > 0:
            print(f"[RL] ingested {label}: states={n} total={len(self.q)}")
        return n

    def _load_json_qtable(self, path: str) -> dict:
        if not path or not os.path.exists(path):
            return {}
        try:
            data = read_json_locked(path)
            return data if isinstance(data, dict) else {}
        except Exception as e:
            print(f"[RL] failed to read {path}: {e}")
            return {}

    def load(self):
        self.q = {}
        try:
            use_12 = bool(getattr(CFG, "RL_USE_12_PRETRAIN_POLICY", False))
            pretrain_path = self.resolve_pretrain_policy_path()
            local_path = os.path.normpath(
                os.path.join(_PKG_DIR, getattr(CFG, "RL_SAVE_PATH", "outputs/rl_collision_qtable.json"))
            )

            if use_12:
                pre = self._load_json_qtable(pretrain_path)
                if pre:
                    self._ingest_qtable(pre, "12_test pretrain", overlay=False)
                    print(f"[RL] policy base: {pretrain_path}")
                else:
                    print(
                        f"[RL] WARN: 12_test policy not found at {pretrain_path} "
                        f"(run: cd 12_test && python run_fast.py)"
                    )

            merge_local = bool(getattr(CFG, "RL_PRETRAIN_MERGE_LOCAL", True))
            if (not use_12 or merge_local) and os.path.exists(local_path):
                local = self._load_json_qtable(local_path)
                if local:
                    self._ingest_qtable(
                        local,
                        "11_test local fine-tune",
                        overlay=bool(use_12 and len(self.q) > 0),
                    )
                    if not use_12:
                        print(f"[RL] loaded q table: {local_path}")

            if not self.q and os.path.exists(local_path):
                local = self._load_json_qtable(local_path)
                self._ingest_qtable(local, "local fallback", overlay=False)

            if self.q:
                print(
                    f"[RL] ready states={len(self.q)} "
                    f"epsilon={getattr(CFG, 'RL_EPSILON', '?')} "
                    f"updates={self.update_count}"
                )
            else:
                print("[RL] starting with empty Q-table (no pretrain/local file)")
        except Exception as e:
            print(f"[RL] failed to load q table: {e}")
            self.q = {}


def _decay_epsilon(self):
    try:
        CFG.RL_EPSILON = max(CFG.RL_EPSILON_MIN, CFG.RL_EPSILON * CFG.RL_EPSILON_DECAY)
    except Exception:
        pass


try:
    RLCollisionLearner.decay_epsilon = _decay_epsilon
except Exception:
    pass
