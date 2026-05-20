import math
import time
import threading
from typing import Dict, Tuple, Optional, List

from config import CFG
from airsim_rpc import rpc_call
from utils import connect_client, distance_2d, inside_geofence, geofence_penalty
from belief_map import BeliefMap, GridValue
from rl_collision_learner import RLCollisionLearner


class CommandHub:
    def __init__(self):
        self.lock = threading.Lock()
        self.positions: Dict[str, Tuple[float, float, float]] = {}
        self.status: Dict[str, str] = {}
        self.targets = []
        self.stop_event = threading.Event()
        self.stop_reason = ""
        self.target_detection_active = False
        self.latest_coverage = 0.0
        self.latest_observed_coverage = 0.0
        self.latest_visited_coverage = 0.0
        self.latest_obstacle_coverage = 0.0
        self.latest_unknown_coverage = 1.0
        self.latest_global_visited_coverage = 0.0
        self.latest_episode_obstacle_coverage = 0.0
        self.reservations = {}
        self.priority = {name: idx for idx, name in enumerate(CFG.AGENTS)}
        self.global_goals: Dict[str, Tuple[float, float]] = {}
        self.last_goal_assign_time = 0.0
        self.local_area = None
        self.local_window_index = 0

        # Dynamically discovered physical wall boundaries. None means unknown.
        # Values are AirSim world coordinates in meters.
        self.discovered_bounds = {
            "x_min": None,
            "x_max": None,
            "y_min": None,
            "y_max": None,
        }

        self.rl_learner = RLCollisionLearner()
        self.rl_current_state: Dict[str, Tuple] = {}
        self.rl_step_committed: Dict[str, bool] = {agent: False for agent in CFG.AGENTS}
        self.adaptive_obstacle_dist = {agent: CFG.OBSTACLE_DIST for agent in CFG.AGENTS}
        self.last_safe_reward_time = {agent: 0.0 for agent in CFG.AGENTS}
        self.rewards = {agent: 0.0 for agent in CFG.AGENTS}
        self.collision_count = 0

        # Episode shaping metrics (logged + end-of-episode reward).
        self.episode_start_global_visited = 0.0
        self._step_reward_last_visited = 0.0
        self.episode_min_target_dist_m = float("inf")
        self.episode_coverage_shaping_reward = 0.0
        self.episode_target_shaping_reward = 0.0
        self.episode_global_visited_delta = 0.0
        self._cached_target_ref_xy: Optional[Tuple[float, float]] = None
        self._last_target_pose_poll = 0.0

        # Target tracking state. Target detection does not stop the mission;
        # Agent1 approaches the target while the command drone moves above it.
        self.target_tracking_active = False
        self.target_info = None
        self.target_map_marked = False
        self.primary_tracker_agent = getattr(CFG, "TRACKER_AGENT", "Agent1")
        self.tracker_arrived = False
        self.tracker_arrived_agents = set()
        self.command_target_sent = False


    def set_local_area(self, area, window_index: int = None):
        with self.lock:
            self.local_area = dict(area)
            if window_index is not None:
                self.local_window_index = int(window_index)
            self.global_goals = {}
            self.reservations = {}
        if getattr(CFG, "WINDOW_DEBUG_LOG_ENABLED", False):
            print(
                f"[LOCAL AREA] window={self.local_window_index} "
                f"x={area['x_min']:.1f}~{area['x_max']:.1f}, "
                f"y={area['y_min']:.1f}~{area['y_max']:.1f}"
            )

    def get_local_area(self):
        with self.lock:
            return None if self.local_area is None else dict(self.local_area)

    def point_in_local_area(self, x: float, y: float) -> bool:
        area = self.get_local_area()
        if area is None:
            return True
        return area["x_min"] <= x <= area["x_max"] and area["y_min"] <= y <= area["y_max"]

    def get_agent_area(self, agent: str):
        """Dynamic strip inside the current 100x100 local window."""
        area = self.get_local_area()
        if area is None:
            return None
        idx = list(CFG.AGENTS).index(agent) if agent in CFG.AGENTS else 0
        h = (area["y_max"] - area["y_min"]) / max(len(CFG.AGENTS), 1)
        return {
            "x_min": area["x_min"],
            "x_max": area["x_max"],
            "y_min": area["y_min"] + idx * h,
            "y_max": area["y_min"] + (idx + 1) * h,
        }

    # ============================================================
    # Dynamic wall / boundary discovery
    # ============================================================

    def is_wall_object(self, object_name: str) -> bool:
        name = (object_name or "").lower()
        return any(key in name for key in ("wall", "room", "frozen_room", "boundary", "cube"))

    def get_discovered_bounds(self):
        with self.lock:
            return dict(self.discovered_bounds)

    def all_boundaries_discovered(self) -> bool:
        if not getattr(CFG, "DYNAMIC_BOUNDARY_ENABLED", True):
            return True
        with self.lock:
            return all(v is not None for v in self.discovered_bounds.values())

    def get_discovered_area(self):
        if not self.all_boundaries_discovered():
            return None
        with self.lock:
            b = dict(self.discovered_bounds)
        width = float(b["x_max"]) - float(b["x_min"])
        height = float(b["y_max"]) - float(b["y_min"])
        min_span = getattr(CFG, "WALL_BOUNDARY_MIN_SPAN", CFG.LOCAL_AREA_SIZE * 0.8)
        if width < min_span or height < min_span:
            # Reject physically impossible/tiny discovered map areas caused by false wall detections.
            return None
        return {
            "x_min": float(b["x_min"]),
            "x_max": float(b["x_max"]),
            "y_min": float(b["y_min"]),
            "y_max": float(b["y_max"]),
        }

    def point_inside_dynamic_bounds(self, x: float, y: float, margin: float = 0.0) -> bool:
        if not getattr(CFG, "DYNAMIC_BOUNDARY_ENABLED", True):
            return True
        with self.lock:
            b = dict(self.discovered_bounds)
        if b["x_min"] is not None and x < b["x_min"] + margin:
            return False
        if b["x_max"] is not None and x > b["x_max"] - margin:
            return False
        if b["y_min"] is not None and y < b["y_min"] + margin:
            return False
        if b["y_max"] is not None and y > b["y_max"] - margin:
            return False
        return True

    def area_inside_dynamic_bounds(self, area, margin: float = 0.0) -> bool:
        if not getattr(CFG, "DYNAMIC_BOUNDARY_ENABLED", True):
            return True
        return (
            self.point_inside_dynamic_bounds(area["x_min"], area["y_min"], margin)
            and self.point_inside_dynamic_bounds(area["x_min"], area["y_max"], margin)
            and self.point_inside_dynamic_bounds(area["x_max"], area["y_min"], margin)
            and self.point_inside_dynamic_bounds(area["x_max"], area["y_max"], margin)
        )

    def clamp_area_to_dynamic_bounds(self, area):
        if not getattr(CFG, "DYNAMIC_BOUNDARY_ENABLED", True):
            return dict(area)
        with self.lock:
            b = dict(self.discovered_bounds)
        out = dict(area)
        m = getattr(CFG, "WALL_BOUNDARY_MARGIN", 4.0)
        if b["x_min"] is not None:
            out["x_min"] = max(out["x_min"], b["x_min"] + m)
        if b["x_max"] is not None:
            out["x_max"] = min(out["x_max"], b["x_max"] - m)
        if b["y_min"] is not None:
            out["y_min"] = max(out["y_min"], b["y_min"] + m)
        if b["y_max"] is not None:
            out["y_max"] = min(out["y_max"], b["y_max"] - m)
        return out

    def update_wall_boundary_from_pose(self, source: str, x: float, y: float, yaw_deg: float, wall_distance: float) -> bool:
        """
        Update physical boundary from a wall detected in front of a drone.

        Guard rules:
        - Agent depth-wall detections are accepted only near the current local-window edge.
        - A detected set of four walls must not shrink the map to a tiny rectangle.
        - This avoids false termination where trees/floor are mistaken as walls.
        """
        if not getattr(CFG, "DYNAMIC_BOUNDARY_ENABLED", True):
            return False

        if wall_distance is None or not math.isfinite(wall_distance):
            return False

        is_collision_source = ":collision:" in str(source)
        if not is_collision_source and not getattr(CFG, "AGENT_WALL_BOUNDARY_UPDATE_ENABLED", True):
            return False

        dist = max(0.0, float(wall_distance))
        if not is_collision_source and dist < getattr(CFG, "WALL_BOUNDARY_MIN_DISTANCE", 12.0):
            # Very close wall-like depth is usually floor/tree/foliage in the local area, not the physical room wall.
            return False

        rad = math.radians(yaw_deg)
        dx = math.cos(rad)
        dy = math.sin(rad)
        wx = x + dx * dist
        wy = y + dy * dist
        m = getattr(CFG, "WALL_BOUNDARY_MARGIN", 4.0)
        side = None
        new_val = None

        if abs(dx) >= abs(dy):
            if dx > 0:
                side = "x_max"
                new_val = wx - m
            else:
                side = "x_min"
                new_val = wx + m
        else:
            if dy > 0:
                side = "y_max"
                new_val = wy - m
            else:
                side = "y_min"
                new_val = wy + m

        # Agent depth wall detections must be near the current local window edge.
        if not is_collision_source:
            area = self.get_local_area()
            if area is not None:
                edge_tol = getattr(CFG, "WALL_BOUNDARY_EDGE_TOL", 8.0)
                near_edge = True
                if side == "x_min":
                    near_edge = wx <= area["x_min"] + edge_tol
                elif side == "x_max":
                    near_edge = wx >= area["x_max"] - edge_tol
                elif side == "y_min":
                    near_edge = wy <= area["y_min"] + edge_tol
                elif side == "y_max":
                    near_edge = wy >= area["y_max"] - edge_tol
                if not near_edge:
                    # Avoid false bounds from trees/floor inside the 100x100 local area.
                    return False

        updated = False
        with self.lock:
            old = self.discovered_bounds[side]
            candidate = dict(self.discovered_bounds)

            if side in ("x_max", "y_max"):
                if old is None or new_val < old:
                    candidate[side] = new_val
                else:
                    return False
            else:
                if old is None or new_val > old:
                    candidate[side] = new_val
                else:
                    return False

            # Reject impossible reversed or tiny bounds when opposing side is known.
            min_span = getattr(CFG, "WALL_BOUNDARY_MIN_SPAN", CFG.LOCAL_AREA_SIZE * 0.8)
            if candidate["x_min"] is not None and candidate["x_max"] is not None:
                if candidate["x_max"] <= candidate["x_min"]:
                    return False
                if candidate["x_max"] - candidate["x_min"] < min_span:
                    return False
            if candidate["y_min"] is not None and candidate["y_max"] is not None:
                if candidate["y_max"] <= candidate["y_min"]:
                    return False
                if candidate["y_max"] - candidate["y_min"] < min_span:
                    return False

            self.discovered_bounds = candidate
            updated = True
            bounds_copy = dict(self.discovered_bounds)

        if updated and getattr(CFG, "WALL_BOUNDARY_LOG_ENABLED", False):
            print(
                f"[WALL BOUNDARY] source={source} side={side} "
                f"wall=({wx:.1f},{wy:.1f}) dist={dist:.1f} "
                f"bounds={bounds_copy}"
            )
        return updated

    def request_stop(self, reason: str):
        with self.lock:
            if not self.stop_event.is_set():
                self.stop_reason = reason
                print("\n" + "=" * 70)
                print(f"[EPISODE STOP] {reason}")
                print("=" * 70 + "\n")
        self.stop_event.set()

    def update_position(self, drone: str, x: float, y: float, z: float):
        with self.lock:
            self.positions[drone] = (x, y, z)

    def update_status(self, drone: str, status: str):
        with self.lock:
            old = self.status.get(drone)
            if old == status:
                return
            self.status[drone] = status

        if getattr(CFG, "STATUS_LOG_ENABLED", False):
            print(f"[STATUS] {drone}: {status}")
            return

        if getattr(CFG, "IMPORTANT_STATUS_LOG_ENABLED", True):
            keywords = getattr(CFG, "IMPORTANT_STATUS_KEYWORDS", ("ERROR", "COLLISION"))
            if any(str(k) in status for k in keywords):
                print(f"[STATUS] {drone}: {status}")

    def update_coverage(self, ratio: float, stats=None):
        with self.lock:
            self.latest_coverage = float(ratio)
            self.latest_observed_coverage = float(ratio)
            if stats is not None:
                self.latest_observed_coverage = float(stats.get("observed", ratio))
                self.latest_visited_coverage = float(stats.get("visited", 0.0))
                self.latest_obstacle_coverage = float(stats.get("obstacle", 0.0))
                self.latest_unknown_coverage = float(stats.get("unknown", 1.0))
                self.latest_global_visited_coverage = float(stats.get("global_visited", stats.get("visited", 0.0)))
                self.latest_episode_obstacle_coverage = float(stats.get("episode_obstacle", stats.get("obstacle", 0.0)))

    def add_reward(self, agent: str, value: float):
        with self.lock:
            self.rewards[agent] = self.rewards.get(agent, 0.0) + float(value)

    def total_reward(self) -> float:
        with self.lock:
            return float(sum(self.rewards.values()))

    def begin_episode_shaping(self, global_visited: float = 0.0):
        gv = float(global_visited or 0.0)
        with self.lock:
            self.episode_start_global_visited = gv
            self._step_reward_last_visited = gv
            self.episode_min_target_dist_m = float("inf")
            self.episode_coverage_shaping_reward = 0.0
            self.episode_target_shaping_reward = 0.0
            self.episode_global_visited_delta = 0.0
            self._cached_target_ref_xy = None
            self._last_target_pose_poll = 0.0

    def get_target_reference_xy(self) -> Optional[Tuple[float, float]]:
        tx, ty, tz = self.get_target_ground_truth_pose()
        if tx is not None and ty is not None:
            return float(tx), float(ty)
        with self.lock:
            if self.target_info is not None:
                return (
                    float(self.target_info["target_x"]),
                    float(self.target_info["target_y"]),
                )
        return None

    def note_agent_target_distance(self, agent: str, x: float, y: float):
        poll_dt = float(getattr(CFG, "TARGET_POSE_POLL_DT", 0.0) or 0.0)
        now = time.time()
        with self.lock:
            if poll_dt > 0 and (now - self._last_target_pose_poll) < poll_dt:
                ref = self._cached_target_ref_xy
                if ref is None and self.target_info is not None:
                    ref = (
                        float(self.target_info["target_x"]),
                        float(self.target_info["target_y"]),
                    )
            else:
                self._last_target_pose_poll = now
                ref = self.get_target_reference_xy()
                if ref is not None:
                    self._cached_target_ref_xy = ref
        if ref is None:
            return
        dist = distance_2d((x, y), ref)
        with self.lock:
            self.episode_min_target_dist_m = min(self.episode_min_target_dist_m, dist)

    def record_visited_delta_for_step_reward(self, global_visited: float) -> float:
        gv = float(global_visited or 0.0)
        with self.lock:
            delta = max(0.0, gv - self._step_reward_last_visited)
            self._step_reward_last_visited = max(self._step_reward_last_visited, gv)
        return delta

    def finalize_episode_shaping(self, final_global_visited: float, stop_reason: str = "") -> Dict[str, float]:
        gv_end = float(final_global_visited or 0.0)
        max_dist = float(getattr(CFG, "EPISODE_TARGET_NORM_DIST_M", 150.0) or 150.0)
        with self.lock:
            visited_delta = max(0.0, gv_end - self.episode_start_global_visited)
            self.episode_global_visited_delta = visited_delta

            coverage_reward = visited_delta * float(
                getattr(CFG, "EPISODE_COVERAGE_REWARD_SCALE", 80.0)
            )
            self.episode_coverage_shaping_reward = coverage_reward

            min_dist = self.episode_min_target_dist_m
            if not math.isfinite(min_dist):
                min_dist = max_dist
            progress = max(0.0, min(1.0, (max_dist - min_dist) / max(max_dist, 1e-3)))
            target_reward = progress * float(
                getattr(CFG, "TARGET_PROXIMITY_REWARD_SCALE", 45.0)
            )
            if str(stop_reason).startswith("TARGET_REACHED"):
                target_reward += float(getattr(CFG, "TARGET_REACHED_SHAPING_BONUS", 30.0))
            self.episode_target_shaping_reward = target_reward

            n_agents = max(len(CFG.AGENTS), 1)
            per_agent = (coverage_reward + target_reward) / float(n_agents)
            for agent in CFG.AGENTS:
                self.rewards[agent] = self.rewards.get(agent, 0.0) + per_agent

            out_min_dist = float(min_dist)
            out_progress = float(progress)

        return {
            "global_visited_delta": visited_delta,
            "min_target_dist_m": out_min_dist,
            "target_progress": out_progress,
            "coverage_shaping_reward": coverage_reward,
            "target_shaping_reward": target_reward,
        }

    def clear_reservation(self, agent: str):
        with self.lock:
            self.reservations.pop(agent, None)

    def _clean_reservations_locked(self):
        now = time.time()
        dead = [agent for agent, r in self.reservations.items() if now - r["time"] > CFG.RESERVATION_TTL]
        for agent in dead:
            del self.reservations[agent]

    def get_conflict_altitude_shift_z(self, agent: str) -> float:
        """
        Return a temporary altitude for vertical deconfliction.
        Agents alternate between slightly higher and slightly lower bands, but remain in low-altitude flight.
        """
        priority = self.priority.get(agent, 0)
        if priority % 2 == 0:
            return max(CFG.AGENT_Z_HIGH_LIMIT, min(CFG.AGENT_Z_LOW_LIMIT, CFG.ALTITUDE_SHIFT_Z))
        lower_z = CFG.AGENT_Z + CFG.AGENT_VERTICAL_AVOID_STEP
        return max(CFG.AGENT_Z_HIGH_LIMIT, min(CFG.AGENT_Z_LOW_LIMIT, lower_z))

    def request_move(self, agent: str, from_cell, to_cell, target_xy, current_xyz):
        x, y, z = current_xyz
        with self.lock:
            self._clean_reservations_locked()
            my_priority = self.priority.get(agent, 999)
            for other, r in self.reservations.items():
                if other == agent:
                    continue
                other_priority = self.priority.get(other, 999)
                if r["to_cell"] == to_cell and my_priority > other_priority:
                    return {"action": "hold", "reason": "same_cell", "target_z": CFG.AGENT_Z, "hold_time": CFG.SAME_CELL_HOLD_TIME}
                if r["from_cell"] == to_cell and r["to_cell"] == from_cell and my_priority > other_priority:
                    return {"action": "hold", "reason": "swap", "target_z": CFG.AGENT_Z, "hold_time": CFG.SWAP_HOLD_TIME}

            for other, pos in self.positions.items():
                if other == agent or other == CFG.COMMAND:
                    continue
                ox, oy, oz = pos
                if abs(z - oz) > getattr(CFG, "PROXIMITY_Z_IGNORE_DISTANCE", 1.2):
                    continue
                d_current = math.hypot(x - ox, y - oy)
                d_target = math.hypot(target_xy[0] - ox, target_xy[1] - oy)
                other_priority = self.priority.get(other, 999)
                if min(d_current, d_target) < CFG.EMERGENCY_DISTANCE and my_priority > other_priority:
                    return {"action": "hold", "reason": "emergency_proximity", "target_z": CFG.AGENT_Z, "hold_time": 0.3}
                if min(d_current, d_target) < CFG.PROXIMITY_DISTANCE and my_priority > other_priority:
                    return {"action": "altitude_shift", "reason": "proximity", "target_z": self.get_conflict_altitude_shift_z(agent), "hold_time": 0.0}

            self.reservations[agent] = {"from_cell": from_cell, "to_cell": to_cell, "target_xy": target_xy, "time": time.time()}
            return {"action": "go", "reason": "clear", "target_z": CFG.AGENT_Z, "hold_time": 0.0}

    def get_repulsion_velocity(self, agent: str, x: float, y: float, z: float):
        rx = ry = 0.0
        emergency = False
        with self.lock:
            items = list(self.positions.items())
        for other, pos in items:
            if other == agent or other == CFG.COMMAND:
                continue
            ox, oy, oz = pos
            if abs(z - oz) > getattr(CFG, "PROXIMITY_Z_IGNORE_DISTANCE", 1.2):
                continue
            dx, dy = x - ox, y - oy
            d = math.hypot(dx, dy)
            if d < 1e-6:
                continue
            if d < CFG.EMERGENCY_DISTANCE:
                emergency = True
            if d < CFG.REPULSION_DISTANCE:
                strength = (CFG.REPULSION_DISTANCE - d) / CFG.REPULSION_DISTANCE
                rx += (dx / d) * strength * CFG.REPULSION_GAIN
                ry += (dy / d) * strength * CFG.REPULSION_GAIN
        return rx, ry, emergency

    def nearest_agent_penalty(self, agent: str, x: float, y: float, z: float) -> float:
        penalty = 0.0
        with self.lock:
            items = list(self.positions.items())
        for other, pos in items:
            if other == agent or other == CFG.COMMAND:
                continue
            ox, oy, oz = pos
            if abs(z - oz) > getattr(CFG, "PROXIMITY_Z_IGNORE_DISTANCE", 1.2):
                continue
            d = math.hypot(x - ox, y - oy)
            if d < CFG.PROXIMITY_DISTANCE:
                penalty += (CFG.PROXIMITY_DISTANCE - d) / CFG.PROXIMITY_DISTANCE
        return penalty

    def get_adaptive_obstacle_dist(self, agent: str) -> float:
        if not CFG.RL_COLLISION_LEARNING_ENABLED:
            return CFG.OBSTACLE_DIST
        with self.lock:
            return float(self.adaptive_obstacle_dist.get(agent, CFG.OBSTACLE_DIST))

    def rl_prepare_step(self, agent: str, snapshot) -> Tuple:
        """Cache discretized state for this control tick (before action choice)."""
        if not CFG.RL_COLLISION_LEARNING_ENABLED:
            return ()
        obstacle_dist = self.get_adaptive_obstacle_dist(agent)
        state = self.rl_learner.make_state(snapshot, obstacle_dist)
        self.rl_current_state[agent] = state
        return state

    def rl_commit_action(self, agent: str, snapshot, action: Dict):
        """Record the action actually executed (RL choice or safety shield)."""
        if not CFG.RL_COLLISION_LEARNING_ENABLED:
            return
        state = self.rl_current_state.get(agent)
        if state is None:
            obstacle_dist = self.get_adaptive_obstacle_dist(agent)
            state = self.rl_learner.make_state(snapshot, obstacle_dist)
        executed_name = str(action.get("name", "safe_side_hard"))
        self.rl_learner.commit_action(agent, state, executed_name, snapshot)
        self.rl_step_committed[agent] = True

    def select_rl_avoidance_action(self, agent: str, snapshot):
        """
        RL action + hard safety shield.

        Q-table은 보조 정책으로만 사용한다. 장애물이 보일 때는 정지/잘못된 방향
        선택을 강제로 막는다. 발표용 안정성을 위해 depth 기반 안전 규칙이 최종 우선권을 가진다.
        """
        fallback = {
            "name": "safe_side_hard",
            "forward": -0.40,
            "side": 1.20,
            "speed_scale": 1.00,
            "force_side": "safe",
        }

        if not CFG.RL_COLLISION_LEARNING_ENABLED:
            return fallback

        obstacle = bool(snapshot.get("obstacle", False))
        left_clear = float(snapshot.get("left_clear", 100.0))
        right_clear = float(snapshot.get("right_clear", 100.0))
        obstacle_dist = self.get_adaptive_obstacle_dist(agent)

        try:
            state = self.rl_current_state.get(agent)
            if state is None:
                state = self.rl_learner.make_state(snapshot, obstacle_dist)
            action = self.rl_learner.select_action(agent, state)
        except Exception:
            return fallback

        # No obstacle: do not allow stop-like behavior.
        if not obstacle:
            if action.get("force_side") == "hold" or action.get("name") == "slow_hold":
                return {
                    "name": "no_obstacle_keep_moving",
                    "forward": 0.10,
                    "side": 0.40,
                    "speed_scale": 0.60,
                    "force_side": "safe",
                }
            return action

        # Obstacle present: stop/hold is forbidden.
        if action.get("force_side") == "hold" or action.get("name") == "slow_hold":
            action = dict(fallback)

        min_clear = min(left_clear, right_clear)

        # Body-corridor safety shield.  When a side is already too close,
        # deterministic escape has priority over learned Q-values.
        hard_side = float(getattr(CFG, "HARD_SIDE_CLEAR_DIST", 5.5))
        if min_clear < hard_side:
            return {
                "name": "shield_side_scrape_escape",
                "forward": float(getattr(CFG, "HARD_AVOID_FORWARD", -1.25)),
                "side": float(getattr(CFG, "HARD_AVOID_SIDE", 1.35)),
                "speed_scale": 1.0,
                "force_side": "right" if right_clear >= left_clear else "left",
            }

        # Both sides blocked: back out first.
        if min_clear < obstacle_dist * 0.45:
            return {
                "name": "safety_reverse_both_blocked",
                "forward": -1.00,
                "side": 0.20,
                "speed_scale": 0.85,
                "force_side": "safe",
            }

        # If one side is clearly open, force that side.
        if right_clear > left_clear + 1.0:
            if action.get("force_side") == "left":
                return {
                    "name": "safety_force_right",
                    "forward": -0.35,
                    "side": 1.20,
                    "speed_scale": 1.00,
                    "force_side": "right",
                }
            if action.get("force_side") == "safe":
                action["side"] = max(float(action.get("side", 1.0)), 1.10)
                action["forward"] = min(float(action.get("forward", -0.3)), -0.25)
                action["speed_scale"] = max(float(action.get("speed_scale", 0.8)), 0.85)
            return action

        if left_clear > right_clear + 1.0:
            if action.get("force_side") == "right":
                return {
                    "name": "safety_force_left",
                    "forward": -0.35,
                    "side": 1.20,
                    "speed_scale": 1.00,
                    "force_side": "left",
                }
            if action.get("force_side") == "safe":
                action["side"] = max(float(action.get("side", 1.0)), 1.10)
                action["forward"] = min(float(action.get("forward", -0.3)), -0.25)
                action["speed_scale"] = max(float(action.get("speed_scale", 0.8)), 0.85)
            return action

        # Balanced: move laterally with a reverse component.
        if action.get("force_side") in ("left", "right", "safe"):
            action["forward"] = min(float(action.get("forward", -0.3)), -0.35)
            action["side"] = max(float(action.get("side", 1.0)), 1.10)
            action["speed_scale"] = max(float(action.get("speed_scale", 0.8)), 0.90)
            action["name"] = f"safety_balanced_{action.get('name', 'rl')}"
            return action

        return fallback

    def reward_step(self, agent: str, coverage_delta: float, snapshot, proximity_penalty: float = 0.0):
        reward = CFG.STEP_ALIVE_REWARD
        reward += coverage_delta * CFG.MAP_COVERAGE_REWARD_SCALE
        if snapshot.get("obstacle", False):
            reward += getattr(CFG, "RL_OBSTACLE_STEP_REWARD", CFG.OBSTACLE_AVOID_REWARD)
        reward -= proximity_penalty * CFG.PROXIMITY_PENALTY_SCALE
        self.add_reward(agent, reward)
        if not CFG.RL_COLLISION_LEARNING_ENABLED:
            return
        if not self.rl_step_committed.get(agent, False) or not self.rl_learner.has_pending(agent):
            return
        obstacle_dist = self.get_adaptive_obstacle_dist(agent)
        next_state = self.rl_learner.make_state(snapshot, obstacle_dist)
        self.rl_learner.update(agent, reward=reward, next_state=next_state)
        self.rl_step_committed[agent] = False

    def reward_safe_if_due(self, agent: str, snapshot):
        if not CFG.RL_COLLISION_LEARNING_ENABLED:
            return
        if snapshot.get("obstacle", False):
            return
        left = float(snapshot.get("left_clear", 100.0))
        right = float(snapshot.get("right_clear", 100.0))
        obstacle_dist = self.get_adaptive_obstacle_dist(agent)
        if min(left, right) < obstacle_dist * 0.85:
            return
        now = time.time()
        with self.lock:
            last_t = self.last_safe_reward_time.get(agent, 0.0)
        if now - last_t < CFG.RL_SAFE_REWARD_DT:
            return
        if self.rl_step_committed.get(agent, False):
            return
        self.rl_commit_action(agent, snapshot, {"name": "safe_side_soft"})
        next_state = self.rl_learner.make_state(snapshot, obstacle_dist)
        self.rl_learner.update(agent, reward=CFG.RL_SAFE_REWARD, next_state=next_state)
        self.rl_step_committed[agent] = False
        self.add_reward(agent, CFG.RL_SAFE_REWARD)
        with self.lock:
            self.last_safe_reward_time[agent] = now
            current = max(
                CFG.ADAPTIVE_OBSTACLE_DIST_MIN,
                self.adaptive_obstacle_dist.get(agent, CFG.OBSTACLE_DIST) - CFG.ADAPTIVE_OBSTACLE_DIST_DECAY_ON_SAFE,
            )
            self.adaptive_obstacle_dist[agent] = current

    def reward_collision(
        self,
        agent: str,
        object_name: str,
        drone_collision: bool = False,
        snapshot=None,
    ):
        self.collision_count += 1
        penalty = CFG.DRONE_COLLISION_REWARD if drone_collision else CFG.RL_COLLISION_REWARD
        self.add_reward(agent, penalty)
        if not CFG.RL_COLLISION_LEARNING_ENABLED:
            return
        if snapshot is None:
            snapshot = {
                "obstacle": True,
                "left_clear": 0.5,
                "right_clear": 0.5,
                "front_clear": 0.5,
            }
        obstacle_dist = self.get_adaptive_obstacle_dist(agent)
        if not self.rl_learner.has_pending(agent):
            state = self.rl_learner.make_state(snapshot, obstacle_dist)
            self.rl_learner.commit_action(agent, state, "reverse", snapshot)
        action_used = self.rl_learner.last_action.get(agent, "?")
        next_state = self.rl_learner.make_state(snapshot, obstacle_dist)
        self.rl_learner.update(agent, reward=penalty, next_state=next_state)
        self.rl_step_committed[agent] = False
        with self.lock:
            current = min(
                CFG.ADAPTIVE_OBSTACLE_DIST_MAX,
                self.adaptive_obstacle_dist.get(agent, CFG.OBSTACLE_DIST)
                + CFG.ADAPTIVE_OBSTACLE_DIST_INC_ON_COLLISION,
            )
            self.adaptive_obstacle_dist[agent] = current
        if getattr(CFG, "RL_LOG_ENABLED", True):
            print(
                f"[RL] collision learned: agent={agent} action={action_used} "
                f"object={object_name} new_obstacle_dist={current:.2f} reward={penalty:.1f} "
                f"q_states={len(self.rl_learner.q)} updates={self.rl_learner.update_count}"
            )
        self.rl_learner.save()

    def _frontier_candidates(self, belief_map: BeliefMap) -> List[Tuple[float, float, float]]:
        """Unknown-cell candidates inside the active 100x100 local window."""
        candidates = []
        step = max(1, int(CFG.FRONTIER_SAMPLE_STEP))
        local_area = self.get_local_area()

        with belief_map.lock:
            grid = belief_map.grid.copy()
            h, w = grid.shape

        if local_area is not None:
            g0 = belief_map.world_to_grid(local_area["x_min"], local_area["y_min"])
            g1 = belief_map.world_to_grid(local_area["x_max"], local_area["y_max"])
            if g0 is None or g1 is None:
                gx_range = range(0, w, step)
                gy_range = range(0, h, step)
            else:
                gx0, gy0 = g0
                gx1, gy1 = g1
                gx0, gx1 = min(gx0, gx1), max(gx0, gx1)
                gy0, gy1 = min(gy0, gy1), max(gy0, gy1)
                gx_range = range(max(0, gx0), min(w, gx1 + 1), step)
                gy_range = range(max(0, gy0), min(h, gy1 + 1), step)
        else:
            gx_range = range(0, w, step)
            gy_range = range(0, h, step)

        for gy in gy_range:
            for gx in gx_range:
                if grid[gy, gx] != GridValue.UNKNOWN:
                    continue
                x = belief_map.x_min + gx * belief_map.resolution
                y = belief_map.y_min + gy * belief_map.resolution
                if not inside_geofence(x, y):
                    continue
                if not self.point_inside_dynamic_bounds(x, y, margin=CFG.AREA_MARGIN):
                    continue
                if local_area is not None and not self.point_in_local_area(x, y):
                    continue
                unknown = belief_map.unknown_ratio_radius(x, y, CFG.FRONTIER_UNKNOWN_RADIUS)
                if unknown <= 0.15:
                    continue
                border_bonus = 0.0
                y0, y1 = max(0, gy - 3), min(h, gy + 4)
                x0, x1 = max(0, gx - 3), min(w, gx + 4)
                sub = grid[y0:y1, x0:x1]
                if (sub != GridValue.UNKNOWN).any():
                    border_bonus = 2.0
                score = CFG.FRONTIER_ASSIGNMENT_BONUS * unknown + border_bonus - CFG.P_GEOFENCE * 0.2 * geofence_penalty(x, y)
                candidates.append((score, x, y))
        candidates.sort(reverse=True, key=lambda t: t[0])
        return candidates[:CFG.FRONTIER_MAX_CANDIDATES]

    # ============================================================
    # Moving local-window expansion
    # ============================================================

    def _window_area_from_center(self, cx: float, cy: float):
        half = CFG.LOCAL_AREA_SIZE / 2.0
        return {
            "x_min": cx - half,
            "x_max": cx + half,
            "y_min": cy - half,
            "y_max": cy + half,
        }

    def _window_center_allowed(self, cx: float, cy: float, respect_dynamic_bounds: bool = True) -> bool:
        """Return True if a 100x100 window centered at (cx, cy) can be used."""
        half = CFG.LOCAL_AREA_SIZE / 2.0
        margin = CFG.AREA_MARGIN

        # Virtual map/geofence check. This is only a large safety box, not the discovered wall boundary.
        corners = (
            (cx - half + margin, cy - half + margin),
            (cx - half + margin, cy + half - margin),
            (cx + half - margin, cy - half + margin),
            (cx + half - margin, cy + half - margin),
        )
        for px, py in corners:
            if not inside_geofence(px, py):
                return False

        # If real wall boundaries have been discovered, obey them.
        # If boundaries are unknown or invalid, do not let them stop exploration.
        if respect_dynamic_bounds and getattr(CFG, "DYNAMIC_BOUNDARY_ENABLED", True):
            area = self._window_area_from_center(cx, cy)
            if not self.area_inside_dynamic_bounds(area, margin=margin):
                return False

        return True

    def _window_stats(self, belief_map: BeliefMap, cx: float, cy: float):
        area = self._window_area_from_center(cx, cy)
        try:
            return belief_map.coverage_stats_area(area, ignore_margin=CFG.COVERAGE_IGNORE_MARGIN)
        except Exception:
            # Fallback if an older BeliefMap is accidentally used.
            return {"observed": 0.0, "visited": 0.0, "obstacle": 0.0, "unknown": 1.0}

    def _score_window_center(self, belief_map: BeliefMap, cx: float, cy: float, current_center, visited_centers, respect_dynamic_bounds=True):
        if not self._window_center_allowed(cx, cy, respect_dynamic_bounds=respect_dynamic_bounds):
            return None

        min_sep = getattr(CFG, "NEXT_WINDOW_MIN_CENTER_SEPARATION", 70.0)
        if visited_centers and any(distance_2d((cx, cy), vc) < min_sep for vc in visited_centers):
            return None

        stats = self._window_stats(belief_map, cx, cy)
        unknown = float(stats.get("unknown", 0.0))
        visited = float(stats.get("visited", 0.0))
        obstacle = float(stats.get("obstacle", 0.0))

        min_unknown = getattr(CFG, "NEXT_WINDOW_MIN_UNKNOWN_RATIO", 0.05)
        if unknown < min_unknown:
            return None

        dist = distance_2d((cx, cy), current_center)
        # Unknown-rich windows are preferred, but avoid jumping absurdly far.
        score = unknown * 120.0 - visited * 30.0 - obstacle * 10.0 - CFG.NEXT_WINDOW_DISTANCE_WEIGHT * dist

        # Prefer windows that are roughly adjacent to current center.
        # This makes the map expand like attached tiles instead of teleporting far away.
        preferred_step = getattr(CFG, "NEXT_WINDOW_EXPANSION_STEP", CFG.LOCAL_AREA_SIZE * 0.85)
        if dist <= preferred_step * 1.5:
            score += getattr(CFG, "NEXT_WINDOW_ADJACENT_BONUS", 12.0)

        return (score, cx, cy, unknown)

    def _select_adjacent_window_center(self, belief_map: BeliefMap, current_center, visited_centers, respect_dynamic_bounds=True):
        """Force expansion to adjacent/ring windows when global frontier sampling fails."""
        cx0, cy0 = current_center
        step = getattr(CFG, "NEXT_WINDOW_EXPANSION_STEP", CFG.LOCAL_AREA_SIZE * 0.85)
        max_ring = int(getattr(CFG, "NEXT_WINDOW_MAX_RING", 6))

        # Cardinal first, diagonal second. This makes the search area attach cleanly.
        dirs = [
            (1, 0), (-1, 0), (0, 1), (0, -1),
            (1, 1), (1, -1), (-1, 1), (-1, -1),
        ]

        best = None
        best_score = -1e9

        for ring in range(1, max_ring + 1):
            for dx, dy in dirs:
                cx = cx0 + dx * step * ring
                cy = cy0 + dy * step * ring

                cand = self._score_window_center(
                    belief_map,
                    cx,
                    cy,
                    current_center=current_center,
                    visited_centers=visited_centers,
                    respect_dynamic_bounds=respect_dynamic_bounds,
                )

                if cand is None:
                    continue

                score, wx, wy, unknown = cand
                # Ring penalty keeps the expansion local unless closer rings are exhausted.
                score -= ring * getattr(CFG, "NEXT_WINDOW_RING_PENALTY", 4.0)

                if score > best_score:
                    best_score = score
                    best = (wx, wy, score, unknown)

            if best is not None:
                if getattr(CFG, "WINDOW_DEBUG_LOG_ENABLED", False):
                    print(
                        f"[WINDOW FALLBACK] adjacent/ring candidate selected: "
                        f"x={best[0]:.1f}, y={best[1]:.1f}, unknown={best[3]:.2f}, score={best[2]:.2f}"
                    )
                return best

        return None

    def select_next_window_center(self, belief_map: BeliefMap, current_center: Tuple[float, float], visited_centers=None):
        """
        Pick the next 100x100 local-window center.

        Policy:
        1. Try normal global unknown-rich sampling.
        2. If that fails, try adjacent/ring expansion around the current window.
        3. If dynamic wall bounds block every candidate, retry once ignoring uncertain dynamic bounds.

        This prevents premature NO_MORE_GLOBAL_FRONTIER when the map still contains large unknown areas.
        """
        if visited_centers is None:
            visited_centers = []

        step = max(1, int(CFG.NEXT_WINDOW_SAMPLE_STEP))
        best = None
        best_score = -1e9

        with belief_map.lock:
            grid = belief_map.grid.copy()
            h, w = grid.shape

        # Pass 1: global unknown-rich sampling.
        for gy in range(0, h, step):
            for gx in range(0, w, step):
                cx = belief_map.x_min + gx * belief_map.resolution
                cy = belief_map.y_min + gy * belief_map.resolution

                cand = self._score_window_center(
                    belief_map,
                    cx,
                    cy,
                    current_center=current_center,
                    visited_centers=visited_centers,
                    respect_dynamic_bounds=True,
                )

                if cand is None:
                    continue

                score, wx, wy, unknown = cand

                # Add a small bonus if the sampled center itself is unknown.
                if grid[gy, gx] == GridValue.UNKNOWN:
                    score += 4.0

                if score > best_score:
                    best_score = score
                    best = (wx, wy, score, unknown)

        if best is not None:
            if getattr(CFG, "WINDOW_DEBUG_LOG_ENABLED", False):
                print(
                    f"[WINDOW SELECT] global unknown candidate: "
                    f"x={best[0]:.1f}, y={best[1]:.1f}, unknown={best[3]:.2f}, score={best[2]:.2f}"
                )
            return best

        # Pass 2: adjacent/ring fallback. This is the important part for attached map expansion.
        if getattr(CFG, "NEXT_WINDOW_FORCE_ADJACENT_FALLBACK", True):
            best = self._select_adjacent_window_center(
                belief_map,
                current_center=current_center,
                visited_centers=visited_centers,
                respect_dynamic_bounds=True,
            )
            if best is not None:
                return best

        # Pass 3: if false wall boundaries blocked all candidates, ignore dynamic bounds once.
        # Real virtual geofence is still obeyed.
        if getattr(CFG, "NEXT_WINDOW_IGNORE_DYNAMIC_BOUNDS_ON_LAST_RESORT", True):
            best = self._select_adjacent_window_center(
                belief_map,
                current_center=current_center,
                visited_centers=visited_centers,
                respect_dynamic_bounds=False,
            )
            if best is not None:
                if getattr(CFG, "WINDOW_DEBUG_LOG_ENABLED", False):
                    print("[WINDOW FALLBACK] selected after ignoring uncertain dynamic wall bounds")
                return best

        if getattr(CFG, "WINDOW_DEBUG_LOG_ENABLED", False):
            print("[WINDOW SELECT] no valid next local window candidate")
        return None

    def assign_global_goals(self, belief_map: BeliefMap, force: bool = False):
        now = time.time()
        if not force and now - self.last_goal_assign_time < CFG.GLOBAL_GOAL_ASSIGN_DT:
            return
        self.last_goal_assign_time = now
        candidates = self._frontier_candidates(belief_map)
        if not candidates:
            # Initial spread inside the current local area if no frontier exists yet.
            area = self.get_local_area()
            if area is None:
                cx = cy = 0.0
            else:
                cx = (area["x_min"] + area["x_max"]) / 2.0
                cy = (area["y_min"] + area["y_max"]) / 2.0
            # Spread agents around the window perimeter (works for any agent count).
            spread = CFG.LOCAL_AREA_SIZE * 0.35
            n_agents = max(1, len(CFG.AGENTS))
            fallback = []
            for i in range(n_agents):
                angle = (2.0 * math.pi * i) / n_agents
                fallback.append((cx + spread * math.cos(angle), cy + spread * math.sin(angle)))
            with self.lock:
                for agent, goal in zip(CFG.AGENTS, fallback):
                    self.global_goals[agent] = goal
                    if getattr(CFG, "GOAL_LOG_ENABLED", False):
                        print(f"[GLOBAL GOAL] {agent}: x={goal[0]:.1f}, y={goal[1]:.1f}, reason=aggressive_outer_fallback")
            return

        selected = []
        with self.lock:
            positions = dict(self.positions)
        for agent in CFG.AGENTS:
            ax, ay, _ = positions.get(agent, (0.0, 0.0, CFG.AGENT_Z))
            best = None
            best_score = -1e9
            for base_score, x, y in candidates:
                sep_ok = all(distance_2d((x, y), p) >= CFG.FRONTIER_MIN_SEPARATION for p in selected)
                sep_penalty = 0.0 if sep_ok else 5.0
                dist = math.hypot(x - ax, y - ay)
                score = base_score - CFG.FRONTIER_DISTANCE_WEIGHT * dist - sep_penalty
                if score > best_score:
                    best_score = score
                    best = (x, y)
            if best is not None:
                selected.append(best)
                with self.lock:
                    self.global_goals[agent] = best
                if getattr(CFG, "GOAL_LOG_ENABLED", False):
                    print(f"[GLOBAL GOAL] {agent}: x={best[0]:.1f}, y={best[1]:.1f}, score={best_score:.2f}")

    def get_global_goal(self, agent: str) -> Optional[Tuple[float, float]]:
        with self.lock:
            return self.global_goals.get(agent)

    # ============================================================
    # Target tracking control
    # ============================================================

    def get_target_ground_truth_pose(self):
        target_gt_x = target_gt_y = target_gt_z = None
        try:
            c = connect_client(confirm=False)
            pose = rpc_call(c.simGetObjectPose, CFG.TARGET_OBJECT_NAME)
            x = float(pose.position.x_val)
            y = float(pose.position.y_val)
            z = float(pose.position.z_val)
            if math.isfinite(x) and math.isfinite(y) and math.isfinite(z):
                target_gt_x, target_gt_y, target_gt_z = x, y, z
        except Exception as e:
            print(f"[WARN] failed to get target ground truth pose: {e}")
        return target_gt_x, target_gt_y, target_gt_z

    def has_target(self) -> bool:
        with self.lock:
            return bool(self.target_tracking_active and self.target_info is not None)

    def get_target_xy(self):
        with self.lock:
            if not self.target_tracking_active or self.target_info is None:
                return None
            return (float(self.target_info["target_x"]), float(self.target_info["target_y"]))

    def claim_target_map_mark(self) -> bool:
        """Return True only for the first worker that should mark redball_01 on the map."""
        with self.lock:
            if self.target_map_marked:
                return False
            self.target_map_marked = True
            return True


    def get_target_xy_for_agent(self, agent_name: str, current_xy=None):
        """Return an autonomous target approach point for each agent.

        Commandless policy:
        - The detecting/primary tracker flies to the true redball center.
        - Other agents fly to offset approach points around the target.
        - Completion still uses the true target cell, so any agent entering the
          redball radius can finish the mission.
        """
        base = self.get_target_xy()
        if base is None:
            return None

        tx, ty = base
        with self.lock:
            primary = self.primary_tracker_agent

        spacing = float(getattr(CFG, "TARGET_APPROACH_SPACING", 7.0))
        ordered = list(getattr(CFG, "AGENTS", ()))
        non_primary = [a for a in ordered if a != primary]

        offsets = {primary: (0.0, 0.0)}
        pattern = [
            (spacing, 0.0),
            (-spacing, 0.0),
            (0.0, spacing),
            (0.0, -spacing),
            (spacing, spacing),
            (-spacing, -spacing),
        ]
        for a, off in zip(non_primary, pattern):
            offsets[a] = off

        ox, oy = offsets.get(agent_name, (0.0, 0.0))
        return tx + ox, ty + oy

    def get_target_xyz(self):
        with self.lock:
            if not self.target_tracking_active or self.target_info is None:
                return None
            return (
                float(self.target_info["target_x"]),
                float(self.target_info["target_y"]),
                float(self.target_info["target_z"]),
            )

    def should_hold_after_target(self, agent_name: str) -> bool:
        if not getattr(CFG, "TARGET_TRACKING_ENABLED", True):
            return False
        if getattr(CFG, "ALL_AGENTS_TRACK_TARGET", False):
            return False
        recording = bool(getattr(CFG, "TARGET_RECORDING_MODE", False))
        hold = bool(getattr(CFG, "HOLD_NON_TRACKER_AFTER_TARGET", False))
        if not hold and not recording:
            return False
        tracker = getattr(CFG, "TRACKER_AGENT", "Agent1")
        with self.lock:
            active = self.target_tracking_active
            primary = self.primary_tracker_agent or tracker
        return bool(active and agent_name != primary)

    def is_primary_tracker(self, agent_name: str) -> bool:
        """True if this agent is the one assigned to approach the red target center."""
        with self.lock:
            return bool(self.target_tracking_active and agent_name == self.primary_tracker_agent)

    def mark_tracker_arrived(self, agent_name: str, x: float, y: float, z: float):
        first_arrival = False
        with self.lock:
            if agent_name not in self.tracker_arrived_agents:
                self.tracker_arrived_agents.add(agent_name)

            # Completion rule: one arrived drone is enough.
            if getattr(CFG, "TARGET_COMPLETE_ON_FIRST_ARRIVAL", True):
                if not self.tracker_arrived:
                    self.tracker_arrived = True
                    first_arrival = True
            else:
                if len(self.tracker_arrived_agents) >= len(CFG.AGENTS) and not self.tracker_arrived:
                    self.tracker_arrived = True
                    first_arrival = True

        if first_arrival and getattr(CFG, "TARGET_ARRIVAL_LOG_ENABLED", True):
            print()
            print("=" * 70)
            print("[TARGET APPROACH COMPLETE]")
            print(f"agent : {agent_name}")
            print(f"pos   : x={x:.2f}, y={y:.2f}, z={z:.2f}")
            print("=" * 70)
            print()

        if first_arrival:
            self.request_stop(f"TARGET_REACHED agent={agent_name}")

    def report_target(self, drone: str, x: float, y: float, z: float, camera: str, confidence: float):
        """
        Target detected: do NOT stop mission immediately.
        Store target position, start tracking mode, and let Agent1 approach the target.
        """
        target_gt_x, target_gt_y, target_gt_z = self.get_target_ground_truth_pose()

        if target_gt_x is None:
            target_x, target_y, target_z = x, y, z
        else:
            target_x, target_y, target_z = target_gt_x, target_gt_y, target_gt_z

        target_info = {
            "drone": drone,
            "drone_x": x,
            "drone_y": y,
            "drone_z": z,
            "target_x": target_x,
            "target_y": target_y,
            "target_z": target_z,
            "target_gt_x": target_gt_x,
            "target_gt_y": target_gt_y,
            "target_gt_z": target_gt_z,
            "camera": camera,
            "confidence": confidence,
            "time": time.time(),
        }

        first_report = False
        update_report = False
        with self.lock:
            if self.target_info is None:
                self.target_info = target_info
                self.target_tracking_active = True
                selector = getattr(CFG, "TARGET_TRACKER_SELECTOR", "fixed")
                if selector == "detector" and drone in getattr(CFG, "AGENTS", ()):
                    self.primary_tracker_agent = drone
                else:
                    self.primary_tracker_agent = getattr(CFG, "TRACKER_AGENT", "Agent1")
                self.tracker_arrived = False
                self.command_target_sent = False
                first_report = True
            else:
                old_conf = float(self.target_info.get("confidence", 0.0))
                if confidence > old_conf:
                    self.target_info = target_info
                    update_report = True
            self.targets.append(target_info)

        self.note_agent_target_distance(drone, x, y)

        if first_report:
            print()
            print("=" * 70)
            print("[TARGET DETECTED - TRACKING MODE START]")
            print(f"detected by     : {drone}")
            print(f"drone position  : x={x:.2f}, y={y:.2f}, z={z:.2f}")
            print(f"target position : x={target_x:.2f}, y={target_y:.2f}, z={target_z:.2f}")
            print(f"primary tracker : {self.primary_tracker_agent}")
            if getattr(CFG, "ALL_AGENTS_TRACK_TARGET", False):
                print(f"tracking agents : {', '.join(CFG.AGENTS)}")
            elif getattr(CFG, "TARGET_RECORDING_MODE", False) or getattr(
                CFG, "HOLD_NON_TRACKER_AFTER_TARGET", False
            ):
                hover_agents = [a for a in CFG.AGENTS if a != self.primary_tracker_agent]
                print(f"tracking agents : {self.primary_tracker_agent}")
                print(f"hover agents    : {', '.join(hover_agents)} (recording)")
            else:
                explorers = [a for a in CFG.AGENTS if a != self.primary_tracker_agent]
                print(f"tracking agents : {self.primary_tracker_agent}")
                print(f"map expansion   : {', '.join(explorers)}")
            print(f"camera          : {camera}")
            print(f"confidence      : {confidence:.3f}")
            print("=" * 70)
            print()
        elif update_report and getattr(CFG, "TARGET_UPDATE_LOG_ENABLED", False):
            print(
                f"[TARGET UPDATE] confidence improved by {drone}: "
                f"target=({target_x:.2f}, {target_y:.2f}, {target_z:.2f}), conf={confidence:.3f}"
            )

        # Do not request_stop here. Agent1 will track the target.

    def print_summary(self):
        with self.lock:
            stop_reason = self.stop_reason
            coverage = self.latest_coverage
            observed_cov = self.latest_observed_coverage
            visited_cov = self.latest_visited_coverage
            obstacle_cov = self.latest_obstacle_coverage
            unknown_cov = self.latest_unknown_coverage
            status_copy = dict(self.status)
            targets_copy = list(self.targets)
            adaptive_copy = dict(self.adaptive_obstacle_dist)
            rewards_copy = dict(self.rewards)
            goals_copy = dict(self.global_goals)
            local_area = None if self.local_area is None else dict(self.local_area)
        print("\n" + "=" * 70)
        print("[COMMAND HUB SUMMARY]")
        print(f"- Stop reason     : {stop_reason if stop_reason else 'RUNNING'}")
        print(f"- Belief coverage : {coverage * 100:.2f}%")
        print(f"  observed={observed_cov * 100:.2f}%, visited={visited_cov * 100:.2f}%, obstacle={obstacle_cov * 100:.2f}%, unknown={unknown_cov * 100:.2f}%")
        print(f"- Collisions      : {self.collision_count}")
        print(f"- Rewards         : {rewards_copy}")
        print(f"- Adaptive depth  : {adaptive_copy}")
        print(f"- Wall bounds     : {self.get_discovered_bounds()}")
        print(f"- Boundaries done : {self.all_boundaries_discovered()}")
        print(f"- Local area      : {local_area}")
        print(f"- Global goals    : {goals_copy}")
        if self.has_target():
            print(f"- Target tracking : ACTIVE target={self.get_target_xyz()} arrived_agents={sorted(list(self.tracker_arrived_agents))}")
        print("- Drone status")
        for k, v in status_copy.items():
            print(f"  {k}: {v}")
        if targets_copy:
            print("- Targets")
            for t in targets_copy:
                tx = t.get("target_x", t.get("target_gt_x"))
                ty = t.get("target_y", t.get("target_gt_y"))
                tz = t.get("target_z", t.get("target_gt_z"))
                print(f"  {t['drone']} -> target=({tx}, {ty}, {tz}), conf={t['confidence']:.3f}, camera={t['camera']}")
        print("=" * 70 + "\n")
