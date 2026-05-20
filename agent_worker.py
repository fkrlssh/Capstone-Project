import math
import threading
import time
from typing import Dict

import airsim

from airsim_rpc import rpc_call
from config import CFG
from belief_map import BeliefMap, GridValue
from planner_6grid import SixGridPlanner
from target_detector import check_target_with_cameras
from camera_utils import detect_front_obstacle, detect_front_wall
from utils import (
    connect_client,
    get_pose,
    distance_2d,
    normalize2,
    limit_norm,
    apply_accel_limit,
    geofence_push_velocity,
    inside_geofence,
    move_velocity_continuous,
    yaw_from_velocity,
    safe_hover,
)


class AgentWorker(threading.Thread):
    def __init__(self, name: str, strip: Dict[str, float], hub, belief_map: BeliefMap):
        super().__init__()
        self.name = name
        self.strip = strip
        self.hub = hub
        self.belief_map = belief_map
        self.client = None
        self.planner = SixGridPlanner(agent_name=name, strip=strip, belief_map=belief_map)
        self.prev_vx = 0.0
        self.prev_vy = 0.0
        self.current_target_cell = None
        self.current_target_xy = None
        self.perception_stop = threading.Event()
        self.perception_lock = threading.Lock()
        self.cached_obstacle = False
        self.cached_left_clear = 100.0
        self.cached_right_clear = 100.0
        self.cached_front_clear = 100.0
        self.cached_near_ratio = 0.0
        self.cached_wall_detected = False
        self.cached_wall_distance = 100.0
        self.cached_target_found = False
        self.cached_target_camera = ""
        self.cached_target_conf = 0.0
        self.target_confirm_count = 0
        self.target_reported = False
        self.last_collision_time = None
        self.last_move_debug_time = 0.0
        self._obstacle_stable = False
        self._obstacle_raw_since = None
        self._obstacle_clear_since = None
        self._cmd_smooth_init = False
        self._smooth_cmd_vx = 0.0
        self._smooth_cmd_vy = 0.0
        self._last_vertical_avoid_time = 0.0
        self._last_non_tracker_hover_time = 0.0
        self._non_tracker_hold_engaged = False

        # Dynamic low-altitude target. Base altitude is CFG.AGENT_Z,
        # but agents may temporarily move slightly up/down for avoidance.
        self.altitude_target_z = CFG.AGENT_Z
        self.altitude_override_until = 0.0
        self.last_altitude_reason = "base"
        self.last_snapshot: Dict = {}

    def run(self):
        self.client = connect_client(confirm=False)
        if getattr(CFG, "AGENT_VERSION_LOG_ENABLED", False):
            print(f"[AGENT VERSION] {self.name} loaded {CFG.VERSION}")
        self.hub.update_status(self.name, "START_EPISODE_SEARCH")
        stagger = float(getattr(CFG, "AGENT_START_STAGGER_SEC", 0.0) or 0.0)
        if stagger > 0 and self.name in CFG.AGENTS:
            time.sleep(CFG.AGENTS.index(self.name) * stagger)
        perception_thread = threading.Thread(target=self.perception_loop, daemon=True)
        perception_thread.start()
        try:
            self.search_loop()
        except Exception as e:
            self.hub.update_status(self.name, f"ERROR: {e}")
            print(f"[ERROR] {self.name}: {e}")
        finally:
            self.perception_stop.set()
            perception_thread.join(timeout=1.0)
            self.hub.clear_reservation(self.name)
        safe_hover(self.client, self.name)

    def perception_loop(self):
        pclient = connect_client(confirm=False)
        last_obstacle = 0.0
        last_target = 0.0
        while not self.perception_stop.is_set() and not self.hub.stop_event.is_set():
            now = time.time()
            if now - last_obstacle >= CFG.OBSTACLE_CHECK_DT:
                stagger = float(getattr(CFG, "PERCEPTION_OBSTACLE_STAGGER_SEC", 0.0) or 0.0)
                if stagger > 0 and self.name in CFG.AGENTS:
                    time.sleep(CFG.AGENTS.index(self.name) * stagger)
                obstacle_dist = self.hub.get_adaptive_obstacle_dist(self.name)
                obstacle_result = detect_front_obstacle(
                    pclient,
                    self.name,
                    obstacle_dist=obstacle_dist,
                )
                # Backward-compatible unpacking: older camera_utils.py returned 3 values,
                # newer cylinder-aware camera_utils.py returns 5 values.
                if len(obstacle_result) >= 5:
                    obstacle, left_clear, right_clear, front_clear, near_ratio = obstacle_result[:5]
                else:
                    obstacle, left_clear, right_clear = obstacle_result[:3]
                    front_clear = min(float(left_clear), float(right_clear), float(obstacle_dist) if obstacle else 100.0)
                    near_ratio = 0.0

                if getattr(CFG, "PERCEPTION_WALL_WITH_OBSTACLE", True) and getattr(
                    CFG, "WALL_DETECTION_ENABLED", True
                ):
                    wall_detected, wall_distance = detect_front_wall(
                        pclient,
                        self.name,
                        max_distance=getattr(CFG, "WALL_DETECT_MAX_DISTANCE", 45.0),
                    )
                else:
                    wall_detected, wall_distance = False, 100.0
                with self.perception_lock:
                    self.cached_obstacle = obstacle
                    self.cached_left_clear = left_clear
                    self.cached_right_clear = right_clear
                    self.cached_front_clear = front_clear
                    self.cached_near_ratio = near_ratio
                    self.cached_wall_detected = wall_detected
                    self.cached_wall_distance = wall_distance
                last_obstacle = now
            if now - last_target >= CFG.TARGET_CHECK_DT:
                # Once any agent has already detected the target, stop repeated target
                # segmentation checks to keep logs/RPC load low. Tracking uses the shared
                # target pose stored in CommandHub.
                if hasattr(self.hub, "has_target") and self.hub.has_target():
                    found, camera, conf = False, "", 0.0
                else:
                    found, camera, conf = check_target_with_cameras(pclient, self.name, self.hub.target_detection_active)
                with self.perception_lock:
                    self.target_confirm_count = self.target_confirm_count + 1 if found else 0
                    if self.target_confirm_count >= CFG.TARGET_CONFIRM_FRAMES:
                        self.cached_target_found = True
                        self.cached_target_camera = camera
                        self.cached_target_conf = conf
                    else:
                        self.cached_target_found = False
                        self.cached_target_camera = ""
                        self.cached_target_conf = 0.0
                last_target = now
            time.sleep(0.03)

    def get_perception_snapshot(self):
        with self.perception_lock:
            return {
                "obstacle": self.cached_obstacle,
                "left_clear": self.cached_left_clear,
                "right_clear": self.cached_right_clear,
                "front_clear": self.cached_front_clear,
                "near_ratio": self.cached_near_ratio,
                "obstacle_dist": self.hub.get_adaptive_obstacle_dist(self.name),
                "wall_detected": self.cached_wall_detected,
                "wall_distance": self.cached_wall_distance,
                "target_found": self.cached_target_found,
                "target_camera": self.cached_target_camera,
                "target_conf": self.cached_target_conf,
            }



    def stabilize_perception_snapshot(self, snapshot: Dict) -> Dict:
        """Hysteresis on depth obstacle flag to stop rapid avoid/forward toggling."""
        raw = bool(snapshot.get("obstacle", False))
        now = time.time()
        engage = float(getattr(CFG, "OBSTACLE_ENGAGE_DELAY_SEC", 0.18))
        clear = float(getattr(CFG, "OBSTACLE_CLEAR_DELAY_SEC", 0.28))

        if raw:
            self._obstacle_clear_since = None
            if not self._obstacle_stable:
                if self._obstacle_raw_since is None:
                    self._obstacle_raw_since = now
                elif now - self._obstacle_raw_since >= engage:
                    self._obstacle_stable = True
        else:
            self._obstacle_raw_since = None
            if self._obstacle_stable:
                if self._obstacle_clear_since is None:
                    self._obstacle_clear_since = now
                elif now - self._obstacle_clear_since >= clear:
                    self._obstacle_stable = False
            else:
                self._obstacle_clear_since = None

        stable = dict(snapshot)
        stable["obstacle"] = self._obstacle_stable
        return stable

    def soften_previous_velocity(self):
        scale = float(getattr(CFG, "HARD_ESCAPE_PREV_V_SCALE", 0.45))
        self.prev_vx *= scale
        self.prev_vy *= scale

    def finalize_cmd_velocity(self, des_vx: float, des_vy: float, dt: float):
        max_accel = float(getattr(CFG, "MAX_ACCEL", 2.5))
        cmd_vx, cmd_vy = apply_accel_limit(self.prev_vx, self.prev_vy, des_vx, des_vy, dt, max_accel)
        if getattr(CFG, "CMD_VELOCITY_SMOOTH_ENABLED", True):
            alpha = float(getattr(CFG, "CMD_VELOCITY_SMOOTH_ALPHA", 0.42))
            if not self._cmd_smooth_init:
                self._smooth_cmd_vx = cmd_vx
                self._smooth_cmd_vy = cmd_vy
                self._cmd_smooth_init = True
            else:
                self._smooth_cmd_vx = alpha * cmd_vx + (1.0 - alpha) * self._smooth_cmd_vx
                self._smooth_cmd_vy = alpha * cmd_vy + (1.0 - alpha) * self._smooth_cmd_vy
            cmd_vx, cmd_vy = self._smooth_cmd_vx, self._smooth_cmd_vy
        self.prev_vx, self.prev_vy = cmd_vx, cmd_vy
        return cmd_vx, cmd_vy

    def resolve_flight_target_z(self, current_z: float) -> float:
        """Keep altitude steady during search to avoid vertical bobbing."""
        now = time.time()
        if now < self.altitude_override_until:
            return self.clamp_agent_z(self.altitude_target_z)
        if getattr(CFG, "MOTION_FIXED_ALTITUDE_SEARCH", True) and not (
            hasattr(self.hub, "has_target") and self.hub.has_target()
        ):
            return float(CFG.AGENT_Z)
        return self.clamp_agent_z(self.altitude_target_z)

    def is_close_obstacle(self, snapshot: Dict) -> bool:
        """True when the drone is inside an emergency body-safety corridor."""
        front_clear = float(snapshot.get("front_clear", 100.0))
        left_clear = float(snapshot.get("left_clear", 100.0))
        right_clear = float(snapshot.get("right_clear", 100.0))
        hard_front = float(getattr(CFG, "EMERGENCY_FRONT_CLEAR_DIST", 6.0))
        hard_side = float(getattr(CFG, "HARD_SIDE_CLEAR_DIST", 5.5))
        return front_clear < hard_front or min(left_clear, right_clear) < hard_side

    def make_hard_escape_action(self, snapshot: Dict, name: str = "hard_corridor_escape"):
        """Deterministic shield that overrides Q-learning near cylinders.

        If one side is more open, force that side.  This prevents Q-table noise
        from repeatedly choosing the same bad turn near the same cylinder line.
        """
        left_clear = float(snapshot.get("left_clear", 100.0))
        right_clear = float(snapshot.get("right_clear", 100.0))
        force_side = "right" if right_clear >= left_clear else "left"
        return {
            "name": name,
            "forward": float(getattr(CFG, "HARD_AVOID_FORWARD", -1.25)),
            "side": float(getattr(CFG, "HARD_AVOID_SIDE", 1.35)),
            "speed_scale": 1.0,
            "force_side": force_side,
        }

    def smooth_yield_once(self, target_z: float, yaw_deg: float, hold_time: float = 0.0):
        """Yield for collision/priority handling without a hard stop.

        A hard 0-velocity command followed by sleep makes the drone visibly
        stutter.  This keeps a small decayed velocity command alive, so the
        vehicle slows smoothly while still giving priority to the other drone.
        """
        scale = float(getattr(CFG, "YIELD_SLOWDOWN_SCALE", 0.35))
        vx = self.prev_vx * scale
        vy = self.prev_vy * scale
        self.prev_vx = vx
        self.prev_vy = vy
        self.send_velocity(vx, vy, target_z, yaw_deg)
        sleep_t = min(float(hold_time or 0.0), float(getattr(CFG, "YIELD_MAX_SLEEP", 0.03)))
        if sleep_t > 0.0:
            time.sleep(sleep_t)

    def clamp_agent_z(self, z_value: float) -> float:
        """Clamp agent altitude in NED coordinates. More negative z means higher altitude."""
        high_limit = getattr(CFG, "AGENT_Z_HIGH_LIMIT", CFG.AGENT_Z - 2.0)
        low_limit = getattr(CFG, "AGENT_Z_LOW_LIMIT", CFG.AGENT_Z + 1.0)
        return max(high_limit, min(low_limit, z_value))

    def set_temporary_altitude(self, target_z: float, hold_sec: float, reason: str):
        if not getattr(CFG, "AGENT_ALTITUDE_FREE_ENABLED", True):
            self.altitude_target_z = CFG.AGENT_Z
            self.altitude_override_until = 0.0
            return

        target_z = self.clamp_agent_z(target_z)
        now = time.time()
        changed = abs(target_z - self.altitude_target_z) > 0.05 or reason != self.last_altitude_reason
        self.altitude_target_z = target_z
        self.altitude_override_until = max(self.altitude_override_until, now + hold_sec)
        self.last_altitude_reason = reason

        if changed:
            self.hub.update_status(self.name, f"ALTITUDE_TARGET z={target_z:.1f} reason={reason}")

    def update_altitude_target_z(self, current_z: float, snapshot: Dict) -> float:
        """
        Keep low-altitude flight by default, but allow temporary vertical moves.
        Returns the target z to send to moveByVelocityZAsync.
        """
        if not getattr(CFG, "AGENT_ALTITUDE_FREE_ENABLED", True):
            self.altitude_target_z = CFG.AGENT_Z
            return CFG.AGENT_Z

        now = time.time()
        if now < self.altitude_override_until:
            return self.clamp_agent_z(self.altitude_target_z)

        # No active override: return to base low-altitude band.
        self.altitude_target_z = CFG.AGENT_Z
        self.last_altitude_reason = "base"
        return self.altitude_target_z

    def maybe_trigger_vertical_obstacle_avoid(self, current_z: float, snapshot: Dict, rl_action=None):
        """
        If horizontal avoidance is poor, move slightly up or down while staying in the low-altitude band.
        This is not the main avoidance method; it is a secondary safety maneuver.
        """
        if not getattr(CFG, "AGENT_ALTITUDE_FREE_ENABLED", True):
            return

        if not snapshot.get("obstacle", False):
            return

        if getattr(CFG, "VERTICAL_AVOID_ONLY_WHEN_CLOSE", True) and not self.is_close_obstacle(snapshot):
            return

        now = time.time()
        cooldown = float(getattr(CFG, "VERTICAL_AVOID_COOLDOWN_SEC", 4.0))
        if now - self._last_vertical_avoid_time < cooldown:
            return

        # Do not try to overfly walls; wall handling is done by boundary/geofence logic.
        if snapshot.get("wall_detected", False):
            return

        left_clear = float(snapshot.get("left_clear", 100.0))
        right_clear = float(snapshot.get("right_clear", 100.0))
        side_blocked_dist = getattr(CFG, "VERTICAL_AVOID_SIDE_BLOCKED_DIST", 8.0)
        side_blocked = max(left_clear, right_clear) < side_blocked_dist

        slow_hold = False
        if rl_action is not None:
            slow_hold = rl_action.get("name") == "slow_hold" and getattr(CFG, "VERTICAL_AVOID_ON_SLOW_HOLD", True)

        if not side_blocked and not slow_hold:
            return

        step = getattr(CFG, "AGENT_VERTICAL_AVOID_STEP", 1.0)
        high_limit = getattr(CFG, "AGENT_Z_HIGH_LIMIT", CFG.AGENT_Z - 2.0)
        low_limit = getattr(CFG, "AGENT_Z_LOW_LIMIT", CFG.AGENT_Z + 1.0)

        # Prefer going slightly upward first. If already near the upper limit, go slightly downward.
        if current_z - step >= high_limit:
            target_z = current_z - step
            reason = "vertical_obstacle_up"
        else:
            target_z = min(low_limit, current_z + step)
            reason = "vertical_obstacle_down"

        self.set_temporary_altitude(
            target_z,
            getattr(CFG, "AGENT_VERTICAL_AVOID_HOLD_SEC", 2.5),
            reason,
        )
        self._last_vertical_avoid_time = now

    def search_loop(self):
        self.hub.update_status(self.name, "EXECUTOR_RL_FRONTIER_ACTIVE")
        self._cmd_smooth_init = False
        self._obstacle_stable = False
        self._obstacle_raw_since = None
        self._obstacle_clear_since = None
        last_loop_time = time.perf_counter()
        while not self.hub.stop_event.is_set():
            loop_start = time.perf_counter()
            dt = max(1e-3, loop_start - last_loop_time)
            last_loop_time = loop_start

            x, y, z, yaw_deg = get_pose(self.client, self.name)
            self.hub.update_position(self.name, x, y, z)
            self.hub.note_agent_target_distance(self.name, x, y)

            changed = self.belief_map.mark_radius(x, y, GridValue.VISITED, radius=CFG.VISITED_MARK_RADIUS)
            if changed:
                self.hub.add_reward(self.name, min(1.0, changed * 0.015))

            if self.recover_if_collided():
                self.prev_vx = 0.0
                self.prev_vy = 0.0
                self.current_target_cell = None
                self.current_target_xy = None
                continue

            snapshot = self.stabilize_perception_snapshot(self.get_perception_snapshot())
            self.last_snapshot = snapshot
            if getattr(CFG, "RL_COLLISION_LEARNING_ENABLED", True):
                self.hub.rl_prepare_step(self.name, snapshot)
            self.update_altitude_target_z(z, snapshot)
            target_z = self.resolve_flight_target_z(z)

            if snapshot.get("wall_detected", False):
                updated = self.hub.update_wall_boundary_from_pose(
                    source=self.name,
                    x=x,
                    y=y,
                    yaw_deg=yaw_deg,
                    wall_distance=snapshot.get("wall_distance", CFG.WALL_DETECT_MAX_DISTANCE),
                )
                if updated:
                    self.current_target_cell = None
                    self.current_target_xy = None
                    self.hub.clear_reservation(self.name)

            self.hub.reward_safe_if_due(self.name, snapshot)

            if snapshot["target_found"] and not self.target_reported:
                self.target_reported = True

                # Report first, so CommandHub stores the real redball_01 world position.
                # Then mark THAT target position on the belief map in red.
                # Do not mark the detecting drone position as target.
                self.hub.report_target(self.name, x, y, z, snapshot["target_camera"], snapshot["target_conf"])

                target_xyz = self.hub.get_target_xyz()
                should_mark = (
                    not hasattr(self.hub, "claim_target_map_mark")
                    or self.hub.claim_target_map_mark()
                )

                if should_mark:
                    if target_xyz is not None:
                        tx_map, ty_map, tz_map = target_xyz
                    else:
                        # Fallback only if object pose is unavailable.
                        tx_map, ty_map, tz_map = x, y, z

                    self.belief_map.mark_radius(
                        tx_map,
                        ty_map,
                        GridValue.TARGET,
                        radius=getattr(CFG, "TARGET_MAP_MARK_RADIUS", 3.0),
                    )
                    if getattr(CFG, "TARGET_MAP_MARK_LOG_ENABLED", True):
                        print(
                            f"[TARGET MAP MARK] redball_01 marked on map: "
                            f"x={tx_map:.2f}, y={ty_map:.2f}, z={tz_map:.2f}"
                        )

                # Do NOT stop immediately. Target tracking mode handles the next behavior.

            # ============================================================
            # Target tracking / recording hover (Agent2~4)
            # ============================================================
            if self.hub.should_hold_after_target(self.name):
                if not self._non_tracker_hold_engaged:
                    self._non_tracker_hold_engaged = True
                    self.current_target_cell = None
                    self.current_target_xy = None
                    self.hub.clear_reservation(self.name)
                    if getattr(CFG, "IMPORTANT_STATUS_LOG_ENABLED", True):
                        print(
                            f"[RECORDING HOVER] {self.name} holding position "
                            f"(tracker={getattr(CFG, 'TRACKER_AGENT', 'Agent1')})"
                        )
                self.hub.update_status(self.name, "RECORDING_HOVER")
                self.prev_vx = 0.0
                self.prev_vy = 0.0
                hold_z = float(getattr(CFG, "NON_TRACKER_HOLD_Z", CFG.AGENT_Z))
                now_hover = time.time()
                interval = float(getattr(CFG, "NON_TRACKER_HOVER_CMD_INTERVAL_SEC", 2.0))
                if now_hover - self._last_non_tracker_hover_time >= interval:
                    self._last_non_tracker_hover_time = now_hover
                    if getattr(CFG, "NON_TRACKER_USE_SAFE_HOVER", True):
                        safe_hover(self.client, self.name)
                    else:
                        self.send_velocity(0.0, 0.0, hold_z, yaw_deg)
                else:
                    self.send_velocity(0.0, 0.0, hold_z, yaw_deg)
                time.sleep(CFG.EXECUTOR_DT)
                continue

            self._non_tracker_hold_engaged = False

            # After redball_01 is detected: Agent1 tracks; other agents continue search_loop below.
            if self.hub.has_target() and getattr(CFG, "ALL_AGENTS_TRACK_TARGET", False):
                self.track_target_step(x=x, y=y, z=z, yaw_deg=yaw_deg, snapshot=snapshot, dt=dt)
                time.sleep(CFG.EXECUTOR_DT)
                continue

            is_primary_tracker = (
                hasattr(self.hub, "is_primary_tracker")
                and self.hub.is_primary_tracker(self.name)
            )
            if self.hub.has_target() and is_primary_tracker:
                self.track_target_step(x=x, y=y, z=z, yaw_deg=yaw_deg, snapshot=snapshot, dt=dt)
                time.sleep(CFG.EXECUTOR_DT)
                continue

            if snapshot["obstacle"]:
                front_clear = float(snapshot.get("front_clear", snapshot.get("obstacle_dist", CFG.OBSTACLE_DIST)))
                measured_obstacle_dist = min(
                    front_clear,
                    float(snapshot.get("left_clear", CFG.OBSTACLE_DIST)),
                    float(snapshot.get("right_clear", CFG.OBSTACLE_DIST)),
                    float(snapshot.get("obstacle_dist", CFG.OBSTACLE_DIST)),
                )
                max_map_mark = float(getattr(CFG, "OBSTACLE_MAP_MARK_MAX_DIST", 7.0))
                if measured_obstacle_dist <= max_map_mark:
                    self.planner.mark_front_obstacle(x, y, yaw_deg, obstacle_dist=measured_obstacle_dist)
                self.current_target_cell = None
                self.current_target_xy = None

            need_new_target = False
            if self.current_target_xy is None:
                need_new_target = True
            elif distance_2d((x, y), self.current_target_xy) < CFG.CELL_REACH_DIST:
                self.hub.clear_reservation(self.name)
                self.current_target_cell = None
                self.current_target_xy = None
                need_new_target = True
            else:
                # 다른 에이전트가 이미 현재 target 주변을 방문했으면 같은 곳으로 계속 가지 않는다.
                # 완전 금지는 아니고, target 주변이 충분히 visited가 되었을 때만 새 목표를 요청한다.
                if hasattr(self.belief_map, "episode_visited_ratio_radius"):
                    tv = self.belief_map.episode_visited_ratio_radius(
                        self.current_target_xy[0],
                        self.current_target_xy[1],
                        radius=getattr(CFG, "EPISODE_VISITED_AVOID_RADIUS", CFG.PLANNER_CELL_SIZE),
                    )
                    if tv >= getattr(CFG, "CURRENT_TARGET_VISITED_ABORT_RATIO", 0.65):
                        self.hub.clear_reservation(self.name)
                        self.current_target_cell = None
                        self.current_target_xy = None
                        need_new_target = True
                        self.hub.update_status(self.name, f"RETARGET_VISITED ratio={tv:.2f}")

            if need_new_target:
                next_cell, next_xy, reason = self.planner.choose_next_cell(x=x, y=y, z=z, yaw_deg=yaw_deg, snapshot=snapshot, hub=self.hub)
                from_cell = self.planner.cell_of(x, y)
                decision = self.hub.request_move(self.name, from_cell, next_cell, next_xy, (x, y, z))
                if decision["action"] == "hold":
                    self.hub.update_status(self.name, f"YIELD conflict={decision['reason']}")
                    self.smooth_yield_once(self.altitude_target_z, yaw_deg, decision.get("hold_time", 0.0))
                    continue
                if decision.get("action") == "altitude_shift":
                    self.set_temporary_altitude(
                        decision.get("target_z", CFG.ALTITUDE_SHIFT_Z),
                        getattr(CFG, "ALTITUDE_SHIFT_HOLD_SEC", 3.0),
                        f"conflict_{decision.get('reason', 'proximity')}",
                    )
                    target_z = self.altitude_target_z
                self.current_target_cell = next_cell
                self.current_target_xy = next_xy
                self.hub.update_status(self.name, f"PLAN_6GRID {reason}")

            tx, ty = self.current_target_xy
            dx, dy = tx - x, ty - y
            ux, uy = normalize2(dx, dy)
            dist_to_target = math.hypot(dx, dy)
            speed = CFG.FAST_AGENT_SPEED
            if dist_to_target < CFG.PLANNER_CELL_SIZE * 0.7:
                speed = max(CFG.FAST_MIN_SPEED, CFG.FAST_AGENT_SPEED * (dist_to_target / (CFG.PLANNER_CELL_SIZE * 0.7)))
            des_vx, des_vy = ux * speed, uy * speed

            if snapshot["obstacle"]:
                front_clear = float(snapshot.get("front_clear", 100.0))
                if self.is_close_obstacle(snapshot):
                    rl_action = self.make_hard_escape_action(snapshot, name="hard_corridor_escape")
                    self.hub.rl_commit_action(self.name, snapshot, rl_action)
                    self.soften_previous_velocity()
                    des_vx, des_vy = self.obstacle_reactive_velocity(
                        des_vx, des_vy, snapshot["left_clear"], snapshot["right_clear"], rl_action, bold=False
                    )
                else:
                    rl_action = self.hub.select_rl_avoidance_action(self.name, snapshot)
                    self.hub.rl_commit_action(self.name, snapshot, rl_action)
                    bold = getattr(CFG, "SEARCH_BOLD_AVOIDANCE", True)
                    des_vx, des_vy = self.obstacle_reactive_velocity(
                        des_vx, des_vy, snapshot["left_clear"], snapshot["right_clear"], rl_action, bold=bold
                    )
                if self.is_close_obstacle(snapshot):
                    self.maybe_trigger_vertical_obstacle_avoid(z, snapshot, rl_action)
                self.hub.update_status(self.name, f"RL_AVOID action={rl_action['name']} front={front_clear:.1f} z={target_z:.1f}")

            if getattr(CFG, "DRONE_REPULSION_ENABLED", False):
                rep_x, rep_y, emergency = self.hub.get_repulsion_velocity(self.name, x, y, z)
                if emergency:
                    self.hub.update_status(self.name, "EMERGENCY_REPULSION")
                des_vx += rep_x
                des_vy += rep_y
            geo_x, geo_y = geofence_push_velocity(x, y)
            des_vx += geo_x
            des_vy += geo_y
            des_vx, des_vy = limit_norm(des_vx, des_vy, CFG.FAST_AGENT_SPEED)

            next_x, next_y = x + des_vx * CFG.EXECUTOR_DT, y + des_vy * CFG.EXECUTOR_DT
            if not inside_geofence(next_x, next_y):
                self.current_target_cell = None
                self.current_target_xy = None
                center_x = (CFG.MAP_X_MIN + CFG.MAP_X_MAX) / 2.0
                center_y = (CFG.MAP_Y_MIN + CFG.MAP_Y_MAX) / 2.0
                cx_dir, cy_dir = normalize2(center_x - x, center_y - y)
                des_vx, des_vy = cx_dir * CFG.FAST_MIN_SPEED, cy_dir * CFG.FAST_MIN_SPEED
                self.hub.update_status(self.name, "GEOFENCE_CORRECTION")

            cmd_vx, cmd_vy = self.finalize_cmd_velocity(des_vx, des_vy, dt)

            now_debug = time.time()
            if CFG.MOVE_DEBUG_ENABLED and now_debug - self.last_move_debug_time >= CFG.MOVE_DEBUG_INTERVAL:
                des_speed = math.hypot(des_vx, des_vy)
                cmd_speed = math.hypot(cmd_vx, cmd_vy)
                print(
                    f"[MOVE DEBUG] {self.name} "
                    f"pos=({x:.1f},{y:.1f},{z:.1f}) "
                    f"target=({tx:.1f},{ty:.1f}) "
                    f"dist={dist_to_target:.1f} "
                    f"des=({des_vx:.2f},{des_vy:.2f}|{des_speed:.2f}) "
                    f"cmd=({cmd_vx:.2f},{cmd_vy:.2f}|{cmd_speed:.2f}) "
                    f"yaw={yaw_deg:.1f} target_z={target_z:.1f} obstacle={snapshot.get('obstacle')} "
                    f"wall={snapshot.get('wall_detected')} wall_dist={snapshot.get('wall_distance'):.1f} "
                    f"L={snapshot.get('left_clear'):.1f} R={snapshot.get('right_clear'):.1f} "
                    f"obs_dist={snapshot.get('obstacle_dist'):.1f}"
                )
                self.last_move_debug_time = now_debug

            self.send_velocity(cmd_vx, cmd_vy, target_z, yaw_deg)

            proximity = self.hub.nearest_agent_penalty(self.name, x, y, z)
            cov_stats = self.belief_map.coverage_stats(ignore_margin=CFG.COVERAGE_IGNORE_MARGIN)
            gv = float(cov_stats.get("global_visited", cov_stats.get("visited", 0.0)))
            coverage_delta = self.hub.record_visited_delta_for_step_reward(gv)
            self.hub.reward_step(self.name, coverage_delta, snapshot, proximity_penalty=proximity)

            elapsed = time.perf_counter() - loop_start
            time.sleep(max(0.0, CFG.EXECUTOR_DT - elapsed))

        self.smooth_stop()
        self.hub.update_status(self.name, "SEARCH_STOPPED")

    def track_target_step(self, x: float, y: float, z: float, yaw_deg: float, snapshot: dict, dt: float):
        """
        Primary tracker (Agent1) flies straight to redball_01 center.
        Other agents should not call this when ALL_AGENTS_TRACK_TARGET is False.
        """
        if getattr(CFG, "RL_COLLISION_LEARNING_ENABLED", True):
            self.hub.rl_prepare_step(self.name, snapshot)
        self.hub.note_agent_target_distance(self.name, x, y)

        is_primary = (
            hasattr(self.hub, "is_primary_tracker")
            and self.hub.is_primary_tracker(self.name)
        )
        if is_primary or self.name == getattr(CFG, "TRACKER_AGENT", "Agent1"):
            target = self.hub.get_target_xy() if hasattr(self.hub, "get_target_xy") else None
        elif hasattr(self.hub, "get_target_xy_for_agent"):
            target = self.hub.get_target_xy_for_agent(self.name, current_xy=(x, y))
        else:
            target = self.hub.get_target_xy()
        if target is None:
            return False

        tx, ty = target
        dx, dy = tx - x, ty - y
        dist = math.hypot(dx, dy)

        base_target = self.hub.get_target_xy() if hasattr(self.hub, "get_target_xy") else target
        if base_target is None:
            base_target = target
        base_dist = math.hypot(base_target[0] - x, base_target[1] - y)
        arrival_radius = getattr(CFG, "TRACKER_APPROACH_RADIUS", 2.5)

        # Completion rule: if any drone enters the target cell/radius around the redball,
        # the mission is complete. TARGET marking is written again so red remains above VISITED.
        if base_dist <= arrival_radius:
            target_xyz = self.hub.get_target_xyz() if hasattr(self.hub, "get_target_xyz") else None
            if target_xyz is not None:
                tx_map, ty_map, _ = target_xyz
            else:
                tx_map, ty_map = base_target
            self.belief_map.mark_radius(
                tx_map,
                ty_map,
                GridValue.TARGET,
                radius=getattr(CFG, "TARGET_MAP_MARK_RADIUS", 3.0),
            )
            self.hub.mark_tracker_arrived(self.name, x, y, z)
            self.prev_vx = 0.0
            self.prev_vy = 0.0
            if getattr(CFG, "TRACKER_HOVER_ON_ARRIVAL", True):
                self.send_velocity(0.0, 0.0, getattr(CFG, "TRACKER_TARGET_Z", CFG.AGENT_Z), yaw_deg)
            return True

        ux, uy = normalize2(dx, dy)
        speed = getattr(CFG, "TRACKER_SPEED", CFG.FAST_AGENT_SPEED)
        if dist < 8.0:
            speed = max(getattr(CFG, "TRACKER_MIN_SPEED", 0.8), speed * dist / 8.0)

        des_vx, des_vy = ux * speed, uy * speed
        target_z = float(getattr(CFG, "TRACKER_TARGET_Z", CFG.AGENT_Z))

        straight_track = (
            getattr(CFG, "TRACK_STRAIGHT_TO_TARGET", False)
            and (is_primary or self.name == getattr(CFG, "TRACKER_AGENT", "Agent1"))
        )
        recording_primary = bool(getattr(CFG, "TARGET_RECORDING_MODE", False)) and is_primary
        straight_skips_avoid = straight_track and (
            getattr(CFG, "TRACK_STRAIGHT_IGNORE_ALL_AVOID", False) or recording_primary
        )

        ignore_target_obstacle = False
        if snapshot.get("obstacle", False) or recording_primary:
            target_visible = bool(snapshot.get("target_found", False))
            front_clear = float(snapshot.get("front_clear", 100.0))
            depth_tol = float(getattr(CFG, "TARGET_OBSTACLE_DEPTH_TOL", 2.5))
            target_depth_like = target_visible and front_clear >= max(0.0, dist - depth_tol)
            very_close_to_target = dist <= getattr(CFG, "TRACKER_APPROACH_RADIUS", 2.5) + 1.0
            ignore_dist = float(getattr(CFG, "TARGET_OBSTACLE_IGNORE_DIST", 18.0))
            ignore_target_obstacle = (
                getattr(CFG, "IGNORE_TARGET_AS_OBSTACLE", True)
                and (
                    recording_primary
                    or dist <= ignore_dist
                    or target_depth_like
                    or very_close_to_target
                )
            )

        emergency_only = recording_primary and getattr(CFG, "TRACK_RECORDING_EMERGENCY_ONLY", True)
        emergency_min_dist = float(getattr(CFG, "TRACK_RECORDING_EMERGENCY_MIN_DIST_M", 5.0))

        run_track_avoid = (
            snapshot.get("obstacle", False)
            and not ignore_target_obstacle
            and (
                (not straight_skips_avoid)
                or (
                    emergency_only
                    and self.is_close_obstacle(snapshot)
                    and dist > emergency_min_dist
                )
            )
        )

        if run_track_avoid:
            if self.is_close_obstacle(snapshot):
                rl_action = self.make_hard_escape_action(snapshot, name="track_hard_corridor_escape")
                self.hub.rl_commit_action(self.name, snapshot, rl_action)
                self.soften_previous_velocity()
                des_vx, des_vy = self.obstacle_reactive_velocity(
                    des_vx,
                    des_vy,
                    snapshot.get("left_clear", 100.0),
                    snapshot.get("right_clear", 100.0),
                    rl_action,
                    bold=False,
                )
            else:
                rl_action = self.hub.select_rl_avoidance_action(self.name, snapshot)
                self.hub.rl_commit_action(self.name, snapshot, rl_action)
                blend = float(getattr(CFG, "TRACK_BOLD_PATH_BLEND", 0.45)) if straight_track else 0.0
                if emergency_only:
                    blend = max(blend, 0.88)
                bold = straight_track or getattr(CFG, "SEARCH_BOLD_AVOIDANCE", False)
                des_vx, des_vy = self.obstacle_reactive_velocity(
                    des_vx,
                    des_vy,
                    snapshot.get("left_clear", 100.0),
                    snapshot.get("right_clear", 100.0),
                    rl_action,
                    bold=bold,
                )
                if straight_track and blend > 0.0:
                    des_vx = blend * ux * speed + (1.0 - blend) * des_vx
                    des_vy = blend * uy * speed + (1.0 - blend) * des_vy
                    des_vx, des_vy = limit_norm(des_vx, des_vy, getattr(CFG, "TRACKER_SPEED", CFG.FAST_AGENT_SPEED))
            if self.is_close_obstacle(snapshot):
                self.maybe_trigger_vertical_obstacle_avoid(z, snapshot, rl_action)
            tag = "TRACK_STRAIGHT_AVOID" if straight_track else "TRACK_AVOID"
            self.hub.update_status(self.name, f"{tag} action={rl_action['name']} dist={dist:.1f}")
        elif straight_skips_avoid or recording_primary:
            tag = "TRACK_RECORDING_STRAIGHT" if recording_primary else "TRACK_STRAIGHT"
            self.hub.update_status(self.name, f"{tag} dist={dist:.1f}")
        elif snapshot.get("obstacle", False) and ignore_target_obstacle:
            self.hub.update_status(self.name, f"TRACK_TARGET_IGNORE_BALL_DEPTH dist={dist:.1f}")
        else:
            self.hub.update_status(self.name, f"TRACK_TARGET dist={dist:.1f}")

        # Drone-drone artificial repulsion is disabled by default.
        # Collision prevention is handled by priority/yield logic, not by pushing
        # the tracker away from the target path.
        if getattr(CFG, "DRONE_REPULSION_ENABLED", False):
            rep_x, rep_y, emergency = self.hub.get_repulsion_velocity(self.name, x, y, z)
            if emergency:
                self.hub.update_status(self.name, "TRACK_EMERGENCY_REPULSION")
            des_vx += rep_x
            des_vy += rep_y

        geo_x, geo_y = geofence_push_velocity(x, y)
        des_vx += geo_x
        des_vy += geo_y
        des_vx, des_vy = limit_norm(des_vx, des_vy, getattr(CFG, "TRACKER_SPEED", CFG.FAST_AGENT_SPEED))

        cmd_vx, cmd_vy = self.finalize_cmd_velocity(des_vx, des_vy, dt)

        cmd_speed = math.hypot(cmd_vx, cmd_vy)
        min_cmd_speed = getattr(CFG, "MIN_START_CMD_SPEED", 0.35)
        if cmd_speed < min_cmd_speed and dist > getattr(CFG, "TRACKER_APPROACH_RADIUS", 2.5):
            cmd_vx = ux * min_cmd_speed
            cmd_vy = uy * min_cmd_speed
            self.prev_vx, self.prev_vy = cmd_vx, cmd_vy

        self.send_velocity(cmd_vx, cmd_vy, target_z, yaw_deg)
        if getattr(CFG, "RL_COLLISION_LEARNING_ENABLED", True):
            proximity = self.hub.nearest_agent_penalty(self.name, x, y, z)
            cov_stats = self.belief_map.coverage_stats(ignore_margin=CFG.COVERAGE_IGNORE_MARGIN)
            gv = float(cov_stats.get("global_visited", cov_stats.get("visited", 0.0)))
            coverage_delta = self.hub.record_visited_delta_for_step_reward(gv)
            self.hub.reward_step(self.name, coverage_delta, snapshot, proximity_penalty=proximity)
        return False

    def obstacle_reactive_velocity(self, path_vx, path_vy, left_clear, right_clear, rl_action=None, bold: bool = False):
        ux, uy = normalize2(path_vx, path_vy)
        if abs(ux) + abs(uy) < 1e-6:
            ux, uy = 1.0, 0.0
        left_x, left_y = -uy, ux
        right_x, right_y = uy, -ux
        if rl_action is None:
            rl_action = {"name": "safe_side_hard", "forward": -0.40, "side": 1.20, "speed_scale": 1.00, "force_side": "safe"}
        force_side = rl_action.get("force_side", "safe")
        if force_side == "left":
            sx, sy = left_x, left_y
        elif force_side == "right":
            sx, sy = right_x, right_y
        elif force_side == "hold":
            sx, sy = -ux, -uy
        else:
            sx, sy = (right_x, right_y) if right_clear > left_clear else (left_x, left_y)
        avoid_speed = CFG.AVOID_SPEED * float(rl_action.get("speed_scale", 1.0))
        forward_coeff = float(rl_action.get("forward", -0.35))
        side_coeff = float(rl_action.get("side", 1.10))
        if bold:
            forward_coeff = max(forward_coeff, float(getattr(CFG, "SEARCH_AVOID_FORWARD_COEFF", -0.12)))
            side_coeff = min(side_coeff, float(getattr(CFG, "SEARCH_AVOID_SIDE_COEFF", 0.95)))
            # Blend more of the planned path so the drone threads aisles instead of orbiting open space.
            path_scale = float(getattr(CFG, "SEARCH_BOLD_PATH_BLEND", 0.55))
            avx = path_scale * path_vx + (1.0 - path_scale) * (
                forward_coeff * ux * avoid_speed + side_coeff * sx * avoid_speed
            )
            avy = path_scale * path_vy + (1.0 - path_scale) * (
                forward_coeff * uy * avoid_speed + side_coeff * sy * avoid_speed
            )
            return limit_norm(avx, avy, CFG.FAST_AGENT_SPEED)
        avx = forward_coeff * ux * avoid_speed + side_coeff * sx * avoid_speed
        avy = forward_coeff * uy * avoid_speed + side_coeff * sy * avoid_speed
        return limit_norm(avx, avy, CFG.FAST_AGENT_SPEED)

    def send_velocity(self, vx: float, vy: float, target_z: float, yaw_deg: float):
        move_velocity_continuous(self.client, self.name, vx, vy, target_z, CFG.VELOCITY_CMD_DURATION, yaw_deg)

    def smooth_stop(self):
        for _ in range(6):
            try:
                _, _, _, yaw_deg = get_pose(self.client, self.name)
                self.prev_vx *= 0.65
                self.prev_vy *= 0.65
                move_velocity_continuous(self.client, self.name, self.prev_vx, self.prev_vy, CFG.AGENT_Z, CFG.VELOCITY_CMD_DURATION, yaw_deg)
            except Exception:
                pass
            time.sleep(CFG.EXECUTOR_DT)
        safe_hover(self.client, self.name)

    def recover_if_collided(self) -> bool:
        try:
            info = rpc_call(self.client.simGetCollisionInfo, vehicle_name=self.name)
        except Exception:
            return False
        if not info.has_collided:
            return False
        current_collision_time = info.time_stamp
        if self.last_collision_time == current_collision_time:
            return False
        self.last_collision_time = current_collision_time
        object_name = getattr(info, "object_name", "unknown")
        self.hub.update_status(self.name, f"COLLISION_DETECTED: {object_name}")

        if hasattr(self.hub, "is_wall_object") and self.hub.is_wall_object(object_name):
            try:
                xw, yw, _, yaw_w = get_pose(self.client, self.name)
                self.hub.update_wall_boundary_from_pose(
                    source=f"{self.name}:collision:{object_name}",
                    x=xw,
                    y=yw,
                    yaw_deg=yaw_w,
                    wall_distance=0.0,
                )
            except Exception:
                pass

        snap = self.last_snapshot if self.last_snapshot else {
            "obstacle": True,
            "left_clear": 0.5,
            "right_clear": 0.5,
            "front_clear": 0.5,
        }
        self.hub.reward_collision(
            self.name,
            object_name,
            drone_collision=("Agent" in object_name or "Drone" in object_name),
            snapshot=snap,
        )
        try:
            x, y, _, _ = get_pose(self.client, self.name)
            self.belief_map.mark_radius(x, y, GridValue.OBSTACLE, radius=CFG.OBSTACLE_MARK_RADIUS + 1.5)
            self.planner.mark_blocked_cell(self.planner.cell_of(x, y))
        except Exception:
            pass
        if CFG.COLLISION_RESTART_ENABLED:
            self.hub.request_stop(f"EPISODE_COLLISION agent={self.name} object={object_name}")
            safe_hover(self.client, self.name)
            return True
        return False
