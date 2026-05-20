import csv
import os
import threading
import time

from airsim_rpc import reset_shared_client, rpc_call
from config import CFG
from belief_map import BeliefMap
from command_hub import CommandHub
from agent_worker import AgentWorker
from training_log_utils import (
    EPISODE_LOG_FIELDS,
    read_max_global_episode,
    count_episode_rows,
)
from safe_io import file_lock
from utils import (
    connect_client,
    get_position,
    get_pose,
    create_square_area,
    split_area_into_strips,
    clamp_area_to_geofence,
    safe_hover,
    wait_for_z,
)


class MissionController:
    """Commandless autonomous multi-agent mission controller.

    Physical CommandDrone is removed. CommandHub remains as a software-only
    coordinator for shared target state, reservations, map/window management,
    rewards, and collision learning.
    """

    def __init__(self):
        self.client = connect_client()
        self.hub = CommandHub()
        self.belief_map = None
        self.plans = None
        self.current_local_area = None
        self.current_window_center = None
        self.visited_window_centers = []
        self.local_window_index = 0
        self.last_collision_stamp = {}
        self.collision_monitor_stop = threading.Event()
        self.collision_monitor_thread = None
        self._global_episode_counter = read_max_global_episode()
        self._session_episode_counter = 0
        if self._global_episode_counter > 0:
            print(
                f"[EPISODE LOG] Resuming global episode index after "
                f"{self._global_episode_counter} (CSV rows={count_episode_rows()})"
            )

    def allocate_global_episode(self) -> int:
        self._session_episode_counter += 1
        if getattr(CFG, "EPISODE_LOG_APPEND_GLOBAL", True):
            self._global_episode_counter += 1
            return self._global_episode_counter
        return self._session_episode_counter

    def active_drones(self):
        """Only physical agent drones are controlled in commandless mode."""
        return list(CFG.AGENTS)

    def reconnect(self):
        self.client = connect_client(confirm=False)

    def safe_rpc(self, desc: str, fn) -> bool:
        for attempt in range(CFG.RPC_RETRY_COUNT):
            try:
                fn()
                return True
            except BufferError as e:
                print(f"[RPC WARN] BufferError during {desc}, retry={attempt + 1}: {e}")
                time.sleep(CFG.RPC_RETRY_SLEEP)
                self.reconnect()
            except Exception as e:
                print(f"[RPC WARN] failed during {desc}, retry={attempt + 1}: {e}")
                time.sleep(CFG.RPC_RETRY_SLEEP)
                self.reconnect()
        return False

    # ============================================================
    # Setup
    # ============================================================

    def setup_drones(self):
        if getattr(CFG, "INIT_LOG_ENABLED", False):
            print("[INIT] Enable API control and arm agent drones")
        drones = list(self.active_drones())
        fpv = getattr(CFG, "FPV_VEHICLE", "Agent1")
        if fpv in drones:
            drones.remove(fpv)
            drones.insert(0, fpv)
        for drone in drones:
            self.safe_rpc(
                f"enableApiControl {drone}",
                lambda d=drone: self.client.enableApiControl(True, vehicle_name=d),
            )
            self.safe_rpc(
                f"armDisarm {drone}",
                lambda d=drone: self.client.armDisarm(True, vehicle_name=d),
            )
            time.sleep(0.1)

    def setup_target_detection(self):
        if not CFG.TARGET_DETECTION_ENABLED:
            self.hub.target_detection_active = False
            print("[TARGET] Target detection disabled by config.")
            return
        try:
            objects = self.client.simListSceneObjects(CFG.TARGET_OBJECT_REGEX)
        except Exception as e:
            self.hub.target_detection_active = False
            print(f"[TARGET] Could not list scene objects. Target detection disabled. reason={e}")
            return
        if CFG.REQUIRE_TARGET_OBJECT_IN_MAP and not objects:
            self.hub.target_detection_active = False
            print("[TARGET] redball_01 object not found in map. Target detection disabled.")
            return

        try:
            self.client.simSetSegmentationObjectID(".*", 0, True)
            ok = self.client.simSetSegmentationObjectID(
                CFG.TARGET_OBJECT_REGEX, CFG.TARGET_SEGMENTATION_ID, True
            )
            if not ok:
                self.hub.target_detection_active = False
                print("[TARGET] Failed to assign segmentation ID to redball_01.")
                return
        except Exception as e:
            self.hub.target_detection_active = False
            print(f"[TARGET] Failed to setup segmentation. reason={e}")
            return
        self.hub.target_detection_active = True
        if getattr(CFG, "INIT_LOG_ENABLED", False):
            print("[TARGET] Target detection active.")

    def initialize_global_belief_map(self):
        if self.belief_map is None:
            self.belief_map = BeliefMap(
                CFG.MAP_X_MIN,
                CFG.MAP_X_MAX,
                CFG.MAP_Y_MIN,
                CFG.MAP_Y_MAX,
                CFG.GRID_RESOLUTION,
            )
            print(
                f"[MAP] belief grid {self.belief_map.width}x{self.belief_map.height} "
                f"x=[{CFG.MAP_X_MIN:.0f},{CFG.MAP_X_MAX:.0f}] "
                f"y=[{CFG.MAP_Y_MIN:.0f},{CFG.MAP_Y_MAX:.0f}] res={CFG.GRID_RESOLUTION}m"
            )
            if getattr(CFG, "WINDOW_DEBUG_LOG_ENABLED", False):
                print("\n[GLOBAL BELIEF MAP]")
                print(f"x: {CFG.MAP_X_MIN:.1f} ~ {CFG.MAP_X_MAX:.1f}")
                print(f"y: {CFG.MAP_Y_MIN:.1f} ~ {CFG.MAP_Y_MAX:.1f}")

    def takeoff_agents(self):
        if getattr(CFG, "INIT_LOG_ENABLED", False):
            print("\n[TAKEOFF] Agent drones")
        for agent in CFG.AGENTS:
            if self.hub.stop_event.is_set():
                return
            self.safe_rpc(f"takeoff {agent}", lambda a=agent: self.client.takeoffAsync(vehicle_name=a))
            time.sleep(0.7)
            self.safe_rpc(
                f"moveToZ {agent}",
                lambda a=agent: self.client.moveToZAsync(
                    CFG.AGENT_Z, CFG.FAST_AGENT_SPEED, vehicle_name=a
                ),
            )
            ok = wait_for_z(self.client, agent, CFG.AGENT_Z, CFG.AGENT_TAKEOFF_TIMEOUT_SEC)
            safe_hover(self.client, agent)
            if not ok:
                self.hub.request_stop(f"EPISODE_TAKEOFF_TIMEOUT agent={agent}")
                return
            x, y, z = get_position(self.client, agent)
            self.hub.update_position(agent, x, y, z)
            self.hub.update_status(agent, "LOW_ALTITUDE_READY")
        self.prime_collision_stamps()

    def autonomous_start_center(self):
        mode = getattr(CFG, "AUTONOMOUS_START_CENTER_MODE", "agent_mean")
        if mode == "fixed":
            return float(CFG.AUTONOMOUS_START_CENTER_X), float(CFG.AUTONOMOUS_START_CENTER_Y)

        xs, ys = [], []
        for agent in CFG.AGENTS:
            try:
                x, y, z = get_position(self.client, agent)
                xs.append(x)
                ys.append(y)
                self.hub.update_position(agent, x, y, z)
            except Exception:
                pass
        if xs and ys:
            return sum(xs) / len(xs), sum(ys) / len(ys)
        return float(CFG.AUTONOMOUS_START_CENTER_X), float(CFG.AUTONOMOUS_START_CENTER_Y)

    # ============================================================
    # Map/window planning
    # ============================================================

    def make_local_area(self, cx: float, cy: float):
        area = create_square_area(cx, cy, CFG.LOCAL_AREA_SIZE)
        area = clamp_area_to_geofence(area)
        if hasattr(self.hub, "clamp_area_to_dynamic_bounds"):
            area = self.hub.clamp_area_to_dynamic_bounds(area)
        return area

    def set_local_window(self, center_x: float, center_y: float, window_index: int):
        area = self.make_local_area(center_x, center_y)
        self.current_local_area = area
        self.current_window_center = (
            (area["x_min"] + area["x_max"]) / 2.0,
            (area["y_min"] + area["y_max"]) / 2.0,
        )
        self.local_window_index = int(window_index)
        self.hub.set_local_area(area, window_index=window_index)
        self.visited_window_centers.append(self.current_window_center)

        strips = split_area_into_strips(area, len(CFG.AGENTS))
        plans = {}
        for agent, strip in zip(CFG.AGENTS, strips):
            plans[agent] = {"strip": strip, "start": (0.0, 0.0, CFG.AGENT_Z)}
        self.plans = plans

        if getattr(CFG, "WINDOW_DEBUG_LOG_ENABLED", False):
            print("\n[LOCAL SEARCH WINDOW]")
            print(f"window={window_index}")
            print(f"x: {area['x_min']:.2f} ~ {area['x_max']:.2f}")
            print(f"y: {area['y_min']:.2f} ~ {area['y_max']:.2f}")
        return plans

    def create_search_plan(self, cx: float, cy: float):
        self.initialize_global_belief_map()
        return self.set_local_window(cx, cy, window_index=1)

    def effective_global_stats(self):
        if self.belief_map is None:
            return {"observed": 0.0, "visited": 0.0, "obstacle": 0.0, "unknown": 1.0}
        area = self.hub.get_discovered_area() if hasattr(self.hub, "get_discovered_area") else None
        if area is not None:
            return self.belief_map.coverage_stats_area(area, ignore_margin=CFG.COVERAGE_IGNORE_MARGIN)
        return self.belief_map.coverage_stats(ignore_margin=CFG.COVERAGE_IGNORE_MARGIN)

    def is_global_complete(self):
        stats = self.effective_global_stats()
        coverage_ok = (
            stats["observed"] >= CFG.TARGET_MAP_COVERAGE
            and stats["visited"] >= CFG.TARGET_VISITED_COVERAGE
        )
        boundary_required = getattr(CFG, "REQUIRE_ALL_BOUNDARIES_FOR_GLOBAL_COMPLETE", True)
        boundary_ok = True
        if boundary_required and hasattr(self.hub, "all_boundaries_discovered"):
            boundary_ok = self.hub.all_boundaries_discovered()
        return stats, coverage_ok and boundary_ok

    # ============================================================
    # Collision monitor
    # ============================================================

    def should_ignore_collision(self, agent: str, info) -> bool:
        object_name = getattr(info, "object_name", "") or ""
        try:
            _, _, z = get_position(self.client, agent)
        except Exception:
            z = 0.0
        if z > CFG.COLLISION_ENABLE_Z:
            for key in CFG.COLLISION_IGNORE_NEAR_GROUND_OBJECTS:
                if key in object_name:
                    return True
        return False

    def prime_collision_stamps(self):
        self.last_collision_stamp = {}
        for agent in CFG.AGENTS:
            try:
                info = self.client.simGetCollisionInfo(vehicle_name=agent)
                if info.has_collided:
                    self.last_collision_stamp[agent] = info.time_stamp
            except Exception:
                pass

    def start_collision_monitor(self):
        self.collision_monitor_stop = threading.Event()
        self.prime_collision_stamps()

        def loop():
            c = connect_client(confirm=False)
            while not self.collision_monitor_stop.is_set() and not self.hub.stop_event.is_set():
                for agent in CFG.AGENTS:
                    try:
                        info = rpc_call(c.simGetCollisionInfo, vehicle_name=agent)
                    except Exception:
                        continue
                    if not info.has_collided:
                        continue
                    if self.should_ignore_collision(agent, info):
                        self.last_collision_stamp[agent] = info.time_stamp
                        continue
                    if self.last_collision_stamp.get(agent) == info.time_stamp:
                        continue

                    self.last_collision_stamp[agent] = info.time_stamp
                    object_name = getattr(info, "object_name", "unknown")
                    print("\n" + "=" * 70)
                    print("[GLOBAL COLLISION DETECTED]")
                    print("stage  : GLOBAL_MONITOR")
                    print(f"agent  : {agent}")
                    print(f"object : {object_name}")
                    print("=" * 70 + "\n")

                    if hasattr(self.hub, "is_wall_object") and self.hub.is_wall_object(object_name):
                        try:
                            xw, yw, _, yaw_w = get_pose(c, agent)
                            self.hub.update_wall_boundary_from_pose(
                                source=f"{agent}:collision:{object_name}",
                                x=xw,
                                y=yw,
                                yaw_deg=yaw_w,
                                wall_distance=0.0,
                            )
                        except Exception:
                            pass
                    self.hub.reward_collision(
                        agent,
                        object_name,
                        drone_collision=("Agent" in object_name or "Drone" in object_name),
                    )
                    self.hub.request_stop(
                        f"EPISODE_COLLISION stage=GLOBAL_MONITOR agent={agent} object={object_name}"
                    )
                    break
                time.sleep(CFG.COLLISION_MONITOR_DT)

        self.collision_monitor_thread = threading.Thread(target=loop, daemon=True)
        self.collision_monitor_thread.start()

    def stop_collision_monitor(self):
        self.collision_monitor_stop.set()
        if self.collision_monitor_thread is not None:
            self.collision_monitor_thread.join(timeout=1.0)
        self.collision_monitor_thread = None

    # ============================================================
    # Mission execution
    # ============================================================

    def run_search(self, plans, episode_start_time: float):
        if getattr(CFG, "INIT_LOG_ENABLED", False):
            print("\n[MISSION] Start autonomous frontier episode")
        self.hub.assign_global_goals(self.belief_map, force=True)
        workers = [
            AgentWorker(name=agent, strip=plans[agent]["strip"], hub=self.hub, belief_map=self.belief_map)
            for agent in CFG.AGENTS
        ]
        for w in workers:
            w.start()

        last_coverage_check = 0.0
        last_coverage_print = 0.0
        last_summary_print = 0.0
        prev_global_coverage = self.belief_map.coverage_stats(
            ignore_margin=CFG.COVERAGE_IGNORE_MARGIN
        )["observed"]
        prev_local_visited = 0.0
        local_stall_count = 0
        local_window_count = 1

        try:
            while any(w.is_alive() for w in workers):
                now = time.time()
                if now - episode_start_time >= CFG.MAX_EPISODE_TIME_SEC:
                    self.hub.request_stop("EPISODE_TIMEOUT")
                    break

                if now - self.hub.last_goal_assign_time >= CFG.GLOBAL_GOAL_ASSIGN_DT:
                    self.hub.assign_global_goals(self.belief_map, force=True)

                if now - last_coverage_check >= CFG.COVERAGE_CHECK_DT:
                    global_stats = self.effective_global_stats()
                    local_stats = self.belief_map.coverage_stats_area(
                        self.current_local_area, ignore_margin=CFG.COVERAGE_IGNORE_MARGIN
                    )
                    global_coverage = global_stats["observed"]
                    delta = max(0.0, global_coverage - prev_global_coverage)
                    prev_global_coverage = global_coverage
                    self.hub.update_coverage(global_coverage, stats=global_stats)
                    if delta > 0:
                        per_agent_reward = delta * CFG.MAP_COVERAGE_REWARD_SCALE / max(len(CFG.AGENTS), 1)
                        for agent in CFG.AGENTS:
                            self.hub.add_reward(agent, per_agent_reward)

                    if (
                        getattr(CFG, "COVERAGE_LOG_ENABLED", False)
                        and now - last_coverage_print >= getattr(CFG, "COVERAGE_LOG_INTERVAL", 30.0)
                    ):
                        print(
                            f"[WINDOW {self.local_window_index}] "
                            f"local observed={local_stats['observed'] * 100:.2f}% "
                            f"visited={local_stats['visited'] * 100:.2f}% "
                            f"unknown={local_stats['unknown'] * 100:.2f}% | "
                            f"global observed={global_stats['observed'] * 100:.2f}% "
                            f"visited={global_stats['visited'] * 100:.2f}%"
                        )
                        last_coverage_print = now

                    global_stats, global_complete = self.is_global_complete()
                    target_tracking_active = self.hub.has_target() if hasattr(self.hub, "has_target") else False
                    tracker_arrived = getattr(self.hub, "tracker_arrived", False)

                    if global_complete and not (target_tracking_active and not tracker_arrived):
                        self.hub.request_stop(
                            f"GLOBAL_MAP_COMPLETE observed={global_stats['observed'] * 100:.2f}% "
                            f"visited={global_stats['visited'] * 100:.2f}%"
                        )
                        break

                    # While Agent1 tracks the ball, other agents keep expanding the map via local windows.
                    # Do not skip window relocation here (old continue blocked all map growth).

                    local_complete = (
                        local_stats["observed"] >= CFG.LOCAL_TARGET_MAP_COVERAGE
                        and local_stats["visited"] >= CFG.LOCAL_TARGET_VISITED_COVERAGE
                    )

                    local_visited_delta = max(0.0, local_stats["visited"] - prev_local_visited)
                    prev_local_visited = local_stats["visited"]
                    if local_visited_delta < getattr(CFG, "LOCAL_STALL_DELTA", 0.005):
                        local_stall_count += 1
                    else:
                        local_stall_count = 0

                    local_stalled = (
                        getattr(CFG, "LOCAL_STALL_MOVE_ENABLED", True)
                        and local_stats["visited"] >= getattr(CFG, "LOCAL_STALL_MIN_VISITED", 0.45)
                        and local_stall_count >= getattr(CFG, "LOCAL_STALL_CHECKS", 4)
                    )

                    if (local_complete or local_stalled) and getattr(CFG, "COMMAND_RELOCATE_ENABLED", True):
                        if local_window_count >= CFG.MAX_LOCAL_WINDOWS_PER_EPISODE:
                            self.hub.request_stop("MAX_LOCAL_WINDOWS_REACHED")
                            break
                        next_center = self.hub.select_next_window_center(
                            self.belief_map,
                            current_center=self.current_window_center,
                            visited_centers=self.visited_window_centers,
                        )
                        if next_center is None:
                            self.hub.request_stop("NO_MORE_GLOBAL_FRONTIER")
                            break
                        nx, ny, score, unknown = next_center
                        local_window_count += 1
                        move_reason = "complete" if local_complete else f"stall count={local_stall_count}"
                        print(
                            f"[WINDOW MOVE] window={self.local_window_index} reason={move_reason} "
                            f"next=({nx:.1f},{ny:.1f}), unknown={unknown:.2f}, score={score:.2f}"
                        )
                        self.set_local_window(nx, ny, window_index=local_window_count)
                        prev_local_visited = 0.0
                        local_stall_count = 0
                        self.hub.assign_global_goals(self.belief_map, force=True)

                    last_coverage_check = now

                if (
                    getattr(CFG, "SUMMARY_PRINT_ENABLED", False)
                    and now - last_summary_print >= getattr(CFG, "SUMMARY_PRINT_INTERVAL", 30.0)
                ):
                    self.hub.print_summary()
                    last_summary_print = now

                if self.hub.stop_event.is_set():
                    break
                time.sleep(0.2)
        except KeyboardInterrupt:
            self.hub.request_stop("KEYBOARD_INTERRUPT")

        for w in workers:
            w.join(timeout=3.0)
        if getattr(CFG, "INIT_LOG_ENABLED", False):
            print("[MISSION] Episode finished")

    def land_all(self):
        print("\n[LAND] Landing all agent drones")
        for drone in self.active_drones():
            safe_hover(self.client, drone)
        for drone in self.active_drones():
            try:
                self.client.landAsync(vehicle_name=drone)
            except Exception as e:
                print(f"[WARN] land failed for {drone}: {e}")
            time.sleep(0.2)
        time.sleep(3.0)
        for drone in self.active_drones():
            try:
                self.client.armDisarm(False, vehicle_name=drone)
                self.client.enableApiControl(False, vehicle_name=drone)
            except Exception:
                pass

    # ============================================================
    # Output / episode management
    # ============================================================

    def save_outputs(self, episode: int = None):
        os.makedirs(CFG.OUTPUT_DIR, exist_ok=True)
        if self.belief_map is not None:
            self.belief_map.save(CFG.OUTPUT_DIR)
            if episode is not None and hasattr(self.belief_map, "save_episode"):
                self.belief_map.save_episode(CFG.OUTPUT_DIR, episode)
            if getattr(CFG, "PRESENTATION_UNION_ENABLED", True):
                exclude_ep = None
                if getattr(CFG, "PRESENTATION_UNION_EXCLUDE_LATEST_EPISODE", False) and episode is not None:
                    exclude_ep = int(episode)
                BeliefMap.save_presentation_union(CFG.OUTPUT_DIR, exclude_episode=exclude_ep)
        if getattr(self.hub, "targets", None):
            target_csv = os.path.join(CFG.OUTPUT_DIR, "targets.csv")
            with open(target_csv, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "drone",
                    "drone_x", "drone_y", "drone_z",
                    "target_x", "target_y", "target_z",
                    "target_gt_x", "target_gt_y", "target_gt_z",
                    "camera", "confidence", "time",
                ])
                for t in self.hub.targets:
                    writer.writerow([
                        t.get("drone"),
                        t.get("drone_x"), t.get("drone_y"), t.get("drone_z"),
                        t.get("target_x"), t.get("target_y"), t.get("target_z"),
                        t.get("target_gt_x"), t.get("target_gt_y"), t.get("target_gt_z"),
                        t.get("camera"), t.get("confidence"), t.get("time"),
                    ])
            print(f"[OUTPUT] saved: {target_csv}")
        self.hub.rl_learner.save()

    def append_episode_log(
        self,
        global_episode: int,
        session_episode: int,
        shaping: dict = None,
    ):
        os.makedirs(CFG.OUTPUT_DIR, exist_ok=True)
        exists = os.path.exists(CFG.EPISODE_CSV)
        global_stats = self.effective_global_stats() if self.belief_map is not None else {}
        local_stats = (
            self.belief_map.coverage_stats_area(self.current_local_area, ignore_margin=CFG.COVERAGE_IGNORE_MARGIN)
            if self.belief_map is not None and self.current_local_area is not None
            else {}
        )
        shaping = shaping or {}
        row = [
            global_episode,
            session_episode,
            f"{float(getattr(CFG, 'OBSTACLE_DENSITY', 0.1)):.4f}",
            self.hub.stop_reason,
            f"{global_stats.get('observed', 0.0):.6f}",
            f"{global_stats.get('visited', 0.0):.6f}",
            f"{global_stats.get('obstacle', 0.0):.6f}",
            f"{global_stats.get('unknown', 1.0):.6f}",
            self.local_window_index,
            f"{local_stats.get('observed', 0.0):.6f}",
            f"{local_stats.get('visited', 0.0):.6f}",
            f"{local_stats.get('obstacle', 0.0):.6f}",
            f"{local_stats.get('unknown', 1.0):.6f}",
            self.hub.collision_count,
            f"{shaping.get('global_visited_delta', 0.0):.6f}",
            f"{shaping.get('min_target_dist_m', 0.0):.3f}",
            f"{shaping.get('target_progress', 0.0):.6f}",
            f"{shaping.get('coverage_shaping_reward', 0.0):.3f}",
            f"{shaping.get('target_shaping_reward', 0.0):.3f}",
            f"{self.hub.total_reward():.3f}",
            dict(self.hub.rewards),
        ]
        with file_lock(CFG.EPISODE_CSV):
            with open(CFG.EPISODE_CSV, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                if not exists:
                    writer.writerow(EPISODE_LOG_FIELDS)
                writer.writerow(row)

    def reset_for_next_episode(self):
        print("\n[RESET] Resetting AirSim for next episode...\n")
        try:
            rpc_call(self.client.reset)
        except Exception as e:
            print(f"[WARN] reset failed: {e}")
        reset_shared_client()
        time.sleep(CFG.RESET_SLEEP_SEC)
        self.client = connect_client(confirm=False)
        if hasattr(self.hub.rl_learner, "decay_epsilon"):
            self.hub.rl_learner.decay_epsilon()

        # Keep only behavior-level learning.
        # Spatial memory is reset because obstacle/target positions can change every episode.
        old_learner = self.hub.rl_learner
        old_adaptive_obstacle_dist = dict(getattr(self.hub, "adaptive_obstacle_dist", {}))

        self.hub = CommandHub()
        self.hub.rl_learner = old_learner
        if old_adaptive_obstacle_dist:
            self.hub.adaptive_obstacle_dist.update(old_adaptive_obstacle_dist)

        # Reset all coordinate-dependent state.
        self.belief_map = None
        self.plans = None
        self.current_local_area = None
        self.current_window_center = None
        self.local_window_index = 0
        self.visited_window_centers = []
        self.last_collision_stamp = {}

    def run_episode(self, global_episode: int, session_episode: int):
        print("\n" + "=" * 70)
        print(
            f"[EPISODE] global={global_episode} session={session_episode}/{CFG.MAX_EPISODES}  "
            f"VERSION={CFG.VERSION}"
        )
        print("=" * 70 + "\n")
        episode_start = time.time()
        try:
            self.setup_drones()
            self.setup_target_detection()
            self.initialize_global_belief_map()
            if self.belief_map is not None and hasattr(self.belief_map, "start_new_episode"):
                self.belief_map.start_new_episode(global_episode)
            start_stats = self.effective_global_stats() if self.belief_map is not None else {}
            start_visited = float(
                start_stats.get("global_visited", start_stats.get("visited", 0.0))
            )
            self.hub.begin_episode_shaping(start_visited)
            self.takeoff_agents()
            if self.hub.stop_event.is_set():
                return self.hub.stop_reason

            cx, cy = self.autonomous_start_center()
            plans = self.create_search_plan(cx, cy)

            self.start_collision_monitor()
            try:
                self.run_search(plans, episode_start)
            finally:
                self.stop_collision_monitor()

            final_stats = self.effective_global_stats()
            self.hub.update_coverage(final_stats["observed"], stats=final_stats)
            if getattr(CFG, "FINAL_SUMMARY_PRINT_ENABLED", False):
                self.hub.print_summary()
            end_visited = float(
                final_stats.get("global_visited", final_stats.get("visited", 0.0))
            )
            shaping = self.hub.finalize_episode_shaping(
                end_visited, stop_reason=self.hub.stop_reason
            )
            if getattr(CFG, "RL_LOG_ENABLED", True):
                print(
                    f"[SHAPING] visited_delta={shaping['global_visited_delta']:.4f} "
                    f"min_target_dist={shaping['min_target_dist_m']:.1f}m "
                    f"progress={shaping['target_progress']:.3f} "
                    f"+cov={shaping['coverage_shaping_reward']:.1f} "
                    f"+tgt={shaping['target_shaping_reward']:.1f}"
                )
            self.save_outputs(global_episode)
            self.append_episode_log(global_episode, session_episode, shaping)
            return self.hub.stop_reason
        except Exception as e:
            print(f"[MISSION ERROR] {e}")
            self.hub.request_stop(f"EPISODE_EXCEPTION {e}")
            shaping = self.hub.finalize_episode_shaping(
                float(
                    self.effective_global_stats().get(
                        "global_visited",
                        self.effective_global_stats().get("visited", 0.0),
                    )
                )
                if self.belief_map is not None
                else 0.0,
                stop_reason=self.hub.stop_reason,
            )
            self.append_episode_log(global_episode, session_episode, shaping)
            return self.hub.stop_reason

    def run(self):
        print(
            f"[ENV] agents={len(CFG.AGENTS)} {list(CFG.AGENTS)} | "
            f"foliage in UE only; OBSTACLE_DENSITY(log)={float(getattr(CFG, 'OBSTACLE_DENSITY', 0.1)):.2f}"
        )
        if getattr(CFG, "API_RPC_LIGHT_PRESET", False):
            print(
                "[API-LIGHT] preset on — "
                f"executor={CFG.EXECUTOR_HZ}Hz, vel_dur={CFG.VELOCITY_CMD_DURATION}s, "
                f"obstacle_dt={CFG.OBSTACLE_CHECK_DT}s, wall=off, "
                f"camera={getattr(CFG, 'CAMERA_WIDTH', '?')}x{getattr(CFG, 'CAMERA_HEIGHT', '?')} "
                f"(sync settings.json + restart UE)"
            )
        if getattr(CFG, "RPC_SHARED_CLIENT_ENABLED", False):
            print(
                "[RPC] shared client + serialized calls, "
                f"image_max_concurrent={getattr(CFG, 'RPC_IMAGE_MAX_CONCURRENT', 1)} "
                f"(reduces API-not-received hover)"
            )
        if getattr(CFG, "FAST_TEST_PRESET", False):
            print(
                "[FAST-TEST] preset on — "
                f"FAST_AGENT_SPEED={CFG.FAST_AGENT_SPEED}m/s, TRACKER_SPEED={CFG.TRACKER_SPEED}m/s, "
                f"MAX_ACCEL={CFG.MAX_ACCEL}, ClockSpeed={getattr(CFG, 'SIM_CLOCK_SPEED', 1.0)} "
                f"(cmd speed only; ClockSpeed 2+ often causes API hover with many agents)"
            )
        if getattr(CFG, "RL_COLLISION_LEARNING_ENABLED", False) and getattr(
            CFG, "RL_USE_12_PRETRAIN_POLICY", False
        ):
            from rl_collision_learner import RLCollisionLearner

            p12 = RLCollisionLearner.resolve_pretrain_policy_path()
            exists = os.path.exists(p12)
            print(
                f"[RL] 12_test pretrain policy: {'ON' if exists else 'MISSING'} "
                f"path={p12} merge_local={getattr(CFG, 'RL_PRETRAIN_MERGE_LOCAL', True)}"
            )
        for session_episode in range(1, CFG.MAX_EPISODES + 1):
            global_episode = self.allocate_global_episode()
            reason = self.run_episode(global_episode, session_episode)
            if self.belief_map is None:
                stats = {"observed": 0.0, "visited": 0.0}
            else:
                stats = self.effective_global_stats()
            if (
                stats["observed"] >= CFG.TARGET_MAP_COVERAGE
                and stats["visited"] >= CFG.TARGET_VISITED_COVERAGE
            ) or reason.startswith("GLOBAL_MAP_COMPLETE"):
                print("[TRAINING] Global observed+visited coverage reached. Stopping training.")
                self.land_all()
                break

            if reason == "KEYBOARD_INTERRUPT":
                self.land_all()
                break
            if reason.startswith("TARGET_FOUND") or reason.startswith("TARGET_REACHED"):
                if getattr(CFG, "CONTINUE_AFTER_TARGET_REACHED", True):
                    print("[TRAINING] Target success — continuing next episode.")
                    self.reset_for_next_episode()
                    continue
                self.land_all()
                break
            self.reset_for_next_episode()
        else:
            print("[TRAINING] Max episodes reached. Landing all drones.")
            self.land_all()


if __name__ == "__main__":
    mission = MissionController()
    mission.run()
