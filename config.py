from dataclasses import dataclass
from operator import truediv
from typing import Tuple


@dataclass
class Config:
    VERSION: str = "four_agents_rpc_stable_v26_recording_hover"
    API_RPC_LIGHT_PRESET: bool = True
    # Faster episodes via cmd speed (FAST_AGENT_SPEED). ClockSpeed>1 often triggers
    # "API call was not received" with 6 drones — keep SIM_CLOCK_SPEED at 1.0 unless stable.
    FAST_TEST_PRESET: bool = True
    SIM_CLOCK_SPEED: float = 1.0

    # ============================================================
    # Vehicles
    # ============================================================
    COMMAND: str = "CommandDrone"
    COMMAND_ENABLED: bool = False  # Physical command drone is not used; CommandHub remains software-only.
    AGENTS: Tuple[str, str, str, str] = (
        "Agent1", "Agent2", "Agent3", "Agent4",
    )

    FRONT_CAMERA: str = "front_center"
    BOTTOM_CAMERA: str = "bottom_center"
    USE_BOTTOM_CAMERA: bool = False
    # AirSim settings.json ViewMode=Fpv; tracker agent for first-person tracking view.
    FPV_VEHICLE: str = "Agent1"

    # ============================================================
    # AirSim NED coordinate
    # More negative z = higher altitude.
    # ============================================================
    COMMAND_Z: float = -35.0
    AGENT_Z: float = -5.0
    COMMAND_SPEED: float = 8.0
    ALTITUDE_SHIFT_Z: float = -5.5
    ALTITUDE_REACH_TOL: float = 1.0

    # ============================================================
    # Agent altitude freedom
    # Agents stay low, but can shift slightly up/down for avoidance.
    # ============================================================
    AGENT_ALTITUDE_FREE_ENABLED: bool = True
    AGENT_Z_HIGH_LIMIT: float = -7.0
    AGENT_Z_LOW_LIMIT: float = -4.0
    AGENT_VERTICAL_AVOID_STEP: float = 0.6
    AGENT_VERTICAL_AVOID_HOLD_SEC: float = 2.5
    ALTITUDE_SHIFT_HOLD_SEC: float = 3.0
    VERTICAL_AVOID_SIDE_BLOCKED_DIST: float = 5.5
    VERTICAL_AVOID_ON_SLOW_HOLD: bool = False
    VERTICAL_AVOID_ONLY_WHEN_CLOSE: bool = True
    VERTICAL_AVOID_COOLDOWN_SEC: float = 4.0
    PROXIMITY_Z_IGNORE_DISTANCE: float = 1.2

    # ============================================================
    # Global map + moving local area
    # Physical map size is unknown. These are only large virtual bounds.
    # Actual usable boundaries are discovered dynamically by wall detection.
    # ============================================================
    # ±100 m: covers redball_01 near x≈89 and forest/play extent (geofence flyable x≈±90).
    VIRTUAL_MAP_EXTENT_M: float = 100.0
    MAP_X_MIN: float = -100.0
    MAP_X_MAX: float = 100.0
    MAP_Y_MIN: float = -100.0
    MAP_Y_MAX: float = 100.0

    # Local area is fixed to 100x100.
    LOCAL_AREA_SIZE: float = 100.0
    AREA_SIZE: float = 100.0
    AREA_MARGIN: float = 5.0

    # ============================================================
    # Fully autonomous mode without a physical command drone
    # ============================================================
    AUTONOMOUS_START_CENTER_MODE: str = "agent_mean"  # agent_mean or fixed
    AUTONOMOUS_START_CENTER_X: float = 0.0
    AUTONOMOUS_START_CENTER_Y: float = 0.0
    INIT_LOG_ENABLED: bool = False

    # Local window completion.
    # This is intentionally lower than global completion so the system expands faster.
    LOCAL_TARGET_MAP_COVERAGE: float = 0.58
    LOCAL_TARGET_VISITED_COVERAGE: float = 0.48

    # Global completion.
    TARGET_MAP_COVERAGE: float = 0.90
    TARGET_VISITED_COVERAGE: float = 0.90

    # ============================================================
    # Moving-window expansion
    # ============================================================
    LOCAL_STALL_MOVE_ENABLED: bool = True
    LOCAL_STALL_MIN_VISITED: float = 0.45
    LOCAL_STALL_DELTA: float = 0.005
    LOCAL_STALL_CHECKS: int = 4

    COMMAND_RELOCATE_ENABLED: bool = True
    NEXT_WINDOW_SAMPLE_STEP: int = 8
    NEXT_WINDOW_UNKNOWN_RADIUS: float = 18.0
    NEXT_WINDOW_DISTANCE_WEIGHT: float = 0.02
    NEXT_WINDOW_MIN_CENTER_SEPARATION: float = 70.0

    # Robust map expansion fallback.
    # If frontier sampling fails, move to adjacent 100x100 windows instead of stopping.
    NEXT_WINDOW_FORCE_ADJACENT_FALLBACK: bool = True
    NEXT_WINDOW_EXPANSION_STEP: float = 90.0
    NEXT_WINDOW_MAX_RING: int = 6
    NEXT_WINDOW_MIN_UNKNOWN_RATIO: float = 0.04
    NEXT_WINDOW_ADJACENT_BONUS: float = 12.0
    NEXT_WINDOW_RING_PENALTY: float = 4.0
    NEXT_WINDOW_IGNORE_DYNAMIC_BOUNDS_ON_LAST_RESORT: bool = True

    MAX_LOCAL_WINDOWS_PER_EPISODE: int = 20
    COMMAND_RELOCATE_TIMEOUT_SEC: float = 60.0
    COMMAND_RELOCATE_REACH_DIST: float = 3.0

    # ============================================================
    # Geofence / wall boundary
    # ============================================================
    USE_GEOFENCE: bool = True
    WALL_MARGIN: float = 10.0
    GEOFENCE_PUSH_MARGIN: float = 8.0

    DYNAMIC_BOUNDARY_ENABLED: bool = True
    # Off in API-light preset: saves one depth RPC per agent per obstacle tick (6x load).
    WALL_DETECTION_ENABLED: bool = False
    PERCEPTION_WALL_WITH_OBSTACLE: bool = False
    WALL_DETECT_MAX_DISTANCE: float = 80.0
    WALL_BOUNDARY_MARGIN: float = 4.0

    AGENT_WALL_BOUNDARY_UPDATE_ENABLED: bool = False
    WALL_BOUNDARY_EDGE_TOL: float = 8.0
    WALL_BOUNDARY_MIN_SPAN: float = 80.0
    WALL_BOUNDARY_MIN_DISTANCE: float = 12.0

    # Wall detector uses broad upper/middle front depth.
    WALL_ROI_TOP: float = 0.08
    WALL_ROI_BOTTOM: float = 0.55
    WALL_NEAR_RATIO_THRESHOLD: float = 0.70
    WALL_SIDE_NEAR_RATIO_THRESHOLD: float = 0.55
    WALL_DISTANCE_BALANCE_TOL: float = 4.0

    # Keep True for unknown physical map size.
    REQUIRE_ALL_BOUNDARIES_FOR_GLOBAL_COMPLETE: bool = True

    # ============================================================
    # Episode training / termination
    # ============================================================
    MAX_EPISODES: int = 500
    MAX_EPISODE_TIME_SEC: float = 420.0
    RESET_SLEEP_SEC: float = 3.0
    EPISODE_CSV: str = "outputs/episode_log_dynamic_v7_clock1_fast_visit.csv"
    EPISODE_MAP_DIR: str = "episodes"
    # Continue episode index across main.py restarts; plot TB only appends new rows.
    EPISODE_LOG_APPEND_GLOBAL: bool = True
    RL_PLOT_STATE_JSON: str = "rl_plot_state.json"
    NORMALIZED_INDICATORS_CSV: str = "outputs/normalized_indicators.csv"
    NORMALIZED_INDICATORS_JSONL: str = "outputs/normalized_indicators.jsonl"
    NORMALIZED_SUMMARY_JSON: str = "normalized_summary.json"

    # Normalized metric scales (0–1 dashboards)
    METRIC_REWARD_NORM_CEILING: float = 120.0
    METRIC_VISITED_DELTA_NORM: float = 0.15
    METRIC_Q_STATES_NORM: float = 64.0
    METRIC_COMPOSITE_WEIGHTS: Tuple[str, ...] = ()  # empty => use defaults in normalized_metrics.py

    # Background supervisor (train_background.py)
    BG_METRICS_INTERVAL_SEC: float = 180.0

    # ============================================================
    # Collision monitor
    # ============================================================
    COLLISION_RESTART_ENABLED: bool = True
    COLLISION_MONITOR_DT: float = 0.20
    COLLISION_ENABLE_Z: float = -2.0
    COLLISION_IGNORE_NEAR_GROUND_OBJECTS: Tuple[str, ...] = (
        "frozen_Room", "Floor", "floor", "Ground", "ground"
    )
    MAX_COLLISIONS_BEFORE_RESTART: int = 1
    MAX_MISSION_RESTARTS: int = 3

    # ============================================================
    # Global/local frontier assignment
    # ============================================================
    GLOBAL_GOAL_ASSIGN_DT: float = 3.0
    FRONTIER_SAMPLE_STEP: int = 4
    FRONTIER_MAX_CANDIDATES: int = 260
    FRONTIER_MIN_SEPARATION: float = 18.0
    FRONTIER_UNKNOWN_RADIUS: float = 12.0
    FRONTIER_DISTANCE_WEIGHT: float = 0.02
    FRONTIER_ASSIGNMENT_BONUS: float = 16.0

    # ============================================================
    # 6-grid local planner
    # ============================================================
    PLANNER_CELL_SIZE: float = 5.0
    SIX_GRID_ANGLES_DEG: Tuple[float, ...] = (0.0, 60.0, 120.0, 180.0, 240.0, 300.0)

    # FAST_TEST_PRESET: raise FAST_AGENT_SPEED / SIM_CLOCK_SPEED for wall-clock speed.
    FAST_AGENT_SPEED: float = 5.0
    FAST_MIN_SPEED: float = 1.2

    R_UNKNOWN: float = 14.0
    R_DIRECTION: float = 0.6
    R_STRIP_CENTER: float = 0.15
    R_BACKTRACK: float = 1.0
    R_GLOBAL_GOAL: float = 12.0
    R_GOAL_DISTANCE: float = 5.0
    # Planner scores unknown using this radius (m), not just the next 5m cell.
    PLANNER_UNKNOWN_SCORE_RADIUS: float = 12.0
    R_RL_MOTION: float = 1.0

    P_VISITED: float = 1.0

    # 에피소드 내에서 이미 방문한 곳으로 되돌아가는 것을 억제.
    # 완전 금지는 아니며, 막힌 상황에서는 backtracking/우회 가능.
    P_EPISODE_VISITED_CELL: float = 3.5
    P_EPISODE_VISITED_AREA: float = 4.5
    P_RECENT_CELL: float = 3.0
    EPISODE_VISITED_AVOID_RADIUS: float = 4.0
    RECENT_CELL_MEMORY: int = 18
    CURRENT_TARGET_VISITED_ABORT_RATIO: float = 0.65

    P_OBSTACLE: float = 5.0
    P_GEOFENCE: float = 10.0
    P_AGENT: float = 12.0
    P_BLOCKED: float = 14.0
    P_FRONT_OBSTACLE: float = 4.0
    P_OUT_OF_STRIP: float = 0.5

    BACKTRACK_SCORE_THRESHOLD: float = -6.0

    # ============================================================
    # Continuous executor / heading alignment (API-light preset)
    # Lower Hz + longer cmd duration => fewer RPCs, less "API not received" hover.
    # ============================================================
    EXECUTOR_HZ: float = 10.0
    EXECUTOR_DT: float = 1.0 / EXECUTOR_HZ
    # Keep each velocity command alive longer than one control tick.
    VELOCITY_CMD_DURATION: float = 2.5
    AGENT_START_STAGGER_SEC: float = 0.25
    CELL_REACH_DIST: float = 2.8

    YAW_AXIS_SIGN: float = 1.0
    YAW_OFFSET_DEG: float = 0.0
    HEADING_ALIGN_THRESHOLD_DEG: float = 18.0
    HEADING_SLOW_THRESHOLD_DEG: float = 55.0
    # Do not slow down too aggressively while yaw catches up.
    HEADING_ALIGN_SPEED_SCALE: float = 0.97
    HEADING_SLOW_SPEED_SCALE: float = 0.88
    MIN_START_CMD_SPEED: float = 1.0
    MAX_YAW_RATE_DEG_PER_SEC: float = 90.0

    # Motion stability: reduce stop-and-go / bobbing from flickering depth + abrupt commands.
    OBSTACLE_ENGAGE_DELAY_SEC: float = 0.18
    OBSTACLE_CLEAR_DELAY_SEC: float = 0.28
    CMD_VELOCITY_SMOOTH_ENABLED: bool = True
    CMD_VELOCITY_SMOOTH_ALPHA: float = 0.32
    HARD_ESCAPE_PREV_V_SCALE: float = 0.45
    MOTION_FIXED_ALTITUDE_SEARCH: bool = True
    MAX_ACCEL: float = 5.0

    TARGET_CHECK_DT: float = 0.55
    OBSTACLE_CHECK_DT: float = 0.30
    TARGET_POSE_POLL_DT: float = 1.0

    # ============================================================
    # UE foliage obstacles (InstancedFoliage) — install/density in editor only.
    # Log label for episode CSV; must match your UE foliage density setting.
    # ============================================================
    OBSTACLE_DENSITY: float = 0.07

    # Keep training after TARGET_REACHED (500-episode runs).
    CONTINUE_AFTER_TARGET_REACHED: bool = True

    # ============================================================
    # Belief map
    # ============================================================
    GRID_RESOLUTION: float = 1.0
    VISITED_MARK_RADIUS: float = 5.0
    OBSTACLE_MARK_RADIUS: float = 3.5
    # Belief-map OBSTACLE paint only when the cylinder is actually close (avoids blocking forest entry from far depth).
    OBSTACLE_MAP_MARK_MAX_DIST: float = 6.5
    COVERAGE_CHECK_DT: float = 3.0
    COVERAGE_IGNORE_MARGIN: float = 4.0

    # ============================================================
    # Obstacle avoidance
    # ============================================================
    OBSTACLE_DIST: float = 10.0
    # Depth trigger for reactive avoidance (still on). Map marking uses OBSTACLE_MAP_MARK_MAX_DIST.
    OBSTACLE_ROI_TOP: float = 0.08
    OBSTACLE_ROI_BOTTOM: float = 0.92
    OBSTACLE_CENTER_LEFT: float = 0.18
    OBSTACLE_CENTER_RIGHT: float = 0.82
    OBSTACLE_LEFT_LEFT: float = 0.03
    OBSTACLE_LEFT_RIGHT: float = 0.35
    OBSTACLE_RIGHT_LEFT: float = 0.65
    OBSTACLE_RIGHT_RIGHT: float = 0.97
    FRONT_THIN_OBSTACLE_PERCENTILE: float = 1.0
    FRONT_THIN_NEAR_RATIO_THRESHOLD: float = 0.0025
    FRONT_NEAR_PERCENTILE: float = 4.0
    FRONT_NEAR_RATIO_THRESHOLD: float = 0.012
    AVOID_SPEED: float = 4.2
    EMERGENCY_FRONT_CLEAR_DIST: float = 4.5
    EMERGENCY_FRONT_NEAR_RATIO_THRESHOLD: float = 0.0010
    HARD_AVOID_FORWARD: float = -1.25
    HARD_AVOID_SIDE: float = 1.35

    # Safety corridor check: the drone is not a point.  If the side bands are
    # too close, treat it as an obstacle even when the exact center looks open.
    SIDE_SAFETY_DIST: float = 5.0
    CORRIDOR_SAFETY_DIST: float = 5.5
    SIDE_NEAR_RATIO_THRESHOLD: float = 0.008
    CORRIDOR_NEAR_RATIO_THRESHOLD: float = 0.010
    HARD_SIDE_CLEAR_DIST: float = 4.0

    # Search: keep moving forward between pillars; hard escape only when truly tight.
    SEARCH_BOLD_AVOIDANCE: bool = True
    SEARCH_AVOID_FORWARD_COEFF: float = -0.12
    SEARCH_AVOID_SIDE_COEFF: float = 0.95
    SEARCH_BOLD_PATH_BLEND: float = 0.55

    # ============================================================
    # RPC / command polling safety (reduces "API call was not received")
    # ============================================================
    RPC_SHARED_CLIENT_ENABLED: bool = True
    RPC_SERIALIZE_CALLS: bool = True
    RPC_IMAGE_MAX_CONCURRENT: int = 2
    RPC_CONFIRM_ON_FIRST_CONNECT: bool = True
    PERCEPTION_OBSTACLE_STAGGER_SEC: float = 0.04
    RPC_RETRY_COUNT: int = 3
    RPC_RETRY_SLEEP: float = 0.5
    # AirSim camera resolution (must match settings.json CaptureSettings).
    CAMERA_WIDTH: int = 192
    CAMERA_HEIGHT: int = 108
    COMMAND_TAKEOFF_TIMEOUT_SEC: float = 40.0
    AGENT_TAKEOFF_TIMEOUT_SEC: float = 35.0
    DEPLOY_SINGLE_TIMEOUT_SEC: float = 45.0
    DEPLOY_RETRY_COUNT: int = 3
    DEPLOY_RETRY_SLEEP: float = 1.0
    DEPLOY_REACH_DIST: float = 1.5

    # ============================================================
    # Multi-drone conflict manager
    # ============================================================
    RESERVATION_TTL: float = 1.2
    SAME_CELL_HOLD_TIME: float = 0.08
    SWAP_HOLD_TIME: float = 0.10
    PROXIMITY_DISTANCE: float = 16.0
    EMERGENCY_DISTANCE: float = 5.0
    REPULSION_DISTANCE: float = 18.0
    REPULSION_GAIN: float = 4.0

    # No artificial push between drones. Only priority/yield collision prevention.
    DRONE_REPULSION_ENABLED: bool = False
    DRONE_PROXIMITY_HOLD_ENABLED: bool = True
    DRONE_COLLISION_HOLD_DISTANCE: float = 8.0
    DRONE_COLLISION_HOLD_TIME: float = 0.05

    COLLISION_RECOVERY_SPEED: float = 3.5
    COLLISION_RECOVERY_TIME: float = 1.2

    # ============================================================
    # RL-style online collision / avoidance learning
    # ============================================================
    RL_COLLISION_LEARNING_ENABLED: bool = True
    RL_ALPHA: float = 0.45
    RL_GAMMA: float = 0.85
    # Warm-start from 12_test fast virtual Q-table, then fine-tune in UE.
    RL_USE_12_PRETRAIN_POLICY: bool = True
    # Empty => auto: ../12_test/outputs/rl_collision_qtable.json
    RL_PRETRAIN_POLICY_PATH: str = ""
    # False = outputs 비었을 때 12_test 정책만 사용 (실측 fresh start)
    # True  = 11_test/outputs/rl_collision_qtable.json 이 있으면 그걸로 덮어씀
    RL_PRETRAIN_MERGE_LOCAL: bool = False
    # Lower exploration when starting from 12_test policy.
    RL_EPSILON: float = 0.10
    RL_EPSILON_DECAY: float = 0.992
    RL_EPSILON_MIN: float = 0.05
    RL_SAVE_PATH: str = "outputs/rl_collision_qtable.json"
    RL_SAVE_EVERY_UPDATES: int = 20
    RL_OBSTACLE_STEP_REWARD: float = 0.35

    STEP_ALIVE_REWARD: float = 0.01
    MAP_COVERAGE_REWARD_SCALE: float = 180.0
    # Episode-end bonus: global visited fraction gained this episode.
    EPISODE_COVERAGE_REWARD_SCALE: float = 80.0
    # Episode-end bonus: fleet got closer to redball (GT pose when available).
    EPISODE_TARGET_NORM_DIST_M: float = 200.0
    TARGET_PROXIMITY_REWARD_SCALE: float = 45.0
    TARGET_REACHED_SHAPING_BONUS: float = 30.0
    OBSTACLE_AVOID_REWARD: float = 0.20
    PROXIMITY_PENALTY_SCALE: float = 0.08
    RL_COLLISION_REWARD: float = -80.0
    DRONE_COLLISION_REWARD: float = -120.0
    RL_SAFE_REWARD: float = 0.5
    RL_SAFE_REWARD_DT: float = 2.0

    ADAPTIVE_OBSTACLE_DIST_MIN: float = 8.0
    ADAPTIVE_OBSTACLE_DIST_MAX: float = 11.5
    ADAPTIVE_OBSTACLE_DIST_INC_ON_COLLISION: float = 1.0
    ADAPTIVE_OBSTACLE_DIST_DECAY_ON_SAFE: float = 0.02

    # ============================================================
    # Target detection: keep OFF until 90% map-visit experiment is stable.
    # Turn these ON later for redball_01 search.
    # ============================================================
    TARGET_DETECTION_ENABLED: bool = True
    REQUIRE_TARGET_OBJECT_IN_MAP: bool = True
    TARGET_OBJECT_NAME: str = "redball_01"
    TARGET_OBJECT_REGEX: str = ".*redball_01.*"
    TARGET_SEGMENTATION_ID: int = 1
    TARGET_USE_BOTTOM_CAMERA: bool = False
    TARGET_USE_FRONT_CAMERA: bool = True
    REQUIRE_RED_RGB_CONFIRM: bool = False
    TARGET_RED_R_MIN: int = 180
    TARGET_RED_G_MAX: int = 120
    TARGET_RED_B_MAX: int = 120
    TARGET_RED_RATIO_MIN: float = 0.01
    TARGET_MIN_AREA: int = 1
    TARGET_MIN_PIXELS: int = 1
    TARGET_MIN_CIRCULARITY: float = 0.0
    TARGET_CONF_THRESHOLD: float = 0.30
    TARGET_CONFIRM_FRAMES: int = 1
    STOP_ON_FIRST_TARGET: bool = False



    # ============================================================
    # Target tracking mode (녹화용: 타겟 반견 시 Agent1만 접근, 나머지 호버)
    # ============================================================
    TARGET_RECORDING_MODE: bool = True
    TARGET_TRACKING_ENABLED: bool = True
    TRACKER_AGENT: str = "Agent1"
    TARGET_TRACKER_SELECTOR: str = "fixed"
    # True = Agent2~4는 타겟 반견 후 제자리 호버 (탐색 중단)
    HOLD_NON_TRACKER_AFTER_TARGET: bool = True
    # 녹화 중에는 창 이동·맵 확장 최소화 (Agent1 추적만 강조)
    EXPAND_MAP_DURING_TARGET_TRACKING: bool = False

    # If True, all agents move toward the detected target instead of only Agent1.
    ALL_AGENTS_TRACK_TARGET: bool = False
    # Agent1: 타겟 반견 후 직진 (녹화용 — 회피로 궤적이 휘지 않게)
    TRACK_STRAIGHT_TO_TARGET: bool = True
    TRACK_STRAIGHT_IGNORE_ALL_AVOID: bool = True
    # True = 정말 가까운 실린더만 hard escape, 그 외 직진
    TRACK_RECORDING_EMERGENCY_ONLY: bool = True
    TRACK_RECORDING_EMERGENCY_MIN_DIST_M: float = 5.0
    TRACK_BOLD_PATH_BLEND: float = 0.92
    TARGET_APPROACH_SPACING: float = 7.0
    # Far from the ball, all agents fly toward the actual redball center.
    # Only near the ball do they spread slightly to avoid stacking.
    TARGET_FORMATION_ACTIVATE_DIST: float = 999999.0

    # During final target approach the red ball itself appears as a depth obstacle.
    # Ignore that obstacle response so drones do not orbit away from the ball.
    IGNORE_TARGET_AS_OBSTACLE: bool = True
    TARGET_OBSTACLE_IGNORE_DIST: float = 35.0
    # Only ignore depth obstacle as target when the front depth roughly matches the target distance.
    TARGET_OBSTACLE_DEPTH_TOL: float = 4.0
    # Non-tracker hover altitude (slightly above Agent1 to reduce FPV clutter / collisions).
    NON_TRACKER_HOLD_Z: float = -6.0
    NON_TRACKER_USE_SAFE_HOVER: bool = True
    NON_TRACKER_HOVER_CMD_INTERVAL_SEC: float = 2.0


    TRACKER_APPROACH_RADIUS: float = 4.0
    TRACKER_TARGET_Z: float = -5.0
    TRACKER_SPEED: float = 5.0
    TRACKER_MIN_SPEED: float = 1.2
    TRACKER_HOVER_ON_ARRIVAL: bool = True
    STOP_AFTER_TRACKER_ARRIVAL: bool = False
    # Mission is complete as soon as at least one agent reaches the red target cell/radius.
    TARGET_COMPLETE_ON_FIRST_ARRIVAL: bool = True

    COMMAND_MOVE_TO_TARGET: bool = False
    COMMAND_TARGET_Z: float = -35.0
    COMMAND_TARGET_SPEED: float = 8.0
    COMMAND_TARGET_REACH_DIST: float = 5.0

    # redball_01 detected world position is marked on the belief map in red.
    TARGET_MAP_MARK_RADIUS: float = 6.0

    # ============================================================
    # Debug / output / quiet logging
    # ============================================================
    MOVE_DEBUG_ENABLED: bool = False
    MOVE_DEBUG_INTERVAL: float = 1.5

    # Keep terminal output minimal. Important events are still printed:
    # episode start/stop, target detected/marked/arrived, collision, window move, output path.
    STATUS_LOG_ENABLED: bool = False
    IMPORTANT_STATUS_LOG_ENABLED: bool = True
    IMPORTANT_STATUS_KEYWORDS: Tuple[str, ...] = (
        "ERROR",
        "COLLISION_DETECTED",
    )

    TARGET_DEBUG_LOG_ENABLED: bool = False
    TARGET_UPDATE_LOG_ENABLED: bool = False
    TARGET_MAP_MARK_LOG_ENABLED: bool = True
    TARGET_ARRIVAL_LOG_ENABLED: bool = True

    GOAL_LOG_ENABLED: bool = False
    WINDOW_DEBUG_LOG_ENABLED: bool = False
    WALL_BOUNDARY_LOG_ENABLED: bool = False
    RL_LOG_ENABLED: bool = True
    AGENT_VERSION_LOG_ENABLED: bool = False

    COVERAGE_LOG_ENABLED: bool = False
    COVERAGE_LOG_INTERVAL: float = 30.0
    SUMMARY_PRINT_ENABLED: bool = False
    SUMMARY_PRINT_INTERVAL: float = 30.0
    FINAL_SUMMARY_PRINT_ENABLED: bool = False
    MAP_SAVE_LOG_ENABLED: bool = False

    # Union of per-episode maps -> one presentation PNG (obstacles omitted by default).
    PRESENTATION_UNION_ENABLED: bool = True
    PRESENTATION_UNION_BASENAME: str = "belief_presentation_union"
    PRESENTATION_UNION_SKIP_OBSTACLE: bool = True
    # If True, the episode index just saved is excluded from the union (past episodes only).
    PRESENTATION_UNION_EXCLUDE_LATEST_EPISODE: bool = False

    OUTPUT_DIR: str = "outputs"


CFG = Config()
