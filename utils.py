import math
import time
from typing import Dict, List, Tuple

import airsim

from airsim_rpc import get_client, rpc_call
from config import CFG


def connect_client(confirm: bool = True) -> airsim.MultirotorClient:
    return get_client(confirm=confirm)


def get_pose(client: airsim.MultirotorClient, vehicle_name: str) -> Tuple[float, float, float, float]:
    state = rpc_call(client.getMultirotorState, vehicle_name=vehicle_name)
    p = state.kinematics_estimated.position
    q = state.kinematics_estimated.orientation
    try:
        _, _, yaw = airsim.to_eularian_angles(q)
        yaw_deg = math.degrees(yaw)
    except Exception:
        yaw_deg = 0.0
    return float(p.x_val), float(p.y_val), float(p.z_val), float(yaw_deg)


def get_position(client: airsim.MultirotorClient, vehicle_name: str) -> Tuple[float, float, float]:
    x, y, z, _ = get_pose(client, vehicle_name)
    return x, y, z


def distance_2d(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def normalize2(vx: float, vy: float) -> Tuple[float, float]:
    d = math.hypot(vx, vy)
    if d < 1e-6:
        return 0.0, 0.0
    return vx / d, vy / d


def limit_norm(vx: float, vy: float, max_norm: float) -> Tuple[float, float]:
    d = math.hypot(vx, vy)
    if d <= max_norm or d < 1e-6:
        return vx, vy
    scale = max_norm / d
    return vx * scale, vy * scale


def normalize_angle_deg(angle: float) -> float:
    return (angle + 180.0) % 360.0 - 180.0


def angle_diff_deg(target: float, current: float) -> float:
    return normalize_angle_deg(target - current)


def yaw_from_velocity(vx: float, vy: float) -> float:
    if abs(vx) + abs(vy) < 1e-6:
        return 0.0
    yaw = math.degrees(math.atan2(vy, vx))
    yaw = CFG.YAW_AXIS_SIGN * yaw + CFG.YAW_OFFSET_DEG
    return normalize_angle_deg(yaw)


def apply_accel_limit(
    prev_vx: float,
    prev_vy: float,
    des_vx: float,
    des_vy: float,
    dt: float,
    max_accel: float,
) -> Tuple[float, float]:
    dvx = des_vx - prev_vx
    dvy = des_vy - prev_vy
    max_delta = max_accel * max(dt, 1e-3)
    dvx, dvy = limit_norm(dvx, dvy, max_delta)
    vx = prev_vx + dvx
    vy = prev_vy + dvy

    des_speed = math.hypot(des_vx, des_vy)
    cmd_speed = math.hypot(vx, vy)
    min_start = getattr(CFG, "MIN_START_CMD_SPEED", 0.0)
    if des_speed > min_start and cmd_speed < min_start:
        ux, uy = normalize2(des_vx, des_vy)
        vx, vy = ux * min_start, uy * min_start

    return vx, vy


def safe_hover(client: airsim.MultirotorClient, vehicle: str):
    try:
        rpc_call(client.hoverAsync, vehicle_name=vehicle)
    except Exception:
        pass


def wait_for_z(client, vehicle: str, target_z: float, timeout_sec: float, tol: float = None) -> bool:
    if tol is None:
        tol = CFG.ALTITUDE_REACH_TOL
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        try:
            _, _, z = get_position(client, vehicle)
            if abs(z - target_z) <= tol:
                return True
        except Exception:
            pass
        time.sleep(0.1)
    return False


def wait_for_xy_z(client, vehicle: str, target, timeout_sec: float, xy_tol: float, z_tol: float) -> bool:
    tx, ty, tz = target
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        try:
            x, y, z = get_position(client, vehicle)
            if math.hypot(x - tx, y - ty) <= xy_tol and abs(z - tz) <= z_tol:
                return True
        except Exception:
            pass
        time.sleep(0.1)
    return False


def get_safe_bounds():
    return {
        "x_min": CFG.MAP_X_MIN + CFG.WALL_MARGIN,
        "x_max": CFG.MAP_X_MAX - CFG.WALL_MARGIN,
        "y_min": CFG.MAP_Y_MIN + CFG.WALL_MARGIN,
        "y_max": CFG.MAP_Y_MAX - CFG.WALL_MARGIN,
    }


def inside_geofence(x: float, y: float) -> bool:
    if not CFG.USE_GEOFENCE:
        return True
    b = get_safe_bounds()
    return b["x_min"] <= x <= b["x_max"] and b["y_min"] <= y <= b["y_max"]


def clamp_to_geofence(x: float, y: float) -> Tuple[float, float]:
    if not CFG.USE_GEOFENCE:
        return x, y
    b = get_safe_bounds()
    return clamp(x, b["x_min"], b["x_max"]), clamp(y, b["y_min"], b["y_max"])


def clamp_area_to_geofence(area):
    if not CFG.USE_GEOFENCE:
        return area
    b = get_safe_bounds()
    return {
        "x_min": max(area["x_min"], b["x_min"]),
        "x_max": min(area["x_max"], b["x_max"]),
        "y_min": max(area["y_min"], b["y_min"]),
        "y_max": min(area["y_max"], b["y_max"]),
    }


def geofence_penalty(x: float, y: float) -> float:
    if not CFG.USE_GEOFENCE:
        return 0.0
    b = get_safe_bounds()
    dx = min(abs(x - b["x_min"]), abs(b["x_max"] - x))
    dy = min(abs(y - b["y_min"]), abs(b["y_max"] - y))
    d = min(dx, dy)
    if d >= CFG.GEOFENCE_PUSH_MARGIN:
        return 0.0
    return (CFG.GEOFENCE_PUSH_MARGIN - d) / CFG.GEOFENCE_PUSH_MARGIN


def geofence_push_velocity(x: float, y: float) -> Tuple[float, float]:
    if not CFG.USE_GEOFENCE:
        return 0.0, 0.0
    b = get_safe_bounds()
    m = CFG.GEOFENCE_PUSH_MARGIN
    px = 0.0
    py = 0.0
    if x < b["x_min"] + m:
        px += (b["x_min"] + m - x) / m
    elif x > b["x_max"] - m:
        px -= (x - (b["x_max"] - m)) / m
    if y < b["y_min"] + m:
        py += (b["y_min"] + m - y) / m
    elif y > b["y_max"] - m:
        py -= (y - (b["y_max"] - m)) / m
    px, py = limit_norm(px, py, 1.0)
    return px * CFG.FAST_AGENT_SPEED, py * CFG.FAST_AGENT_SPEED


def move_velocity_continuous(
    client: airsim.MultirotorClient,
    vehicle: str,
    vx: float,
    vy: float,
    target_z: float,
    duration: float,
    current_yaw_deg: float,
):
    speed = math.hypot(vx, vy)
    if speed > 0.05:
        target_yaw = yaw_from_velocity(vx, vy)
    else:
        target_yaw = current_yaw_deg

    yaw_error = angle_diff_deg(target_yaw, current_yaw_deg)
    abs_error = abs(yaw_error)

    cmd_vx, cmd_vy = vx, vy
    if speed > 0.05:
        if abs_error > CFG.HEADING_SLOW_THRESHOLD_DEG:
            scale = float(getattr(CFG, "HEADING_SLOW_SPEED_SCALE", 0.75))
            cmd_vx *= scale
            cmd_vy *= scale
        elif abs_error > CFG.HEADING_ALIGN_THRESHOLD_DEG:
            scale = float(getattr(CFG, "HEADING_ALIGN_SPEED_SCALE", 0.90))
            cmd_vx *= scale
            cmd_vy *= scale

        min_cmd = getattr(CFG, "MIN_START_CMD_SPEED", 0.5)
        cmd_speed = math.hypot(cmd_vx, cmd_vy)
        if cmd_speed < min_cmd:
            ux, uy = normalize2(vx, vy)
            cmd_vx, cmd_vy = ux * min_cmd, uy * min_cmd

    try:
        rpc_call(
            client.moveByVelocityZAsync,
            cmd_vx,
            cmd_vy,
            target_z,
            duration,
            drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
            yaw_mode=airsim.YawMode(False, target_yaw),
            vehicle_name=vehicle,
        )
    except Exception as e:
        print(f"[WARN] move_velocity_continuous failed: {vehicle}: {e}")


def create_square_area(cx: float, cy: float, size: float):
    half = size / 2.0
    return {"x_min": cx - half, "x_max": cx + half, "y_min": cy - half, "y_max": cy + half}


def split_area_into_strips(area: Dict, num_strips: int) -> List[Dict]:
    """Split a rectangular area into num_strips horizontal bands (along Y)."""
    n = max(1, int(num_strips))
    x_min, x_max = area["x_min"], area["x_max"]
    y_min, y_max = area["y_min"], area["y_max"]
    strip_h = (y_max - y_min) / float(n)
    strips = []
    for i in range(n):
        strips.append(
            {
                "x_min": x_min,
                "x_max": x_max,
                "y_min": y_min + i * strip_h,
                "y_max": y_min + (i + 1) * strip_h,
            }
        )
    return strips


def split_area_into_4_strips(area):
    return split_area_into_strips(area, 4)


def point_in_strip(x: float, y: float, strip) -> bool:
    return strip["x_min"] <= x <= strip["x_max"] and strip["y_min"] <= y <= strip["y_max"]


def clamp_to_strip(x: float, y: float, strip) -> Tuple[float, float]:
    return (
        clamp(x, strip["x_min"] + CFG.AREA_MARGIN, strip["x_max"] - CFG.AREA_MARGIN),
        clamp(y, strip["y_min"] + CFG.AREA_MARGIN, strip["y_max"] - CFG.AREA_MARGIN),
    )
