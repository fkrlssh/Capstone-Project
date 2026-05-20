from typing import Optional, Tuple

import airsim
import numpy as np

from airsim_rpc import sim_get_images
from config import CFG


def _image_response_to_rgb(response) -> Optional[np.ndarray]:
    img1d = np.frombuffer(response.image_data_uint8, dtype=np.uint8)

    if response.width == 0 or response.height == 0:
        return None

    if img1d.size == response.width * response.height * 4:
        img = img1d.reshape(response.height, response.width, 4)
        img = img[:, :, :3]
    elif img1d.size == response.width * response.height * 3:
        img = img1d.reshape(response.height, response.width, 3)
    else:
        return None

    return img


def get_scene_image(client: airsim.MultirotorClient, vehicle: str, camera_name: str) -> Optional[np.ndarray]:
    try:
        responses = sim_get_images(client, [
            airsim.ImageRequest(camera_name, airsim.ImageType.Scene, False, False)
        ], vehicle_name=vehicle)

        if not responses:
            return None

        return _image_response_to_rgb(responses[0])

    except Exception as e:
        print(f"[WARN] get_scene_image failed: {vehicle}/{camera_name}: {e}")
        return None


def get_segmentation_image(client: airsim.MultirotorClient, vehicle: str, camera_name: str) -> Optional[np.ndarray]:
    try:
        responses = sim_get_images(client, [
            airsim.ImageRequest(camera_name, airsim.ImageType.Segmentation, False, False)
        ], vehicle_name=vehicle)

        if not responses:
            return None

        return _image_response_to_rgb(responses[0])

    except Exception as e:
        print(f"[WARN] get_segmentation_image failed: {vehicle}/{camera_name}: {e}")
        return None


def get_depth_image(client: airsim.MultirotorClient, vehicle: str, camera_name: str) -> Optional[np.ndarray]:
    try:
        responses = sim_get_images(client, [
            airsim.ImageRequest(camera_name, airsim.ImageType.DepthPlanar, True, False)
        ], vehicle_name=vehicle)

        if not responses or responses[0].width == 0 or responses[0].height == 0:
            return None

        response = responses[0]
        depth = np.array(response.image_data_float, dtype=np.float32)
        depth = depth.reshape(response.height, response.width)

        depth = np.nan_to_num(depth, nan=100.0, posinf=100.0, neginf=0.0)
        depth = np.clip(depth, 0.0, 100.0)

        return depth

    except Exception as e:
        print(f"[WARN] get_depth_image failed: {vehicle}/{camera_name}: {e}")
        return None


def detect_front_obstacle(
    client: airsim.MultirotorClient,
    vehicle: str,
    obstacle_dist: float = None,
) -> Tuple[bool, float, float, float, float]:
    """
    Safety-corridor obstacle detector for cylinder/foliage obstacles.

    Returns:
        obstacle, left_near, right_near, front_clear, near_ratio

    The drone must not be treated as a point.  A visually open center line can
    still collide if a propeller/body edge scrapes a nearby cylinder.  This
    detector therefore checks a wide central corridor plus left/right clearance.
    """
    depth = get_depth_image(client, vehicle, CFG.FRONT_CAMERA)
    if depth is None:
        return False, 100.0, 100.0, 100.0, 0.0
    if obstacle_dist is None:
        obstacle_dist = CFG.OBSTACLE_DIST

    h, w = depth.shape

    top = int(h * getattr(CFG, "OBSTACLE_ROI_TOP", 0.08))
    bottom = int(h * getattr(CFG, "OBSTACLE_ROI_BOTTOM", 0.92))
    top = max(0, min(h - 2, top))
    bottom = max(top + 2, min(h, bottom))

    c_l = int(w * getattr(CFG, "OBSTACLE_CENTER_LEFT", 0.18))
    c_r = int(w * getattr(CFG, "OBSTACLE_CENTER_RIGHT", 0.82))
    l_l = int(w * getattr(CFG, "OBSTACLE_LEFT_LEFT", 0.03))
    l_r = int(w * getattr(CFG, "OBSTACLE_LEFT_RIGHT", 0.35))
    r_l = int(w * getattr(CFG, "OBSTACLE_RIGHT_LEFT", 0.65))
    r_r = int(w * getattr(CFG, "OBSTACLE_RIGHT_RIGHT", 0.97))

    center = depth[top:bottom, max(0, c_l):min(w, c_r)]
    left = depth[top:bottom, max(0, l_l):min(w, l_r)]
    right = depth[top:bottom, max(0, r_l):min(w, r_r)]
    # Body corridor: wider than the old center ROI, but not full image.
    corridor = depth[top:bottom, int(w * 0.12):int(w * 0.88)]

    if center.size == 0 or left.size == 0 or right.size == 0 or corridor.size == 0:
        return False, 100.0, 100.0, 100.0, 0.0

    p = float(getattr(CFG, "FRONT_NEAR_PERCENTILE", 4.0))
    thin_p = float(getattr(CFG, "FRONT_THIN_OBSTACLE_PERCENTILE", 1.0))

    front_clear = float(np.percentile(center, p))
    thin_front_clear = float(np.percentile(center, thin_p))
    left_near = float(np.percentile(left, p))
    right_near = float(np.percentile(right, p))
    corridor_near = float(np.percentile(corridor, thin_p))

    obstacle_dist = float(obstacle_dist)
    side_safety = float(getattr(CFG, "SIDE_SAFETY_DIST", 7.0))
    corridor_safety = float(getattr(CFG, "CORRIDOR_SAFETY_DIST", 8.5))

    near_ratio = float(np.mean(center < obstacle_dist))
    ratio_threshold = float(getattr(CFG, "FRONT_NEAR_RATIO_THRESHOLD", 0.006))
    thin_ratio_threshold = float(getattr(CFG, "FRONT_THIN_NEAR_RATIO_THRESHOLD", 0.0025))

    side_near_ratio = float(np.mean(left < side_safety) + np.mean(right < side_safety)) / 2.0
    corridor_near_ratio = float(np.mean(corridor < corridor_safety))

    emergency_dist = float(getattr(CFG, "EMERGENCY_FRONT_CLEAR_DIST", 6.0))
    emergency_ratio = float(getattr(CFG, "EMERGENCY_FRONT_NEAR_RATIO_THRESHOLD", 0.0010))

    side_scrape = (
        min(left_near, right_near) < side_safety
        or side_near_ratio > float(getattr(CFG, "SIDE_NEAR_RATIO_THRESHOLD", 0.003))
    )
    corridor_blocked = (
        corridor_near < corridor_safety
        or corridor_near_ratio > float(getattr(CFG, "CORRIDOR_NEAR_RATIO_THRESHOLD", 0.004))
    )

    obstacle = (
        front_clear < obstacle_dist
        or near_ratio > ratio_threshold
        or (thin_front_clear < obstacle_dist and near_ratio > thin_ratio_threshold)
        or (thin_front_clear < emergency_dist and near_ratio > emergency_ratio)
        or side_scrape
        or corridor_blocked
    )

    return bool(obstacle), left_near, right_near, min(front_clear, corridor_near), near_ratio


def detect_front_wall(
    client: airsim.MultirotorClient,
    vehicle: str,
    max_distance: float = None,
) -> Tuple[bool, float]:
    """
    Detect a broad physical wall using front depth.

    Important:
    - Use only upper/middle image region so the floor is not mistaken as a wall.
    - Require a wide, balanced, near planar surface across left/center/right.
    - Thin cylinders/foliage should usually fail this test because they are narrow.
    """
    if not getattr(CFG, "WALL_DETECTION_ENABLED", True):
        return False, 100.0

    if max_distance is None:
        max_distance = getattr(CFG, "WALL_DETECT_MAX_DISTANCE", 80.0)

    depth = get_depth_image(client, vehicle, CFG.FRONT_CAMERA)
    if depth is None:
        return False, 100.0

    h, w = depth.shape
    top = int(h * getattr(CFG, "WALL_ROI_TOP", 0.08))
    bottom = int(h * getattr(CFG, "WALL_ROI_BOTTOM", 0.55))
    bottom = max(top + 5, min(h, bottom))

    roi = depth[top:bottom, int(w * 0.05):int(w * 0.95)]
    left = depth[top:bottom, int(w * 0.05):int(w * 0.35)]
    center = depth[top:bottom, int(w * 0.35):int(w * 0.65)]
    right = depth[top:bottom, int(w * 0.65):int(w * 0.95)]

    p = getattr(CFG, "FRONT_NEAR_PERCENTILE", 8.0)
    l_near = float(np.percentile(left, p))
    c_near = float(np.percentile(center, p))
    r_near = float(np.percentile(right, p))
    wall_dist = min(l_near, c_near, r_near)

    full_ratio = float(np.mean(roi < max_distance))
    left_ratio = float(np.mean(left < max_distance))
    center_ratio = float(np.mean(center < max_distance))
    right_ratio = float(np.mean(right < max_distance))

    near_ratio_threshold = getattr(CFG, "WALL_NEAR_RATIO_THRESHOLD", 0.70)
    side_ratio_threshold = getattr(CFG, "WALL_SIDE_NEAR_RATIO_THRESHOLD", 0.55)
    balance_tol = getattr(CFG, "WALL_DISTANCE_BALANCE_TOL", 4.0)

    balanced = (max(l_near, c_near, r_near) - min(l_near, c_near, r_near)) <= balance_tol

    wall = (
        wall_dist < max_distance
        and full_ratio >= near_ratio_threshold
        and left_ratio >= side_ratio_threshold
        and center_ratio >= side_ratio_threshold
        and right_ratio >= side_ratio_threshold
        and balanced
    )

    return bool(wall), float(wall_dist)
