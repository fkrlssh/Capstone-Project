from typing import Tuple

import airsim
import cv2
import numpy as np

from config import CFG
from camera_utils import get_scene_image, get_segmentation_image


def red_ratio_in_mask(scene_img: np.ndarray, mask: np.ndarray) -> float:
    """
    Scene 이미지에서 mask 영역이 빨간색인지 확인.
    AirSim 환경에 따라 RGB/BGR 순서가 달라 보일 수 있으므로 둘 다 검사한다.
    """
    if scene_img is None or mask is None:
        return 0.0

    if mask.sum() <= 0:
        return 0.0

    c0 = scene_img[:, :, 0]
    c1 = scene_img[:, :, 1]
    c2 = scene_img[:, :, 2]

    # RGB 가정: R,G,B = 0,1,2
    red_rgb = (
        (c0 >= CFG.TARGET_RED_R_MIN) &
        (c1 <= CFG.TARGET_RED_G_MAX) &
        (c2 <= CFG.TARGET_RED_B_MAX) &
        mask
    )

    # BGR 가정: B,G,R = 0,1,2
    red_bgr = (
        (c2 >= CFG.TARGET_RED_R_MIN) &
        (c1 <= CFG.TARGET_RED_G_MAX) &
        (c0 <= CFG.TARGET_RED_B_MAX) &
        mask
    )

    ratio_rgb = float(red_rgb.sum()) / float(mask.sum())
    ratio_bgr = float(red_bgr.sum()) / float(mask.sum())

    return max(ratio_rgb, ratio_bgr)


def detect_target_sphere_by_segmentation(
    client: airsim.MultirotorClient,
    vehicle: str,
    camera_name: str
) -> Tuple[bool, float]:

    seg_img = get_segmentation_image(client, vehicle, camera_name)

    if seg_img is None:
        return False, 0.0

    h, w = seg_img.shape[:2]

    # 화면 모서리를 배경색으로 추정
    corner_pixels = np.concatenate([
        seg_img[0:10, 0:10].reshape(-1, 3),
        seg_img[0:10, w - 10:w].reshape(-1, 3),
        seg_img[h - 10:h, 0:10].reshape(-1, 3),
        seg_img[h - 10:h, w - 10:w].reshape(-1, 3),
    ], axis=0)

    bg_color = np.median(corner_pixels, axis=0).astype(np.int16)

    diff = np.linalg.norm(
        seg_img.astype(np.int16) - bg_color.reshape(1, 1, 3),
        axis=2
    )

    # redball_01은 segmentation ID가 다르므로 배경과 다른 색으로 보인다.
    mask = diff > 10

    pixel_count = int(mask.sum())
    min_pixels = getattr(CFG, "TARGET_MIN_PIXELS", 1)

    if pixel_count < min_pixels:
        return False, 0.0

    mask_u8 = (mask.astype(np.uint8) * 255)

    # 작은 점 형태 target도 지우지 않기 위해 morphology open 사용 안 함
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask_u8,
        connectivity=8
    )

    if num_labels <= 1:
        return False, 0.0

    best_label = 1
    best_area = 0

    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])

        if area > best_area:
            best_area = area
            best_label = label

    if best_area < min_pixels:
        return False, 0.0

    target_mask = labels == best_label

    red_ratio = 0.0

    if getattr(CFG, "REQUIRE_RED_RGB_CONFIRM", False):
        scene_img = get_scene_image(client, vehicle, camera_name)
        red_ratio = red_ratio_in_mask(scene_img, target_mask)

        if red_ratio < CFG.TARGET_RED_RATIO_MIN:
            if getattr(CFG, "TARGET_DEBUG_LOG_ENABLED", False):
                print(
                    f"[TARGET DEBUG] {vehicle}/{camera_name} "
                    f"seg_pixels={best_area}, red_ratio={red_ratio:.3f} "
                    f"-> rejected by RGB"
                )
            return False, red_ratio

    # segmentation 기반 confidence
    # 30픽셀 이상이면 확실한 target으로 취급
    confidence = min(1.0, best_area / 30.0)

    # 너무 작은 점도 segmentation으로 redball_01이면 최소 confidence 부여
    if confidence < 0.20:
        confidence = 0.20

    threshold = getattr(CFG, "TARGET_CONF_THRESHOLD", 0.30)

    if getattr(CFG, "TARGET_DEBUG_LOG_ENABLED", False):
        print(
            f"[TARGET DEBUG] {vehicle}/{camera_name} "
            f"seg_pixels={best_area}, red_ratio={red_ratio:.3f}, "
            f"conf={confidence:.3f}"
        )

    if confidence < threshold:
        if getattr(CFG, "TARGET_DEBUG_LOG_ENABLED", False):
            print(
                f"[TARGET DEBUG] {vehicle}/{camera_name} "
                f"conf={confidence:.3f} -> rejected by low confidence"
            )
        return False, confidence

    return True, confidence


def check_target_with_cameras(
    client: airsim.MultirotorClient,
    vehicle: str,
    target_detection_active: bool
) -> Tuple[bool, str, float]:

    if not target_detection_active:
        return False, "", 0.0

    cameras = []

    if CFG.TARGET_USE_BOTTOM_CAMERA and CFG.USE_BOTTOM_CAMERA:
        cameras.append(CFG.BOTTOM_CAMERA)

    if CFG.TARGET_USE_FRONT_CAMERA:
        cameras.append(CFG.FRONT_CAMERA)

    best_found = False
    best_camera = ""
    best_conf = 0.0

    for cam in cameras:
        found, conf = detect_target_sphere_by_segmentation(
            client,
            vehicle,
            cam
        )

        if found and conf > best_conf:
            best_found = True
            best_camera = cam
            best_conf = conf

    return best_found, best_camera, best_conf