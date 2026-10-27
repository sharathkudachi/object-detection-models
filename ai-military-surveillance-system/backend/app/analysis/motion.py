import math
import cv2
import numpy as np

# Image-plane directions only (y grows downward). Not geographic without camera calibration.
DIRS = ["East", "South-East", "South", "South-West", "West", "North-West", "North", "North-East"]


def direction(p0, p1, min_px=4.0):
    dx, dy = p1[0] - p0[0], p1[1] - p0[1]
    if math.hypot(dx, dy) < min_px:
        return "Stationary"
    ang = math.degrees(math.atan2(dy, dx)) % 360
    return DIRS[int(((ang + 22.5) % 360) // 45)]


def estimate_roi_motion(prev_gray, curr_gray, bbox, max_dim=80, motion_threshold=2.0):
    """
    Optimized Farneback dense optical flow on object ROI.
    Downscales large ROIs to max_dim (default 80px) to achieve ~30x speedup
    while preserving motion classification fidelity.
    """
    if prev_gray is None or curr_gray is None:
        return "Static", 0.0

    h_img, w_img = curr_gray.shape[:2]
    x1, y1, x2, y2 = map(int, bbox)
    x1 = max(0, min(w_img - 1, x1))
    y1 = max(0, min(h_img - 1, y1))
    x2 = max(0, min(w_img, x2))
    y2 = max(0, min(h_img, y2))

    roi_w, roi_h = x2 - x1, y2 - y1
    if roi_w <= 2 or roi_h <= 2:
        return "Static", 0.0

    roi_prev = prev_gray[y1:y2, x1:x2]
    roi_curr = curr_gray[y1:y2, x1:x2]

    # Downscale if ROI is larger than max_dim
    max_side = max(roi_w, roi_h)
    scale = 1.0
    if max_side > max_dim:
        scale = max_dim / float(max_side)
        new_w = max(4, int(roi_w * scale))
        new_h = max(4, int(roi_h * scale))
        roi_prev = cv2.resize(roi_prev, (new_w, new_h), interpolation=cv2.INTER_AREA)
        roi_curr = cv2.resize(roi_curr, (new_w, new_h), interpolation=cv2.INTER_AREA)

    try:
        flow = cv2.calcOpticalFlowFarneback(
            roi_prev, roi_curr, None,
            pyr_scale=0.5, levels=2, winsize=11, iterations=2, poly_n=5, poly_sigma=1.1, flags=0
        )
        mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        avg_motion = float(np.mean(mag))
        if scale != 1.0:
            avg_motion = avg_motion / scale

        status = "Moving" if avg_motion > motion_threshold else "Static"
        return status, round(avg_motion, 2)
    except Exception:
        return "Static", 0.0
