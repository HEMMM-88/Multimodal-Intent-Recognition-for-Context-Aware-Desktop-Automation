"""
gesture_detector.py
Core gesture detection using MediaPipe hand landmarks.
"""

import math


def _tip_ids():
    return [4, 8, 12, 16, 20]


def _dist(p1, p2):
    return math.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2)


def _finger_states(landmarks) -> dict:
    lm = landmarks
    tips = _tip_ids()
    fingers = {}

    thumb_tip = lm[4]
    thumb_ip = lm[3]
    thumb_mcp = lm[2]
    wrist = lm[0]

    palm_span = _dist(lm[5], lm[17])
    thumb_vertical_diff = thumb_mcp.y - thumb_tip.y
    thumb_extension = _dist(thumb_tip, wrist) > _dist(thumb_ip, wrist) * 1.08
    thumb_margin = max(0.008, palm_span * 0.10)

    fingers["thumb_is_up"] = (
        thumb_extension and thumb_vertical_diff > thumb_margin
    )
    fingers["thumb_is_down"] = (
        thumb_extension and thumb_vertical_diff < -thumb_margin
    )
    fingers["thumb"] = thumb_extension

    for i, name in enumerate(["index", "middle", "ring", "pinky"]):
        tip_id = tips[i + 1]
        tip = lm[tip_id]
        pip = lm[tip_id - 2]
        mcp = lm[tip_id - 3]

        tip_above_pip = tip.y < pip.y
        tip_mcp_dist = _dist(tip, mcp)
        pip_mcp_dist = _dist(pip, mcp)

        fingers[name] = (
            tip_above_pip and tip_mcp_dist > pip_mcp_dist * 0.62
        )

    return fingers


def detect_gesture(landmarks, return_details: bool = False):
    f = _finger_states(landmarks)

    thumb = f["thumb"]
    thumb_is_up = f["thumb_is_up"]
    thumb_is_down = f["thumb_is_down"]
    index = f["index"]
    middle = f["middle"]
    ring = f["ring"]
    pinky = f["pinky"]

    palm_span = _dist(landmarks[5], landmarks[17])
    pinch_dist = _dist(landmarks[4], landmarks[8])
    two_finger_dist = _dist(landmarks[8], landmarks[12])

    size_conf = max(0.10, min(1.0, palm_span / 0.10))
    pinch_threshold = max(0.015, palm_span * 0.45)
    two_finger_threshold = max(0.015, palm_span * 0.60)
    pinch_conf = max(
        0.0,
        min(1.0, 1.0 - pinch_dist / pinch_threshold),
    )

    thumb_only = not index and not middle and not ring and not pinky

    gesture = "none"
    confidence = 0.0

    # Thumb gestures must be checked only when the other fingers are folded.
    if thumb_only and thumb_is_up:
        gesture = "thumbs_up"
        confidence = 0.90 * size_conf

    elif thumb_only and thumb_is_down:
        gesture = "thumbs_down"
        confidence = 0.90 * size_conf

    elif index and middle and ring and pinky:
        gesture = "open_palm"
        confidence = (0.97 if thumb else 0.90) * size_conf

    elif index and not middle and not ring and not pinky:
        gesture = "pointing_up"
        confidence = (0.95 if not thumb else 0.88) * size_conf

    elif pinch_dist < pinch_threshold:
        gesture = "pinch"
        confidence = pinch_conf

    elif index and middle and not ring and not pinky:
        gesture = "two_finger_tap"
        confidence = 0.88 * size_conf

    elif index and middle and ring and not pinky:
        gesture = "three_fingers_up"
        confidence = 0.92 * size_conf

    elif not thumb and not index and not middle and not ring and not pinky:
        gesture = "closed_fist"
        confidence = 0.96 * size_conf

    details = {
        "index_tip": (landmarks[8].x, landmarks[8].y),
        "palm_center": (landmarks[9].x, landmarks[9].y),
        "pinch_dist": pinch_dist,
        "two_finger_dist": two_finger_dist,
        "palm_span": palm_span,
        "size_conf": size_conf,
        "pinch_threshold": pinch_threshold,
        "two_finger_threshold": two_finger_threshold,
        "fingers": f,
    }

    if return_details:
        return gesture, confidence, details

    return gesture
