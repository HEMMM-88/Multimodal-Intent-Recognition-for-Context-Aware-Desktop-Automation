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
    lm   = landmarks
    tips = _tip_ids()
    fingers = {}

    thumb_tip = lm[tips[0]]
    thumb_ip  = lm[tips[0] - 1]
    wrist     = lm[0]
    if wrist.x < lm[17].x:
        fingers["thumb"] = thumb_tip.x < thumb_ip.x
    else:
        fingers["thumb"] = thumb_tip.x > thumb_ip.x

    for i, name in enumerate(["index", "middle", "ring", "pinky"]):
        tip_id        = tips[i + 1]
        tip           = lm[tip_id]
        pip           = lm[tip_id - 2]
        mcp           = lm[tip_id - 3]
        tip_above_pip = tip.y < pip.y
        tip_mcp_dist  = math.sqrt((tip.x - mcp.x) ** 2 + (tip.y - mcp.y) ** 2)
        pip_mcp_dist  = math.sqrt((pip.x - mcp.x) ** 2 + (pip.y - mcp.y) ** 2)
        tip_extended  = tip_mcp_dist > pip_mcp_dist * 0.62
        fingers[name] = tip_above_pip and tip_extended

    return fingers


def detect_gesture(landmarks, return_details: bool = False):
    """
    Detect a hand gesture from MediaPipe landmarks.

    Returns (gesture, confidence, details) when return_details=True,
    otherwise just the gesture string.

    Gestures: open_palm, pointing_up, pinch, two_finger_tap,
              three_fingers_up, thumbs_up, thumbs_down, closed_fist, none.
    """
    f      = _finger_states(landmarks)
    thumb  = f["thumb"]
    index  = f["index"]
    middle = f["middle"]
    ring   = f["ring"]
    pinky  = f["pinky"]

    pinch_dist      = _dist(landmarks[4], landmarks[8])
    two_finger_dist = _dist(landmarks[8], landmarks[12])
    palm_span       = _dist(landmarks[5], landmarks[17])

    # All thresholds scale with palm_span for distance-invariant detection.
    pinch_threshold      = max(0.010, min(0.14, palm_span * 0.72))
    two_finger_threshold = max(0.008, min(0.10, palm_span * 0.60))
    size_conf  = max(0.10, min(1.0, (palm_span - 0.006) / 0.09))
    pinch_conf = max(0.0, 1.0 - (pinch_dist / pinch_threshold)) * size_conf

    gesture    = "none"
    confidence = 0.0

    if index and middle and ring and pinky:
        gesture    = "open_palm"
        confidence = (0.97 if thumb else 0.90) * size_conf

    elif index and not middle and not ring and not pinky:
        gesture    = "pointing_up"
        confidence = (0.95 if not thumb else 0.88) * size_conf

    elif thumb and not index and not middle and not ring and not pinky:
        thumb_vertical_diff = landmarks[2].y - landmarks[4].y
        margin = max(0.006, palm_span * 0.10)
        if thumb_vertical_diff > margin:
            gesture = "thumbs_up"
        elif thumb_vertical_diff < -margin:
            gesture = "thumbs_down"
        confidence = 0.90 * size_conf

    elif pinch_dist < pinch_threshold:
        gesture    = "pinch"
        confidence = pinch_conf

    elif index and middle and not ring and not pinky:
        gesture    = "two_finger_tap"
        confidence = 0.88 * size_conf

    elif index and middle and ring and not pinky:
        gesture    = "three_fingers_up"
        confidence = 0.92 * size_conf

    elif not index and not middle and not ring and not pinky:
        gesture    = "closed_fist"
        confidence = 0.96 * size_conf

    details = {
        "index_tip":            (landmarks[8].x, landmarks[8].y),
        "palm_center":          (landmarks[9].x, landmarks[9].y),
        "pinch_dist":           pinch_dist,
        "two_finger_dist":      two_finger_dist,
        "palm_span":            palm_span,
        "size_conf":            size_conf,
        "pinch_threshold":      pinch_threshold,
        "two_finger_threshold": two_finger_threshold,
        "fingers":              f,
    }

    if return_details:
        return gesture, confidence, details
    return gesture
