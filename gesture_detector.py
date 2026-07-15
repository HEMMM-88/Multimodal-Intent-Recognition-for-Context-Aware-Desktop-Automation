"""
gesture_detector.py
Core gesture detection for a compact, human-friendly gesture set.
"""

import math


def _tip_ids():
    """MediaPipe finger tip landmark indices."""
    return [4, 8, 12, 16, 20]


def _dist(p1, p2):
    return math.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2)


def _finger_states(landmarks) -> dict:
    """
    Returns a dict of which fingers are 'up' (extended).
    Keys: thumb, index, middle, ring, pinky
    """
    lm = landmarks
    tips = _tip_ids()
    fingers = {}

    # Thumb: compare x positions (left vs right hand heuristic)
    thumb_tip = lm[tips[0]]
    thumb_ip = lm[tips[0] - 1]
    wrist = lm[0]
    if wrist.x < lm[17].x:  # right hand
        fingers["thumb"] = thumb_tip.x < thumb_ip.x
    else:  # left hand
        fingers["thumb"] = thumb_tip.x > thumb_ip.x

    # Other fingers: tip y < pip y means finger is up
    names = ["index", "middle", "ring", "pinky"]
    for i, name in enumerate(names):
        tip = lm[tips[i + 1]]
        pip = lm[tips[i + 1] - 2]
        fingers[name] = tip.y < pip.y

    return fingers


def detect_gesture(landmarks, return_details: bool = False):
    """
    Detect compact core gestures and optionally return confidence + metadata.

    Core gestures:
      - open_palm
      - pointing_up
      - pinch
      - two_finger_tap
      - three_fingers_up
      - thumbs_up
      - thumbs_down
      - closed_fist
      - none
    """
    f = _finger_states(landmarks)
    thumb = f["thumb"]
    index = f["index"]
    middle = f["middle"]
    ring = f["ring"]
    pinky = f["pinky"]

    pinch_dist = _dist(landmarks[4], landmarks[8])
    two_finger_dist = _dist(landmarks[8], landmarks[12])
    palm_span = _dist(landmarks[5], landmarks[17])  # robust hand-size proxy

    # Adaptive thresholds improve reliability across camera distance.
    pinch_threshold = max(0.03, min(0.1, palm_span * 0.5))
    two_finger_threshold = max(0.024, min(0.08, palm_span * 0.4))

    # Keep confidence usable even for farther/smaller hands.
    size_conf = max(0.35, min(1.0, (palm_span - 0.04) / 0.09))
    pinch_conf = max(0.0, 1.0 - (pinch_dist / pinch_threshold)) * size_conf
    two_finger_conf = max(0.0, 1.0 - (two_finger_dist / two_finger_threshold)) * size_conf

    gesture = "none"
    confidence = 0.0

    # Pointing: only index is up (thumb tolerant for comfort).
    if index and not middle and not ring and not pinky:
        gesture = "pointing_up"
        confidence = 0.92 * size_conf

    # Thumb-only gestures for volume control.
    elif thumb and not index and not middle and not ring and not pinky:
        # Relative to wrist, thumb tip higher => thumbs up.
        if landmarks[4].y < landmarks[0].y:
            gesture = "thumbs_up"
        else:
            gesture = "thumbs_down"
        confidence = 0.88 * size_conf

    # Pinch: thumb + index close. Allow one trailing finger to be noisy/up for robustness.
    elif pinch_dist < pinch_threshold and not (ring and pinky):
        gesture = "pinch"
        confidence = pinch_conf

    # Two-finger tap: index + middle up and close, with ring/pinky down.
    elif index and middle and not ring and not pinky and two_finger_dist < two_finger_threshold:
        gesture = "two_finger_tap"
        confidence = two_finger_conf

    # Three-finger mode: index + middle + ring up, pinky down.
    elif index and middle and ring and not pinky:
        gesture = "three_fingers_up"
        confidence = 0.9 * size_conf

    # Open palm: all non-thumb fingers up (thumb tolerant for comfort).
    elif index and middle and ring and pinky:
        gesture = "open_palm"
        confidence = (0.95 if thumb else 0.85) * size_conf

    # Closed fist: all fingers down.
    elif not any([thumb, index, middle, ring, pinky]):
        gesture = "closed_fist"
        confidence = 0.95 * size_conf

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
