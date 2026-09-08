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

    Deliberately relaxed thresholds — a finger only needs to be *mostly*
    extended, not fully straight. This means natural, low-effort hand
    positions work instead of requiring rigid fully-splayed poses.
    The 0.75 margin (was 0.85) allows fingers to be slightly curled
    and still count as 'up', which is much less tiring to hold.
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

    names = ["index", "middle", "ring", "pinky"]
    for i, name in enumerate(names):
        tip_id = tips[i + 1]
        tip = lm[tip_id]
        pip = lm[tip_id - 2]
        mcp = lm[tip_id - 3]

        # Primary: tip above pip (relaxed — tip just needs to be higher, not strictly)
        tip_above_pip = tip.y < pip.y

        # Secondary: loose extension check.
        # At long range the landmark coordinates are noisier and slightly
        # compressed, so we drop the ratio to 0.62 (was 0.72). A finger
        # only needs to be ~60% as extended as its MCP–PIP segment to count
        # as "up" — much less strain, and still rejects a clearly closed fist.
        tip_mcp_dist = math.sqrt((tip.x - mcp.x)**2 + (tip.y - mcp.y)**2)
        pip_mcp_dist = math.sqrt((pip.x - mcp.x)**2 + (pip.y - mcp.y)**2)
        tip_extended = tip_mcp_dist > pip_mcp_dist * 0.62

        fingers[name] = tip_above_pip and tip_extended

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

    # ── Long-range adaptive thresholds ────────────────────────────────────
    # palm_span shrinks proportionally with distance.  All thresholds below
    # scale with palm_span so detection stays reliable whether the hand is
    # 30 cm or 3 m from the lens.
    #
    # Typical palm_span values (normalised 0-1 in a 1280×720 frame):
    #   ~0.20   very close  (arm fully extended toward camera)
    #   ~0.10   normal desk distance (~60 cm)
    #   ~0.04   far         (~1.5 m)
    #   ~0.015  very far    (~3 m, outer detection limit)
    #
    # The floor values (first arg to max()) are the minimum span that still
    # makes geometric sense; the ceil values (second arg to min()) cap the
    # threshold so close-up hands don't trigger on coarse movements.

    # Pinch: thumb tip ↔ index tip must be closer than this to count.
    # Scale factor 0.72 (was 0.65) accepts a slightly wider gap — at long
    # range finger tips blur together and the gap overestimates.
    pinch_threshold = max(0.010, min(0.14, palm_span * 0.72))

    # Two-finger tap: index tip ↔ middle tip distance.
    two_finger_threshold = max(0.008, min(0.10, palm_span * 0.60))

    # ── Long-range size confidence ─────────────────────────────────────────
    # Maps palm_span to a [0, 1] confidence multiplier.
    #
    # Old formula: (palm_span - 0.015) / 0.12  → floor 0.20
    #   – At palm_span=0.04 (1.5 m) this gave only (0.04-0.015)/0.12 ≈ 0.21
    #   – At palm_span=0.015 (3 m) this hit the floor of 0.20
    #
    # New formula: (palm_span - 0.006) / 0.09  → floor 0.10
    #   – At palm_span=0.015 (3 m) → (0.015-0.006)/0.09 ≈ 0.10 (still usable)
    #   – At palm_span=0.04  (1.5m) → (0.040-0.006)/0.09 ≈ 0.38
    #   – At palm_span=0.10  (60cm) → (0.100-0.006)/0.09 ≈ 1.00 (capped)
    #
    # The lower floor (0.10 vs 0.20) lets very distant hands get through the
    # confidence gate when the threshold is also relaxed in config.yaml.
    size_conf = max(0.10, min(1.0, (palm_span - 0.006) / 0.09))
    pinch_conf = max(0.0, 1.0 - (pinch_dist / pinch_threshold)) * size_conf
    two_finger_conf = max(0.0, 1.0 - (two_finger_dist / two_finger_threshold)) * size_conf

    gesture = "none"
    confidence = 0.0

    # Open palm: most common gesture — cursor movement.
    # Accept thumb-down too (relaxed hand). Only needs 3+ fingers up
    # so a slightly curled ring/pinky doesn't break it.
    # Checked FIRST so a natural resting-open hand always moves cursor.
    if index and middle and ring and pinky:
        gesture = "open_palm"
        confidence = (0.97 if thumb else 0.90) * size_conf

    # Pointing: index only up. Thumb tolerant — tucking thumb is unnatural at rest.
    elif index and not middle and not ring and not pinky:
        gesture = "pointing_up"
        confidence = (0.95 if not thumb else 0.88) * size_conf

    # Thumb-only gestures — relaxed fist with thumb out is a natural pose.
    elif thumb and not index and not middle and not ring and not pinky:
        thumb_tip_y = landmarks[4].y
        thumb_mcp_y = landmarks[2].y
        thumb_vertical_diff = thumb_mcp_y - thumb_tip_y
        # Scale margin with hand size. Lower floor (0.006 was 0.012) so
        # thumbs_up/down still fires at long range where the hand is tiny.
        margin = max(0.006, palm_span * 0.10)
        if thumb_vertical_diff > margin:
            gesture = "thumbs_up"
        elif thumb_vertical_diff < -margin:
            gesture = "thumbs_down"
        else:
            gesture = "none"
        confidence = 0.90 * size_conf

    # Pinch: thumb + index close. Ring/pinky can be relaxed/up — don't penalize.
    elif pinch_dist < pinch_threshold:
        gesture = "pinch"
        confidence = pinch_conf

    # Two-finger tap: index + middle up. Ring/pinky don't need to be fully tucked —
    # just not extended past pip. Wider two_finger_dist threshold for relaxed fingers.
    elif index and middle and not ring and not pinky:
        gesture = "two_finger_tap"
        confidence = 0.88 * size_conf  # don't require fingers to be close together

    # Three-finger scroll: index + middle + ring up. Pinky relaxed/down.
    elif index and middle and ring and not pinky:
        gesture = "three_fingers_up"
        confidence = 0.92 * size_conf

    # Closed fist: all fingers down. Thumb tolerant.
    elif not index and not middle and not ring and not pinky:
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
