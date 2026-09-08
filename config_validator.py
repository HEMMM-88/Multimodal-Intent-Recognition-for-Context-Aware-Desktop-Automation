"""
config_validator.py
Validates config.yaml at startup and reports all problems at once instead of
letting the system silently use wrong defaults.

Design goals:
- No external dependencies (stdlib only + PyYAML which is already required).
- Fail-soft: returns a list of warnings and a list of errors; the caller decides
  whether errors are fatal.
- Catches the most common mistakes: wrong types, out-of-range values, unknown
  context names, and missing required keys.
"""

from __future__ import annotations
import logging

logger = logging.getLogger(__name__)

# ── Valid context bucket names (must match intent_classifier.RULES keys) ──────
VALID_CONTEXTS = frozenset({
    "default", "browser", "video_player", "code_editor", "document_editor", "terminal"
})

# ── Valid gesture names (from gesture_detector) ───────────────────────────────
VALID_GESTURES = frozenset({
    "open_palm", "pointing_up", "pinch", "two_finger_tap",
    "three_fingers_up", "thumbs_up", "thumbs_down", "closed_fist", "none",
    # Synthetic trigger keys used by main.py
    "open_palm_horizontal", "primary_click_swipe",
    "two_hand_swipe", "two_hand_swipe_vertical",
    "two_hand_pointing_up", "two_hand_closed_fist",
})

# ── Schema: (key_path, type, min, max, required) ──────────────────────────────
# key_path uses "." as separator, e.g. "settings.control.debounce_seconds"
_FLOAT_RULES: list[tuple[str, float, float, bool]] = [
    ("settings.detection_confidence",                  0.05, 1.0,  True),
    ("settings.tracking_confidence",                   0.05, 1.0,  True),
    ("settings.startup_delay",                         0.0,  30.0, False),
    ("settings.app_refresh_seconds",                   0.05, 5.0,  False),
    ("settings.detection_upscale",                     1.0,  2.0,  False),
    ("settings.control.gesture_confidence_threshold",  0.10, 1.0,  False),
    ("settings.control.gesture_stability_seconds",     0.0,  2.0,  False),
    ("settings.control.min_palm_span",                 0.005, 0.20, False),
    ("settings.control.debounce_seconds",              0.05, 2.0,  False),
    ("settings.control.pause_toggle_debounce",         0.1,  5.0,  False),
    ("settings.control.pinch_click_hold_seconds",      0.0,  2.0,  False),
    ("settings.control.pinch_drag_hold_seconds",       0.05, 3.0,  False),
    ("settings.control.pinch_break_grace_seconds",     0.0,  1.0,  False),
    ("settings.control.drag_release_grace_seconds",    0.0,  2.0,  False),
    ("settings.control.two_finger_tap_stable_seconds", 0.0,  2.0,  False),
    ("settings.control.scroll_mode_inactivity_seconds",0.1,  10.0, False),
    ("settings.control.scroll_motion_threshold",       0.001,0.20, False),
    ("settings.control.modifier_entry_seconds",        0.0,  2.0,  False),
    ("settings.control.two_hand_swipe_threshold",      0.01, 0.50, False),
    ("settings.control.two_hand_swipe_debounce_seconds",0.1, 5.0,  False),
    ("settings.mouse_control.ema_alpha",               0.05, 1.0,  False),
    ("settings.mouse_control.drag_ema_alpha",          0.05, 1.0,  False),
    ("settings.mouse_control.deadzone_px",             0.0,  50.0, False),
]

_INT_RULES: list[tuple[str, int, int, bool]] = [
    ("settings.camera_index",                 0,  9,    True),
    ("settings.capture_width",                160, 4096, False),
    ("settings.capture_height",               120, 2160, False),
    ("settings.capture_fps",                  1,  120,  False),
    ("settings.control.two_hand_settle_frames",0,  30,   False),
]

_BOOL_KEYS = frozenset({
    "settings.show_overlay",
    "settings.control.two_hand_enabled",
    "settings.mouse_control.reverse_horizontal_motion",
    "settings.mouse_control.reverse_vertical_motion",
    "settings.mouse_control.mirror_sideways",
    "settings.mouse_control.invert_x",
    "settings.mouse_control.invert_y",
    "settings.control.swap_handedness",
})


def _get_nested(d: dict, path: str):
    """Traverse a dotted key path. Returns (value, found: bool)."""
    keys = path.split(".")
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return None, False
        cur = cur[k]
    return cur, True


def validate_config(config: dict) -> tuple[list[str], list[str]]:
    """
    Validate a loaded config dict.

    Returns:
        (warnings, errors) — warnings are suggestions, errors indicate broken behavior.
        An empty errors list means the config is safe to use.
    """
    warnings: list[str] = []
    errors:   list[str] = []

    # ── Float range checks ────────────────────────────────────────────────
    for path, lo, hi, required in _FLOAT_RULES:
        val, found = _get_nested(config, path)
        if not found:
            if required:
                errors.append(f"MISSING required key: {path}")
            continue
        try:
            fval = float(val)
        except (TypeError, ValueError):
            errors.append(f"BAD TYPE: {path} = {val!r} (expected a number)")
            continue
        if not (lo <= fval <= hi):
            errors.append(f"OUT OF RANGE: {path} = {fval} (expected {lo}–{hi})")

    # ── Integer range checks ───────────────────────────────────────────────
    for path, lo, hi, required in _INT_RULES:
        val, found = _get_nested(config, path)
        if not found:
            if required:
                errors.append(f"MISSING required key: {path}")
            continue
        try:
            ival = int(val)
        except (TypeError, ValueError):
            errors.append(f"BAD TYPE: {path} = {val!r} (expected an integer)")
            continue
        if not (lo <= ival <= hi):
            errors.append(f"OUT OF RANGE: {path} = {ival} (expected {lo}–{hi})")

    # ── Boolean checks ─────────────────────────────────────────────────────
    for path in _BOOL_KEYS:
        val, found = _get_nested(config, path)
        if found and not isinstance(val, bool):
            warnings.append(f"EXPECTED bool: {path} = {val!r} (use true/false in YAML)")

    # ── primary_click_gesture ─────────────────────────────────────────────
    pcg, found = _get_nested(config, "settings.control.primary_click_gesture")
    if found and pcg not in ("pointing_up", "pinch"):
        errors.append(
            f"INVALID primary_click_gesture: {pcg!r} — must be 'pointing_up' or 'pinch'"
        )

    # ── primary_hand ──────────────────────────────────────────────────────
    ph, found = _get_nested(config, "settings.control.primary_hand")
    if found and ph not in ("Left", "Right"):
        errors.append(f"INVALID primary_hand: {ph!r} — must be 'Left' or 'Right'")

    # ── Apps section: check for duplicate window title entries ────────────
    apps = config.get("apps", {})
    all_titles: list[str] = []
    for app_key, app_data in apps.items():
        if not isinstance(app_data, dict):
            errors.append(f"apps.{app_key} must be a mapping, got {type(app_data).__name__}")
            continue
        titles = app_data.get("window_titles", [])
        if not isinstance(titles, list):
            errors.append(f"apps.{app_key}.window_titles must be a list")
            continue
        for t in titles:
            if t in all_titles:
                warnings.append(f"DUPLICATE window title '{t}' in apps.{app_key} — earlier entry takes priority")
            all_titles.append(t)

    # ── context_rules: validate context names and gesture names ───────────
    context_rules = config.get("context_rules", {})
    for ctx, gestures in context_rules.items():
        if ctx not in VALID_CONTEXTS:
            warnings.append(
                f"context_rules.{ctx} — '{ctx}' is not a known context "
                f"(known: {', '.join(sorted(VALID_CONTEXTS))}). Rule will be ignored."
            )
        if not isinstance(gestures, dict):
            continue
        for gest, spec in gestures.items():
            if gest not in VALID_GESTURES:
                warnings.append(
                    f"context_rules.{ctx}.{gest} — '{gest}' is not a known gesture/trigger. Rule will be ignored."
                )
            if isinstance(spec, dict):
                if "intent" not in spec:
                    errors.append(f"context_rules.{ctx}.{gest} is missing required 'intent' key")
                if "label" not in spec:
                    errors.append(f"context_rules.{ctx}.{gest} is missing required 'label' key")

    # ── combo_rules: validate context and gesture names ───────────────────
    combo_rules = config.get("combo_rules", {})
    for ctx, gestures in combo_rules.items():
        if ctx not in VALID_CONTEXTS:
            warnings.append(
                f"combo_rules.{ctx} — '{ctx}' is not a known context. Rule will be ignored."
            )
        if not isinstance(gestures, dict):
            continue
        for gest, spec in gestures.items():
            if gest not in VALID_GESTURES:
                warnings.append(
                    f"combo_rules.{ctx}.{gest} — '{gest}' is not a known gesture. Rule will be ignored."
                )
            if isinstance(spec, dict):
                if "intent" not in spec:
                    errors.append(f"combo_rules.{ctx}.{gest} is missing required 'intent' key")
                if "label" not in spec:
                    errors.append(f"combo_rules.{ctx}.{gest} is missing required 'label' key")

    return warnings, errors


def validate_and_report(config: dict, fatal_on_error: bool = True) -> bool:
    """
    Run validation and log all findings.

    Returns True if the config is safe to use.
    If fatal_on_error is True and there are errors, calls sys.exit(1).
    """
    import sys
    warnings, errors = validate_config(config)

    for w in warnings:
        logger.warning("[config] %s", w)

    for e in errors:
        logger.error("[config] %s", e)

    if errors:
        logger.error(
            "[config] %d error(s) found — fix config.yaml before running. "
            "Run with --skip-validation to bypass (not recommended).",
            len(errors),
        )
        if fatal_on_error:
            sys.exit(1)
        return False

    if warnings:
        logger.info("[config] Validation passed with %d warning(s).", len(warnings))
    else:
        logger.info("[config] Validation passed — config looks good.")

    return True
