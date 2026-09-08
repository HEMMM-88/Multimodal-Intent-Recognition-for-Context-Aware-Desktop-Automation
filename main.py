"""
main.py
Gesture Control System - main entry point.

This version intentionally uses a compact, human-friendly gesture set:
  1) open_palm        -> cursor movement
  2) pointing_up      -> left click (hold), drag on longer hold
  3) two_finger_tap   -> right click (150ms stable)
  4) three_fingers_up -> scroll mode
  5) thumbs_up/down   -> volume up/down
  6) closed_fist      -> pause/resume toggle

Two-hand improvements (all 8 fixes applied):
  FIX-1: Modifier hand requires a minimum hold time before activating
          (no more accidental combos from a resting second hand).
  FIX-2: Cursor movement is suppressed while a two-hand swipe is detected
          so the cursor doesn't jump during tab/window switches.
  FIX-3: Horizontal and vertical two-hand swipe states are fully separated;
          the y-delta bug (modifier_palm_x used as a y value) is removed.
  FIX-4: HUD shows "COMBO READY" when the modifier is active and the
          primary hand is holding a combo-eligible gesture.
  FIX-5: Window-management two-hand gestures are noted in comments; users
          are advised to run --no-overlay for cleanest focus behavior.
  FIX-6: Directional swipe debounce uses separate keys per direction
          ("two_hand_swipe_right" / "two_hand_swipe_left") so a quick
          reverse swipe doesn't get swallowed by the shared clock.
  FIX-7: Handedness is resolved by wrist X position (left wrist = left hand)
          rather than MediaPipe's label, which is unreliable after mirroring.
          swap_handedness is kept as an escape hatch but defaults to false.
  FIX-8: A "settle gate" requires both hands to be seen for at least
          two_hand_settle_frames consecutive frames before swipe detection
          starts, preventing accidental swipes as hands enter/leave frame.
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from logging.handlers import RotatingFileHandler

import cv2
import mediapipe as mp
import yaml

from gesture_detector import detect_gesture
from action_executor import execute_action
from app_detector import get_active_window_info, match_app_config
from events import GestureEvent, ContextState
from intent_classifier import (
    classify as classify_intent,
    classify_combo,
    load_rules_from_config,
    load_combo_rules_from_config,
    RULES,
)
from config_validator import validate_and_report
from gesture_logger import GestureLogger

try:
    import pyautogui
    PYAUTOGUI_AVAILABLE = True
except ImportError:
    pyautogui = None
    PYAUTOGUI_AVAILABLE = False


LOG_FILE = Path(__file__).resolve().parent / "gesture_control.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        RotatingFileHandler(LOG_FILE, maxBytes=128 * 1024, backupCount=2, encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

COLORS = {
    "green":  (50, 220, 50),
    "yellow": (30, 220, 220),
    "red":    (20, 20, 255),
    "blue":   (255, 160, 30),
    "white":  (245, 245, 245),
    "orange": (30, 165, 255),   # used for COMBO READY indicator
    "bg":     (22, 22, 22),
}

# Gestures that are eligible for two-hand combo resolution.
_COMBO_ELIGIBLE = frozenset({"thumbs_up", "thumbs_down", "two_finger_tap", "three_fingers_up"})


def load_config(config_path: str) -> dict:
    path = Path(config_path)
    if not path.exists():
        logger.error("Config file not found: %s", config_path)
        sys.exit(1)
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data


def _debounced(name: str, clock: dict[str, float], interval: float, now: float) -> bool:
    """Generic gesture debounce helper."""
    last = clock.get(name, 0.0)
    if now - last < interval:
        return False
    clock[name] = now
    return True


# Known context buckets and their keyword fallbacks, used only when the
# active window didn't match a config.yaml `apps.<key>` entry.
_CONTEXT_FALLBACK_TOKENS = {
    "browser":         ("chrome", "firefox", "edge", "brave", "opera", "browser"),
    "video_player":    ("vlc", "youtube", "netflix", "potplayer", "mpc", "media player", "video"),
    "code_editor":     ("vscode", "visual studio", "code", "pycharm", "intellij", "sublime", "notepad++"),
    "document_editor": ("word", "winword", "notepad", "wordpad", "acrobat", "foxit", "pdf", "libreoffice writer"),
    "terminal":        ("cmd.exe", "powershell", "windows terminal", "wt.exe", "conemu", "bash", "wsl"),
}

# FIX-6: Each direction now has its own debounce key so an immediate reverse
# swipe is not eaten by the shared cooldown from the previous direction.
TWO_HAND_SWIPE_ACTIONS = {
    "window_switch":          ("key:alt+tab",        "key:alt+shift+tab",   "NEXT_WINDOW",   "PREV_WINDOW"),
    "browser_tab_switch":     ("key:ctrl+tab",        "key:ctrl+shift+tab",  "NEXT_TAB",      "PREV_TAB"),
    "virtual_desktop_switch": ("key:ctrl+win+right",  "key:ctrl+win+left",   "NEXT_DESKTOP",  "PREV_DESKTOP"),
    "code_tab_switch":        ("key:ctrl+tab",        "key:ctrl+shift+tab",  "NEXT_EDITOR",   "PREV_EDITOR"),
    "history_nav":            ("key:alt+right",       "key:alt+left",        "FORWARD",       "BACK"),
}

TWO_HAND_SWIPE_VERTICAL_ACTIONS = {
    "fast_scroll": ("scroll_up_fast", "scroll_down_fast", "FAST_SCROLL_UP",   "FAST_SCROLL_DOWN"),
    "zoom_inout":  ("key:ctrl+=",     "key:ctrl+-",        "ZOOM_IN",          "ZOOM_OUT"),
    "slide_nav":   ("key:pageup",     "key:pagedown",      "PREV_SLIDE",       "NEXT_SLIDE"),
    "media_seek":  ("key:shift+right","key:shift+left",    "SEEK_FWD_LARGE",   "SEEK_BACK_LARGE"),
}


def _classify_context(app_key: str | None, title: str, proc: str) -> str:
    """
    Resolve the active window into a context bucket.
    Priority: config app_key match > keyword fallback > "default".
    """
    if app_key in _CONTEXT_FALLBACK_TOKENS:
        return app_key
    text = f"{title} {proc}".lower()
    for context, tokens in _CONTEXT_FALLBACK_TOKENS.items():
        if any(t in text for t in tokens):
            return context
    return "default"


def _move_cursor_ema(
    pointer_norm: tuple[float, float],
    screen_size: tuple[int, int],
    ema_pos: tuple[float, float] | None,
    alpha: float,
    deadzone_px: float,
    reverse_horizontal: bool,
    reverse_vertical: bool,
) -> tuple[float, float]:
    """EMA-smoothed cursor movement for stable virtual mouse control."""
    x_norm, y_norm = pointer_norm
    if reverse_horizontal:
        x_norm = 1.0 - x_norm
    if reverse_vertical:
        y_norm = 1.0 - y_norm

    x_norm = max(0.0, min(1.0, x_norm))
    y_norm = max(0.0, min(1.0, y_norm))

    screen_w, screen_h = screen_size
    target_x = x_norm * (screen_w - 1)
    target_y = y_norm * (screen_h - 1)

    if ema_pos is None:
        smooth_x, smooth_y = target_x, target_y
    else:
        smooth_x = alpha * target_x + (1.0 - alpha) * ema_pos[0]
        smooth_y = alpha * target_y + (1.0 - alpha) * ema_pos[1]

    if ema_pos is None or abs(smooth_x - ema_pos[0]) >= deadzone_px or abs(smooth_y - ema_pos[1]) >= deadzone_px:
        pyautogui.moveTo(int(smooth_x), int(smooth_y))

    return smooth_x, smooth_y


def draw_overlay(
    frame,
    app_name: str,
    context: str,
    gesture_label: str,
    action_label: str,
    paused: bool,
    scroll_mode: bool,
    fps: float,
    modifier_gesture: str = "none",
    modifier_active: bool = False,      # FIX-1/4: True once the modifier hold timer has elapsed
    combo_ready: bool = False,          # FIX-4: True when modifier is active + primary is combo-eligible
):
    h, _ = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (390, h), COLORS["bg"], -1)
    cv2.addWeighted(overlay, 0.62, frame, 0.38, 0, frame)

    y = 30

    def put(txt, color=COLORS["white"], scale=0.58, thickness=1):
        nonlocal y
        cv2.putText(frame, txt, (12, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)
        y += int(30 * scale + 10)

    put("Gesture Control", COLORS["yellow"], 0.7, 2)
    put(f"App: {app_name}", COLORS["blue"])
    put(f"Context: {context}", COLORS["blue"])
    put(f"Gesture: {gesture_label}", COLORS["green"], 0.62, 2)

    # FIX-4: Show modifier hand status with graduated feedback.
    if modifier_gesture != "none":
        if combo_ready:
            # Modifier is locked in and primary gesture is combo-eligible — tell the user.
            put(">> COMBO READY <<", COLORS["orange"], 0.58, 2)
        elif modifier_active:
            put(f"Modifier: {modifier_gesture} (active)", COLORS["orange"], 0.52)
        else:
            # Modifier hand detected but still in the entry hold window.
            put(f"Modifier: {modifier_gesture} (holding...)", COLORS["white"], 0.50)

    put(f"Action: {action_label}", COLORS["yellow"])
    put(f"Scroll Mode: {'ON' if scroll_mode else 'OFF'}")
    put(f"Status: {'PAUSED' if paused else 'ACTIVE'}", COLORS["red"] if paused else COLORS["green"])
    put(f"FPS: {fps:.1f}", COLORS["white"])
    put("Q = Quit | P = Manual Pause", COLORS["white"], 0.5)


def print_gesture_reference(config_path: str = "config.yaml"):
    print(
        """
Core Gestures (6):
  open_palm        -> Cursor movement
  pointing_up      -> Left click (hold), drag on long hold
  two_finger_tap   -> Right click (stable >= 150ms)
  three_fingers_up -> Scroll mode (auto-exit after inactivity)
  thumbs_up/down   -> Volume control
  closed_fist      -> Pause/Resume
  none             -> Idle / no action
"""
    )

    try:
        config = load_config(config_path)
        load_rules_from_config(config)
    except Exception as exc:
        print(f"(Could not load '{config_path}' to show context-specific overrides: {exc})")
        return

    print("Context-Specific Overrides (live from config.yaml -> context_rules):")
    for context in sorted(RULES):
        if context == "default":
            continue
        print(f"  [{context}]")
        for gesture_key, (intent, action, label) in sorted(RULES[context].items()):
            action_str = f", action={action}" if action else ""
            print(f"    {gesture_key:22s} -> {label}  (intent={intent}{action_str})")
    print()


def run(config_path: str, no_overlay: bool = False, skip_validation: bool = False):
    config = load_config(config_path)

    # ── Config validation ──────────────────────────────────────────────────
    # Catches YAML typos and out-of-range values before they cause silent bugs.
    if not skip_validation:
        validate_and_report(config, fatal_on_error=True)
    load_rules_from_config(config)
    load_combo_rules_from_config(config)
    settings = config.get("settings", {})
    control  = settings.get("control", {})
    mouse_cfg = settings.get("mouse_control", {})

    camera_idx        = int(settings.get("camera_index", 0))
    det_conf          = float(settings.get("detection_confidence", 0.65))
    track_conf        = float(settings.get("tracking_confidence", 0.65))
    show_overlay      = bool(settings.get("show_overlay", True)) and not no_overlay
    startup_delay     = float(settings.get("startup_delay", 1.0))
    app_refresh_seconds = float(settings.get("app_refresh_seconds", 0.2))

    # Stability / usability
    conf_threshold            = float(control.get("gesture_confidence_threshold", 0.72))
    gesture_stability_seconds = float(control.get("gesture_stability_seconds", 0.08))
    min_palm_span             = float(control.get("min_palm_span", 0.075))
    debounce_seconds          = float(control.get("debounce_seconds", 0.25))
    pause_toggle_debounce     = float(control.get("pause_toggle_debounce", 0.8))
    pinch_click_hold          = float(control.get("pinch_click_hold_seconds", 0.2))
    pinch_drag_hold           = float(control.get("pinch_drag_hold_seconds", 0.45))
    pinch_break_grace         = float(control.get("pinch_break_grace_seconds", 0.12))
    drag_release_grace        = float(control.get("drag_release_grace_seconds", 0.18))
    primary_click_gesture     = str(control.get("primary_click_gesture", "pointing_up")).strip().lower()
    if primary_click_gesture not in {"pointing_up", "pinch"}:
        primary_click_gesture = "pointing_up"
    two_finger_stable            = float(control.get("two_finger_tap_stable_seconds", 0.15))
    scroll_inactivity_timeout    = float(control.get("scroll_mode_inactivity_seconds", 1.0))
    scroll_motion_threshold      = float(control.get("scroll_motion_threshold", 0.01))
    browser_swipe_threshold      = float(control.get("browser_pinch_swipe_threshold", 0.12))
    video_seek_threshold         = float(control.get("video_seek_horizontal_threshold", 0.06))
    video_seek_debounce          = float(control.get("video_seek_debounce_seconds", 0.35))
    volume_step_debounce         = float(control.get("volume_step_debounce_seconds", 0.22))

    # Cursor smoothing (EMA)
    ema_alpha = float(mouse_cfg.get("ema_alpha", 0.35))
    ema_alpha = max(0.05, min(1.0, ema_alpha))
    drag_ema_alpha = float(mouse_cfg.get("drag_ema_alpha", max(0.12, ema_alpha * 0.75)))
    drag_ema_alpha = max(0.05, min(1.0, drag_ema_alpha))
    mouse_deadzone     = float(mouse_cfg.get("deadzone_px", 1.0))
    reverse_horizontal = bool(
        mouse_cfg.get(
            "reverse_horizontal_motion",
            mouse_cfg.get("mirror_sideways", mouse_cfg.get("invert_x", False)),
        )
    )
    reverse_vertical = bool(mouse_cfg.get("reverse_vertical_motion", mouse_cfg.get("invert_y", False)))

    browser_scroll_scale = float(control.get("browser_scroll_scale", 1400.0))
    default_scroll_scale = float(control.get("default_scroll_scale", 1000.0))
    code_scroll_scale    = float(control.get("code_scroll_scale", 900.0))

    two_hand_enabled       = bool(control.get("two_hand_enabled", True))
    primary_hand_label     = str(control.get("primary_hand", "Right")).strip().capitalize()
    if primary_hand_label not in ("Left", "Right"):
        primary_hand_label = "Right"
    modifier_hand_label    = "Left" if primary_hand_label == "Right" else "Right"
    # FIX-7: swap_handedness is kept as an escape hatch but position-based
    # assignment is now the primary strategy (see hand-sorting block below).
    swap_handedness        = bool(control.get("swap_handedness", False))
    combo_debounce_seconds = float(control.get("combo_debounce_seconds", 0.35))

    two_hand_swipe_threshold          = float(control.get("two_hand_swipe_threshold", 0.09))
    two_hand_swipe_debounce           = float(control.get("two_hand_swipe_debounce_seconds", 0.6))
    two_hand_swipe_vertical_threshold = float(control.get("two_hand_swipe_vertical_threshold", 0.09))
    two_hand_swipe_vertical_debounce  = float(control.get("two_hand_swipe_vertical_debounce_seconds", 0.6))

    # FIX-1: Modifier hand must be held for this long before combos / swipes activate.
    modifier_entry_seconds = float(control.get("modifier_entry_seconds", 0.20))

    # FIX-8: Both hands must be continuously visible for this many frames before
    # swipe detection starts, preventing accidental triggers on hand entry/exit.
    two_hand_settle_frames = int(control.get("two_hand_settle_frames", 4))

    logger.info("Starting Gesture Control | config=%s camera=%s", config_path, camera_idx)

    if not PYAUTOGUI_AVAILABLE:
        logger.error("pyautogui is required for virtual mouse control.")
        sys.exit(1)

    pyautogui.FAILSAFE = False
    # NOTE: FAILSAFE disabled — moving the mouse to the top-left corner will NOT
    # abort the program. Set FAILSAFE = True in development if you need that escape hatch.
    screen_size = pyautogui.size()

    cap = cv2.VideoCapture(camera_idx)
    if not cap.isOpened():
        logger.error("Cannot open camera index %s", camera_idx)
        sys.exit(1)

    # ── Long-range camera setup ────────────────────────────────────────────
    # Request higher resolution so distant hands occupy more pixels, giving
    # MediaPipe more signal to work with.  The values are hints; the driver
    # will use the nearest mode it actually supports.
    capture_width  = int(settings.get("capture_width",  1280))
    capture_height = int(settings.get("capture_height",  720))
    capture_fps    = int(settings.get("capture_fps",      30))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  capture_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, capture_height)
    cap.set(cv2.CAP_PROP_FPS,          capture_fps)

    # detection_upscale > 1.0 enlarges the frame before MediaPipe sees it.
    # Helps on cheap 640×480 webcams where a distant hand has too few pixels.
    # The display frame is kept at native resolution (no upscale on screen).
    detection_upscale = float(settings.get("detection_upscale", 1.0))
    detection_upscale = max(1.0, min(2.0, detection_upscale))  # clamp to safe range
    if detection_upscale > 1.0:
        logger.info(
            "Detection upscale: %.1fx (processing at ~%dx%d)",
            detection_upscale,
            int(capture_width * detection_upscale),
            int(capture_height * detection_upscale),
        )

    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        model_complexity=1,
        min_detection_confidence=det_conf,
        min_tracking_confidence=track_conf,
    )

    if startup_delay > 0:
        logger.info("Starting in %.1f seconds...", startup_delay)
        time.sleep(startup_delay)

    # ── Runtime state ──────────────────────────────────────────────────────
    paused      = False
    dragging    = False
    scroll_mode = False
    combo_ready = False   # defined here to fix scope — updated each frame

    ema_pos: tuple[float, float] | None = None
    debounce_clock: dict[str, float] = {}

    pinch_start      = None
    pinch_last_seen  = 0.0
    pinch_armed_click  = False
    pinch_swipe_done   = False
    pinch_start_x      = None

    two_finger_start = None
    two_finger_fired = False

    scroll_prev_y      = None
    scroll_last_active = 0.0
    open_palm_prev_x   = None
    lost_hand_since    = None

    last_raw_gesture  = "none"
    raw_gesture_since = time.time()

    modifier_first_seen: float | None = None
    modifier_active     = False

    both_swipe_prev_x: tuple[float, float] | None = None
    both_swipe_prev_y: tuple[float, float] | None = None
    both_hands_settle_count = 0

    app_name         = "Unknown"
    app_context      = "default"
    last_app_refresh = 0.0
    title = ""
    proc  = ""

    display_gesture = "none"
    display_action  = "-"

    fps              = 0.0
    fps_counter      = 0
    fps_window_start = time.time()

    try:
      while True:

        frame = cv2.flip(frame, 1)

        # ── Long-range upscale for detection ──────────────────────────────
        # Enlarge the frame before MediaPipe so distant hands get more pixels.
        # We keep `frame` at native resolution for display; `rgb` (the
        # MediaPipe input) is the optionally upscaled version.
        if detection_upscale > 1.0:
            h_orig, w_orig = frame.shape[:2]
            detect_frame = cv2.resize(
                frame,
                (int(w_orig * detection_upscale), int(h_orig * detection_upscale)),
                interpolation=cv2.INTER_LINEAR,
            )
        else:
            detect_frame = frame

        rgb = cv2.cvtColor(detect_frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)
        now = time.time()

        # ── Landmark coordinate normalisation ──────────────────────────────
        # MediaPipe always returns landmarks normalised to [0, 1] of whatever
        # frame it processed.  When we upscaled the detection frame the
        # normalised values are already correct (they're still 0-1 fractions
        # of the enlarged image), BUT the draw_landmarks call uses `frame`
        # (native resolution) so we don't need to rescale — MediaPipe's
        # drawing utils normalise internally.  Nothing extra needed here;
        # the upscale/detection is fully transparent to the rest of the loop.

        # Refresh active app context at a fixed interval.
        if now - last_app_refresh >= app_refresh_seconds:
            title, proc = get_active_window_info()
            app_key     = match_app_config(title, proc, config.get("apps", {}))
            app_name    = app_key or (title[:40] if title else "Unknown")
            app_context = _classify_context(app_key, title, proc)
            last_app_refresh = now

        gesture         = "none"
        confidence      = 0.0
        details: dict   = {}
        modifier_gesture = "none"
        modifier_palm_x: float | None = None
        modifier_palm_y: float | None = None
        mod_details: dict = {}
        primary_hand_lm = None
        modifier_hand_lm = None

        if results.multi_hand_landmarks:
            lost_hand_since = None

            # ── FIX-7: Assign Left/Right by wrist X position ───────────────
            # MediaPipe's handedness label is unreliable after the frame is
            # horizontally flipped.  Sorting by wrist X (landmark 0) is far
            # more stable: smaller x → more to the left of the image → "Left".
            all_lms = results.multi_hand_landmarks
            if len(all_lms) == 2:
                # Sort ascending by wrist x — index 0 is the left hand,
                # index 1 is the right hand (in the mirrored/user-facing frame).
                sorted_lms = sorted(all_lms, key=lambda h: h.landmark[0].x)
                hands_by_label: dict[str, object] = {
                    "Left":  sorted_lms[0],
                    "Right": sorted_lms[1],
                }
            else:
                # Single hand: fall back to MediaPipe label + optional swap.
                hand_lm = all_lms[0]
                handedness_list = results.multi_handedness or []
                if handedness_list:
                    label = handedness_list[0].classification[0].label
                else:
                    label = "Right"
                if swap_handedness:
                    label = "Left" if label == "Right" else "Right"
                hands_by_label = {label: hand_lm}

            # Draw landmarks for all detected hands.
            if show_overlay:
                for hand_lm in all_lms:
                    mp_drawing.draw_landmarks(
                        frame,
                        hand_lm,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style(),
                    )

            primary_hand_lm  = hands_by_label.get(primary_hand_label)
            modifier_hand_lm = hands_by_label.get(modifier_hand_label) if two_hand_enabled else None

            if primary_hand_lm is None and len(hands_by_label) == 1:
                # Single hand visible, didn't match primary label — treat as primary.
                primary_hand_lm  = next(iter(hands_by_label.values()))
                modifier_hand_lm = None

            # ── Modifier hand gesture detection ────────────────────────────
            if modifier_hand_lm is not None:
                mod_gesture, mod_conf, mod_details = detect_gesture(
                    modifier_hand_lm.landmark, return_details=True
                )
                mod_palm_span     = float(mod_details.get("palm_span", 0.0))
                mod_conf_threshold = conf_threshold * 0.8
                if mod_conf >= mod_conf_threshold and mod_palm_span >= min_palm_span * 0.8:
                    modifier_gesture = mod_gesture
                    mod_center = mod_details.get("palm_center")
                    if mod_center:
                        modifier_palm_x = mod_center[0]
                        modifier_palm_y = mod_center[1]

                # FIX-1: Start (or keep) the modifier entry timer.
                if modifier_first_seen is None:
                    modifier_first_seen = now
                modifier_active = (now - modifier_first_seen) >= modifier_entry_seconds
            else:
                # Modifier hand gone — reset entry timer and active flag.
                modifier_first_seen = None
                modifier_active     = False
                mod_details         = {}

            if primary_hand_lm is None:
                results.multi_hand_landmarks = None
        else:
            # No hands at all — reset modifier state.
            modifier_first_seen = None
            modifier_active     = False

        # ── FIX-8: Both-hands settle gate ─────────────────────────────────
        # Increment the settle counter only when both hands are present and
        # both have passed their respective confidence gates (primary lm set,
        # modifier_active means the modifier has passed its entry timer too).
        both_hands_present = (
            primary_hand_lm is not None
            and modifier_hand_lm is not None
            and modifier_active
        )
        if both_hands_present:
            both_hands_settle_count = min(both_hands_settle_count + 1, two_hand_settle_frames + 1)
        else:
            both_hands_settle_count = 0

        # Swipe detection is gated: both hands must have been present and
        # settled for at least two_hand_settle_frames consecutive frames.
        swipe_gate_open = both_hands_settle_count >= two_hand_settle_frames

        # When the settle gate is not open, reset swipe prev positions so
        # the first frame after the gate opens doesn't carry stale deltas.
        if not swipe_gate_open:
            both_swipe_prev_x = None
            both_swipe_prev_y = None

        # ── Primary hand gesture processing ───────────────────────────────
        if results.multi_hand_landmarks and primary_hand_lm is not None:
            landmarks = primary_hand_lm.landmark
            raw_gesture, confidence, details = detect_gesture(landmarks, return_details=True)

            palm_span  = float(details.get("palm_span", 0.0))
            size_conf  = float(details.get("size_conf", 1.0))
            effective_conf_threshold = conf_threshold * max(0.50, size_conf)
            if confidence < effective_conf_threshold or palm_span < min_palm_span:
                raw_gesture = "none"
                confidence  = 0.0

            if raw_gesture != last_raw_gesture:
                last_raw_gesture  = raw_gesture
                raw_gesture_since = now

            # Keep cursor responsive on open_palm; stabilize all action gestures.
            if raw_gesture in ("open_palm", "none"):
                gesture = raw_gesture
            elif now - raw_gesture_since >= gesture_stability_seconds:
                gesture = raw_gesture
            else:
                gesture = "none"

            # FIX-4: Determine combo_ready for HUD feedback.
            combo_ready = modifier_active and gesture in _COMBO_ELIGIBLE

            # Closed fist pause toggle — always works, even while paused.
            if gesture == "closed_fist" and _debounced("pause_toggle", debounce_clock, pause_toggle_debounce, now):
                paused = not paused
                display_action = "PAUSED" if paused else "RESUMED"
                if dragging:
                    pyautogui.mouseUp()
                    dragging = False
                scroll_mode          = False
                pinch_start          = None
                pinch_armed_click    = False
                pinch_swipe_done     = False
                pinch_start_x        = None
                two_finger_start     = None
                two_finger_fired     = False
                scroll_prev_y        = None
                open_palm_prev_x     = None
                ema_pos              = None
                pinch_last_seen      = 0.0
                lost_hand_since      = None

            if not paused:
                pointer_norm = details.get("index_tip")
                palm_center  = details.get("palm_center")

                # ── 1) Open palm => cursor movement ────────────────────────
                # FIX-2: Suppress cursor movement while a two-hand open_palm
                # swipe is actively happening (swipe_gate_open AND both hands
                # open_palm) so the cursor doesn't jump during tab/window switches.
                two_hand_open_palm_active = (
                    swipe_gate_open
                    and gesture == "open_palm"
                    and modifier_gesture == "open_palm"
                    and modifier_palm_x is not None
                )

                if gesture == "open_palm" and pointer_norm and not two_hand_open_palm_active:
                    ema_pos = _move_cursor_ema(
                        pointer_norm=pointer_norm,
                        screen_size=screen_size,
                        ema_pos=ema_pos,
                        alpha=ema_alpha,
                        deadzone_px=mouse_deadzone,
                        reverse_horizontal=reverse_horizontal,
                        reverse_vertical=reverse_vertical,
                    )

                    horiz_intent = classify_intent("open_palm_horizontal", app_context).name
                    HORIZONTAL_MOTION_ACTIONS = {
                        "video_seek": ("key:right", "key:left", "SEEK_FORWARD", "SEEK_BACK"),
                        "page_nav":   ("key:pagedown", "key:pageup", "PAGE_DOWN", "PAGE_UP"),
                    }
                    if horiz_intent in HORIZONTAL_MOTION_ACTIONS:
                        fwd_action, back_action, fwd_label, back_label = HORIZONTAL_MOTION_ACTIONS[horiz_intent]
                        if open_palm_prev_x is not None:
                            dx = pointer_norm[0] - open_palm_prev_x
                            if abs(dx) >= video_seek_threshold and _debounced("horizontal_motion", debounce_clock, video_seek_debounce, now):
                                execute_action(fwd_action if dx > 0 else back_action)
                                display_action = fwd_label if dx > 0 else back_label
                        open_palm_prev_x = pointer_norm[0]
                else:
                    open_palm_prev_x = None

                # ── Two-hand horizontal swipe ──────────────────────────────
                # FIX-3: Separate state block for horizontal swipe.
                # FIX-6: Use per-direction debounce keys.
                # FIX-8: Gated by swipe_gate_open.
                if (
                    two_hand_enabled
                    and swipe_gate_open
                    and gesture == "open_palm"
                    and modifier_gesture == "open_palm"
                    and modifier_palm_x is not None
                    and palm_center is not None
                ):
                    if both_swipe_prev_x is not None:
                        prev_primary_x, prev_modifier_x = both_swipe_prev_x
                        dx_primary  = palm_center[0] - prev_primary_x
                        dx_modifier = modifier_palm_x - prev_modifier_x

                        if (
                            abs(dx_primary) >= two_hand_swipe_threshold
                            and abs(dx_modifier) >= two_hand_swipe_threshold
                            and (dx_primary > 0) == (dx_modifier > 0)
                        ):
                            # FIX-6: Separate debounce key per direction.
                            direction_key = "two_hand_swipe_right" if dx_primary > 0 else "two_hand_swipe_left"
                            if _debounced(direction_key, debounce_clock, two_hand_swipe_debounce, now):
                                swipe_intent = classify_intent("two_hand_swipe", app_context)
                                if swipe_intent.name in TWO_HAND_SWIPE_ACTIONS:
                                    fwd, back, fwd_label, back_label = TWO_HAND_SWIPE_ACTIONS[swipe_intent.name]
                                    execute_action(fwd if dx_primary > 0 else back)
                                    display_action = fwd_label if dx_primary > 0 else back_label

                    both_swipe_prev_x = (palm_center[0], modifier_palm_x)
                else:
                    # Don't null out here — the settle gate handles the reset above.
                    if not both_hands_present:
                        both_swipe_prev_x = None

                # ── Two-hand vertical swipe ────────────────────────────────
                # FIX-3: Completely separate state (both_swipe_prev_y) and
                # logic from the horizontal block. The old y-delta bug
                # (modifier_palm_x used as a y value) is gone.
                # FIX-8: Gated by swipe_gate_open.
                if (
                    two_hand_enabled
                    and swipe_gate_open
                    and gesture == "open_palm"
                    and modifier_gesture == "open_palm"
                    and modifier_palm_y is not None
                    and palm_center is not None
                ):
                    if both_swipe_prev_y is not None:
                        prev_primary_y, prev_modifier_y = both_swipe_prev_y
                        primary_y_now  = palm_center[1]
                        modifier_y_now = modifier_palm_y

                        dy_primary  = primary_y_now - prev_primary_y
                        dy_modifier = modifier_y_now - prev_modifier_y

                        # Require the motion to be more vertical than horizontal
                        # (compare against the horizontal delta from this same frame).
                        horiz_dx_primary = abs(palm_center[0] - both_swipe_prev_x[0]) if both_swipe_prev_x else 0.0

                        if (
                            abs(dy_primary) >= two_hand_swipe_vertical_threshold
                            and abs(dy_modifier) >= two_hand_swipe_vertical_threshold
                            and (dy_primary > 0) == (dy_modifier > 0)
                            and abs(dy_primary) > horiz_dx_primary * 1.2
                        ):
                            # FIX-6: Per-direction debounce for vertical swipes too.
                            vert_dir_key = "two_hand_swipe_v_up" if dy_primary < 0 else "two_hand_swipe_v_down"
                            if _debounced(vert_dir_key, debounce_clock, two_hand_swipe_vertical_debounce, now):
                                vert_intent = classify_intent("two_hand_swipe_vertical", app_context)
                                if vert_intent.name in TWO_HAND_SWIPE_VERTICAL_ACTIONS:
                                    up_a, down_a, up_lbl, down_lbl = TWO_HAND_SWIPE_VERTICAL_ACTIONS[vert_intent.name]
                                    execute_action(up_a if dy_primary < 0 else down_a)
                                    display_action = up_lbl if dy_primary < 0 else down_lbl

                    both_swipe_prev_y = (palm_center[1], modifier_palm_y)
                else:
                    if not both_hands_present:
                        both_swipe_prev_y = None

                # ── Two-hand pointing_up => new window/tab ─────────────────
                # FIX-5 note: run with --no-overlay for cleanest focus behavior
                # so the action hotkey lands on the intended target app.
                if (
                    two_hand_enabled
                    and modifier_active
                    and gesture == "pointing_up"
                    and modifier_gesture == "pointing_up"
                    and _debounced("two_hand_pointing", debounce_clock, 1.2, now)
                ):
                    new_win_intent = classify_intent("two_hand_pointing_up", app_context)
                    if new_win_intent.action:
                        execute_action(new_win_intent.action)
                        display_action = new_win_intent.label

                # ── Two-hand closed_fist => show desktop / close tab ────────
                # FIX-5 note: same focus caveat as above.
                if (
                    two_hand_enabled
                    and modifier_active
                    and gesture == "closed_fist"
                    and modifier_gesture == "closed_fist"
                    and _debounced("two_hand_fist", debounce_clock, 1.5, now)
                ):
                    fist_intent = classify_intent("two_hand_closed_fist", app_context)
                    if fist_intent.action:
                        execute_action(fist_intent.action)
                        display_action = fist_intent.label

                # ── Two-hand screenshot: modifier three_fingers + primary open_palm ──
                if (
                    two_hand_enabled
                    and modifier_active
                    and gesture == "open_palm"
                    and modifier_gesture == "three_fingers_up"
                    and _debounced("two_hand_screenshot", debounce_clock, 1.5, now)
                ):
                    execute_action("screenshot")
                    display_action = "SCREENSHOT (2H)"

                # ── 2+3) Primary click / drag ──────────────────────────────
                if gesture == primary_click_gesture and pointer_norm:
                    pinch_last_seen = now
                    ema_pos = _move_cursor_ema(
                        pointer_norm=pointer_norm,
                        screen_size=screen_size,
                        ema_pos=ema_pos,
                        alpha=drag_ema_alpha if dragging else ema_alpha,
                        deadzone_px=mouse_deadzone,
                        reverse_horizontal=reverse_horizontal,
                        reverse_vertical=reverse_vertical,
                    )

                    if pinch_start is None:
                        pinch_start       = now
                        pinch_start_x     = pointer_norm[0]
                        pinch_armed_click = False
                        pinch_swipe_done  = False

                    hold_time = now - pinch_start
                    if hold_time >= pinch_click_hold:
                        pinch_armed_click = True

                    if (
                        classify_intent("primary_click_swipe", app_context).name == "browser_nav"
                        and primary_click_gesture == "pinch"
                        and not pinch_swipe_done
                        and pinch_start_x is not None
                    ):
                        dx = pointer_norm[0] - pinch_start_x
                        if abs(dx) >= browser_swipe_threshold and _debounced("browser_nav", debounce_clock, 0.45, now):
                            execute_action("key:alt+right" if dx > 0 else "key:alt+left")
                            display_action   = "FORWARD" if dx > 0 else "BACK"
                            pinch_swipe_done = True

                    if hold_time >= pinch_drag_hold and not dragging and not pinch_swipe_done:
                        pyautogui.mouseDown()
                        dragging       = True
                        display_action = "DRAG_START"

                else:
                    pinch_briefly_lost = pinch_start is not None and (now - pinch_last_seen) <= pinch_break_grace

                    if pinch_briefly_lost and pointer_norm and dragging:
                        ema_pos = _move_cursor_ema(
                            pointer_norm=pointer_norm,
                            screen_size=screen_size,
                            ema_pos=ema_pos,
                            alpha=drag_ema_alpha,
                            deadzone_px=mouse_deadzone,
                            reverse_horizontal=reverse_horizontal,
                            reverse_vertical=reverse_vertical,
                        )
                    elif pinch_start is not None:
                        if dragging:
                            pyautogui.mouseUp()
                            dragging       = False
                            display_action = "DROP"
                        elif pinch_armed_click and not pinch_swipe_done and _debounced("left_click", debounce_clock, debounce_seconds, now):
                            execute_action("click")
                            display_action = "LEFT_CLICK"
                        pinch_start       = None
                        pinch_armed_click = False
                        pinch_swipe_done  = False
                        pinch_start_x     = None

                # ── Two-hand combo resolution ──────────────────────────────
                # FIX-1: Only classify_combo when modifier_active (hold timer passed).
                combo_intent = (
                    classify_combo(modifier_active, gesture, app_context)
                    if two_hand_enabled and gesture in _COMBO_ELIGIBLE
                    else None
                )

                # ── 4) Two-finger tap => right click or combo ──────────────
                if gesture == "two_finger_tap":
                    if two_finger_start is None:
                        two_finger_start = now
                        two_finger_fired = False
                    if (
                        not two_finger_fired
                        and now - two_finger_start >= two_finger_stable
                        and _debounced("right_click", debounce_clock, debounce_seconds, now)
                    ):
                        if combo_intent is not None:
                            execute_action(combo_intent.action)
                            display_action = combo_intent.label
                        else:
                            intent = classify_intent("two_finger_tap", app_context)
                            execute_action(intent.action)
                            display_action = intent.label
                        two_finger_fired = True
                else:
                    two_finger_start = None
                    two_finger_fired = False

                # ── 5) Three-fingers up => scroll or combo ─────────────────
                if gesture == "three_fingers_up" and palm_center:
                    if combo_intent is not None and _debounced("three_finger_combo", debounce_clock, combo_debounce_seconds, now):
                        execute_action(combo_intent.action)
                        display_action = combo_intent.label
                    else:
                        if not scroll_mode:
                            scroll_mode   = True
                            scroll_prev_y = palm_center[1]
                        scroll_last_active = now

                        if scroll_prev_y is not None:
                            dy = palm_center[1] - scroll_prev_y
                            if abs(dy) >= scroll_motion_threshold:
                                three_finger_intent = classify_intent("three_fingers_up", app_context)
                                if three_finger_intent.name == "volume_step":
                                    if _debounced("volume_step", debounce_clock, 0.08, now):
                                        execute_action("volume_down" if dy > 0 else "volume_up")
                                        display_action = "VOLUME_DOWN" if dy > 0 else "VOLUME_UP"
                                else:
                                    scale = (
                                        browser_scroll_scale if app_context == "browser"
                                        else code_scroll_scale if app_context == "code_editor"
                                        else default_scroll_scale
                                    )
                                    pyautogui.scroll(int(-dy * scale))
                                    display_action = three_finger_intent.label
                                scroll_last_active = now
                            scroll_prev_y = palm_center[1]
                else:
                    if scroll_mode and now - scroll_last_active > scroll_inactivity_timeout:
                        scroll_mode   = False
                        scroll_prev_y = None
                        display_action = "SCROLL_MODE_OFF"

                # ── Volume gestures ────────────────────────────────────────
                if gesture == "thumbs_up":
                    if combo_intent is not None:
                        if _debounced("volume_up_gesture_combo", debounce_clock, combo_debounce_seconds, now):
                            execute_action(combo_intent.action)
                            display_action = combo_intent.label
                    elif _debounced("volume_up_gesture", debounce_clock, volume_step_debounce, now):
                        intent = classify_intent("thumbs_up", app_context)
                        execute_action(intent.action)
                        display_action = intent.label

                elif gesture == "thumbs_down":
                    if combo_intent is not None:
                        if _debounced("volume_down_gesture_combo", debounce_clock, combo_debounce_seconds, now):
                            execute_action(combo_intent.action)
                            display_action = combo_intent.label
                    elif _debounced("volume_down_gesture", debounce_clock, volume_step_debounce, now):
                        intent = classify_intent("thumbs_down", app_context)
                        execute_action(intent.action)
                        display_action = intent.label

        else:
            # No primary hand: clean up states and safely release drag.
            gesture    = "none"
            confidence = 0.0
            if dragging:
                if lost_hand_since is None:
                    lost_hand_since = now
                elif now - lost_hand_since >= drag_release_grace:
                    pyautogui.mouseUp()
                    dragging       = False
                    display_action = "DROP"
            pinch_start       = None
            pinch_last_seen   = 0.0
            pinch_armed_click = False
            pinch_swipe_done  = False
            pinch_start_x     = None
            two_finger_start  = None
            two_finger_fired  = False
            open_palm_prev_x  = None

            if scroll_mode and now - scroll_last_active > scroll_inactivity_timeout:
                scroll_mode    = False
                scroll_prev_y  = None
                display_action = "SCROLL_MODE_OFF"

            combo_ready = False

        display_gesture = f"{gesture} ({confidence:.2f})" if gesture != "none" else "none"

        # ── Gesture CSV logging ────────────────────────────────────────────
        if gesture != "none":
            _intent_name = classify_intent(gesture, app_context).name if gesture != "none" else "none"
            _intent      = classify_intent(gesture, app_context)
            gesture_logger.log(
                gesture=gesture,
                confidence=confidence,
                palm_span=float(details.get("palm_span", 0.0)),
                size_conf=float(details.get("size_conf", 0.0)),
                context=app_context,
                app_name=app_name,
                intent=_intent.name,
                action=_intent.action,
                label=_intent.label,
                modifier_active=modifier_active,
                combo=combo_ready,
                fps=fps,
            )

        # FPS tracking
        fps_counter += 1
        elapsed = now - fps_window_start
        if elapsed >= 0.5:
            fps              = fps_counter / elapsed
            fps_counter      = 0
            fps_window_start = now

        # ── Render overlay and handle keys ─────────────────────────────────
        if show_overlay:
            draw_overlay(
                frame=frame,
                app_name=app_name,
                context=app_context,
                gesture_label=display_gesture,
                action_label=display_action,
                paused=paused,
                scroll_mode=scroll_mode,
                fps=fps,
                modifier_gesture=modifier_gesture,
                modifier_active=modifier_active,
                combo_ready=combo_ready,
            )
            if paused:
                cv2.putText(
                    frame,
                    "PAUSED",
                    (frame.shape[1] // 2 - 70, frame.shape[0] // 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.3,
                    COLORS["red"],
                    3,
                    cv2.LINE_AA,
                )

            cv2.imshow("Gesture Control", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("p"):
                paused         = not paused
                display_action = "PAUSED" if paused else "RESUMED"

    finally:
        cap.release()
        cv2.destroyAllWindows()
        hands.close()
        gesture_logger.print_summary()
        gesture_logger.close()
        logger.info("Gesture Control stopped")


def main():
    parser = argparse.ArgumentParser(description="Hand Gesture Control System")
    parser.add_argument("--config",           default="config.yaml", help="Path to config YAML file")
    parser.add_argument("--no-overlay",       action="store_true",   help="Disable camera overlay window")
    parser.add_argument("--list-gestures",    action="store_true",   help="Print core gesture reference and exit")
    parser.add_argument("--skip-validation",  action="store_true",   help="Skip config.yaml validation on startup")
    args = parser.parse_args()

    if args.list_gestures:
        print_gesture_reference(args.config)
        return
    run(args.config, no_overlay=args.no_overlay, skip_validation=args.skip_validation)


if __name__ == "__main__":
    main()
