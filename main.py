"""
main.py
Gesture Control System — entry point.

Gestures:
  open_palm        -> cursor movement
  pointing_up      -> left click (hold), drag on longer hold
  two_finger_tap   -> right click
  three_fingers_up -> scroll mode
  thumbs_up/down   -> volume up/down
  closed_fist      -> pause/resume toggle
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

mp_hands        = mp.solutions.hands
mp_drawing      = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

COLORS = {
    "green":  (50, 220, 50),
    "yellow": (30, 220, 220),
    "red":    (20, 20, 255),
    "blue":   (255, 160, 30),
    "white":  (245, 245, 245),
    "orange": (30, 165, 255),
    "bg":     (22, 22, 22),
}

_COMBO_ELIGIBLE = frozenset({"thumbs_up", "thumbs_down", "two_finger_tap", "three_fingers_up"})

TWO_HAND_SWIPE_ACTIONS = {
    "window_switch":          ("key:alt+tab",       "key:alt+shift+tab",  "NEXT_WINDOW",  "PREV_WINDOW"),
    "browser_tab_switch":     ("key:ctrl+tab",       "key:ctrl+shift+tab", "NEXT_TAB",     "PREV_TAB"),
    "virtual_desktop_switch": ("key:ctrl+win+right", "key:ctrl+win+left",  "NEXT_DESKTOP", "PREV_DESKTOP"),
    "code_tab_switch":        ("key:ctrl+tab",       "key:ctrl+shift+tab", "NEXT_EDITOR",  "PREV_EDITOR"),
    "history_nav":            ("key:alt+right",      "key:alt+left",       "FORWARD",      "BACK"),
}

TWO_HAND_SWIPE_VERTICAL_ACTIONS = {
    "fast_scroll": ("scroll_up_fast", "scroll_down_fast", "FAST_SCROLL_UP",  "FAST_SCROLL_DOWN"),
    "zoom_inout":  ("key:ctrl+=",     "key:ctrl+-",       "ZOOM_IN",         "ZOOM_OUT"),
    "slide_nav":   ("key:pageup",     "key:pagedown",     "PREV_SLIDE",      "NEXT_SLIDE"),
    "media_seek":  ("key:shift+right","key:shift+left",   "SEEK_FWD_LARGE",  "SEEK_BACK_LARGE"),
}

_CONTEXT_FALLBACK_TOKENS = {
    "browser":         ("chrome", "firefox", "edge", "brave", "opera", "browser"),
    "video_player":    ("vlc", "youtube", "netflix", "potplayer", "mpc", "media player", "video"),
    "code_editor":     ("vscode", "visual studio", "code", "pycharm", "intellij", "sublime", "notepad++"),
    "document_editor": ("word", "winword", "notepad", "wordpad", "acrobat", "foxit", "pdf", "libreoffice writer"),
    "terminal":        ("cmd.exe", "powershell", "windows terminal", "wt.exe", "conemu", "bash", "wsl"),
}


def load_config(config_path: str) -> dict:
    path = Path(config_path)
    if not path.exists():
        logger.error("Config file not found: %s", config_path)
        sys.exit(1)
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _debounced(name: str, clock: dict[str, float], interval: float, now: float) -> bool:
    last = clock.get(name, 0.0)
    if now - last < interval:
        return False
    clock[name] = now
    return True


def _classify_context(app_key: str | None, title: str, proc: str) -> str:
    # YouTube in a browser tab — title contains "YouTube" but app_key is "browser".
    # Check this first so it lands in video_player, not browser.
    if "youtube" in title.lower():
        return "video_player"
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
    modifier_active: bool = False,
    combo_ready: bool = False,
):
    h, _ = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (390, h), COLORS["bg"], -1)
    cv2.addWeighted(overlay, 0.62, frame, 0.38, 0, frame)
    y = 30

    def put(txt, color=COLORS["white"], scale=0.58, thickness=1):
        nonlocal y
        cv2.putText(frame, txt, (12, y), cv2.FONT_HERSHEY_SIMPLEX,
                    scale, color, thickness, cv2.LINE_AA)
        y += int(30 * scale + 10)

    put("Gesture Control", COLORS["yellow"], 0.7, 2)
    put(f"App: {app_name}", COLORS["blue"])
    put(f"Context: {context}", COLORS["blue"])
    put(f"Gesture: {gesture_label}", COLORS["green"], 0.62, 2)
    if modifier_gesture != "none":
        if combo_ready:
            put(">> COMBO READY <<", COLORS["orange"], 0.58, 2)
        elif modifier_active:
            put(f"Modifier: {modifier_gesture} (active)", COLORS["orange"], 0.52)
        else:
            put(f"Modifier: {modifier_gesture} (holding...)", COLORS["white"], 0.50)
    put(f"Action: {action_label}", COLORS["yellow"])
    put(f"Scroll: {'ON' if scroll_mode else 'OFF'}")
    put(f"{'PAUSED' if paused else 'ACTIVE'}", COLORS["red"] if paused else COLORS["green"])
    put(f"FPS: {fps:.1f}", COLORS["white"])
    put("Q = Quit  |  P = Pause", COLORS["white"], 0.5)


def print_gesture_reference(config_path: str = "config.yaml"):
    print("""
Gestures:
  open_palm        -> Cursor movement
  pointing_up      -> Left click / drag
  two_finger_tap   -> Right click
  three_fingers_up -> Scroll mode
  thumbs_up/down   -> Volume up/down
  closed_fist      -> Pause / Resume
""")
    try:
        config = load_config(config_path)
        load_rules_from_config(config)
    except Exception as exc:
        print(f"Could not load config: {exc}")
        return
    print("Context overrides:")
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
    if not skip_validation:
        validate_and_report(config, fatal_on_error=True)
    load_rules_from_config(config)
    load_combo_rules_from_config(config)

    settings  = config.get("settings", {})
    control   = settings.get("control", {})
    mouse_cfg = settings.get("mouse_control", {})

    camera_idx          = int(settings.get("camera_index", 0))
    det_conf            = float(settings.get("detection_confidence", 0.65))
    track_conf          = float(settings.get("tracking_confidence", 0.65))
    show_overlay        = bool(settings.get("show_overlay", True)) and not no_overlay
    startup_delay       = float(settings.get("startup_delay", 1.0))
    app_refresh_seconds = float(settings.get("app_refresh_seconds", 0.2))

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

    ema_alpha      = max(0.05, min(1.0, float(mouse_cfg.get("ema_alpha", 0.35))))
    drag_ema_alpha = max(0.05, min(1.0, float(mouse_cfg.get("drag_ema_alpha", max(0.12, ema_alpha * 0.75)))))
    mouse_deadzone     = float(mouse_cfg.get("deadzone_px", 1.0))
    reverse_horizontal = bool(mouse_cfg.get("reverse_horizontal_motion",
                              mouse_cfg.get("mirror_sideways", mouse_cfg.get("invert_x", False))))
    reverse_vertical   = bool(mouse_cfg.get("reverse_vertical_motion", mouse_cfg.get("invert_y", False)))

    browser_scroll_scale = float(control.get("browser_scroll_scale", 1400.0))
    default_scroll_scale = float(control.get("default_scroll_scale", 1000.0))
    code_scroll_scale    = float(control.get("code_scroll_scale", 900.0))

    two_hand_enabled   = bool(control.get("two_hand_enabled", True))
    primary_hand_label = str(control.get("primary_hand", "Right")).strip().capitalize()
    if primary_hand_label not in ("Left", "Right"):
        primary_hand_label = "Right"
    modifier_hand_label    = "Left" if primary_hand_label == "Right" else "Right"
    swap_handedness        = bool(control.get("swap_handedness", False))
    combo_debounce_seconds = float(control.get("combo_debounce_seconds", 0.35))

    two_hand_swipe_threshold          = float(control.get("two_hand_swipe_threshold", 0.09))
    two_hand_swipe_debounce           = float(control.get("two_hand_swipe_debounce_seconds", 0.6))
    two_hand_swipe_vertical_threshold = float(control.get("two_hand_swipe_vertical_threshold", 0.09))
    two_hand_swipe_vertical_debounce  = float(control.get("two_hand_swipe_vertical_debounce_seconds", 0.6))
    modifier_entry_seconds            = float(control.get("modifier_entry_seconds", 0.20))
    two_hand_settle_frames            = int(control.get("two_hand_settle_frames", 4))

    logger.info("Starting | config=%s camera=%s", config_path, camera_idx)

    if not PYAUTOGUI_AVAILABLE:
        logger.error("pyautogui is required.")
        sys.exit(1)

    pyautogui.FAILSAFE = False
    screen_size = pyautogui.size()

    cap = cv2.VideoCapture(camera_idx)
    if not cap.isOpened():
        logger.error("Cannot open camera %s", camera_idx)
        sys.exit(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  int(settings.get("capture_width",  1280)))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(settings.get("capture_height",  720)))
    cap.set(cv2.CAP_PROP_FPS,          int(settings.get("capture_fps",      30)))

    detection_upscale = max(1.0, min(2.0, float(settings.get("detection_upscale", 1.0))))
    if detection_upscale > 1.0:
        logger.info("Detection upscale: %.1fx", detection_upscale)

    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        model_complexity=1,
        min_detection_confidence=det_conf,
        min_tracking_confidence=track_conf,
    )

    log_cfg        = settings.get("logging", {})
    gesture_logger = GestureLogger(
        log_dir=Path(__file__).resolve().parent / "logs",
        enabled=bool(log_cfg.get("gesture_csv_enabled", True)),
    )

    if startup_delay > 0:
        logger.info("Starting in %.1fs...", startup_delay)
        time.sleep(startup_delay)

    paused      = False
    dragging    = False
    scroll_mode = False
    combo_ready = False

    ema_pos: tuple[float, float] | None = None
    debounce_clock: dict[str, float]    = {}

    pinch_start       = None
    pinch_last_seen   = 0.0
    pinch_armed_click = False
    pinch_swipe_done  = False
    pinch_start_x     = None

    two_finger_start = None
    two_finger_fired = False

    scroll_prev_y      = None
    scroll_last_active = 0.0
    open_palm_prev_x   = None
    lost_hand_since    = None

    last_raw_gesture  = "none"
    raw_gesture_since = time.time()

    modifier_first_seen: float | None     = None
    modifier_active                        = False
    both_swipe_prev_x: tuple | None        = None
    both_swipe_prev_y: tuple | None        = None
    both_hands_settle_count                = 0

    app_name         = "Unknown"
    app_context      = "default"
    last_app_refresh = 0.0

    display_gesture  = "none"
    display_action   = "-"
    fps              = 0.0
    fps_counter      = 0
    fps_window_start = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.warning("Failed to read frame")
                break

            frame = cv2.flip(frame, 1)

            if detection_upscale > 1.0:
                h0, w0 = frame.shape[:2]
                detect_frame = cv2.resize(
                    frame,
                    (int(w0 * detection_upscale), int(h0 * detection_upscale)),
                    interpolation=cv2.INTER_LINEAR,
                )
            else:
                detect_frame = frame

            rgb     = cv2.cvtColor(detect_frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)
            now     = time.time()

            if now - last_app_refresh >= app_refresh_seconds:
                title, proc  = get_active_window_info()
                app_key      = match_app_config(title, proc, config.get("apps", {}))
                app_name     = app_key or (title[:40] if title else "Unknown")
                app_context  = _classify_context(app_key, title, proc)
                last_app_refresh = now

            gesture          = "none"
            confidence       = 0.0
            details: dict    = {}
            modifier_gesture = "none"
            modifier_palm_x: float | None = None
            modifier_palm_y: float | None = None
            mod_details: dict = {}
            primary_hand_lm  = None
            modifier_hand_lm = None

            if results.multi_hand_landmarks:
                lost_hand_since = None
                all_lms = results.multi_hand_landmarks

                if len(all_lms) == 2:
                    sorted_lms = sorted(all_lms, key=lambda h: h.landmark[0].x)
                    hands_by_label: dict[str, object] = {
                        "Left": sorted_lms[0], "Right": sorted_lms[1]
                    }
                else:
                    hl   = results.multi_handedness or []
                    lbl  = hl[0].classification[0].label if hl else "Right"
                    if swap_handedness:
                        lbl = "Left" if lbl == "Right" else "Right"
                    hands_by_label = {lbl: all_lms[0]}

                if show_overlay:
                    for hl in all_lms:
                        mp_drawing.draw_landmarks(
                            frame, hl, mp_hands.HAND_CONNECTIONS,
                            mp_drawing_styles.get_default_hand_landmarks_style(),
                            mp_drawing_styles.get_default_hand_connections_style(),
                        )

                primary_hand_lm  = hands_by_label.get(primary_hand_label)
                modifier_hand_lm = hands_by_label.get(modifier_hand_label) if two_hand_enabled else None

                if primary_hand_lm is None and len(hands_by_label) == 1:
                    primary_hand_lm  = next(iter(hands_by_label.values()))
                    modifier_hand_lm = None

                if modifier_hand_lm is not None:
                    mg, mc, mod_details = detect_gesture(modifier_hand_lm.landmark, return_details=True)
                    if mc >= conf_threshold * 0.8 and float(mod_details.get("palm_span", 0)) >= min_palm_span * 0.8:
                        modifier_gesture = mg
                        ctr = mod_details.get("palm_center")
                        if ctr:
                            modifier_palm_x = ctr[0]
                            modifier_palm_y = ctr[1]
                    if modifier_first_seen is None:
                        modifier_first_seen = now
                    modifier_active = (now - modifier_first_seen) >= modifier_entry_seconds
                else:
                    modifier_first_seen = None
                    modifier_active     = False
                    mod_details         = {}

                if primary_hand_lm is None:
                    results.multi_hand_landmarks = None
            else:
                modifier_first_seen = None
                modifier_active     = False

            both_hands_present = (
                primary_hand_lm is not None
                and modifier_hand_lm is not None
                and modifier_active
            )
            both_hands_settle_count = (
                min(both_hands_settle_count + 1, two_hand_settle_frames + 1)
                if both_hands_present else 0
            )
            swipe_gate_open = both_hands_settle_count >= two_hand_settle_frames
            if not swipe_gate_open:
                both_swipe_prev_x = None
                both_swipe_prev_y = None

            if results.multi_hand_landmarks and primary_hand_lm is not None:
                landmarks = primary_hand_lm.landmark
                raw_gesture, confidence, details = detect_gesture(landmarks, return_details=True)

                palm_span = float(details.get("palm_span", 0.0))
                size_conf = float(details.get("size_conf", 1.0))
                if confidence < conf_threshold * max(0.50, size_conf) or palm_span < min_palm_span:
                    raw_gesture = "none"
                    confidence  = 0.0

                if raw_gesture != last_raw_gesture:
                    last_raw_gesture  = raw_gesture
                    raw_gesture_since = now

                if raw_gesture in ("open_palm", "none"):
                    gesture = raw_gesture
                elif now - raw_gesture_since >= gesture_stability_seconds:
                    gesture = raw_gesture
                else:
                    gesture = "none"

                combo_ready = modifier_active and gesture in _COMBO_ELIGIBLE

                # Two-hand closed_fist must be checked BEFORE the single-hand
                # pause toggle so it doesn't get swallowed. Only fires when
                # modifier is active (held long enough).
                two_hand_fist_fired = False
                if (two_hand_enabled and modifier_active
                        and gesture == "closed_fist" and modifier_gesture == "closed_fist"
                        and _debounced("two_hand_fist", debounce_clock, 1.5, now)):
                    fi = classify_intent("two_hand_closed_fist", app_context)
                    if fi.action:
                        execute_action(fi.action)
                        display_action = fi.label
                        two_hand_fist_fired = True

                # Single-hand pause toggle — skip if two-hand fist already fired.
                if (not two_hand_fist_fired
                        and gesture == "closed_fist"
                        and _debounced("pause_toggle", debounce_clock, pause_toggle_debounce, now)):
                    paused = not paused
                    display_action = "PAUSED" if paused else "RESUMED"
                    if dragging:
                        pyautogui.mouseUp()
                        dragging = False
                    scroll_mode       = False
                    pinch_start       = None
                    pinch_armed_click = False
                    pinch_swipe_done  = False
                    pinch_start_x     = None
                    two_finger_start  = None
                    two_finger_fired  = False
                    scroll_prev_y     = None
                    open_palm_prev_x  = None
                    ema_pos           = None
                    pinch_last_seen   = 0.0
                    lost_hand_since   = None

                if not paused:
                    pointer_norm = details.get("index_tip")
                    palm_center  = details.get("palm_center")

                    two_hand_open_palm_active = (
                        swipe_gate_open
                        and gesture == "open_palm"
                        and modifier_gesture == "open_palm"
                        and modifier_palm_x is not None
                    )

                    if gesture == "open_palm" and pointer_norm and not two_hand_open_palm_active:
                        ema_pos = _move_cursor_ema(pointer_norm, screen_size, ema_pos,
                                                   ema_alpha, mouse_deadzone,
                                                   reverse_horizontal, reverse_vertical)
                        horiz_intent = classify_intent("open_palm_horizontal", app_context).name
                        HORIZ_ACTIONS = {
                            "video_seek": ("key:right", "key:left", "SEEK_FWD", "SEEK_BACK"),
                            "page_nav":   ("key:pagedown", "key:pageup", "PAGE_DOWN", "PAGE_UP"),
                        }
                        if horiz_intent in HORIZ_ACTIONS:
                            fa, ba, fl, bl = HORIZ_ACTIONS[horiz_intent]
                            if open_palm_prev_x is not None:
                                dx = pointer_norm[0] - open_palm_prev_x
                                if abs(dx) >= video_seek_threshold and _debounced("horiz_motion", debounce_clock, video_seek_debounce, now):
                                    execute_action(fa if dx > 0 else ba)
                                    display_action = fl if dx > 0 else bl
                            open_palm_prev_x = pointer_norm[0]
                    else:
                        open_palm_prev_x = None

                    if (two_hand_enabled and swipe_gate_open
                            and gesture == "open_palm" and modifier_gesture == "open_palm"
                            and modifier_palm_x is not None and palm_center is not None):
                        if both_swipe_prev_x is not None:
                            dx_p = palm_center[0] - both_swipe_prev_x[0]
                            dx_m = modifier_palm_x - both_swipe_prev_x[1]
                            if (abs(dx_p) >= two_hand_swipe_threshold
                                    and abs(dx_m) >= two_hand_swipe_threshold
                                    and (dx_p > 0) == (dx_m > 0)):
                                dk = "two_hand_swipe_right" if dx_p > 0 else "two_hand_swipe_left"
                                if _debounced(dk, debounce_clock, two_hand_swipe_debounce, now):
                                    si = classify_intent("two_hand_swipe", app_context)
                                    if si.name in TWO_HAND_SWIPE_ACTIONS:
                                        fwd, bk, fl, bl = TWO_HAND_SWIPE_ACTIONS[si.name]
                                        execute_action(fwd if dx_p > 0 else bk)
                                        display_action = fl if dx_p > 0 else bl
                        both_swipe_prev_x = (palm_center[0], modifier_palm_x)
                    elif not both_hands_present:
                        both_swipe_prev_x = None

                    if (two_hand_enabled and swipe_gate_open
                            and gesture == "open_palm" and modifier_gesture == "open_palm"
                            and modifier_palm_y is not None and palm_center is not None):
                        if both_swipe_prev_y is not None:
                            dy_p    = palm_center[1] - both_swipe_prev_y[0]
                            dy_m    = modifier_palm_y - both_swipe_prev_y[1]
                            hdx     = abs(palm_center[0] - both_swipe_prev_x[0]) if both_swipe_prev_x else 0.0
                            if (abs(dy_p) >= two_hand_swipe_vertical_threshold
                                    and abs(dy_m) >= two_hand_swipe_vertical_threshold
                                    and (dy_p > 0) == (dy_m > 0)
                                    and abs(dy_p) > hdx * 1.2):
                                vk = "two_hand_swipe_v_up" if dy_p < 0 else "two_hand_swipe_v_down"
                                if _debounced(vk, debounce_clock, two_hand_swipe_vertical_debounce, now):
                                    vi = classify_intent("two_hand_swipe_vertical", app_context)
                                    if vi.name in TWO_HAND_SWIPE_VERTICAL_ACTIONS:
                                        ua, da, ul, dl = TWO_HAND_SWIPE_VERTICAL_ACTIONS[vi.name]
                                        execute_action(ua if dy_p < 0 else da)
                                        display_action = ul if dy_p < 0 else dl
                        both_swipe_prev_y = (palm_center[1], modifier_palm_y)
                    elif not both_hands_present:
                        both_swipe_prev_y = None

                    if (two_hand_enabled and modifier_active
                            and gesture == "pointing_up" and modifier_gesture == "pointing_up"
                            and _debounced("two_hand_pointing", debounce_clock, 1.2, now)):
                        ni = classify_intent("two_hand_pointing_up", app_context)
                        if ni.action:
                            execute_action(ni.action)
                            display_action = ni.label

                    if (two_hand_enabled and modifier_active
                            and gesture == "open_palm" and modifier_gesture == "three_fingers_up"
                            and _debounced("two_hand_screenshot", debounce_clock, 1.5, now)):
                        execute_action("screenshot")
                        display_action = "SCREENSHOT (2H)"

                    if gesture == primary_click_gesture and pointer_norm:
                        pinch_last_seen = now
                        ema_pos = _move_cursor_ema(pointer_norm, screen_size, ema_pos,
                                                   drag_ema_alpha if dragging else ema_alpha,
                                                   mouse_deadzone, reverse_horizontal, reverse_vertical)
                        if pinch_start is None:
                            pinch_start       = now
                            pinch_start_x     = pointer_norm[0]
                            pinch_armed_click = False
                            pinch_swipe_done  = False
                        hold_time = now - pinch_start
                        if hold_time >= pinch_click_hold:
                            pinch_armed_click = True
                        if (classify_intent("primary_click_swipe", app_context).name == "browser_nav"
                                and primary_click_gesture == "pinch"
                                and not pinch_swipe_done and pinch_start_x is not None):
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
                        briefly_lost = pinch_start is not None and (now - pinch_last_seen) <= pinch_break_grace
                        if briefly_lost and pointer_norm and dragging:
                            ema_pos = _move_cursor_ema(pointer_norm, screen_size, ema_pos,
                                                       drag_ema_alpha, mouse_deadzone,
                                                       reverse_horizontal, reverse_vertical)
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

                    combo_intent = (
                        classify_combo(modifier_active, gesture, app_context)
                        if two_hand_enabled and gesture in _COMBO_ELIGIBLE else None
                    )

                    if gesture == "two_finger_tap":
                        if two_finger_start is None:
                            two_finger_start = now
                            two_finger_fired = False
                        if (not two_finger_fired
                                and now - two_finger_start >= two_finger_stable
                                and _debounced("right_click", debounce_clock, debounce_seconds, now)):
                            ti = combo_intent or classify_intent("two_finger_tap", app_context)
                            execute_action(ti.action)
                            display_action   = ti.label
                            two_finger_fired = True
                    else:
                        two_finger_start = None
                        two_finger_fired = False

                    if gesture == "three_fingers_up" and palm_center:
                        if combo_intent is not None and _debounced("3f_combo", debounce_clock, combo_debounce_seconds, now):
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
                                    tfi = classify_intent("three_fingers_up", app_context)
                                    if tfi.name == "volume_step":
                                        if _debounced("vol_step", debounce_clock, 0.08, now):
                                            execute_action("volume_down" if dy > 0 else "volume_up")
                                            display_action = "VOL_DOWN" if dy > 0 else "VOL_UP"
                                    else:
                                        scale = (browser_scroll_scale if app_context == "browser"
                                                 else code_scroll_scale if app_context == "code_editor"
                                                 else default_scroll_scale)
                                        pyautogui.scroll(int(-dy * scale))
                                        display_action = tfi.label
                                    scroll_last_active = now
                                scroll_prev_y = palm_center[1]
                    else:
                        if scroll_mode and now - scroll_last_active > scroll_inactivity_timeout:
                            scroll_mode    = False
                            scroll_prev_y  = None
                            display_action = "SCROLL_OFF"

                    if gesture == "thumbs_up":
                        if combo_intent and _debounced("tu_combo", debounce_clock, combo_debounce_seconds, now):
                            execute_action(combo_intent.action)
                            display_action = combo_intent.label
                        elif _debounced("thumbs_up", debounce_clock, volume_step_debounce, now):
                            ti = classify_intent("thumbs_up", app_context)
                            execute_action(ti.action)
                            display_action = ti.label
                    elif gesture == "thumbs_down":
                        if combo_intent and _debounced("td_combo", debounce_clock, combo_debounce_seconds, now):
                            execute_action(combo_intent.action)
                            display_action = combo_intent.label
                        elif _debounced("thumbs_down", debounce_clock, volume_step_debounce, now):
                            ti = classify_intent("thumbs_down", app_context)
                            execute_action(ti.action)
                            display_action = ti.label

            else:
                gesture     = "none"
                confidence  = 0.0
                combo_ready = False
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
                    display_action = "SCROLL_OFF"

            display_gesture = f"{gesture} ({confidence:.2f})" if gesture != "none" else "none"

            if gesture != "none":
                _i = classify_intent(gesture, app_context)
                gesture_logger.log(
                    gesture=gesture, confidence=confidence,
                    palm_span=float(details.get("palm_span", 0.0)),
                    size_conf=float(details.get("size_conf", 0.0)),
                    context=app_context, app_name=app_name,
                    intent=_i.name, action=_i.action, label=_i.label,
                    modifier_active=modifier_active, combo=combo_ready, fps=fps,
                )

            fps_counter += 1
            elapsed = now - fps_window_start
            if elapsed >= 0.5:
                fps              = fps_counter / elapsed
                fps_counter      = 0
                fps_window_start = now

            if show_overlay:
                draw_overlay(
                    frame=frame, app_name=app_name, context=app_context,
                    gesture_label=display_gesture, action_label=display_action,
                    paused=paused, scroll_mode=scroll_mode, fps=fps,
                    modifier_gesture=modifier_gesture,
                    modifier_active=modifier_active,
                    combo_ready=combo_ready,
                )
                if paused:
                    cv2.putText(frame, "PAUSED",
                                (frame.shape[1] // 2 - 70, frame.shape[0] // 2),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.3, COLORS["red"], 3, cv2.LINE_AA)
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
        logger.info("Stopped.")


def main():
    parser = argparse.ArgumentParser(description="Gesture Control System")
    parser.add_argument("--config",          default="config.yaml")
    parser.add_argument("--no-overlay",      action="store_true")
    parser.add_argument("--list-gestures",   action="store_true")
    parser.add_argument("--skip-validation", action="store_true")
    args = parser.parse_args()
    if args.list_gestures:
        print_gesture_reference(args.config)
        return
    run(args.config, no_overlay=args.no_overlay, skip_validation=args.skip_validation)


if __name__ == "__main__":
    main()
