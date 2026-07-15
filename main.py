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
    "green": (50, 220, 50),
    "yellow": (30, 220, 220),
    "red": (20, 20, 255),
    "blue": (255, 160, 30),
    "white": (245, 245, 245),
    "bg": (22, 22, 22),
}


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


def _classify_context(app_key: str | None, title: str, proc: str) -> str:
    """Map foreground app info into coarse contexts without expensive logic."""
    text = f"{app_key or ''} {title} {proc}".lower()

    browser_tokens = ("chrome", "firefox", "edge", "brave", "opera", "browser")
    video_tokens = ("vlc", "youtube", "netflix", "potplayer", "mpc", "media player", "video")
    code_tokens = ("vscode", "visual studio", "code", "pycharm", "intellij", "sublime", "notepad++")

    if any(t in text for t in browser_tokens):
        return "browser"
    if any(t in text for t in video_tokens):
        return "video_player"
    if any(t in text for t in code_tokens):
        return "code_editor"
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
    put(f"Action: {action_label}", COLORS["yellow"])
    put(f"Scroll Mode: {'ON' if scroll_mode else 'OFF'}")
    put(f"Status: {'PAUSED' if paused else 'ACTIVE'}", COLORS["red"] if paused else COLORS["green"])
    put(f"FPS: {fps:.1f}", COLORS["white"])
    put("Q = Quit | P = Manual Pause", COLORS["white"], 0.5)


def print_gesture_reference():
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


def run(config_path: str, no_overlay: bool = False):
    config = load_config(config_path)
    settings = config.get("settings", {})
    control = settings.get("control", {})
    mouse_cfg = settings.get("mouse_control", {})

    camera_idx = int(settings.get("camera_index", 0))
    det_conf = float(settings.get("detection_confidence", 0.65))
    track_conf = float(settings.get("tracking_confidence", 0.65))
    show_overlay = bool(settings.get("show_overlay", True)) and not no_overlay
    startup_delay = float(settings.get("startup_delay", 1.0))
    app_refresh_seconds = float(settings.get("app_refresh_seconds", 0.2))

    # Stability / usability settings
    conf_threshold = float(control.get("gesture_confidence_threshold", 0.72))
    gesture_stability_seconds = float(control.get("gesture_stability_seconds", 0.08))
    min_palm_span = float(control.get("min_palm_span", 0.075))
    debounce_seconds = float(control.get("debounce_seconds", 0.25))
    pause_toggle_debounce = float(control.get("pause_toggle_debounce", 0.8))
    pinch_click_hold = float(control.get("pinch_click_hold_seconds", 0.2))
    pinch_drag_hold = float(control.get("pinch_drag_hold_seconds", 0.45))
    pinch_break_grace = float(control.get("pinch_break_grace_seconds", 0.12))
    drag_release_grace = float(control.get("drag_release_grace_seconds", 0.18))
    primary_click_gesture = str(control.get("primary_click_gesture", "pointing_up")).strip().lower()
    if primary_click_gesture not in {"pointing_up", "pinch"}:
        primary_click_gesture = "pointing_up"
    two_finger_stable = float(control.get("two_finger_tap_stable_seconds", 0.15))
    scroll_inactivity_timeout = float(control.get("scroll_mode_inactivity_seconds", 1.0))
    scroll_motion_threshold = float(control.get("scroll_motion_threshold", 0.01))
    browser_swipe_threshold = float(control.get("browser_pinch_swipe_threshold", 0.12))
    video_seek_threshold = float(control.get("video_seek_horizontal_threshold", 0.06))
    video_seek_debounce = float(control.get("video_seek_debounce_seconds", 0.35))
    volume_step_debounce = float(control.get("volume_step_debounce_seconds", 0.22))

    # Cursor smoothing (EMA)
    ema_alpha = float(mouse_cfg.get("ema_alpha", 0.35))
    ema_alpha = max(0.05, min(1.0, ema_alpha))
    drag_ema_alpha = float(mouse_cfg.get("drag_ema_alpha", max(0.12, ema_alpha * 0.75)))
    drag_ema_alpha = max(0.05, min(1.0, drag_ema_alpha))
    mouse_deadzone = float(mouse_cfg.get("deadzone_px", 1.0))
    reverse_horizontal = bool(
        mouse_cfg.get(
            "reverse_horizontal_motion",
            mouse_cfg.get("mirror_sideways", mouse_cfg.get("invert_x", False)),
        )
    )
    reverse_vertical = bool(mouse_cfg.get("reverse_vertical_motion", mouse_cfg.get("invert_y", False)))

    browser_scroll_scale = float(control.get("browser_scroll_scale", 1400.0))
    default_scroll_scale = float(control.get("default_scroll_scale", 1000.0))
    code_scroll_scale = float(control.get("code_scroll_scale", 900.0))

    logger.info("Starting Gesture Control | config=%s camera=%s", config_path, camera_idx)

    if not PYAUTOGUI_AVAILABLE:
        logger.error("pyautogui is required for virtual mouse control.")
        sys.exit(1)

    pyautogui.FAILSAFE = False
    screen_size = pyautogui.size()

    cap = cv2.VideoCapture(camera_idx)
    if not cap.isOpened():
        logger.error("Cannot open camera index %s", camera_idx)
        sys.exit(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        model_complexity=0,  # lower compute for better FPS stability
        min_detection_confidence=det_conf,
        min_tracking_confidence=track_conf,
    )

    # Runtime states
    paused = False
    dragging = False
    scroll_mode = False

    ema_pos = None
    debounce_clock: dict[str, float] = {}

    pinch_start = None
    pinch_last_seen = 0.0
    pinch_armed_click = False
    pinch_swipe_done = False
    pinch_start_x = None

    two_finger_start = None
    two_finger_fired = False

    scroll_prev_y = None
    scroll_last_active = 0.0
    open_palm_prev_x = None
    lost_hand_since = None

    # Gesture stabilization state (filters short misclassifications).
    last_raw_gesture = "none"
    raw_gesture_since = time.time()

    app_name = "Unknown"
    app_context = "default"
    last_app_refresh = 0.0
    title = ""
    proc = ""

    display_gesture = "none"
    display_action = "-"

    fps = 0.0
    fps_counter = 0
    fps_window_start = time.time()

    if startup_delay > 0:
        logger.info("Starting in %.1f seconds...", startup_delay)
        time.sleep(startup_delay)

    while True:
        ret, frame = cap.read()
        if not ret:
            logger.warning("Failed to read frame")
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)
        now = time.time()

        # Refresh active app context at a fixed interval (low overhead, no lag spikes).
        if now - last_app_refresh >= app_refresh_seconds:
            title, proc = get_active_window_info()
            app_key = match_app_config(title, proc, config.get("apps", {}))
            app_name = app_key or (title[:40] if title else "Unknown")
            app_context = _classify_context(app_key, title, proc)
            last_app_refresh = now

        gesture = "none"
        confidence = 0.0
        details = {}

        if results.multi_hand_landmarks:
            lost_hand_since = None
            hand_lm = results.multi_hand_landmarks[0]
            landmarks = hand_lm.landmark
            raw_gesture, confidence, details = detect_gesture(landmarks, return_details=True)

            if show_overlay:
                mp_drawing.draw_landmarks(
                    frame,
                    hand_lm,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style(),
                )

            palm_span = float(details.get("palm_span", 0.0))
            size_conf = float(details.get("size_conf", 1.0))
            # Lower confidence requirement for farther hands while keeping a safe floor.
            effective_conf_threshold = conf_threshold * max(0.62, size_conf)
            if confidence < effective_conf_threshold or palm_span < min_palm_span:
                raw_gesture = "none"
                confidence = 0.0

            if raw_gesture != last_raw_gesture:
                last_raw_gesture = raw_gesture
                raw_gesture_since = now

            # Keep cursor responsive on open_palm, but stabilize action gestures.
            if raw_gesture in ("open_palm", "none"):
                gesture = raw_gesture
            elif now - raw_gesture_since >= gesture_stability_seconds:
                gesture = raw_gesture
            else:
                gesture = "none"

            # Closed fist pause toggle should work even while paused.
            if gesture == "closed_fist" and _debounced("pause_toggle", debounce_clock, pause_toggle_debounce, now):
                paused = not paused
                display_action = "PAUSED" if paused else "RESUMED"
                if dragging:
                    pyautogui.mouseUp()
                    dragging = False
                scroll_mode = False
                pinch_start = None
                pinch_armed_click = False
                pinch_swipe_done = False
                pinch_start_x = None
                two_finger_start = None
                two_finger_fired = False
                scroll_prev_y = None
                open_palm_prev_x = None
                ema_pos = None
                pinch_last_seen = 0.0
                lost_hand_since = None

            if not paused:
                pointer_norm = details.get("index_tip")
                palm_center = details.get("palm_center")

                # 1) Open palm => cursor movement (EMA smoothing)
                if gesture == "open_palm" and pointer_norm:
                    ema_pos = _move_cursor_ema(
                        pointer_norm=pointer_norm,
                        screen_size=screen_size,
                        ema_pos=ema_pos,
                        alpha=ema_alpha,
                        deadzone_px=mouse_deadzone,
                        reverse_horizontal=reverse_horizontal,
                        reverse_vertical=reverse_vertical,
                    )

                    # Video context: open palm + horizontal move => seek
                    if app_context == "video_player":
                        if open_palm_prev_x is not None:
                            dx = pointer_norm[0] - open_palm_prev_x
                            if abs(dx) >= video_seek_threshold and _debounced("video_seek", debounce_clock, video_seek_debounce, now):
                                execute_action("key:right" if dx > 0 else "key:left")
                                display_action = "SEEK_FORWARD" if dx > 0 else "SEEK_BACK"
                        open_palm_prev_x = pointer_norm[0]
                else:
                    open_palm_prev_x = None

                # 2 + 3) Primary click gesture => click(hold) / drag(hold longer)
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
                        pinch_start = now
                        pinch_start_x = pointer_norm[0]
                        pinch_armed_click = False
                        pinch_swipe_done = False

                    hold_time = now - pinch_start
                    if hold_time >= pinch_click_hold:
                        pinch_armed_click = True

                    # Browser context: swipe nav is only enabled for pinch mode to avoid accidental triggers.
                    if (
                        app_context == "browser"
                        and primary_click_gesture == "pinch"
                        and not pinch_swipe_done
                        and pinch_start_x is not None
                    ):
                        dx = pointer_norm[0] - pinch_start_x
                        if abs(dx) >= browser_swipe_threshold and _debounced("browser_nav", debounce_clock, 0.45, now):
                            execute_action("key:alt+right" if dx > 0 else "key:alt+left")
                            display_action = "FORWARD" if dx > 0 else "BACK"
                            pinch_swipe_done = True

                    if hold_time >= pinch_drag_hold and not dragging and not pinch_swipe_done:
                        pyautogui.mouseDown()
                        dragging = True
                        display_action = "DRAG_START"

                else:
                    pinch_briefly_lost = pinch_start is not None and (now - pinch_last_seen) <= pinch_break_grace

                    if pinch_briefly_lost and pointer_norm and dragging:
                        # Keep drag smooth if pinch briefly drops due tracking jitter.
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
                            dragging = False
                            display_action = "DROP"
                        elif pinch_armed_click and not pinch_swipe_done and _debounced("left_click", debounce_clock, debounce_seconds, now):
                            execute_action("click")
                            display_action = "LEFT_CLICK"
                        pinch_start = None
                        pinch_armed_click = False
                        pinch_swipe_done = False
                        pinch_start_x = None

                # 4) Two-finger tap => right click (stable 150ms)
                if gesture == "two_finger_tap":
                    if two_finger_start is None:
                        two_finger_start = now
                        two_finger_fired = False
                    if (
                        not two_finger_fired
                        and now - two_finger_start >= two_finger_stable
                        and _debounced("right_click", debounce_clock, debounce_seconds, now)
                    ):
                        execute_action("right_click")
                        display_action = "CONTEXT_MENU" if app_context == "code_editor" else "RIGHT_CLICK"
                        two_finger_fired = True
                else:
                    two_finger_start = None
                    two_finger_fired = False

                # 5) Three-fingers up => scroll mode
                if gesture == "three_fingers_up" and palm_center:
                    if not scroll_mode:
                        scroll_mode = True
                        scroll_prev_y = palm_center[1]
                    scroll_last_active = now

                    if scroll_prev_y is not None:
                        dy = palm_center[1] - scroll_prev_y
                        if abs(dy) >= scroll_motion_threshold:
                            if app_context == "video_player":
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
                                display_action = "SCROLL"
                            scroll_last_active = now
                        scroll_prev_y = palm_center[1]
                else:
                    if scroll_mode and now - scroll_last_active > scroll_inactivity_timeout:
                        scroll_mode = False
                        scroll_prev_y = None
                        display_action = "SCROLL_MODE_OFF"

                # Dedicated thumb volume gestures.
                if gesture == "thumbs_up" and _debounced("volume_up_gesture", debounce_clock, volume_step_debounce, now):
                    execute_action("volume_up")
                    display_action = "VOLUME_UP"
                elif gesture == "thumbs_down" and _debounced("volume_down_gesture", debounce_clock, volume_step_debounce, now):
                    execute_action("volume_down")
                    display_action = "VOLUME_DOWN"

        else:
            # No hand: cleanup states and safely stop drag
            gesture = "none"
            confidence = 0.0
            if dragging:
                if lost_hand_since is None:
                    lost_hand_since = now
                elif now - lost_hand_since >= drag_release_grace:
                    pyautogui.mouseUp()
                    dragging = False
                    display_action = "DROP"
            pinch_start = None
            pinch_last_seen = 0.0
            pinch_armed_click = False
            pinch_swipe_done = False
            pinch_start_x = None
            two_finger_start = None
            two_finger_fired = False
            open_palm_prev_x = None

            if scroll_mode and now - scroll_last_active > scroll_inactivity_timeout:
                scroll_mode = False
                scroll_prev_y = None
                display_action = "SCROLL_MODE_OFF"

        display_gesture = f"{gesture} ({confidence:.2f})" if gesture != "none" else "none"

        # FPS tracking
        fps_counter += 1
        elapsed = now - fps_window_start
        if elapsed >= 0.5:
            fps = fps_counter / elapsed
            fps_counter = 0
            fps_window_start = now

        # Render and key handling
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
                paused = not paused
                display_action = "PAUSED" if paused else "RESUMED"

    cap.release()
    cv2.destroyAllWindows()
    hands.close()
    logger.info("Gesture Control stopped")


def main():
    parser = argparse.ArgumentParser(description="Hand Gesture Control System")
    parser.add_argument("--config", default="config.yaml", help="Path to config YAML file")
    parser.add_argument("--no-overlay", action="store_true", help="Disable camera overlay window")
    parser.add_argument("--list-gestures", action="store_true", help="Print core gesture reference and exit")
    args = parser.parse_args()

    if args.list_gestures:
        print_gesture_reference()
        return
    run(args.config, no_overlay=args.no_overlay)


if __name__ == "__main__":
    main()
