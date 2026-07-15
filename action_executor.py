"""
action_executor.py
Maps gesture action strings to actual system actions on Windows.
"""

import logging
import time

try:
    import pyautogui
    PYAUTOGUI_AVAILABLE = True
except ImportError:
    PYAUTOGUI_AVAILABLE = False
    logging.warning("pyautogui not available — keyboard/mouse actions disabled")

try:
    from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
    from ctypes import cast, POINTER
    from comtypes import CLSCTX_ALL
    PYCAW_AVAILABLE = True
except ImportError:
    PYCAW_AVAILABLE = False
    logging.warning("pycaw not available — native volume control disabled")

logger = logging.getLogger(__name__)

# User-facing action aliases mapped to internal actions/hotkeys.
ACTION_ALIASES = {
    # Generic media/navigation aliases
    "stop_media": "media_stop",
    "next_item": "media_next",
    "previous_item": "media_prev",

    # YouTube-style aliases
    "play_pause_video": "key:k",
    "full_screen_toggle": "key:f",
    "forward_10_seconds": "key:l",
    "rewind_10_seconds": "key:j",
    "volume_increase": "volume_up",
    "volume_decrease": "volume_down",

    # PowerPoint aliases
    "start_slideshow": "key:f5",
    "end_slideshow": "key:escape",
    "next_slide": "slide_next",
    "previous_slide": "slide_prev",
    "pointer_toggle": "key:ctrl+l",

    # Browser aliases
    "refresh_page": "key:f5",
    "next_tab": "key:ctrl+tab",
    "previous_tab": "key:ctrl+shift+tab",
    "new_tab": "key:ctrl+t",
    "close_tab": "key:ctrl+w",

    # Spotify-style aliases
    "play_pause_music": "media_play_pause",
    "stop_music": "media_stop",
    "next_track": "media_next",
    "previous_track": "media_prev",

    # PDF/document aliases
    "zoom_toggle": "key:ctrl+0",
    "next_page": "key:pagedown",
    "previous_page": "key:pageup",
    "zoom_in": "key:ctrl+=",
    "zoom_out": "key:ctrl+-",
}


def _resolve_action_alias(action: str) -> str:
    """Resolve alias chains safely."""
    resolved = action
    for _ in range(8):
        nxt = ACTION_ALIASES.get(resolved)
        if not nxt:
            return resolved
        resolved = nxt
    return resolved

# ── Volume helpers ──────────────────────────────────────────────────────────

def _get_volume_interface():
    """Get Windows audio endpoint volume interface via pycaw."""
    if not PYCAW_AVAILABLE:
        return None
    try:
        devices = AudioUtilities.GetSpeakers()
        interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
        volume = cast(interface, POINTER(IAudioEndpointVolume))
        return volume
    except Exception as e:
        logger.error(f"Could not get volume interface: {e}")
        return None


_volume_interface = None  # lazy init


def _volume_ctrl():
    global _volume_interface
    if _volume_interface is None:
        _volume_interface = _get_volume_interface()
    return _volume_interface


def _change_volume(delta: float):
    """Change master volume by delta (-1.0 to 1.0)."""
    vol = _volume_ctrl()
    if vol:
        current = vol.GetMasterVolumeLevelScalar()
        new_vol = max(0.0, min(1.0, current + delta))
        vol.SetMasterVolumeLevelScalar(new_vol, None)
        logger.info(f"Volume: {int(new_vol * 100)}%")
    else:
        # Fallback to keyboard
        if delta > 0:
            pyautogui.press("volumeup")
        else:
            pyautogui.press("volumedown")


def _mute_toggle():
    vol = _volume_ctrl()
    if vol:
        current = vol.GetMute()
        vol.SetMute(not current, None)
        logger.info(f"Mute: {'ON' if not current else 'OFF'}")
    else:
        pyautogui.press("volumemute")


# ── Screenshot helper ────────────────────────────────────────────────────────

def _take_screenshot():
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    path = f"screenshot_{timestamp}.png"
    if PYAUTOGUI_AVAILABLE:
        img = pyautogui.screenshot()
        img.save(path)
        logger.info(f"Screenshot saved: {path}")


# ── Key press helper ─────────────────────────────────────────────────────────

def _press_key(key_str: str):
    """
    Press a key combination. Supports formats like:
      'space', 'ctrl+z', 'alt+left', 'ctrl+shift+t'
    """
    if not PYAUTOGUI_AVAILABLE:
        return
    parts = key_str.lower().split("+")
    if len(parts) == 1:
        pyautogui.press(parts[0])
    else:
        pyautogui.hotkey(*parts)
    logger.debug(f"Key pressed: {key_str}")


# ── Main dispatcher ──────────────────────────────────────────────────────────

def execute_action(action: str):
    """Execute an action string."""
    if not action or action == "nothing":
        return

    original_action = action
    action = _resolve_action_alias(action)

    if original_action == action:
        logger.info(f"Executing action: {action}")
    else:
        logger.info(f"Executing action: {action} (from alias: {original_action})")

    if not PYAUTOGUI_AVAILABLE and not action.startswith("volume") and action != "mute":
        logger.warning("pyautogui not installed — skipping action")
        return

    try:
        match action:
            # ── Scroll ──
            case "scroll_up":
                pyautogui.scroll(5)
            case "scroll_down":
                pyautogui.scroll(-5)

            # ── Mouse ──
            case "click":
                pyautogui.click()
            case "right_click":
                pyautogui.rightClick()

            # ── Media ──
            case "media_play_pause":
                pyautogui.press("playpause")
            case "media_next":
                pyautogui.press("nexttrack")
            case "media_prev":
                pyautogui.press("prevtrack")
            case "media_stop":
                pyautogui.press("stop")

            # ── Volume ──
            case "volume_up":
                _change_volume(0.05)
            case "volume_down":
                _change_volume(-0.05)
            case "mute":
                _mute_toggle()

            # ── Slides ──
            case "slide_next":
                pyautogui.press("right")
            case "slide_prev":
                pyautogui.press("left")

            # ── Utilities ──
            case "screenshot":
                _take_screenshot()
            case "show_desktop":
                pyautogui.hotkey("win", "d")

            # ── key:<combo> ──
            case _ if action.startswith("key:"):
                key_combo = action[4:]
                _press_key(key_combo)

            case _:
                logger.warning(f"Unknown action: {action}")

    except Exception as e:
        logger.error(f"Action '{action}' failed: {e}")
