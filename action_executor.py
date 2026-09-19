"""
action_executor.py
Translates action strings into OS-level actions on Windows.
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

ACTION_ALIASES = {
    "stop_media":         "media_stop",
    "next_item":          "media_next",
    "previous_item":      "media_prev",
    "play_pause_video":   "key:k",
    "full_screen_toggle": "key:f",
    "forward_10_seconds": "key:l",
    "rewind_10_seconds":  "key:j",
    "volume_increase":    "volume_up",
    "volume_decrease":    "volume_down",
    "start_slideshow":    "key:f5",
    "end_slideshow":      "key:escape",
    "next_slide":         "slide_next",
    "previous_slide":     "slide_prev",
    "pointer_toggle":     "key:ctrl+l",
    "refresh_page":       "key:f5",
    "next_tab":           "key:ctrl+tab",
    "previous_tab":       "key:ctrl+shift+tab",
    "new_tab":            "key:ctrl+t",
    "close_tab":          "key:ctrl+w",
    "play_pause_music":   "media_play_pause",
    "stop_music":         "media_stop",
    "next_track":         "media_next",
    "previous_track":     "media_prev",
    "zoom_toggle":        "key:ctrl+0",
    "next_page":          "key:pagedown",
    "previous_page":      "key:pageup",
    "zoom_in":            "key:ctrl+=",
    "zoom_out":           "key:ctrl+-",
}


def _resolve_action_alias(action: str) -> str:
    resolved = action
    for _ in range(8):
        nxt = ACTION_ALIASES.get(resolved)
        if not nxt:
            return resolved
        resolved = nxt
    return resolved


def _get_volume_interface():
    if not PYCAW_AVAILABLE:
        return None
    try:
        device = AudioUtilities.GetSpeakers()
        if hasattr(device, "EndpointVolume"):
            return device.EndpointVolume
        interface = device.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
        return cast(interface, POINTER(IAudioEndpointVolume))
    except Exception as e:
        logger.error("Could not get volume interface: %s", e)
        return None


_volume_interface = None


def _volume_ctrl():
    global _volume_interface
    if _volume_interface is None:
        _volume_interface = _get_volume_interface()
    return _volume_interface


def _change_volume(delta: float):
    vol = _volume_ctrl()
    if vol:
        new_vol = max(0.0, min(1.0, vol.GetMasterVolumeLevelScalar() + delta))
        vol.SetMasterVolumeLevelScalar(new_vol, None)
        logger.info("Volume: %d%%", int(new_vol * 100))
    else:
        pyautogui.press("volumeup" if delta > 0 else "volumedown")


def _mute_toggle():
    vol = _volume_ctrl()
    if vol:
        current = vol.GetMute()
        vol.SetMute(not current, None)
        logger.info("Mute: %s", "ON" if not current else "OFF")
    else:
        pyautogui.press("volumemute")


def _take_screenshot():
    from pathlib import Path
    screenshots_dir = Path(__file__).resolve().parent / "screenshots"
    screenshots_dir.mkdir(exist_ok=True)
    path = screenshots_dir / f"screenshot_{time.strftime('%Y%m%d_%H%M%S')}.png"
    if PYAUTOGUI_AVAILABLE:
        pyautogui.screenshot().save(str(path))
        logger.info("Screenshot saved: %s", path)


def _press_key(key_str: str):
    if not PYAUTOGUI_AVAILABLE:
        return
    parts = key_str.lower().split("+")
    if len(parts) == 1:
        pyautogui.press(parts[0])
    else:
        pyautogui.hotkey(*parts)
    logger.debug("Key pressed: %s", key_str)


def execute_action(action: str):
    """Execute an action string."""
    if not action or action == "nothing":
        return
    original = action
    action   = _resolve_action_alias(action)
    logger.info("Action: %s%s", action, f" (alias: {original})" if original != action else "")

    if not PYAUTOGUI_AVAILABLE and not action.startswith("volume") and action != "mute":
        logger.warning("pyautogui not installed — skipping action")
        return

    try:
        match action:
            case "scroll_up":        pyautogui.scroll(5)
            case "scroll_down":      pyautogui.scroll(-5)
            case "scroll_up_fast":   pyautogui.scroll(20)
            case "scroll_down_fast": pyautogui.scroll(-20)
            case "click":            pyautogui.click()
            case "right_click":      pyautogui.rightClick()
            case "media_play_pause": pyautogui.press("playpause")
            case "media_next":       pyautogui.press("nexttrack")
            case "media_prev":       pyautogui.press("prevtrack")
            case "media_stop":       pyautogui.press("stop")
            case "volume_up":        _change_volume(0.05)
            case "volume_down":      _change_volume(-0.05)
            case "mute":             _mute_toggle()
            case "slide_next":       pyautogui.press("right")
            case "slide_prev":       pyautogui.press("left")
            case "screenshot":       _take_screenshot()
            case "show_desktop":     pyautogui.hotkey("win", "d")
            case _ if action.startswith("key:"): _press_key(action[4:])
            case _: logger.warning("Unknown action: %s", action)
    except Exception as e:
        logger.error("Action '%s' failed: %s", action, e)
