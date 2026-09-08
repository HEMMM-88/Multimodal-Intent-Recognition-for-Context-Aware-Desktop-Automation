"""
app_detector.py
Detects the currently active foreground window on Windows.
"""

import logging
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

try:
    import win32gui
    import win32process
    import psutil
    WIN32_AVAILABLE = True
except ImportError:
    WIN32_AVAILABLE = False
    logger.warning("win32gui/psutil not available — app detection disabled")


def get_active_window_info() -> Tuple[str, str]:
    """Return (window_title, process_name) of the focused window, or ('', '') on failure."""
    if not WIN32_AVAILABLE:
        return ("", "")
    try:
        hwnd  = win32gui.GetForegroundWindow()
        title = win32gui.GetWindowText(hwnd)
        _, pid = win32process.GetWindowThreadProcessId(hwnd)
        try:
            proc_name = psutil.Process(pid).name()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            proc_name = ""
        return (title, proc_name)
    except Exception as e:
        logger.debug("get_active_window_info error: %s", e)
        return ("", "")


def match_app_config(title: str, proc_name: str, apps_config: dict) -> Optional[str]:
    """Return the matching app key from the apps config section, or None."""
    title_lower = title.lower()
    proc_lower  = proc_name.lower()
    for app_key, app_data in apps_config.items():
        for wt in app_data.get("window_titles", []):
            if wt.lower() in title_lower:
                return app_key
        if app_key.lower() in proc_lower:
            return app_key
    return None
