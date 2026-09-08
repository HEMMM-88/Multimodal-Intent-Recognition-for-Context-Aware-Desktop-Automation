"""
gesture_logger.py
Structured CSV logging of every gesture event + outcome.

Writes one row per gesture classification decision to a CSV file so the
session can be analysed afterward for:
  - Per-gesture accuracy / false-positive rate
  - Context detection correctness
  - Action hit rate
  - FPS / latency over time

CSV schema
----------
timestamp_iso   : ISO-8601 wall-clock time of the event
elapsed_s       : seconds since logger was created (float, for easy time-series plotting)
gesture         : detected gesture label ("open_palm", "none", etc.)
confidence      : classifier confidence (0.0 – 1.0)
palm_span       : normalised palm span (proxy for hand distance from camera)
size_conf       : distance-adjusted confidence multiplier
context         : app context bucket ("browser", "default", …)
app_name        : foreground window title / app key
intent          : resolved Intent.name
action          : resolved Intent.action (empty string if None)
label           : HUD label shown to user
modifier_active : whether the modifier (second) hand was active (0 or 1)
combo           : whether a two-hand combo fired (0 or 1)
fps             : instantaneous FPS at the time of logging

Usage
-----
    from gesture_logger import GestureLogger
    logger = GestureLogger()            # starts writing to gesture_log_<timestamp>.csv
    logger.log(...)                     # call once per gesture decision
    logger.close()                      # flush and close (also called by __del__)
    logger.print_summary()              # print per-gesture accuracy table to stdout
"""

from __future__ import annotations

import csv
import time
from datetime import datetime, timezone
from pathlib import Path


# Column names in order — change here and the writer stays in sync automatically.
_COLUMNS = [
    "timestamp_iso",
    "elapsed_s",
    "gesture",
    "confidence",
    "palm_span",
    "size_conf",
    "context",
    "app_name",
    "intent",
    "action",
    "label",
    "modifier_active",
    "combo",
    "fps",
]


class GestureLogger:
    """
    Writes structured gesture event data to a rotating CSV file.

    Parameters
    ----------
    log_dir : path to the directory where CSV files are written.
              Defaults to the same directory as this module.
    enabled : set False to make all calls no-ops without changing call sites.
    """

    def __init__(self, log_dir: str | Path | None = None, enabled: bool = True):
        self.enabled = enabled
        self._start_time = time.time()
        self._file = None
        self._writer = None
        self._row_count = 0
        # For per-gesture summary stats: gesture -> [total, fired_action]
        self._stats: dict[str, list[int]] = {}

        if not enabled:
            return

        log_dir = Path(log_dir) if log_dir else Path(__file__).resolve().parent
        log_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = log_dir / f"gesture_log_{timestamp}.csv"

        self._file = open(log_path, "w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._file, fieldnames=_COLUMNS)
        self._writer.writeheader()
        self._log_path = log_path

    def log(
        self,
        gesture: str,
        confidence: float,
        palm_span: float,
        size_conf: float,
        context: str,
        app_name: str,
        intent: str,
        action: str | None,
        label: str,
        modifier_active: bool,
        combo: bool,
        fps: float,
    ) -> None:
        """Record one gesture classification event."""
        if not self.enabled or self._writer is None:
            return

        now = time.time()
        row = {
            "timestamp_iso":  datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "elapsed_s":      round(now - self._start_time, 3),
            "gesture":        gesture,
            "confidence":     round(confidence, 4),
            "palm_span":      round(palm_span, 4),
            "size_conf":      round(size_conf, 4),
            "context":        context,
            "app_name":       app_name,
            "intent":         intent,
            "action":         action or "",
            "label":          label,
            "modifier_active":int(modifier_active),
            "combo":          int(combo),
            "fps":            round(fps, 1),
        }
        self._writer.writerow(row)
        self._row_count += 1

        # Update in-memory summary stats.
        if gesture != "none":
            stats = self._stats.setdefault(gesture, [0, 0])
            stats[0] += 1
            if action:
                stats[1] += 1

        # Flush every 50 rows to keep data on disk if the process crashes.
        if self._row_count % 50 == 0:
            self._file.flush()

    def print_summary(self) -> None:
        """Print a per-gesture event summary table to stdout."""
        if not self._stats:
            print("GestureLogger: no gesture events recorded.")
            return

        print("\n── Gesture Session Summary ──────────────────────────")
        print(f"{'Gesture':<22} {'Events':>7} {'Actions fired':>14} {'Action rate':>12}")
        print("-" * 58)
        total_events = 0
        total_actions = 0
        for gesture in sorted(self._stats):
            events, actions = self._stats[gesture]
            rate = actions / events * 100 if events else 0.0
            print(f"{gesture:<22} {events:>7} {actions:>14} {rate:>11.1f}%")
            total_events  += events
            total_actions += actions
        print("-" * 58)
        overall = total_actions / total_events * 100 if total_events else 0.0
        print(f"{'TOTAL':<22} {total_events:>7} {total_actions:>14} {overall:>11.1f}%")
        print()
        if self._file:
            print(f"Full log saved to: {self._log_path}")
        print()

    def close(self) -> None:
        """Flush and close the CSV file."""
        if self._file and not self._file.closed:
            self._file.flush()
            self._file.close()

    def __del__(self):
        self.close()
