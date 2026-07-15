"""
install_startup.py
Registers (or removes) Gesture Control as a Windows startup application.
Adds a shortcut to the Windows Startup folder so it runs on login.

Usage:
    python install_startup.py install    # Add to startup
    python install_startup.py remove     # Remove from startup
    python install_startup.py status     # Check if installed
"""

import os
import sys
import shutil
from pathlib import Path


STARTUP_FOLDER = Path(os.environ.get("APPDATA", "")) / "Microsoft" / "Windows" / "Start Menu" / "Programs" / "Startup"
SHORTCUT_NAME = "GestureControl.bat"
APP_NAME = "Gesture Control"


def get_script_dir() -> Path:
    return Path(__file__).resolve().parent


def get_shortcut_path() -> Path:
    return STARTUP_FOLDER / SHORTCUT_NAME


def create_bat_launcher(script_dir: Path) -> Path:
    """Create a .bat launcher that runs main.py with pythonw (no console)."""
    python_exe = sys.executable.replace("python.exe", "pythonw.exe")
    if not Path(python_exe).exists():
        python_exe = sys.executable  # fallback

    bat_content = f"""@echo off
cd /d "{script_dir}"
start "" "{python_exe}" "{script_dir / 'main.py'}" --config "{script_dir / 'config.yaml'}"
"""
    bat_path = script_dir / "launch_gesture_control.bat"
    bat_path.write_text(bat_content, encoding="utf-8")
    return bat_path


def install():
    script_dir = get_script_dir()
    shortcut_path = get_shortcut_path()

    if not STARTUP_FOLDER.exists():
        print(f"ERROR: Startup folder not found: {STARTUP_FOLDER}")
        sys.exit(1)

    bat_path = create_bat_launcher(script_dir)
    shutil.copy2(bat_path, shortcut_path)

    print(f"✅ {APP_NAME} installed to startup!")
    print(f"   Startup shortcut: {shortcut_path}")
    print(f"   Launcher script:  {bat_path}")
    print()
    print("It will launch automatically when you log into Windows.")
    print("To remove: python install_startup.py remove")


def remove():
    shortcut_path = get_shortcut_path()
    if shortcut_path.exists():
        shortcut_path.unlink()
        print(f"✅ {APP_NAME} removed from startup.")
    else:
        print(f"ℹ️  {APP_NAME} was not in startup (nothing to remove).")


def status():
    shortcut_path = get_shortcut_path()
    if shortcut_path.exists():
        print(f"✅ {APP_NAME} IS registered as a startup application.")
        print(f"   Path: {shortcut_path}")
    else:
        print(f"❌ {APP_NAME} is NOT registered as a startup application.")
        print(f"   Run: python install_startup.py install")


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in ("install", "remove", "status"):
        print(__doc__)
        sys.exit(0)

    cmd = sys.argv[1]
    if cmd == "install":
        install()
    elif cmd == "remove":
        remove()
    elif cmd == "status":
        status()


if __name__ == "__main__":
    main()
