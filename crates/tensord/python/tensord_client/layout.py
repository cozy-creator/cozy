"""A venv's layout on this OS, as cozy_platform::paths spells it: `bin/python` and
`lib/python3.X/site-packages`, or `Scripts\\python.exe` and `Lib\\site-packages` on Windows."""

from __future__ import annotations

import sys
from pathlib import Path

WINDOWS = sys.platform == "win32"


def interpreter(venv: Path) -> Path:
    return venv / "Scripts" / "python.exe" if WINDOWS else venv / "bin" / "python"


def site_packages(venv: Path) -> list[Path]:
    return [venv / "Lib" / "site-packages"] if WINDOWS else sorted(venv.glob("lib/python*/site-packages"))
