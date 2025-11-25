"""Entry point for launching the Astraia GUI backend with uvicorn."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"

if SRC_DIR.exists():
    sys.path.insert(0, str(SRC_DIR))

from astraia_gui.backend import create_app  # noqa: E402

app = create_app()

