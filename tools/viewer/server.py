#!/usr/bin/env python3
"""Entry point moved to tools/server.py — this file delegates to it."""
import runpy
import sys
from pathlib import Path

sys.argv[0] = str(Path(__file__).parent.parent / "server.py")
runpy.run_path(sys.argv[0], run_name="__main__")
