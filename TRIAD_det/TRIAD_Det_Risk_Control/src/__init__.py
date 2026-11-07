"""TRIAD-Det risk-control study. Makes the shared base-pipeline package `triad` importable."""
import sys
from pathlib import Path

_ROOT = str(Path(__file__).resolve().parents[2])   # TRIAD_det/
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
