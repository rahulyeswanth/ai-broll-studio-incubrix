import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]          # project root (frontend.py, ui_helpers.py)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
