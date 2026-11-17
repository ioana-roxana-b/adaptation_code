import os
import sys

# The pipeline modules in src/ import each other by top-level name (as in main.py)
_SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
